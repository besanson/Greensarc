"""Tests for the SARC composition adapter.

These run only when ``sarc-governance`` is installed (the ``sarc`` extra); they
are skipped otherwise so the core test suite stays dependency-free.
"""

from __future__ import annotations

import pytest

sg = pytest.importorskip("sarc_governance")

from green_sarc.adapters.sarc import (  # noqa: E402 - after importorskip by design
    SarcCostCarbonGovernance,
    default_usage_extractor,
    wrap_toolset,
)
from green_sarc.estimator import ColdStartEstimator  # noqa: E402
from green_sarc.state import Budget  # noqa: E402


class _InnerToolset:
    """SARC-compatible async toolset that reports usage and records calls."""

    def __init__(self, tokens: int = 250) -> None:
        self.tokens = tokens
        self.calls: list[str] = []

    async def call_tool(self, name, tool_args, ctx=None, tool=None):
        self.calls.append(name)
        return {"text": f"ran {name}", "usage": {"total_tokens": self.tokens}}


def _governance(budget, cost_model, carbon_model) -> SarcCostCarbonGovernance:
    return SarcCostCarbonGovernance(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
        region="eu-west",
    )


def test_usage_extractor_shapes():
    assert default_usage_extractor({"usage": {"total_tokens": 42}}) == 42.0
    assert (
        default_usage_extractor({"usage": {"prompt_tokens": 10, "completion_tokens": 5}}) == 15.0
    )
    assert default_usage_extractor({"actual_tokens": 7}) == 7.0
    assert default_usage_extractor("no usage here") == 0.0


def test_constraints_target_the_right_sites(cost_model, carbon_model):
    gov = _governance(Budget(1000.0, 100.0), cost_model, carbon_model)
    constraints = gov.constraints()
    by_id = {c.id: c for c in constraints}
    assert by_id["green_sarc.budget_gate"].verif is sg.EnforcementPoint.PAG
    assert by_id["green_sarc.budget_gate"].klass is sg.ConstraintClass.HARD
    assert by_id["green_sarc.cost_carbon_audit"].verif is sg.EnforcementPoint.PAA
    assert by_id["green_sarc.cost_carbon_audit"].klass is sg.ConstraintClass.SOFT


async def test_admitted_call_runs_spends_and_audits(cost_model, carbon_model):
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    gov = _governance(budget, cost_model, carbon_model)
    inner = _InnerToolset(tokens=250)
    gt = wrap_toolset(inner, gov)

    result = await gt.call_tool(
        "chat.completion", {"model": "m", "prompt_tokens": 100, "max_tokens": 200}, None, None
    )

    assert inner.calls == ["chat.completion"]
    assert result["usage"]["total_tokens"] == 250
    assert budget.remaining_tokens() == 10_000.0 - 250.0  # PAA spent the actual
    audit = gov.store.list()
    assert len(audit) == 1
    assert audit[0].admitted is True
    assert audit[0].actual_cost == 250.0


async def test_over_budget_call_is_blocked_by_sarc(cost_model, carbon_model):
    budget = Budget(token_budget=50.0, carbon_ceiling=100.0)  # too small for the forecast
    gov = _governance(budget, cost_model, carbon_model)
    inner = _InnerToolset()
    gt = wrap_toolset(inner, gov)

    with pytest.raises(sg.ConstraintViolation) as exc:
        await gt.call_tool(
            "chat.completion", {"model": "m", "prompt_tokens": 100, "max_tokens": 200}, None, None
        )

    assert exc.value.constraint_id == "green_sarc.budget_gate"
    assert exc.value.point is sg.EnforcementPoint.PAG
    assert inner.calls == []  # the model was never reached
    assert budget.remaining_tokens() == 50.0  # nothing spent
    # The rejection is still audited.
    assert gov.store.list()[-1].admitted is False


async def test_composes_with_a_caller_safety_constraint(cost_model, carbon_model):
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    gov = _governance(budget, cost_model, carbon_model)
    inner = _InnerToolset()
    safety = sg.Constraint(
        id="safety.no_danger",
        klass=sg.ConstraintClass.HARD,
        verif=sg.EnforcementPoint.PAG,
        response=sg.Response.BLOCK,
        predicate=lambda ctx: ctx.get("tool") == "danger",
        description="block the danger tool",
    )
    gt = wrap_toolset(inner, gov, spec=sg.ConstraintSpec([safety]))

    # Safety rule blocks the unsafe tool (Green SARC cost gate untouched).
    with pytest.raises(sg.ConstraintViolation) as exc:
        await gt.call_tool("danger", {"model": "m"}, None, None)
    assert exc.value.constraint_id == "safety.no_danger"

    # A safe, affordable action still runs and is cost-audited.
    await gt.call_tool(
        "chat.completion", {"model": "m", "prompt_tokens": 10, "max_tokens": 10}, None, None
    )
    assert inner.calls == ["chat.completion"]
    assert any(r.admitted for r in gov.store.list())
