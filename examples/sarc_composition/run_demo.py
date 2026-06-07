"""SARC composition demo: Green SARC plugged into a SARC GovernanceToolset.

Run it::

    pip install 'green-sarc[sarc]'
    python examples/sarc_composition/run_demo.py

This shows the *composition* of the two layers on a single governed toolset:
Green SARC's predictive cost/carbon control is expressed as SARC constraints and
enforced at the same four sites as a caller-supplied SARC **safety** rule. One
``GovernanceToolset`` therefore blocks both an over-budget action (Green SARC's
Pre-Action Gate) and an unsafe action (the caller's SARC constraint).

The dependency direction is unchanged: Green SARC's core does not import SARC;
only the adapter (`green_sarc.adapters.sarc`) does.
"""

from __future__ import annotations

import asyncio

try:
    import sarc_governance as sg
except ImportError:  # pragma: no cover
    raise SystemExit("This demo needs sarc-governance:  pip install 'green-sarc[sarc]'")

from green_sarc import Budget, LearnedEstimator, TableCarbonModel, TableCostModel
from green_sarc.adapters.sarc import SarcCostCarbonGovernance, wrap_toolset


class MockToolset:
    """A SARC-compatible async toolset that reports token usage in its result."""

    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        self.calls: list[str] = []

    async def call_tool(self, name, tool_args, ctx=None, tool=None):
        self.calls.append(name)
        return {"text": f"ran {name}", "usage": {"total_tokens": self.tokens}}


def safety_constraint():
    """A caller-supplied SARC safety rule (unrelated to cost): block 'danger'."""
    return sg.Constraint(
        id="safety.no_danger",
        klass=sg.ConstraintClass.HARD,
        verif=sg.EnforcementPoint.PAG,
        response=sg.Response.BLOCK,
        predicate=lambda ctx: ctx.get("tool") == "danger",
        description="Example safety rule, enforced by SARC.",
    )


async def main() -> None:
    cost_model = TableCostModel()
    carbon_model = TableCarbonModel(default_intensity=230.0)
    budget = Budget(token_budget=600.0, carbon_ceiling=5.0, delta=0.05)

    governance = SarcCostCarbonGovernance(
        budget=budget,
        estimator=LearnedEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
        region="eu-west",
    )
    inner = MockToolset(tokens=250)
    # One GovernanceToolset enforcing BOTH safety (caller) and cost/carbon (Green SARC).
    safety_spec = sg.ConstraintSpec([safety_constraint()])
    gt = wrap_toolset(inner, governance, spec=safety_spec)

    print("One SARC GovernanceToolset enforcing safety + Green SARC cost/carbon:\n")

    await gt.call_tool(
        "chat.completion", {"model": "gpt-x", "prompt_tokens": 120, "max_tokens": 180}, None, None
    )
    print(f"  chat.completion -> OK (ran); budget left {budget.remaining_tokens():.0f} tokens")

    try:
        await gt.call_tool(
            "chat.completion",
            {"model": "gpt-x", "prompt_tokens": 120, "max_tokens": 4000},
            None,
            None,
        )
    except sg.ConstraintViolation as exc:
        print(
            f"  chat.completion -> BLOCKED by {exc.constraint_id} at {exc.point.value} "
            f"(Green SARC budget gate)"
        )

    try:
        await gt.call_tool("danger", {"model": "gpt-x"}, None, None)
    except sg.ConstraintViolation as exc:
        print(
            f"  danger          -> BLOCKED by {exc.constraint_id} at {exc.point.value} "
            f"(SARC safety rule)"
        )

    print(f"\n  inner toolset was actually called for: {inner.calls}")
    print("\nGreen SARC audit log (predicted vs actual):")
    for rec in governance.store.list():
        print(
            f"  {rec.action_kind:<16} admitted={rec.admitted!s:<5} "
            f"pred={rec.predicted_cost:7.1f} actual={rec.actual_cost:7.1f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
