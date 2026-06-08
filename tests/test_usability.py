"""Out-of-the-box usability: reference data + the one-call constructor."""

from __future__ import annotations

import pytest

from green_sarc import (
    Action,
    ActionOutcome,
    GateRejected,
    GreenGovernor,
    default_carbon,
    default_pricing,
)
from green_sarc.data import canonical_model_id


def test_default_pricing_has_known_models():
    cost = default_pricing()
    # Known model: priced from its profile, not the fallback.
    usd = cost.usd("gpt-4o", 1000, 1000)
    assert usd == 1000 * 2.5e-6 + 1000 * 1.0e-5
    # Unknown model: falls back to the default profile (no crash).
    assert cost.usd("some-future-model", 100, 100) > 0.0


def test_canonical_model_id_normalises_real_ids():
    assert canonical_model_id("gpt-4o-2024-08-06") == "gpt-4o"
    assert canonical_model_id("gpt-4o-mini-2024-07-18") == "gpt-4o-mini"
    assert canonical_model_id("claude-3-5-sonnet-20241022") == "claude-sonnet"
    assert canonical_model_id("claude-3-haiku-20240307") == "claude-haiku"
    assert canonical_model_id("meta-llama/Llama-3.1-70B-Instruct".lower()) == "llama-3.1-70b"
    assert canonical_model_id("some-unknown") == "some-unknown"


def test_default_pricing_resolves_dated_ids():
    cost = default_pricing()
    # A dated id must price the same as its canonical slug, not the fallback.
    assert cost.usd("gpt-4o-2024-08-06", 1000, 1000) == cost.usd("gpt-4o", 1000, 1000)


def test_with_defaults_bootstrap_from_jsonl(tmp_path):
    from green_sarc.auditor import AuditRecord
    from green_sarc.stores.jsonl import JSONLAuditStore

    log = tmp_path / "audit.jsonl"
    store = JSONLAuditStore(log)
    for prompt, total in [(100, 250.0), (200, 450.0), (300, 650.0)]:
        store.append(
            AuditRecord(
                action_id="a",
                action_kind="chat.completion",
                model="gpt-4o",
                region="us-east-1",
                predicted_cost=0.0,
                predicted_carbon=0.0,
                confidence=0.0,
                actual_cost=total,
                actual_carbon=0.0,
                budget_remaining_tokens=0.0,
                carbon_remaining=0.0,
                carbon_intensity=370.0,
                admitted=True,
                verdict="admit",
                prompt_tokens=prompt,
            )
        )
    gov = GreenGovernor.with_defaults(token_budget=10_000, bootstrap_jsonl=str(log))
    assert (
        gov.estimator.samples(Action(kind="chat.completion", model="gpt-4o", region="us-east-1"))
        == 3
    )  # learned state was rehydrated from the log


def test_default_carbon_has_known_regions():
    carbon = default_carbon()
    assert carbon.carbon_intensity("eu-north-1") == 30.0  # clean grid
    assert carbon.carbon_intensity("ap-southeast-2") == 520.0  # dirty grid
    assert carbon.carbon_intensity("mars-1") == 400.0  # default


def test_with_defaults_builds_a_working_governor():
    gov = GreenGovernor.with_defaults(token_budget=50_000, usd_budget=1.0)
    assert gov.budget.token_budget == 50_000
    assert gov.budget.usd_budget == 1.0
    # The reference tables are wired, so cost/carbon/USD are all live.
    assert gov.cost_model.usd("gpt-4o", 10, 10) > 0.0


async def test_with_defaults_runs_and_governs():
    gov = GreenGovernor.with_defaults(token_budget=10_000, usd_budget=1.0)

    async def execute(action):
        return ActionOutcome(result="hi", actual_tokens=300.0)

    action = Action(
        kind="chat.completion",
        model="gpt-4o",
        region="us-east-1",
        prompt_tokens=100,
        max_tokens=200,
    )
    result = await gov.run_action(action, execute)
    assert result.actual_cost == 300.0
    assert result.audit.actual_usd > 0.0
    assert gov.budget.remaining_tokens() == 10_000 - 300.0


async def test_with_defaults_enforces_token_budget():
    gov = GreenGovernor.with_defaults(token_budget=50)  # tiny

    async def execute(action):
        return ActionOutcome(result="hi", actual_tokens=300.0)

    action = Action(
        kind="chat.completion",
        model="gpt-4o",
        region="us-east-1",
        prompt_tokens=100,
        max_tokens=200,
    )
    with pytest.raises(GateRejected):
        await gov.run_action(action, execute)
