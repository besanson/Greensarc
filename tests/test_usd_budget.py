"""USD budget enforcement, parallel to tokens and carbon (audit P1-2)."""

from __future__ import annotations


from green_sarc.estimator import ColdStartEstimator
from green_sarc.forecast import Verdict
from green_sarc.gate import PreActionGate
from green_sarc.governor import ActionOutcome, GateRejected, GreenGovernor
from green_sarc.monitor import ActionTimeMonitor
from green_sarc.pricing import ModelProfile, TableCostModel
from green_sarc.state import Budget, GovernanceContext

from .conftest import make_action


def _usd_cost_model() -> TableCostModel:
    return TableCostModel(
        default_profile=ModelProfile(
            energy_per_token_kwh=1.0e-6,
            usd_per_prompt_token=1.0e-5,
            usd_per_completion_token=2.0e-5,
        )
    )


def test_budget_usd_accounting():
    b = Budget(token_budget=10_000.0, carbon_ceiling=1_000.0, usd_budget=1.0)
    assert b.remaining_usd() == 1.0
    b.commit(0.0, 0.0, 0.0, 0.0, actual_usd=0.4)
    assert abs(b.remaining_usd() - 0.6) < 1e-9
    assert not b.is_usd_exhausted()
    b.spend(0.0, 0.0, usd=0.6)
    assert b.is_usd_exhausted()


def test_no_usd_budget_is_unbounded():
    b = Budget(token_budget=10_000.0, carbon_ceiling=1_000.0)
    assert b.remaining_usd() == float("inf")
    assert not b.is_usd_exhausted()


def test_gate_rejects_on_usd(carbon_model):
    gate = PreActionGate(ColdStartEstimator(_usd_cost_model(), carbon_model))
    # cold-start forecast: prompt 100 + completion 200 -> usd = 100*1e-5 + 200*2e-5 = 0.005
    budget = Budget(token_budget=10_000.0, carbon_ceiling=1_000.0, usd_budget=0.0001)
    decision = gate.evaluate(make_action(prompt=100, max_tokens=200), GovernanceContext(budget))
    assert decision.verdict is Verdict.REJECT
    assert "$" in decision.reason


async def test_governor_enforces_usd_budget(carbon_model):
    cost_model = _usd_cost_model()
    budget = Budget(token_budget=1_000_000.0, carbon_ceiling=1.0e9, usd_budget=0.01)
    gov = GreenGovernor(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
        monitor=ActionTimeMonitor(max_loops=100),
    )

    async def execute(action):
        return ActionOutcome(result="ok", actual_tokens=250.0)

    admitted = 0
    rejected = False
    for _ in range(6):
        try:
            await gov.run_action(make_action(prompt=100, max_tokens=200), execute)
            admitted += 1
        except GateRejected as exc:
            assert "$" in exc.decision.reason  # blocked specifically on USD
            rejected = True
            break

    assert admitted >= 1
    assert rejected  # the USD ceiling eventually blocks
    assert 0.0 < budget.usd_spent <= 0.01 + 1e-9


def test_cli_inspect_reports_usd(tmp_path, capsys):
    from green_sarc.auditor import AuditRecord
    from green_sarc.cli import main
    from green_sarc.stores.jsonl import JSONLAuditStore

    store = JSONLAuditStore(tmp_path / "audit.jsonl")
    store.append(
        AuditRecord(
            action_id="a",
            action_kind="chat.completion",
            model="m",
            region="r",
            predicted_cost=120.0,
            predicted_carbon=0.06,
            confidence=0.7,
            actual_cost=100.0,
            actual_carbon=0.05,
            budget_remaining_tokens=0.0,
            carbon_remaining=0.0,
            carbon_intensity=400.0,
            admitted=True,
            verdict="admit",
            predicted_usd=0.006,
            actual_usd=0.005,
        )
    )
    assert main(["inspect", str(tmp_path / "audit.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "total USD spent" in out
    assert "MAE USD" in out
