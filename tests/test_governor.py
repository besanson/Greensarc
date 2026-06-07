"""Tests for the GreenGovernor wiring all four sites end to end."""

from __future__ import annotations

import pytest

from green_sarc.estimator import ColdStartEstimator, LearnedEstimator
from green_sarc.governor import ActionOutcome, GateRejected, GreenGovernor
from green_sarc.monitor import ActionTimeMonitor, CircuitTripped
from green_sarc.state import Budget

from .conftest import RecordingRouter, make_action

pytestmark = pytest.mark.asyncio


def _executor(tokens: float):
    async def execute(action):
        return ActionOutcome(result="ok", actual_tokens=tokens)

    return execute


async def test_admitted_action_spends_budget_and_audits(cost_model, carbon_model):
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    gov = GreenGovernor(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
    )
    res = await gov.run_action(make_action(prompt=100, max_tokens=200), _executor(250.0))

    assert res.result == "ok"
    assert res.actual_cost == 250.0
    assert res.actual_carbon == 250.0 * 1.0e-6 * 500.0
    assert budget.remaining_tokens() == 10_000.0 - 250.0
    assert len(gov.store.list()) == 1
    assert gov.store.list()[0].admitted


async def test_gate_rejection_raises_and_audits(cost_model, carbon_model):
    budget = Budget(token_budget=100.0, carbon_ceiling=100.0)  # too small for 300-token worst case
    gov = GreenGovernor(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
    )
    with pytest.raises(GateRejected):
        await gov.run_action(make_action(prompt=100, max_tokens=200), _executor(250.0))

    # Budget untouched; rejection recorded.
    assert budget.remaining_tokens() == 100.0
    rec = gov.store.list()[0]
    assert not rec.admitted
    assert rec.verdict == "reject"
    assert rec.actual_cost == 0.0


async def test_circuit_breaker_trips_on_loop_count(cost_model, carbon_model):
    budget = Budget(token_budget=100_000.0, carbon_ceiling=1_000.0)
    router = RecordingRouter()
    gov = GreenGovernor(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
        monitor=ActionTimeMonitor(max_loops=1),
        router=router,
    )
    # First action runs; second trips the breaker before executing.
    await gov.run_action(make_action(), _executor(100.0))
    with pytest.raises(CircuitTripped):
        await gov.run_action(make_action(), _executor(100.0))

    assert any(e.reason.value == "circuit_tripped" for e in router.events)
    assert gov.store.list()[-1].circuit_tripped


async def test_post_execution_cost_trip_still_audits(cost_model, carbon_model):
    budget = Budget(token_budget=100_000.0, carbon_ceiling=1_000.0)
    gov = GreenGovernor(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
        monitor=ActionTimeMonitor(max_loops=100, max_marginal_cost=100.0),
    )
    with pytest.raises(CircuitTripped):
        await gov.run_action(make_action(), _executor(500.0))  # over marginal cap
    rec = gov.store.list()[-1]
    assert rec.circuit_tripped
    assert rec.actual_cost == 500.0  # the action did run; budget was spent
    assert budget.remaining_tokens() == 100_000.0 - 500.0


async def test_exhaustion_escalates(cost_model, carbon_model):
    budget = Budget(token_budget=300.0, carbon_ceiling=1_000.0)
    router = RecordingRouter()
    gov = GreenGovernor(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
        router=router,
    )
    # Admitted (worst case 300 == budget), runs, then budget hits zero -> escalate.
    await gov.run_action(make_action(prompt=100, max_tokens=200), _executor(300.0))
    assert budget.is_token_exhausted()
    assert any(e.reason.value == "token_exhausted" for e in router.events)


async def test_learning_loop_predict_act_log_retrain(cost_model, carbon_model):
    budget = Budget(token_budget=1_000_000.0, carbon_ceiling=100_000.0)
    estimator = LearnedEstimator(cost_model, carbon_model, min_samples=3)
    gov = GreenGovernor(
        budget=budget,
        estimator=estimator,
        cost_model=cost_model,
        carbon_model=carbon_model,
        monitor=ActionTimeMonitor(max_loops=1000),
    )
    action = make_action()
    # Actuals are consistently ~250 tokens; the estimator should learn this.
    for _ in range(5):
        await gov.run_action(action, _executor(250.0))

    records = gov.store.list()
    assert len(records) == 5
    # Early records used the cold-start forecast; later ones the learned model.
    assert records[0].forecast_source == "cold_start"
    assert records[-1].forecast_source == "learned"
    # The learned prediction converged on the observed actual.
    assert abs(records[-1].predicted_cost - 250.0) < 1e-6
