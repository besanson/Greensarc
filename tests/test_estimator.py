"""Tests for the predictive estimator (cold start + learned + retrain)."""

from __future__ import annotations

from green_sarc.auditor import AuditRecord
from green_sarc.estimator import ColdStartEstimator, LearnedEstimator
from green_sarc.forecast import GateDecision, Verdict
from green_sarc.state import Budget, GovernanceContext

from .conftest import make_action


def _ctx() -> GovernanceContext:
    return GovernanceContext(budget=Budget(100_000.0, 1_000.0))


def _audit(actual_cost: float, kind="chat.completion", model="test-model") -> AuditRecord:
    return AuditRecord(
        action_id="a",
        action_kind=kind,
        model=model,
        region="eu-west",
        predicted_cost=0.0,
        predicted_carbon=0.0,
        confidence=0.0,
        actual_cost=actual_cost,
        actual_carbon=actual_cost * 5.0e-4,
        budget_remaining_tokens=0.0,
        carbon_remaining=0.0,
        carbon_intensity=500.0,
        admitted=True,
        verdict="admit",
    )


def test_cold_start_is_worst_case(cost_model, carbon_model):
    est = ColdStartEstimator(cost_model, carbon_model)
    f = est.predict(make_action(prompt=100, max_tokens=200), _ctx())
    # Worst case: prompt + the full completion cap.
    assert f.cost_hat == 300.0
    assert f.source == "cold_start"
    assert f.cost_std is None
    # carbon = tokens * 1e-6 kWh/token * 500 gCO2e/kWh
    assert f.carbon_hat == 300.0 * 1.0e-6 * 500.0


def test_cold_start_uses_default_completion_when_uncapped(cost_model, carbon_model):
    est = ColdStartEstimator(cost_model, carbon_model, default_completion_tokens=512)
    action = make_action(prompt=50, max_tokens=200)
    action = action.__class__(
        kind=action.kind,
        model=action.model,
        region=action.region,
        prompt_tokens=50,
        max_tokens=None,
    )
    f = est.predict(action, _ctx())
    assert f.cost_hat == 50 + 512


def test_cold_start_does_not_learn(cost_model, carbon_model):
    est = ColdStartEstimator(cost_model, carbon_model)
    est.update(_audit(123.0))  # no-op, must not raise
    f = est.predict(make_action(), _ctx())
    assert f.source == "cold_start"


def test_learned_defers_until_min_samples(cost_model, carbon_model):
    est = LearnedEstimator(cost_model, carbon_model, min_samples=3)
    action = make_action()
    # Below threshold -> cold start.
    est.update(_audit(250.0))
    est.update(_audit(250.0))
    assert est.predict(action, _ctx()).source == "cold_start"
    # At threshold -> learned, centred on the observed mean.
    est.update(_audit(250.0))
    f = est.predict(action, _ctx())
    assert f.source == "learned"
    assert f.cost_hat == 250.0
    assert est.samples(action) == 3


def test_learned_tracks_mean_and_std(cost_model, carbon_model):
    est = LearnedEstimator(cost_model, carbon_model, min_samples=2)
    action = make_action()
    for v in (100.0, 200.0, 300.0):
        est.update(_audit(v))
    f = est.predict(action, _ctx())
    assert f.cost_hat == 200.0  # mean of 100/200/300
    assert f.cost_std is not None and f.cost_std > 0.0
    # carbon recomputed from the learned token estimate
    assert f.carbon_hat == 200.0 * 1.0e-6 * 500.0


def test_learned_keys_are_per_model(cost_model, carbon_model):
    est = LearnedEstimator(cost_model, carbon_model, min_samples=1)
    est.update(_audit(500.0, model="model-a"))
    a = make_action()
    other = a.__class__(
        kind=a.kind,
        model="model-b",
        region=a.region,
        prompt_tokens=a.prompt_tokens,
        max_tokens=a.max_tokens,
    )
    # model-b has no history -> cold start; model-a is learned.
    assert est.predict(other, _ctx()).source == "cold_start"


def test_gate_decision_helpers():
    from green_sarc.forecast import Forecast

    f = Forecast(cost_hat=1.0, carbon_hat=2.0, confidence=0.5)
    d = GateDecision(verdict=Verdict.ADMIT, forecast=f, reason="ok")
    assert d.admitted
    assert d.to_dict()["verdict"] == "admit"
