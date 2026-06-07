"""Tests for the Pre-Action Gate (SITE 1)."""

from __future__ import annotations

from green_sarc.estimator import ColdStartEstimator
from green_sarc.forecast import Forecast, Verdict
from green_sarc.gate import PreActionGate
from green_sarc.state import Action, Budget, GovernanceContext

from .conftest import make_action


class _FixedEstimator:
    """Estimator returning a preset forecast (for deterministic gate tests)."""

    def __init__(self, forecast: Forecast) -> None:
        self.forecast = forecast

    def predict(self, action: Action, ctx: GovernanceContext) -> Forecast:
        return self.forecast

    def update(self, record) -> None:  # noqa: D102
        return None


def _gate_with(cost_hat, carbon_hat, cost_std=None):
    f = Forecast(cost_hat=cost_hat, carbon_hat=carbon_hat, confidence=0.9, cost_std=cost_std)
    return PreActionGate(_FixedEstimator(f))


def test_admit_within_budget():
    gate = _gate_with(cost_hat=300.0, carbon_hat=0.15)
    ctx = GovernanceContext(budget=Budget(1000.0, 10.0, delta=0.05))
    d = gate.evaluate(make_action(), ctx)
    assert d.verdict is Verdict.ADMIT
    assert d.admitted


def test_reject_when_cost_exceeds_budget():
    gate = _gate_with(cost_hat=300.0, carbon_hat=0.1)
    ctx = GovernanceContext(budget=Budget(100.0, 10.0))
    d = gate.evaluate(make_action(), ctx)
    assert d.verdict is Verdict.REJECT
    assert "cost" in d.reason


def test_reject_when_carbon_exceeds_ceiling():
    gate = _gate_with(cost_hat=10.0, carbon_hat=5.0)
    ctx = GovernanceContext(budget=Budget(1000.0, 1.0))  # only 1 gCO2e left
    d = gate.evaluate(make_action(), ctx)
    assert d.verdict is Verdict.REJECT
    assert "carbon" in d.reason


def test_escalate_when_already_exhausted():
    gate = _gate_with(cost_hat=10.0, carbon_hat=0.1)
    ctx = GovernanceContext(budget=Budget(0.0, 10.0))
    d = gate.evaluate(make_action(), ctx)
    assert d.verdict is Verdict.ESCALATE


def test_confidence_upper_bound_uses_std():
    # cost_hat 100, std 50; at 95% confidence the upper bound is ~182, which
    # exceeds a 150-token budget even though the point estimate fits.
    gate = _gate_with(cost_hat=100.0, carbon_hat=0.1, cost_std=50.0)
    bound = gate.cost_upper_bound(
        Forecast(cost_hat=100.0, carbon_hat=0.1, confidence=0.9, cost_std=50.0), delta=0.05
    )
    assert bound > 150.0
    ctx = GovernanceContext(budget=Budget(150.0, 10.0, delta=0.05))
    assert gate.evaluate(make_action(), ctx).verdict is Verdict.REJECT


def test_zero_std_is_point_estimate():
    gate = _gate_with(cost_hat=100.0, carbon_hat=0.1, cost_std=0.0)
    bound = gate.cost_upper_bound(
        Forecast(cost_hat=100.0, carbon_hat=0.1, confidence=0.9, cost_std=0.0), delta=0.05
    )
    assert bound == 100.0


def test_gate_with_cold_start_estimator(cost_model, carbon_model):
    gate = PreActionGate(ColdStartEstimator(cost_model, carbon_model))
    ctx = GovernanceContext(budget=Budget(1000.0, 10.0))
    d = gate.evaluate(make_action(prompt=100, max_tokens=200), ctx)
    assert d.admitted
    assert d.forecast.source == "cold_start"
