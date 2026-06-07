"""Tests for the predictive estimator (cold start + learned regression + retrain)."""

from __future__ import annotations

from dataclasses import replace

from green_sarc.auditor import AuditRecord
from green_sarc.estimator import ColdStartEstimator, LearnedEstimator
from green_sarc.forecast import GateDecision, Verdict
from green_sarc.state import Budget, GovernanceContext

from .conftest import make_action


def _ctx() -> GovernanceContext:
    return GovernanceContext(budget=Budget(100_000.0, 1_000.0))


def _audit(
    actual_cost: float,
    *,
    prompt_tokens: int = 0,
    kind: str = "chat.completion",
    model: str = "test-model",
) -> AuditRecord:
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
        prompt_tokens=prompt_tokens,
    )


# -- cold start -------------------------------------------------------------


def test_cold_start_is_worst_case(cost_model, carbon_model):
    est = ColdStartEstimator(cost_model, carbon_model)
    f = est.predict(make_action(prompt=100, max_tokens=200), _ctx())
    assert f.cost_hat == 300.0  # prompt + full completion cap
    assert f.source == "cold_start"
    assert f.cost_std is None
    assert f.carbon_hat == 300.0 * 1.0e-6 * 500.0


def test_cold_start_uses_default_completion_when_uncapped(cost_model, carbon_model):
    est = ColdStartEstimator(cost_model, carbon_model, default_completion_tokens=512)
    action = replace(make_action(prompt=50), max_tokens=None)
    assert est.predict(action, _ctx()).cost_hat == 50 + 512


def test_cold_start_does_not_learn(cost_model, carbon_model):
    est = ColdStartEstimator(cost_model, carbon_model)
    est.update(_audit(123.0))  # no-op, must not raise
    assert est.predict(make_action(), _ctx()).source == "cold_start"


# -- learned regression -----------------------------------------------------


def test_learned_defers_until_min_samples(cost_model, carbon_model):
    est = LearnedEstimator(cost_model, carbon_model, min_samples=3)
    action = make_action(prompt=100, max_tokens=200)
    # total 250 with prompt 100 -> completion 150.
    est.update(_audit(250.0, prompt_tokens=100))
    est.update(_audit(250.0, prompt_tokens=100))
    assert est.predict(action, _ctx()).source == "cold_start"
    est.update(_audit(250.0, prompt_tokens=100))
    f = est.predict(action, _ctx())
    assert f.source == "learned"
    assert f.cost_hat == 250.0  # 100 prompt + 150 completion
    assert est.samples(action) == 3


def test_learned_regresses_completion_on_prompt(cost_model, carbon_model):
    est = LearnedEstimator(cost_model, carbon_model, min_samples=3)
    # completion = 50 + 1.0 * prompt (noise-free) -> total = 2*prompt + 50.
    for prompt in (100, 200, 300):
        est.update(_audit(float(2 * prompt + 50), prompt_tokens=prompt))
    action = replace(make_action(prompt=400), max_tokens=None)  # uncapped -> no clamp
    f = est.predict(action, _ctx())
    assert abs(f.cost_hat - 850.0) < 1e-6  # 400 prompt + (50 + 400) completion
    assert f.source == "learned"


def test_learned_clamps_completion_to_max_tokens(cost_model, carbon_model):
    est = LearnedEstimator(cost_model, carbon_model, min_samples=3)
    for _ in range(3):
        est.update(_audit(2000.0, prompt_tokens=100))  # completion 1900
    f = est.predict(make_action(prompt=100, max_tokens=200), _ctx())
    assert f.cost_hat == 300.0  # 100 + min(1900, 200)


def test_learned_exposes_residual_std_under_noise(cost_model, carbon_model):
    import random

    rng = random.Random(0)
    est = LearnedEstimator(cost_model, carbon_model, min_samples=3)
    for _ in range(60):
        prompt = rng.randint(50, 500)
        completion = max(0.0, 100 + 0.5 * prompt + rng.gauss(0, 25))
        est.update(_audit(prompt + completion, prompt_tokens=prompt))
    f = est.predict(replace(make_action(prompt=200), max_tokens=None), _ctx())
    assert f.cost_std is not None and f.cost_std > 0.0


def test_learned_keys_are_per_model(cost_model, carbon_model):
    est = LearnedEstimator(cost_model, carbon_model, min_samples=1)
    est.update(_audit(500.0, prompt_tokens=100, model="model-a"))
    other = replace(make_action(), model="model-b")
    assert est.predict(other, _ctx()).source == "cold_start"  # model-b has no history


def test_gate_decision_helpers():
    from green_sarc.forecast import Forecast

    f = Forecast(cost_hat=1.0, carbon_hat=2.0, confidence=0.5)
    d = GateDecision(verdict=Verdict.ADMIT, forecast=f, reason="ok")
    assert d.admitted
    assert d.to_dict()["verdict"] == "admit"
