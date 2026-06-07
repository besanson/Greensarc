"""Estimator persistence and bootstrap from the audit log (audit P1-1)."""

from __future__ import annotations

from green_sarc.auditor import AuditRecord
from green_sarc.cli import main
from green_sarc.estimator import LearnedEstimator
from green_sarc.state import Budget, GovernanceContext
from green_sarc.stores.jsonl import JSONLAuditStore

from .conftest import make_action


def _ctx() -> GovernanceContext:
    return GovernanceContext(budget=Budget(100_000.0, 1_000.0))


def _rec(prompt: int, total: float) -> AuditRecord:
    return AuditRecord(
        action_id="a",
        action_kind="chat.completion",
        model="test-model",
        region="eu-west",
        predicted_cost=0.0,
        predicted_carbon=0.0,
        confidence=0.0,
        actual_cost=total,
        actual_carbon=total * 5e-4,
        budget_remaining_tokens=0.0,
        carbon_remaining=0.0,
        carbon_intensity=500.0,
        admitted=True,
        verdict="admit",
        prompt_tokens=prompt,
    )


def test_save_load_round_trip(tmp_path, cost_model, carbon_model):
    est = LearnedEstimator(cost_model, carbon_model, min_samples=3)
    for prompt in (100, 200, 300):
        est.update(_rec(prompt, 2 * prompt + 50))
    action = make_action(prompt=400, max_tokens=10_000)
    before = est.predict(action, _ctx()).cost_hat

    path = tmp_path / "state.json"
    est.save(path)
    fresh = LearnedEstimator(cost_model, carbon_model)
    fresh.load(path)
    after = fresh.predict(action, _ctx()).cost_hat
    assert after == before
    assert fresh.min_samples == 3


def test_bootstrap_matches_from_scratch(tmp_path, cost_model, carbon_model):
    log = tmp_path / "audit.jsonl"
    store = JSONLAuditStore(log)
    rows = [(100, 250.0), (200, 450.0), (300, 650.0)]
    for prompt, total in rows:
        store.append(_rec(prompt, total))

    scratch = LearnedEstimator(cost_model, carbon_model, min_samples=3)
    for prompt, total in rows:
        scratch.update(_rec(prompt, total))

    booted = LearnedEstimator(cost_model, carbon_model, min_samples=3)
    n = booted.bootstrap_from_jsonl(log)
    assert n == 3

    action = make_action(prompt=250, max_tokens=10_000)
    assert booted.predict(action, _ctx()).cost_hat == scratch.predict(action, _ctx()).cost_hat


def test_cli_bootstrap(tmp_path, capsys):
    log = tmp_path / "audit.jsonl"
    store = JSONLAuditStore(log)
    for prompt, total in [(100, 250.0), (200, 450.0), (300, 650.0)]:
        store.append(_rec(prompt, total))
    out = tmp_path / "state.json"

    rc = main(["bootstrap", str(log), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "bootstrapped estimator from 3 records" in capsys.readouterr().out
