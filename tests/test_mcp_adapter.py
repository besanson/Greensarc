"""Tests for the KAOS MCP adapter (pure service logic, no mcp dependency)."""

from __future__ import annotations

from green_sarc.adapters.mcp import GreenSarcMCPService
from green_sarc.estimator import ColdStartEstimator
from green_sarc.state import Budget


def _service(budget, cost_model, carbon_model):
    return GreenSarcMCPService(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
    )


def test_gate_then_audit_closes_the_loop(cost_model, carbon_model):
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    svc = _service(budget, cost_model, carbon_model)

    gate_out = svc.pre_action_gate(
        kind="chat.completion",
        model="test-model",
        region="eu-west",
        prompt_tokens=100,
        max_tokens=200,
    )
    assert gate_out["admitted"] is True
    assert gate_out["verdict"] == "admit"
    action_id = gate_out["action_id"]

    audit_out = svc.post_action_auditor(action_id=action_id, actual_tokens=250.0)
    assert audit_out["ok"] is True
    # cold-start predicted 300, actual 250 -> error -50
    assert audit_out["cost_error"] == -50.0
    assert budget.remaining_tokens() == 10_000.0 - 250.0
    assert len(svc.auditor.store.list()) == 1


def test_gate_rejects_over_budget(cost_model, carbon_model):
    budget = Budget(token_budget=100.0, carbon_ceiling=100.0)
    svc = _service(budget, cost_model, carbon_model)
    out = svc.pre_action_gate(
        kind="chat.completion",
        model="test-model",
        region="eu-west",
        prompt_tokens=100,
        max_tokens=200,
    )
    assert out["admitted"] is False
    assert out["verdict"] == "reject"


def test_audit_with_unknown_action_id(cost_model, carbon_model):
    svc = _service(Budget(1000.0, 100.0), cost_model, carbon_model)
    out = svc.post_action_auditor(action_id="nope", actual_tokens=10.0)
    assert out["ok"] is False
    assert "unknown action_id" in out["error"]


def test_explicit_actual_carbon_is_respected(cost_model, carbon_model):
    svc = _service(Budget(10_000.0, 100.0), cost_model, carbon_model)
    gate_out = svc.pre_action_gate(
        kind="k", model="test-model", region="eu-west", prompt_tokens=10, max_tokens=10
    )
    out = svc.post_action_auditor(
        action_id=gate_out["action_id"], actual_tokens=100.0, actual_carbon=0.5
    )
    assert out["ok"] is True
    rec = svc.auditor.store.list()[-1]
    assert rec.actual_carbon == 0.5
