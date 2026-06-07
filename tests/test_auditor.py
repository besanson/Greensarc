"""Tests for the Post-Action Auditor and the audit-log schema (SITE 3)."""

from __future__ import annotations

from green_sarc.auditor import AuditRecord, PostActionAuditor
from green_sarc.forecast import Forecast, GateDecision, Verdict
from green_sarc.stores.memory import MemoryAuditStore


def _decision(cost_hat=300.0, carbon_hat=0.15):
    f = Forecast(cost_hat=cost_hat, carbon_hat=carbon_hat, confidence=0.9, source="learned")
    return f, GateDecision(verdict=Verdict.ADMIT, forecast=f, reason="ok")


def test_record_computes_errors_and_persists():
    store = MemoryAuditStore()
    auditor = PostActionAuditor(store)
    forecast, decision = _decision(cost_hat=300.0, carbon_hat=0.15)

    rec = auditor.record(
        action_id="a1",
        action_kind="chat.completion",
        model="test-model",
        region="eu-west",
        forecast=forecast,
        decision=decision,
        actual_cost=280.0,
        actual_carbon=0.14,
        budget_remaining_tokens=720.0,
        carbon_remaining=9.86,
        carbon_intensity=500.0,
    )
    assert rec.cost_error == 280.0 - 300.0
    assert abs(rec.carbon_error - (0.14 - 0.15)) < 1e-9
    assert rec.forecast_source == "learned"
    assert len(store) == 1


def test_record_feeds_estimator():
    class _SpyEstimator:
        def __init__(self):
            self.updates = []

        def predict(self, action, ctx):  # noqa: D102 - unused here
            raise AssertionError("not called")

        def update(self, record):
            self.updates.append(record)

    spy = _SpyEstimator()
    auditor = PostActionAuditor(MemoryAuditStore(), estimator=spy)
    forecast, decision = _decision()
    auditor.record(
        action_id="a1",
        action_kind="k",
        model="m",
        region="r",
        forecast=forecast,
        decision=decision,
        actual_cost=10.0,
        actual_carbon=0.1,
        budget_remaining_tokens=0.0,
        carbon_remaining=0.0,
        carbon_intensity=500.0,
    )
    assert len(spy.updates) == 1
    assert spy.updates[0].actual_cost == 10.0


def test_audit_record_roundtrip():
    rec = AuditRecord(
        action_id="a1",
        action_kind="chat.completion",
        model="test-model",
        region="eu-west",
        predicted_cost=300.0,
        predicted_carbon=0.15,
        confidence=0.8,
        actual_cost=280.0,
        actual_carbon=0.14,
        budget_remaining_tokens=720.0,
        carbon_remaining=9.86,
        carbon_intensity=500.0,
        admitted=True,
        verdict="admit",
        circuit_tripped=False,
        escalated=False,
        extra={"note": "x"},
    )
    d = rec.to_dict()
    # Derived fields are present for downstream consumers...
    assert d["cost_error"] == -20.0
    # ...and ignored on the way back in.
    back = AuditRecord.from_dict(d)
    assert back.action_id == rec.action_id
    assert back.predicted_cost == rec.predicted_cost
    assert back.actual_carbon == rec.actual_carbon
    assert back.extra == {"note": "x"}
