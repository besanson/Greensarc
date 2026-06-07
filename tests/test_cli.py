"""Tests for the green-sarc CLI."""

from __future__ import annotations

from green_sarc.auditor import AuditRecord
from green_sarc.cli import main
from green_sarc.stores.jsonl import JSONLAuditStore


def _rec(action_id, admitted=True, actual_cost=100.0, predicted_cost=120.0):
    return AuditRecord(
        action_id=action_id,
        action_kind="chat.completion",
        model="test-model",
        region="eu-west",
        predicted_cost=predicted_cost,
        predicted_carbon=0.06,
        confidence=0.7,
        actual_cost=actual_cost,
        actual_carbon=actual_cost * 5.0e-4,
        budget_remaining_tokens=900.0,
        carbon_remaining=9.9,
        carbon_intensity=500.0,
        admitted=admitted,
        verdict="admit" if admitted else "reject",
    )


def test_inspect_reports_summary(tmp_path, capsys):
    path = tmp_path / "audit.jsonl"
    store = JSONLAuditStore(path)
    store.append(_rec("a1", actual_cost=100.0, predicted_cost=120.0))
    store.append(_rec("a2", admitted=False, actual_cost=0.0))

    rc = main(["inspect", str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "records           : 2" in out
    assert "actions executed  : 1" in out
    assert "MAE token cost" in out


def test_inspect_empty_log(tmp_path, capsys):
    path = tmp_path / "empty.jsonl"
    path.touch()
    rc = main(["inspect", str(path)])
    assert rc == 0
    assert "no audit records" in capsys.readouterr().out
