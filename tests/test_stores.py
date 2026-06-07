"""Tests for the audit-store backends."""

from __future__ import annotations

from green_sarc.auditor import AuditRecord
from green_sarc.stores.jsonl import JSONLAuditStore
from green_sarc.stores.memory import MemoryAuditStore
from green_sarc.stores.sqlite import SQLiteAuditStore


def _rec(action_id: str, actual_cost: float = 100.0) -> AuditRecord:
    return AuditRecord(
        action_id=action_id,
        action_kind="chat.completion",
        model="test-model",
        region="eu-west",
        predicted_cost=120.0,
        predicted_carbon=0.06,
        confidence=0.7,
        actual_cost=actual_cost,
        actual_carbon=actual_cost * 5.0e-4,
        budget_remaining_tokens=900.0,
        carbon_remaining=9.9,
        carbon_intensity=500.0,
        admitted=True,
        verdict="admit",
    )


def test_memory_store_append_list_export(tmp_path):
    store = MemoryAuditStore()
    store.append(_rec("a1"))
    store.append(_rec("a2"))
    assert len(store) == 2
    assert [r.action_id for r in store.list()] == ["a1", "a2"]

    out = tmp_path / "audit.jsonl"
    n = store.export_jsonl(out)
    assert n == 2
    assert out.read_text().count("\n") == 2


def test_jsonl_store_roundtrip(tmp_path):
    path = tmp_path / "nested" / "audit.jsonl"
    store = JSONLAuditStore(path)
    store.append(_rec("a1", actual_cost=100.0))
    store.append(_rec("a2", actual_cost=200.0))

    # A fresh store over the same file replays the records.
    reopened = JSONLAuditStore(path)
    records = reopened.list()
    assert [r.action_id for r in records] == ["a1", "a2"]
    assert records[1].actual_cost == 200.0


def test_sqlite_store_roundtrip_and_persist(tmp_path):
    db = tmp_path / "audit.db"
    store = SQLiteAuditStore(db)
    store.append(_rec("a1", actual_cost=100.0))
    store.append(_rec("a2", actual_cost=200.0))
    assert [r.action_id for r in store.list()] == ["a1", "a2"]
    store.close()

    # Reopen the same database file -> records persisted.
    reopened = SQLiteAuditStore(db)
    records = reopened.list()
    assert [r.action_id for r in records] == ["a1", "a2"]
    assert records[1].actual_cost == 200.0
    out = tmp_path / "export.jsonl"
    assert reopened.export_jsonl(out) == 2
    assert out.read_text().count("\n") == 2
    reopened.close()


def test_jsonl_export_copies(tmp_path):
    src = tmp_path / "src.jsonl"
    store = JSONLAuditStore(src)
    store.append(_rec("a1"))
    dest = tmp_path / "dest.jsonl"
    assert store.export_jsonl(dest) == 1
    assert dest.exists()
