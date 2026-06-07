"""Trajectory grouping of the Phase-1 audit log (audit P1-9 scaffolding)."""

from __future__ import annotations

from green_sarc.auditor import AuditRecord
from green_sarc.stores.jsonl import JSONLAuditStore
from green_sarc.trajectory import JSONLTrajectoryStore, TrajectoryStore


def _rec(plan_id=None, session_id=None) -> AuditRecord:
    return AuditRecord(
        action_id="a",
        action_kind="chat.completion",
        model="test-model",
        region="eu-west",
        predicted_cost=10.0,
        predicted_carbon=0.1,
        confidence=0.5,
        actual_cost=10.0,
        actual_carbon=0.1,
        budget_remaining_tokens=0.0,
        carbon_remaining=0.0,
        carbon_intensity=400.0,
        admitted=True,
        verdict="admit",
        plan_id=plan_id,
        session_id=session_id,
    )


def test_groups_by_plan_id(tmp_path):
    log = tmp_path / "audit.jsonl"
    store = JSONLAuditStore(log)
    for _ in range(3):
        store.append(_rec(plan_id="p1"))
    store.append(_rec(plan_id="p2"))
    store.append(_rec())  # no plan or session -> ignored

    trajectories = JSONLTrajectoryStore(log).trajectories()
    assert set(trajectories) == {"p1", "p2"}
    assert len(trajectories["p1"]) == 3
    assert len(trajectories["p2"]) == 1


def test_falls_back_to_session_id(tmp_path):
    log = tmp_path / "audit.jsonl"
    store = JSONLAuditStore(log)
    store.append(_rec(session_id="s1"))
    store.append(_rec(session_id="s1"))
    trajectories = JSONLTrajectoryStore(log).trajectories()
    assert len(trajectories["s1"]) == 2


def test_store_satisfies_protocol(tmp_path):
    assert isinstance(JSONLTrajectoryStore(tmp_path / "x.jsonl"), TrajectoryStore)
