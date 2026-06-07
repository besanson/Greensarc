"""Phase-2 trajectory estimator — INTERFACE STUB ONLY.

Phase 2 will predict the cost and carbon of an ENTIRE PLAN before the agent
starts, enabling rejection of expensive *plans*, not just expensive *steps*.

It is trainable only on the trajectories Phase 1 logs, so it cannot be built
until Phase 1 has produced data.  This module fixes the interface and raises
``NotImplementedError`` — do not implement it as part of Phase 1.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable

from green_sarc.auditor import AuditRecord
from green_sarc.forecast import Forecast
from green_sarc.state import Action, GovernanceContext

__all__ = [
    "Plan",
    "TrajectoryEstimator",
    "NotImplementedTrajectoryEstimator",
    "TrajectoryStore",
    "JSONLTrajectoryStore",
]


@dataclass(frozen=True)
class Plan:
    """An ordered sequence of proposed actions forming a trajectory."""

    actions: List[Action] = field(default_factory=list)


@runtime_checkable
class TrajectoryEstimator(Protocol):
    """Predicts the cost and carbon of an entire plan (Phase 2)."""

    def predict(self, plan: Plan, ctx: GovernanceContext) -> Forecast: ...


class NotImplementedTrajectoryEstimator:
    """Placeholder Phase-2 estimator.

    Exists so the gate and governor can be wired against the trajectory
    interface today.  Building it requires Phase 1's logged trajectories.
    """

    def predict(self, plan: Plan, ctx: GovernanceContext) -> Forecast:
        raise NotImplementedError(
            "Trajectory (whole-plan) estimation is Phase 2; it trains on the "
            "trajectories logged by Phase 1 and is not implemented yet."
        )


@runtime_checkable
class TrajectoryStore(Protocol):
    """Groups Phase-1 audit records into trajectories (the Phase-2 training set)."""

    def trajectories(self) -> Dict[str, List[AuditRecord]]: ...


@dataclass
class JSONLTrajectoryStore:
    """Groups a JSONL audit log into trajectories by ``plan_id`` (or ``session_id``).

    This is the data engine for Phase 2: it reshapes the per-action log that
    Phase 1 produces into ordered, per-plan sequences without committing to a
    Phase-2 estimator algorithm.
    """

    path: Any

    def trajectories(self) -> Dict[str, List[AuditRecord]]:
        from green_sarc.stores.jsonl import JSONLAuditStore

        grouped: "OrderedDict[str, List[AuditRecord]]" = OrderedDict()
        for record in JSONLAuditStore(self.path).iter_records():
            key = record.plan_id or record.session_id
            if key is None:
                continue
            grouped.setdefault(key, []).append(record)
        return dict(grouped)
