"""Phase-2 trajectory estimator — INTERFACE STUB ONLY.

Phase 2 will predict the cost and carbon of an ENTIRE PLAN before the agent
starts, enabling rejection of expensive *plans*, not just expensive *steps*.

It is trainable only on the trajectories Phase 1 logs, so it cannot be built
until Phase 1 has produced data.  This module fixes the interface and raises
``NotImplementedError`` — do not implement it as part of Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable

from green_sarc.forecast import Forecast
from green_sarc.state import Action, GovernanceContext

__all__ = [
    "Plan",
    "TrajectoryEstimator",
    "NotImplementedTrajectoryEstimator",
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
