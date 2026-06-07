"""The Phase-2 trajectory estimator must remain an unimplemented stub."""

from __future__ import annotations

import pytest

from green_sarc.trajectory import (
    NotImplementedTrajectoryEstimator,
    Plan,
    TrajectoryEstimator,
)
from green_sarc.state import Action, Budget, GovernanceContext


def test_trajectory_estimator_raises():
    est = NotImplementedTrajectoryEstimator()
    plan = Plan(actions=[Action(kind="chat.completion")])
    ctx = GovernanceContext(budget=Budget(1000.0, 10.0))
    with pytest.raises(NotImplementedError):
        est.predict(plan, ctx)


def test_stub_satisfies_protocol():
    # The stub conforms structurally to the Phase-2 interface.
    assert isinstance(NotImplementedTrajectoryEstimator(), TrajectoryEstimator)
