"""Tests for the Action-Time Monitor / circuit breaker (SITE 2)."""

from __future__ import annotations

import pytest

from green_sarc.monitor import ActionTimeMonitor, CircuitTripped


def test_loop_count_trips():
    mon = ActionTimeMonitor(max_loops=2)
    mon.before()
    mon.before()
    with pytest.raises(CircuitTripped) as exc:
        mon.before()
    assert mon.tripped
    assert exc.value.loop_count == 3
    assert "loop count" in exc.value.reason


def test_marginal_cost_trips():
    mon = ActionTimeMonitor(max_loops=100, max_marginal_cost=500.0)
    mon.before()
    mon.after(100.0)  # fine
    mon.before()
    with pytest.raises(CircuitTripped):
        mon.after(600.0)
    assert mon.tripped


def test_total_cost_trips():
    mon = ActionTimeMonitor(max_loops=100, max_total_cost=250.0)
    mon.before()
    mon.after(200.0)
    mon.before()
    with pytest.raises(CircuitTripped):
        mon.after(100.0)  # cumulative 300 > 250
    assert mon.total_cost == 300.0


def test_reset_clears_state():
    mon = ActionTimeMonitor(max_loops=1)
    mon.before()
    with pytest.raises(CircuitTripped):
        mon.before()
    mon.reset()
    assert mon.loop_count == 0
    assert not mon.tripped
    mon.before()  # must not raise after reset
