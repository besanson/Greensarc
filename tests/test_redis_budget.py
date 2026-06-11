"""RedisBudget: atomic two-phase reserve/commit/release, races, and TTL reclaim.

Runs against fakeredis (dev dependency); no network. Skips if fakeredis absent.
"""

from __future__ import annotations

import threading

import pytest

fakeredis = pytest.importorskip("fakeredis")

from green_sarc.backends import BudgetBackend, Reservation  # noqa: E402
from green_sarc.backends.redis_budget import RedisBudget  # noqa: E402


def _client():
    # A shared in-memory server so several clients see the same state.
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer())


def _budget(client, **kw):
    kw.setdefault("token_budget", 1000.0)
    kw.setdefault("carbon_ceiling", 500.0)
    return RedisBudget(client, "run-1", **kw)


def test_implements_protocol():
    b = _budget(_client())
    assert isinstance(b, BudgetBackend)


def test_reserve_reduces_net_remaining_and_release_restores():
    b = _budget(_client())
    assert b.remaining_tokens() == 1000.0
    r = b.reserve(300.0, 50.0)
    assert isinstance(r, Reservation)
    assert b.remaining_tokens() == 700.0
    assert b.remaining_carbon() == 450.0
    b.release(r)
    assert b.remaining_tokens() == 1000.0
    assert b.remaining_carbon() == 500.0


def test_reserve_over_budget_returns_none_and_burns_nothing():
    b = _budget(_client(), token_budget=100.0)
    assert b.reserve(150.0, 0.0) is None
    # A rejected reservation must not move the budget.
    assert b.remaining_tokens() == 100.0


def test_commit_spends_actual_not_reserved():
    b = _budget(_client())
    r = b.reserve(400.0, 100.0)  # reserve the worst-case forecast
    assert r is not None
    b.commit(r, actual_tokens=250.0, actual_carbon=40.0, actual_usd=0.5)
    # Reservation released; only the actuals are spent.
    assert b.remaining_tokens() == 750.0
    assert b.remaining_carbon() == 460.0


def test_usd_tracking():
    c = _client()
    b = _budget(c, usd_budget=1.0)
    assert b.remaining_usd() == 1.0
    r = b.reserve(10.0, 0.0)
    b.commit(r, 10.0, 0.0, actual_usd=0.4)
    assert abs(b.remaining_usd() - 0.6) < 1e-9


def test_no_usd_budget_is_unbounded():
    assert _budget(_client()).remaining_usd() == float("inf")


def test_concurrent_reserve_exactly_one_wins():
    # Budget for exactly one reservation of 600 (two would need 1200 > 1000).
    server = fakeredis.FakeServer()
    results = []

    def grab():
        client = fakeredis.FakeStrictRedis(server=server)
        b = RedisBudget(client, "run-1", token_budget=1000.0, carbon_ceiling=500.0)
        results.append(b.reserve(600.0, 0.0))

    threads = [threading.Thread(target=grab) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wins = [r for r in results if r is not None]
    assert len(wins) == 1, f"exactly one of two concurrent reserves must win, got {results}"


def test_ttl_expiry_reclaims_stale_reservation():
    now = [1000.0]
    c = _client()
    b = RedisBudget(
        c,
        "run-1",
        token_budget=100.0,
        carbon_ceiling=100.0,
        reservation_ttl_s=10.0,
        clock=lambda: now[0],
    )
    first = b.reserve(100.0, 0.0)
    assert first is not None
    # No capacity left while the reservation is live.
    assert b.reserve(100.0, 0.0) is None
    # Advance past the TTL: the next reserve reclaims the stale hold and wins.
    now[0] = 1020.0
    reclaimed = b.reserve(100.0, 0.0)
    assert reclaimed is not None
    assert reclaimed.id != first.id
