"""Tests for the TTL + size-bounded correlation map (audit P0-2)."""

from __future__ import annotations

from green_sarc._ttl_map import TTLMap


def test_put_pop_basic():
    m: TTLMap[str, int] = TTLMap()
    m.put("a", 1)
    assert m.pop("a") == 1
    assert m.pop("a") is None
    assert m.pop("missing", -1) == -1


def test_entries_expire_by_ttl():
    clock = {"now": 0.0}
    m: TTLMap[str, int] = TTLMap(ttl_s=10.0, clock=lambda: clock["now"])
    m.put("a", 1)
    clock["now"] = 5.0
    m.put("b", 2)  # sweep on put: "a" still within ttl
    assert len(m) == 2
    clock["now"] = 12.0
    m.put("c", 3)  # sweep evicts "a" (age 12 > 10); "b" age 7 survives
    assert m.pop("a") is None
    assert m.pop("b") == 2
    assert m.evicted >= 1


def test_max_size_evicts_oldest():
    clock = {"now": 0.0}
    m: TTLMap[str, int] = TTLMap(ttl_s=1e9, max_size=2, clock=lambda: clock["now"])
    m.put("a", 1)
    clock["now"] = 1.0
    m.put("b", 2)
    clock["now"] = 2.0
    m.put("c", 3)  # over cap -> evict the oldest ("a")
    assert m.pop("a") is None
    assert m.pop("b") == 2
    assert m.pop("c") == 3
    assert m.evicted == 1


def test_on_evict_fires_on_size_eviction_not_on_pop():
    evicted: list[str] = []
    m: TTLMap[str, int] = TTLMap(ttl_s=1e9, max_size=2, on_evict=lambda k, v: evicted.append(k))
    m.put("a", 1)
    m.put("b", 2)
    m.put("c", 3)  # over cap -> evict "a" -> on_evict("a")
    assert evicted == ["a"]
    assert m.pop("b") == 2  # explicit pop must NOT trigger on_evict
    assert evicted == ["a"]


def test_on_evict_fires_on_ttl_sweep():
    clock = {"now": 0.0}
    evicted: list[str] = []
    m: TTLMap[str, int] = TTLMap(
        ttl_s=10.0, clock=lambda: clock["now"], on_evict=lambda k, v: evicted.append(k)
    )
    m.put("a", 1)
    clock["now"] = 20.0
    m.put("b", 2)  # sweep evicts "a"
    assert evicted == ["a"]


def test_orphaned_entries_stay_bounded():
    # Simulate gate calls whose audit never arrives: the map must not grow without bound.
    m: TTLMap[int, int] = TTLMap(ttl_s=1e9, max_size=100)
    for i in range(1000):
        m.put(i, i)
    assert len(m) <= 100
    assert m.evicted >= 900
