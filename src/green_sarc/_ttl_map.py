"""A small thread-safe TTL + size-bounded map.

The orchestrator adapters correlate a gate call with its later audit call by
stashing the forecast under a key (an ``action_id`` or the request's identity).
If a caller crashes between the two calls the entry would otherwise live forever,
so this map bounds entries by both age (``ttl_s``) and count (``max_size``) —
closing the memory-leak / DoS vector flagged in the audit (P0-2).

It is dependency-free and safe to use from threads and from asyncio (the critical
sections are tiny and never await).
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar

__all__ = ["TTLMap"]

K = TypeVar("K")
V = TypeVar("V")


class TTLMap(Generic[K, V]):
    """A dict-like store whose entries expire by age and are bounded in count.

    Parameters
    ----------
    ttl_s:
        Maximum age of an entry, in seconds, before it is swept.
    max_size:
        Hard cap on entries; when exceeded the oldest entry is evicted.
    clock:
        Monotonic time source (injectable for tests).
    on_evict:
        Optional ``(key, value)`` callback invoked when an entry is dropped by
        the ttl sweep or the size cap (but **not** by an explicit :meth:`pop`).
        Used to release a budget reservation an orphaned entry was holding. It is
        called outside the internal lock.
    """

    def __init__(
        self,
        ttl_s: float = 600.0,
        max_size: int = 10_000,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_evict: Optional[Callable[[K, V], None]] = None,
    ) -> None:
        self._ttl = ttl_s
        self._max = max_size
        self._clock = clock
        self._on_evict = on_evict
        self._data: Dict[K, Tuple[float, V]] = {}
        self._lock = threading.Lock()
        self.evicted = 0

    def put(self, key: K, value: V) -> None:
        evicted: List[Tuple[K, V]] = []
        with self._lock:
            evicted.extend(self._sweep_locked())
            if key not in self._data and len(self._data) >= self._max:
                oldest = min(self._data, key=lambda k: self._data[k][0])
                evicted.append((oldest, self._data.pop(oldest)[1]))
                self.evicted += 1
            self._data[key] = (self._clock(), value)
        self._notify(evicted)

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            entry = self._data.pop(key, None)
            return entry[1] if entry is not None else default

    def _sweep_locked(self) -> List[Tuple[K, V]]:
        now = self._clock()
        dead = [(k, v) for k, (ts, v) in self._data.items() if now - ts > self._ttl]
        for k, _ in dead:
            del self._data[k]
            self.evicted += 1
        return dead

    def _notify(self, evicted: List[Tuple[K, V]]) -> None:
        if self._on_evict is None:
            return
        for k, v in evicted:
            self._on_evict(k, v)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
