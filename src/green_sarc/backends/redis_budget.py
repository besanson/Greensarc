"""Redis reference backend for a distributed :class:`BudgetBackend` (experimental).

Every operation is a single Lua script, so ``reserve``-if-available is one atomic
round trip: two replicas racing for the last of the budget cannot both win.

Consistency semantics
---------------------
- **Atomic within one Redis.** Each op (reserve / commit / release) is one
  ``EVAL``; Redis executes it to completion before any other command. This gives
  linearizable budget accounting *against a single Redis endpoint*.
- **No cross-region guarantee.** Two independent Redis deployments (e.g. active-
  active across regions) are *not* reconciled here; point every governed replica
  at the same logical Redis (or a CP cluster) for the guarantee to hold.
- **TTL on stale reservations.** A reservation is recorded with an expiry; if a
  client crashes between ``reserve`` and ``commit``/``release``, the next
  ``reserve`` reclaims the expired hold and returns its capacity. ``commit``
  spends the *actual* cost regardless of the reserved amount, so a forecast that
  over-reserved never permanently locks budget.

This is a Phase-1 *experimental* backend: it covers token + carbon reservation
and USD spend tracking; it does not implement fair-share reservations or a
Postgres-style durable ledger (Phase 2).
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Any, Callable, Optional

from green_sarc.backends.base import Reservation

__all__ = ["RedisBudget"]


# Reservation amounts are stored as "tokens:carbon" strings in a side hash so the
# TTL sweep and commit/release can return the exact held capacity.
_RECLAIM = """
local now = tonumber(ARGV[1])
local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', now)
for _, rid in ipairs(expired) do
  local amt = redis.call('HGET', KEYS[3], rid)
  if amt then
    local sep = string.find(amt, ':')
    local t = tonumber(string.sub(amt, 1, sep - 1))
    local c = tonumber(string.sub(amt, sep + 1))
    redis.call('HINCRBYFLOAT', KEYS[1], 'reserved_tokens', -t)
    redis.call('HINCRBYFLOAT', KEYS[1], 'reserved_carbon', -c)
    redis.call('HDEL', KEYS[3], rid)
  end
  redis.call('ZREM', KEYS[2], rid)
end
"""

_RESERVE_LUA = (
    _RECLAIM
    + """
local tokens = tonumber(ARGV[2])
local carbon = tonumber(ARGV[3])
local rem_t = tonumber(redis.call('HGET', KEYS[1], 'remaining_tokens'))
local res_t = tonumber(redis.call('HGET', KEYS[1], 'reserved_tokens'))
local rem_c = tonumber(redis.call('HGET', KEYS[1], 'remaining_carbon'))
local res_c = tonumber(redis.call('HGET', KEYS[1], 'reserved_carbon'))
if tokens > rem_t - res_t then return 0 end
if carbon > rem_c - res_c then return 0 end
redis.call('HINCRBYFLOAT', KEYS[1], 'reserved_tokens', tokens)
redis.call('HINCRBYFLOAT', KEYS[1], 'reserved_carbon', carbon)
redis.call('ZADD', KEYS[2], tonumber(ARGV[1]) + tonumber(ARGV[4]), ARGV[5])
redis.call('HSET', KEYS[3], ARGV[5], ARGV[2] .. ':' .. ARGV[3])
return 1
"""
)

_CLEAR_RESERVATION = """
local amt = redis.call('HGET', KEYS[3], ARGV[1])
if amt then
  local sep = string.find(amt, ':')
  local t = tonumber(string.sub(amt, 1, sep - 1))
  local c = tonumber(string.sub(amt, sep + 1))
  redis.call('HINCRBYFLOAT', KEYS[1], 'reserved_tokens', -t)
  redis.call('HINCRBYFLOAT', KEYS[1], 'reserved_carbon', -c)
  redis.call('HDEL', KEYS[3], ARGV[1])
  redis.call('ZREM', KEYS[2], ARGV[1])
end
"""

_COMMIT_LUA = (
    _CLEAR_RESERVATION
    + """
redis.call('HINCRBYFLOAT', KEYS[1], 'remaining_tokens', -tonumber(ARGV[2]))
redis.call('HINCRBYFLOAT', KEYS[1], 'remaining_carbon', -tonumber(ARGV[3]))
redis.call('HINCRBYFLOAT', KEYS[1], 'usd_spent', tonumber(ARGV[4]))
return 1
"""
)

_RELEASE_LUA = _CLEAR_RESERVATION + "return 1\n"


class RedisBudget:
    """A shared cost + carbon budget backed by one Redis (experimental).

    Parameters
    ----------
    client:
        A ``redis.Redis`` (or API-compatible, e.g. ``fakeredis``) client.
    key:
        Logical budget name; all governed replicas must share it.
    token_budget, carbon_ceiling:
        Initial capacities (set once, on first use, via an atomic init).
    usd_budget:
        Optional USD ceiling; ``None`` means unbounded.
    reservation_ttl_s:
        Seconds before an uncommitted reservation is reclaimable.
    clock:
        Injectable time source (seconds); defaults to :func:`time.time`.
    """

    def __init__(
        self,
        client: Any,
        key: str,
        *,
        token_budget: float,
        carbon_ceiling: float,
        usd_budget: Optional[float] = None,
        reservation_ttl_s: float = 300.0,
        namespace: str = "green_sarc",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.client = client
        self._h = f"{namespace}:budget:{key}"
        self._z = f"{self._h}:resv"
        self._a = f"{self._h}:resv_amt"
        self.usd_budget = usd_budget
        self.reservation_ttl_s = float(reservation_ttl_s)
        self._clock = clock
        self._init(token_budget, carbon_ceiling, usd_budget)

    def _eval(self, script: str, *args: Any) -> Any:
        """One atomic EVAL round trip over this budget's three keys.

        Uses ``EVAL`` (not cached ``EVALSHA``) so the backend is portable across
        real Redis and test doubles without a script-cache priming step.
        """
        return self.client.eval(script, 3, self._h, self._z, self._a, *args)

    def _init(
        self, token_budget: float, carbon_ceiling: float, usd_budget: Optional[float]
    ) -> None:
        """Initialise the hash once, atomically; never clobber an existing budget."""
        usd = "inf" if usd_budget is None else repr(float(usd_budget))
        # HSETNX per field is atomic and idempotent across racing replicas.
        c = self.client
        c.hsetnx(self._h, "remaining_tokens", repr(float(token_budget)))
        c.hsetnx(self._h, "reserved_tokens", "0")
        c.hsetnx(self._h, "remaining_carbon", repr(float(carbon_ceiling)))
        c.hsetnx(self._h, "reserved_carbon", "0")
        c.hsetnx(self._h, "usd_spent", "0")
        c.hsetnx(self._h, "usd_budget", usd)

    def reserve(self, tokens: float, carbon: float) -> Optional[Reservation]:
        rid = uuid.uuid4().hex
        ok = self._eval(
            _RESERVE_LUA,
            repr(self._clock()),
            repr(float(tokens)),
            repr(float(carbon)),
            repr(self.reservation_ttl_s),
            rid,
        )
        if int(ok) == 1:
            return Reservation(id=rid, tokens=float(tokens), carbon=float(carbon))
        return None

    def commit(
        self,
        reservation: Reservation,
        actual_tokens: float,
        actual_carbon: float,
        actual_usd: float = 0.0,
    ) -> None:
        self._eval(
            _COMMIT_LUA,
            reservation.id,
            repr(float(actual_tokens)),
            repr(float(actual_carbon)),
            repr(float(actual_usd)),
        )

    def release(self, reservation: Reservation) -> None:
        self._eval(_RELEASE_LUA, reservation.id)

    def _hget_float(self, field: str) -> float:
        v = self.client.hget(self._h, field)
        if v is None:
            return 0.0
        return float(v.decode() if isinstance(v, (bytes, bytearray)) else v)

    def remaining_tokens(self) -> float:
        return self._hget_float("remaining_tokens") - self._hget_float("reserved_tokens")

    def remaining_carbon(self) -> float:
        return self._hget_float("remaining_carbon") - self._hget_float("reserved_carbon")

    def remaining_usd(self) -> float:
        v = self.client.hget(self._h, "usd_budget")
        raw = v.decode() if isinstance(v, (bytes, bytearray)) else v
        if raw is None or raw == "inf" or math.isinf(float(raw)):
            return float("inf")
        return float(raw) - self._hget_float("usd_spent")

    def reset(self) -> None:
        """Delete all keys for this budget (test/operational helper)."""
        self.client.delete(self._h, self._z, self._a)
