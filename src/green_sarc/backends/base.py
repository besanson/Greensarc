"""The :class:`BudgetBackend` protocol and its reservation handle.

Two-phase semantics, mirroring :class:`green_sarc.state.Budget` but handle-based
so a distributed backend can reclaim a crashed client's reservation by TTL:

1. ``reserve(tokens, carbon)`` atomically holds capacity and returns a
   :class:`Reservation`, or ``None`` if either amount would exceed what remains
   (so a rejected action never burns budget).
2. ``commit(reservation, actual_*)`` releases the hold and spends the *actual*
   cost in one atomic step.
3. ``release(reservation)`` returns a hold without spending (the action did not
   run).

``remaining_*`` report capacity net of in-flight reservations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

__all__ = ["BudgetBackend", "Reservation"]


@dataclass(frozen=True)
class Reservation:
    """An in-flight hold returned by :meth:`BudgetBackend.reserve`.

    ``id`` is the backend's handle for the hold; ``tokens`` / ``carbon`` are the
    held amounts (carried so ``commit`` / ``release`` can be validated and so a
    TTL sweep can return the right capacity).
    """

    id: str
    tokens: float
    carbon: float


@runtime_checkable
class BudgetBackend(Protocol):
    """A shared, transactional cost + carbon counter for governed actions."""

    def reserve(self, tokens: float, carbon: float) -> Optional[Reservation]:
        """Atomically hold capacity; return a :class:`Reservation` or ``None``."""
        ...

    def commit(
        self,
        reservation: Reservation,
        actual_tokens: float,
        actual_carbon: float,
        actual_usd: float = 0.0,
    ) -> None:
        """Release ``reservation`` and spend the action's actual cost atomically."""
        ...

    def release(self, reservation: Reservation) -> None:
        """Return ``reservation`` without spending (the action did not run)."""
        ...

    def remaining_tokens(self) -> float:
        """Remaining token budget net of in-flight reservations."""
        ...

    def remaining_carbon(self) -> float:
        """Remaining carbon headroom (gCO2e) net of in-flight reservations."""
        ...

    def remaining_usd(self) -> float:
        """Remaining USD budget, or ``+inf`` when no USD budget is set."""
        ...
