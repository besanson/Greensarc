"""Pluggable budget backends (experimental).

The in-process :class:`green_sarc.state.Budget` is authoritative for a single
replica.  Multi-replica deployments behind a load balancer need a shared,
transactional counter so two replicas cannot collectively over-admit.  This
package defines the :class:`BudgetBackend` protocol and a reference
:class:`~green_sarc.backends.redis_budget.RedisBudget` (optional ``redis``
extra).

Status: **experimental** (Phase 1).  The protocol is handle-based two-phase —
``reserve`` returns a :class:`Reservation`, and ``commit`` / ``release`` consume
it — which suits a distributed store where a crashed client's reservation must
be reclaimed by TTL.
"""

from __future__ import annotations

from green_sarc.backends.base import BudgetBackend, Reservation
from green_sarc.backends.redis_budget import RedisBudget

__all__ = ["BudgetBackend", "Reservation", "RedisBudget"]
