"""Augmented core state for Green SARC (working paper §4).

Green SARC reasons over an augmented agent state:

- ``b_tok``        — remaining token budget (live, decrementing).
- ``kappa(rho,t)`` — carbon intensity (gCO2e/kWh) for compute region ``rho`` at
  time ``t`` (see :mod:`green_sarc.pricing`).
- ``delta_lat``    — latency / SLA headroom.

This module defines the value objects threaded through the four enforcement
sites: the proposed :class:`Action`, the live :class:`Budget`, and the
:class:`GovernanceContext` that carries them alongside identity fields.

The core is framework-agnostic: nothing here imports an agent framework.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

__all__ = [
    "Action",
    "Budget",
    "GovernanceContext",
]


@dataclass(frozen=True)
class Action:
    """A proposed agent action whose cost and carbon are to be forecast.

    Only ``kind`` is required.  ``prompt_tokens`` and ``max_tokens`` describe
    the token shape of an LLM call; ``model`` and ``region`` select the entries
    used from the pricing / carbon tables.
    """

    kind: str
    model: str = ""
    region: str = ""
    prompt_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Budget:
    """Live cost + carbon budget for a governed run.

    ``token_budget`` is ``b_tok``: it is the *remaining* token budget and is
    decremented in place as actions execute.  Carbon is tracked as a fixed
    ceiling ``carbon_ceiling`` (``B_co2``) against a running ``carbon_spent``.

    ``delta`` is the gate confidence parameter: the Pre-Action Gate admits an
    action only when the forecast cost fits the budget with probability at least
    ``1 - delta``.
    """

    token_budget: float
    carbon_ceiling: float
    carbon_spent: float = 0.0
    delta: float = 0.05
    latency_headroom: Optional[float] = None
    reserved_tokens: float = 0.0
    reserved_carbon: float = 0.0
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def remaining_tokens(self) -> float:
        """Remaining token budget ``b_tok`` net of in-flight reservations."""
        return self.token_budget - self.reserved_tokens

    def remaining_carbon(self) -> float:
        """Remaining carbon headroom (gCO2e) net of in-flight reservations."""
        return self.carbon_ceiling - self.carbon_spent - self.reserved_carbon

    def is_token_exhausted(self) -> bool:
        """True once no token budget remains."""
        return self.remaining_tokens() <= 0.0

    def is_carbon_exhausted(self) -> bool:
        """True once the carbon ceiling has been reached."""
        return self.remaining_carbon() <= 0.0

    def reserve(self, tokens: float, carbon: float) -> bool:
        """Atomically hold ``tokens``/``carbon`` against the budget.

        Returns ``False`` (reserving nothing) if either amount would exceed the
        remaining budget, so concurrent callers cannot collectively over-admit
        between their gate check and their spend (audit P0-1).
        """
        with self._lock:
            if tokens > self.token_budget - self.reserved_tokens:
                return False
            if carbon > self.carbon_ceiling - self.carbon_spent - self.reserved_carbon:
                return False
            self.reserved_tokens += tokens
            self.reserved_carbon += carbon
            return True

    def release(self, tokens: float, carbon: float) -> None:
        """Return a reservation without spending (e.g. the action did not run)."""
        with self._lock:
            self.reserved_tokens = max(0.0, self.reserved_tokens - tokens)
            self.reserved_carbon = max(0.0, self.reserved_carbon - carbon)

    def commit(
        self,
        reserve_tokens: float,
        reserve_carbon: float,
        actual_tokens: float,
        actual_carbon: float,
    ) -> None:
        """Release a reservation and spend the action's actual cost atomically."""
        with self._lock:
            self.reserved_tokens = max(0.0, self.reserved_tokens - reserve_tokens)
            self.reserved_carbon = max(0.0, self.reserved_carbon - reserve_carbon)
            self.token_budget -= actual_tokens
            self.carbon_spent += actual_carbon

    def spend(self, tokens: float, carbon: float) -> None:
        """Spend actual cost with no prior reservation, atomically.

        Retained for callers that do not use the reserve/commit protocol.
        """
        with self._lock:
            self.token_budget -= tokens
            self.carbon_spent += carbon


@dataclass
class GovernanceContext:
    """Identity / environment context for a single governed call.

    Mirrors the shape of SARC's ``ExecutionContext`` but carries the live
    :class:`Budget` so the estimator and gate can reason over remaining state.
    """

    budget: Budget
    principal_id: str = ""
    agent_id: str = ""
    session_id: str = ""
    environment: str = ""
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dict of the identity fields (not the budget)."""
        return {
            "principal_id": self.principal_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "environment": self.environment,
            "metadata": dict(self.metadata),
        }
