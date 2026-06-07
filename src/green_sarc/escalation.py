"""Escalation Router (SITE 4).

When budget or carbon is exhausted (or the gate rejects / the breaker trips),
the router dispatches the event to a pluggable handler — human review or a
deterministic fallback — instead of allowing silent overspend.

Like SARC's router, escalation is best-effort: a handler that raises is logged
and suppressed so a broken handler can never break the calling agent loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

__all__ = [
    "EscalationReason",
    "EscalationEvent",
    "EscalationHandler",
    "EscalationRouter",
    "DeterministicFallbackHandler",
    "log_only_handler",
]

logger = logging.getLogger(__name__)


class EscalationReason(str, Enum):
    """Why an action was escalated."""

    TOKEN_EXHAUSTED = "token_exhausted"
    CARBON_EXHAUSTED = "carbon_exhausted"
    GATE_REJECT = "gate_reject"
    CIRCUIT_TRIPPED = "circuit_tripped"


@dataclass
class EscalationEvent:
    """A single escalation: the reason and enough context to act on it."""

    reason: EscalationReason
    action_id: str
    action_kind: str
    detail: str = ""
    budget_remaining_tokens: float = 0.0
    carbon_remaining: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (used by handlers / the MCP adapter)."""
        return {
            "reason": self.reason.value,
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "detail": self.detail,
            "budget_remaining_tokens": self.budget_remaining_tokens,
            "carbon_remaining": self.carbon_remaining,
            "metadata": dict(self.metadata),
        }


EscalationHandler = Callable[[EscalationEvent], Awaitable[None]]


async def log_only_handler(event: EscalationEvent) -> None:
    """Default handler: log the escalation at WARNING level."""
    logger.warning(
        "escalation: reason=%s action=%s detail=%s tokens_left=%.1f carbon_left=%.3f",
        event.reason.value,
        event.action_id,
        event.detail,
        event.budget_remaining_tokens,
        event.carbon_remaining,
    )


@dataclass
class DeterministicFallbackHandler:
    """Invoke a deterministic, side-effect-free fallback on escalation.

    The fallback is any callable taking the :class:`EscalationEvent`; its return
    value is recorded on the event's metadata under ``"fallback_result"`` so the
    caller can route to it instead of overspending.
    """

    fallback: Callable[[EscalationEvent], Any]

    async def __call__(self, event: EscalationEvent) -> None:
        event.metadata["fallback_result"] = self.fallback(event)


class EscalationRouter:
    """Routes escalation events to a pluggable async handler (best-effort)."""

    def __init__(self, handler: Optional[EscalationHandler] = None) -> None:
        self.handler: EscalationHandler = handler if handler is not None else log_only_handler

    async def route(self, event: EscalationEvent) -> None:
        """Dispatch ``event``; never propagate a handler error to the caller."""
        try:
            await self.handler(event)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "escalation handler raised for action=%s reason=%s — suppressed",
                event.action_id,
                event.reason.value,
            )
