"""Forecast and gate-decision value objects.

The estimator produces a :class:`Forecast` for a proposed action; the
Pre-Action Gate turns it into a :class:`GateDecision` carrying a
:class:`Verdict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

__all__ = [
    "Verdict",
    "Forecast",
    "GateDecision",
]


class Verdict(str, Enum):
    """Outcome of the Pre-Action Gate for a proposed action."""

    ADMIT = "admit"  # forecast fits the budget and carbon ceiling
    REJECT = "reject"  # forecast exceeds budget/carbon; do not run
    DOWNROUTE = "downroute"  # too expensive for this route; try a cheaper one
    ESCALATE = "escalate"  # budget/carbon exhausted; hand to the router


@dataclass(frozen=True)
class Forecast:
    """A per-action cost + carbon prediction.

    ``cost_hat`` is the expected token cost; ``carbon_hat`` the expected carbon
    in gCO2e; ``confidence`` is in ``[0, 1]``.  ``cost_std`` is an optional
    standard deviation used by the gate to compute an upper bound at confidence
    ``1 - delta``; when ``None`` the gate treats ``cost_hat`` as the worst case.
    ``source`` records whether the prediction came from a learned model or the
    cold-start fallback.
    """

    cost_hat: float
    carbon_hat: float
    confidence: float
    cost_std: Optional[float] = None
    usd_hat: float = 0.0
    source: str = "estimator"

    def as_tuple(self) -> tuple[float, float, float]:
        """Return ``(cost_hat, carbon_hat, confidence)`` per the §5 interface."""
        return (self.cost_hat, self.carbon_hat, self.confidence)


@dataclass(frozen=True)
class GateDecision:
    """The Pre-Action Gate's verdict for one action, with its forecast."""

    verdict: Verdict
    forecast: Forecast
    reason: str = ""

    @property
    def admitted(self) -> bool:
        """True only when the action was admitted."""
        return self.verdict is Verdict.ADMIT

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (used by the MCP adapter)."""
        return {
            "verdict": self.verdict.value,
            "admitted": self.admitted,
            "reason": self.reason,
            "cost_hat": self.forecast.cost_hat,
            "carbon_hat": self.forecast.carbon_hat,
            "confidence": self.forecast.confidence,
            "source": self.forecast.source,
        }
