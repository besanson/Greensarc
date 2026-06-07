"""Post-Action Auditor (SITE 3) and the audit log schema.

The Post-Action Auditor logs PREDICTED vs ACTUAL cost and carbon for every
action.  This log is, deliberately, two things at once:

1. the ESG / audit record of what the agent actually spent, and
2. the training data for the estimator — the ``(predicted, actual)`` pairs that
   close the **predict -> act -> log -> retrain** loop (§5).

The auditor is framework-agnostic: it appends :class:`AuditRecord` objects to an
:class:`~green_sarc.stores.base.AuditStore` and, if given an estimator, feeds
each record back into ``estimator.update`` so the next forecast is better.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from green_sarc.forecast import Forecast, GateDecision

if TYPE_CHECKING:  # avoid import cycles; these are structural only
    from green_sarc.estimator import Estimator
    from green_sarc.stores.base import AuditStore

__all__ = [
    "AuditRecord",
    "PostActionAuditor",
]


@dataclass
class AuditRecord:
    """One audited action: forecast, actuals, budget snapshot, and outcome.

    ``cost_error`` / ``carbon_error`` are ``actual - predicted`` and are the
    learning signal for the estimator.
    """

    action_id: str
    action_kind: str
    model: str
    region: str
    predicted_cost: float
    predicted_carbon: float
    confidence: float
    actual_cost: float
    actual_carbon: float
    budget_remaining_tokens: float
    carbon_remaining: float
    carbon_intensity: float
    admitted: bool
    verdict: str
    circuit_tripped: bool = False
    escalated: bool = False
    forecast_source: str = "estimator"
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def cost_error(self) -> float:
        """Actual minus predicted token cost."""
        return self.actual_cost - self.predicted_cost

    @property
    def carbon_error(self) -> float:
        """Actual minus predicted carbon (gCO2e)."""
        return self.actual_carbon - self.predicted_carbon

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON Lines storage."""
        d: Dict[str, Any] = {
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "model": self.model,
            "region": self.region,
            "predicted_cost": self.predicted_cost,
            "predicted_carbon": self.predicted_carbon,
            "confidence": self.confidence,
            "actual_cost": self.actual_cost,
            "actual_carbon": self.actual_carbon,
            "cost_error": self.cost_error,
            "carbon_error": self.carbon_error,
            "budget_remaining_tokens": self.budget_remaining_tokens,
            "carbon_remaining": self.carbon_remaining,
            "carbon_intensity": self.carbon_intensity,
            "admitted": self.admitted,
            "verdict": self.verdict,
            "circuit_tripped": self.circuit_tripped,
            "escalated": self.escalated,
            "forecast_source": self.forecast_source,
            "timestamp": self.timestamp,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        """Deserialise from the dict produced by :meth:`to_dict`.

        The derived ``cost_error`` / ``carbon_error`` keys are ignored on load.
        """
        return cls(
            action_id=data["action_id"],
            action_kind=data["action_kind"],
            model=data.get("model", ""),
            region=data.get("region", ""),
            predicted_cost=float(data["predicted_cost"]),
            predicted_carbon=float(data["predicted_carbon"]),
            confidence=float(data.get("confidence", 0.0)),
            actual_cost=float(data["actual_cost"]),
            actual_carbon=float(data["actual_carbon"]),
            budget_remaining_tokens=float(data.get("budget_remaining_tokens", 0.0)),
            carbon_remaining=float(data.get("carbon_remaining", 0.0)),
            carbon_intensity=float(data.get("carbon_intensity", 0.0)),
            admitted=bool(data["admitted"]),
            verdict=data.get("verdict", ""),
            circuit_tripped=bool(data.get("circuit_tripped", False)),
            escalated=bool(data.get("escalated", False)),
            forecast_source=data.get("forecast_source", "estimator"),
            timestamp=float(data.get("timestamp", 0.0)),
            extra=dict(data.get("extra", {})),
        )


class PostActionAuditor:
    """Records predicted-vs-actual per action and feeds the estimator.

    Parameters
    ----------
    store:
        Where :class:`AuditRecord` objects are appended (the ESG/audit log).
    estimator:
        Optional estimator to retrain on each record; this is the ``log ->
        retrain`` half of the learning loop.
    """

    def __init__(
        self,
        store: "AuditStore",
        estimator: Optional["Estimator"] = None,
    ) -> None:
        self.store = store
        self.estimator = estimator

    def record(
        self,
        *,
        action_id: str,
        action_kind: str,
        model: str,
        region: str,
        forecast: Forecast,
        decision: GateDecision,
        actual_cost: float,
        actual_carbon: float,
        budget_remaining_tokens: float,
        carbon_remaining: float,
        carbon_intensity: float,
        circuit_tripped: bool = False,
        escalated: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Build, persist, and learn from one :class:`AuditRecord`."""
        rec = AuditRecord(
            action_id=action_id,
            action_kind=action_kind,
            model=model,
            region=region,
            predicted_cost=forecast.cost_hat,
            predicted_carbon=forecast.carbon_hat,
            confidence=forecast.confidence,
            actual_cost=actual_cost,
            actual_carbon=actual_carbon,
            budget_remaining_tokens=budget_remaining_tokens,
            carbon_remaining=carbon_remaining,
            carbon_intensity=carbon_intensity,
            admitted=decision.admitted,
            verdict=decision.verdict.value,
            circuit_tripped=circuit_tripped,
            escalated=escalated,
            forecast_source=forecast.source,
            extra=dict(extra or {}),
        )
        self.store.append(rec)
        if self.estimator is not None:
            self.estimator.update(rec)
        return rec
