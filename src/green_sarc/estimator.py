"""The predictive estimator (Phase 1).

Interface (§5)::

    estimator.predict(action, context) -> Forecast(cost_hat, carbon_hat, confidence)
    estimator.update(audit_record) -> None

Two concrete estimators are provided:

- :class:`ColdStartEstimator` — the **zero-information** gate.  With no history
  it assumes the worst case (prompt plus the full ``max_tokens`` cap) and a low
  confidence.  This is the static-threshold fallback, used as cold-start
  behaviour, *not* as the primary mechanism.
- :class:`LearnedEstimator` — learns the per-key completion cost online from the
  Auditor's ``(predicted, actual)`` pairs (Welford mean/variance) and predicts
  against it, falling back to the cold-start estimator until it has data.

Both are model-agnostic: carbon is computed from the supplied
:class:`~green_sarc.pricing.CostModel` / :class:`~green_sarc.pricing.CarbonModel`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Protocol, Tuple, runtime_checkable

from green_sarc.auditor import AuditRecord
from green_sarc.forecast import Forecast
from green_sarc.pricing import CarbonModel, CostModel, carbon_for_tokens
from green_sarc.state import Action, GovernanceContext

__all__ = [
    "Estimator",
    "ColdStartEstimator",
    "LearnedEstimator",
]


@runtime_checkable
class Estimator(Protocol):
    """Predicts the cost and carbon of the next action and learns from actuals."""

    def predict(self, action: Action, ctx: GovernanceContext) -> Forecast: ...

    def update(self, record: AuditRecord) -> None: ...


def _key(action: Action) -> Tuple[str, str]:
    """Statistics key: an action's identity for learning purposes."""
    return (action.kind, action.model)


@dataclass
class ColdStartEstimator:
    """Zero-information estimator: conservative worst-case forecast.

    With no history the only safe forecast is the worst case the action could
    produce: its prompt tokens plus the full ``max_tokens`` completion cap (or
    ``default_completion_tokens`` when uncapped).  Confidence is fixed low so the
    gate's ``1 - delta`` upper bound stays conservative.
    """

    cost_model: CostModel
    carbon_model: CarbonModel
    default_completion_tokens: int = 1024
    confidence: float = 0.25

    def predict(self, action: Action, ctx: GovernanceContext) -> Forecast:
        prompt = float(action.prompt_tokens or 0)
        completion = float(
            action.max_tokens if action.max_tokens is not None else self.default_completion_tokens
        )
        cost_hat = prompt + completion
        carbon_hat = carbon_for_tokens(
            self.cost_model,
            self.carbon_model,
            action.model,
            cost_hat,
            action.region,
            ctx.timestamp,
        )
        return Forecast(
            cost_hat=cost_hat,
            carbon_hat=carbon_hat,
            confidence=self.confidence,
            cost_std=None,  # treated as worst case by the gate
            source="cold_start",
        )

    def update(self, record: AuditRecord) -> None:  # noqa: D102 - stateless
        # The zero-information estimator does not learn.
        return None


@dataclass
class _Stat:
    """Online mean/variance accumulator (Welford)."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


@dataclass
class LearnedEstimator:
    """Online per-key estimator that retrains on logged actuals.

    Predicts the total token cost of an action from the running mean of observed
    actuals for its ``(kind, model)`` key, with the standard deviation supplied
    to the gate so it can form a ``1 - delta`` upper bound.  Until a key has at
    least ``min_samples`` observations it defers to :class:`ColdStartEstimator`.
    """

    cost_model: CostModel
    carbon_model: CarbonModel
    min_samples: int = 3
    cold_start: ColdStartEstimator = field(init=False)
    _stats: Dict[Tuple[str, str], _Stat] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cold_start = ColdStartEstimator(self.cost_model, self.carbon_model)

    def predict(self, action: Action, ctx: GovernanceContext) -> Forecast:
        stat = self._stats.get(_key(action))
        if stat is None or stat.n < self.min_samples:
            return self.cold_start.predict(action, ctx)
        cost_hat = stat.mean
        # Confidence grows with sample count, saturating below 1.0.
        confidence = min(0.99, 1.0 - 1.0 / (1.0 + stat.n))
        carbon_hat = carbon_for_tokens(
            self.cost_model,
            self.carbon_model,
            action.model,
            cost_hat,
            action.region,
            ctx.timestamp,
        )
        return Forecast(
            cost_hat=cost_hat,
            carbon_hat=carbon_hat,
            confidence=confidence,
            cost_std=stat.std,
            source="learned",
        )

    def update(self, record: AuditRecord) -> None:
        key = (record.action_kind, record.model)
        stat = self._stats.setdefault(key, _Stat())
        stat.add(record.actual_cost)

    def samples(self, action: Action) -> int:
        """Number of observations learned for ``action``'s key (introspection)."""
        stat = self._stats.get(_key(action))
        return stat.n if stat is not None else 0
