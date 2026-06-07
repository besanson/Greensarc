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

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

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
class _RegStats:
    """Online simple linear regression of ``y`` on ``x`` (least squares)."""

    n: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    sxy: float = 0.0
    syy: float = 0.0

    def update(self, x: float, y: float) -> None:
        self.n += 1
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.sxy += x * y
        self.syy += y * y

    def predict(self, x: float) -> Tuple[float, float]:
        """Return ``(y_hat, residual_std)`` at ``x``.

        Falls back to the mean of ``y`` (with its sample std) when there is too
        little spread in ``x`` to fit a slope.
        """
        mean_y = self.sy / self.n if self.n else 0.0
        denom = self.n * self.sxx - self.sx * self.sx
        if self.n < 3 or denom <= 0.0:
            if self.n < 2:
                return (mean_y, 0.0)
            var = max((self.syy - self.sy * self.sy / self.n) / (self.n - 1), 0.0)
            return (mean_y, math.sqrt(var))
        beta = (self.n * self.sxy - self.sx * self.sy) / denom
        alpha = (self.sy - beta * self.sx) / self.n
        y_hat = alpha + beta * x
        sse = max(self.syy - alpha * self.sy - beta * self.sxy, 0.0)
        sigma = math.sqrt(sse / max(self.n - 2, 1))
        return (y_hat, sigma)


@dataclass
class LearnedEstimator:
    """Online estimator that regresses completion tokens on prompt tokens.

    Completion length correlates with prompt length, so per ``(kind, model)`` key
    this fits an online least-squares line ``completion ~ alpha + beta *
    prompt_tokens`` and predicts ``cost_hat = prompt_tokens + completion`` (the
    completion clamped to ``[0, max_tokens]``).  The regression's residual
    standard deviation is supplied to the gate as ``cost_std`` for its
    ``1 - delta`` upper bound.  Until a key has ``min_samples`` observations it
    defers to the conservative :class:`ColdStartEstimator`.
    """

    cost_model: CostModel
    carbon_model: CarbonModel
    min_samples: int = 3
    cold_start: ColdStartEstimator = field(init=False)
    _stats: Dict[Tuple[str, str], _RegStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cold_start = ColdStartEstimator(self.cost_model, self.carbon_model)

    def predict(self, action: Action, ctx: GovernanceContext) -> Forecast:
        stat = self._stats.get(_key(action))
        if stat is None or stat.n < self.min_samples:
            return self.cold_start.predict(action, ctx)
        prompt = float(action.prompt_tokens or 0)
        completion_hat, sigma = stat.predict(prompt)
        completion_hat = max(0.0, completion_hat)
        if action.max_tokens is not None:
            completion_hat = min(completion_hat, float(action.max_tokens))
        cost_hat = prompt + completion_hat
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
            cost_std=sigma if sigma > 0.0 else None,
            source="learned",
        )

    def update(self, record: AuditRecord) -> None:
        key = (record.action_kind, record.model)
        stat = self._stats.setdefault(key, _RegStats())
        prompt = float(record.prompt_tokens)
        completion = max(0.0, record.actual_cost - prompt)
        stat.update(prompt, completion)

    def samples(self, action: Action) -> int:
        """Number of observations learned for ``action``'s key (introspection)."""
        stat = self._stats.get(_key(action))
        return stat.n if stat is not None else 0

    # -- persistence (audit P1-1) ----------------------------------------

    def save(self, path: Any) -> None:
        """Persist the learned regression state to a JSON file."""
        data = {
            "min_samples": self.min_samples,
            "stats": [
                {
                    "kind": kind,
                    "model": model,
                    "n": s.n,
                    "sx": s.sx,
                    "sy": s.sy,
                    "sxx": s.sxx,
                    "sxy": s.sxy,
                    "syy": s.syy,
                }
                for (kind, model), s in self._stats.items()
            ],
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: Any) -> None:
        """Replace the learned state with one previously :meth:`save`d."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.min_samples = int(data.get("min_samples", self.min_samples))
        self._stats = {}
        for row in data.get("stats", []):
            self._stats[(row["kind"], row["model"])] = _RegStats(
                n=int(row["n"]),
                sx=float(row["sx"]),
                sy=float(row["sy"]),
                sxx=float(row["sxx"]),
                sxy=float(row["sxy"]),
                syy=float(row["syy"]),
            )

    def bootstrap_from_jsonl(self, path: Any) -> int:
        """Rehydrate the estimator by replaying a JSONL audit log; returns count."""
        from green_sarc.stores.jsonl import JSONLAuditStore

        count = 0
        for record in JSONLAuditStore(path).iter_records():
            self.update(record)
            count += 1
        return count
