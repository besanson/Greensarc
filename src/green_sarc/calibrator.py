"""Runtime conformal calibration for the Pre-Action Gate.

The split-conformal upper bound proved in the working paper (Theorem 2) is
exposed here as a runtime strategy object, so the gate can admit on a
distribution-free bound instead of the Normal-:math:`\\sigma` default.

A :class:`Calibrator` is fit once on a log of forecast residuals
``r = actual - predicted`` and then queried at admission time for a one-sided
upper bound on cost.  Two implementations are provided:

- :class:`SplitConformal` — the standard inductive-conformal quantile.  Marginal,
  finite-sample coverage under exchangeability; ``update`` is a no-op.
- :class:`ACIConformal` — adaptive conformal inference (Gibbs & Candès, 2021):
  the miscoverage level is adjusted online from observed coverage, restoring the
  target under distribution shift.  ``update`` is called post-action with the
  realized and predicted cost.

Both are strategy objects passed to :class:`~green_sarc.gate.PreActionGate` via
``calibrator=...``; omitting it preserves the Normal-:math:`\\sigma` behaviour
exactly (backward compatible).  Pure ``numpy``; no SciPy dependency.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

__all__ = ["Calibrator", "SplitConformal", "ACIConformal"]


@runtime_checkable
class Calibrator(Protocol):
    """Maps a point forecast ``mu`` to a one-sided upper bound at risk ``level``."""

    def fit(self, residuals: np.ndarray) -> None: ...

    def upper_bound(self, mu: float, level: float) -> float: ...

    def update(self, observed: float, predicted: float) -> None: ...


def _one_sided_quantile(scores: np.ndarray, coverage: float) -> float:
    """Conformal ``coverage`` quantile of one-sided scores (``+inf`` if degenerate)."""
    if scores.size == 0:
        return float("inf")
    coverage = min(max(coverage, 0.0), 1.0)
    # The finite-sample-valid level uses the (n+1) correction; clamp to +inf when
    # the requested coverage exceeds what the calibration set can certify.
    n = scores.size
    rank = int(np.ceil((n + 1) * coverage))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


class SplitConformal:
    """Split (inductive) conformal upper bound: ``mu + q_{1-level}``."""

    def __init__(self) -> None:
        self._scores: Optional[np.ndarray] = None

    def fit(self, residuals: np.ndarray) -> None:
        self._scores = np.asarray(residuals, dtype=float).ravel()

    def upper_bound(self, mu: float, level: float) -> float:
        if self._scores is None:
            return mu
        q = _one_sided_quantile(self._scores, 1.0 - level)
        return mu + q

    def update(self, observed: float, predicted: float) -> None:  # no-op (offline)
        return None


class ACIConformal:
    """Adaptive conformal inference (Gibbs & Candès, 2021).

    Holds a fixed calibration score set but adapts the effective miscoverage
    level ``alpha_t`` online: ``alpha_{t+1} = alpha_t + gamma (target - err_t)``,
    where ``err_t = 1`` if the last realized residual exceeded the last bound.
    Under distribution shift this drives empirical coverage back to the target.
    """

    def __init__(self, gamma: float = 0.05) -> None:
        self.gamma = gamma
        self._scores: Optional[np.ndarray] = None
        self._target: Optional[float] = None
        self._alpha: float = 0.0
        self._last_q: float = 0.0

    def fit(self, residuals: np.ndarray) -> None:
        self._scores = np.asarray(residuals, dtype=float).ravel()

    def upper_bound(self, mu: float, level: float) -> float:
        if self._scores is None:
            return mu
        if self._target is None:  # first call fixes the coverage target
            self._target = level
            self._alpha = level
        self._last_q = _one_sided_quantile(self._scores, 1.0 - self._alpha)
        if self._last_q == float("inf"):
            # cannot certify at this level; fall back to the empirical max score
            self._last_q = float(self._scores.max()) if self._scores.size else 0.0
        return mu + self._last_q

    def update(self, observed: float, predicted: float) -> None:
        if self._target is None:
            return
        err = 1.0 if (observed - predicted) > self._last_q else 0.0
        self._alpha = float(min(max(self._alpha + self.gamma * (self._target - err), 1e-3), 0.999))

    @property
    def alpha(self) -> float:
        """Current adaptive miscoverage level (for introspection / tests)."""
        return self._alpha
