"""Tests for the runtime conformal calibrators (green_sarc.calibrator)."""

from __future__ import annotations

import numpy as np
import pytest

from green_sarc import ACIConformal, SplitConformal
from green_sarc.gate import PreActionGate


def _coverage(cal: np.ndarray, test: np.ndarray, level: float) -> float:
    sc = SplitConformal()
    sc.fit(cal)
    bound = sc.upper_bound(0.0, level)  # one-sided: predicted = 0, score = actual
    return float(np.mean(test <= bound))


@pytest.mark.parametrize("level", [0.2, 0.1, 0.05])
def test_split_conformal_coverage_gaussian(level: float) -> None:
    rng = np.random.default_rng(0)
    cal, test = rng.normal(0, 10, 5000), rng.normal(0, 10, 5000)
    cov = _coverage(cal, test, level)
    assert abs(cov - (1 - level)) < 0.02  # marginal coverage near nominal


def test_split_conformal_coverage_pareto() -> None:
    # Heavy-tailed (non-Gaussian) residuals: split conformal is distribution-free.
    rng = np.random.default_rng(1)
    cal, test = rng.pareto(2.0, 8000) * 10, rng.pareto(2.0, 8000) * 10
    assert abs(_coverage(cal, test, 0.1) - 0.9) < 0.02


def test_split_conformal_coverage_mixture() -> None:
    rng = np.random.default_rng(2)

    def mix(n: int) -> np.ndarray:
        a = rng.normal(0, 5, n)
        b = rng.normal(40, 20, n)
        pick = rng.random(n) < 0.3
        return np.where(pick, b, a)

    assert abs(_coverage(mix(8000), mix(8000), 0.1) - 0.9) < 0.02


def _aci_rolling_coverage(shift: float, target: float = 0.1) -> tuple[float, float]:
    """Return (fixed_quantile_coverage, aci_coverage) on a post-shift stream."""
    rng = np.random.default_rng(3)
    cal = rng.normal(0, 10, 4000)  # calibration regime
    fixed = SplitConformal()
    fixed.fit(cal)
    aci = ACIConformal(gamma=0.1)
    aci.fit(cal)
    qf = fixed.upper_bound(0.0, target)
    fixed_hits, aci_hits = [], []
    for _ in range(8000):  # deployment regime is shifted
        y = rng.normal(shift, 10)
        fixed_hits.append(1.0 if y <= qf else 0.0)
        qa = aci.upper_bound(0.0, target)
        aci_hits.append(1.0 if y <= qa else 0.0)
        aci.update(observed=y, predicted=0.0)
    # ACI converges after a transient: score it on the post-convergence tail.
    return float(np.mean(fixed_hits)), float(np.mean(aci_hits[4000:]))


def test_aci_restores_coverage_under_upward_shift() -> None:
    fixed_cov, aci_cov = _aci_rolling_coverage(shift=8.0)  # residuals shift up
    assert fixed_cov < 0.9 - 0.03  # fixed quantile under-covers post-shift
    assert abs(aci_cov - 0.9) < 0.07  # ACI restores coverage near target
    assert abs(aci_cov - 0.9) < abs(fixed_cov - 0.9) / 3  # vastly closer than fixed


def test_aci_restores_coverage_under_downward_shift() -> None:
    fixed_cov, aci_cov = _aci_rolling_coverage(shift=-8.0)  # residuals shift down
    assert fixed_cov > 0.9 + 0.03  # fixed quantile over-covers post-shift
    assert abs(aci_cov - 0.9) < 0.07  # ACI restores coverage near target
    assert abs(aci_cov - 0.9) < abs(fixed_cov - 0.9) / 3  # vastly closer than fixed


def test_gate_uses_calibrator_bound() -> None:
    # The gate's upper bound must come from the calibrator when one is supplied,
    # and from the Normal-sigma path otherwise (backward compatibility).
    from green_sarc.forecast import Forecast

    rng = np.random.default_rng(4)
    sc = SplitConformal()
    sc.fit(rng.normal(0, 10, 5000))
    fc = Forecast(cost_hat=100.0, carbon_hat=0.0, confidence=0.9, cost_std=10.0)

    gate_cal = PreActionGate(estimator=None, calibrator=sc)
    gate_norm = PreActionGate(estimator=None)
    assert gate_cal.calibrator_decision == "SplitConformal"
    assert gate_norm.calibrator_decision == "normal_sigma"
    # conformal bound ~ 100 + q_0.9 (~12.8); normal bound ~ 100 + 1.28*10
    assert abs(gate_cal.cost_upper_bound(fc, 0.1) - (100 + 12.8)) < 2.0
    assert gate_norm.cost_upper_bound(fc, 0.1) == pytest.approx(100 + 1.2816 * 10, abs=0.1)
