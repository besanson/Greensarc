"""End-to-end: on heavy-tailed residuals the runtime conformal gate holds nominal
coverage where the Normal-sigma gate under-covers (the §10 finding, in code).

Uses a synthetic skewed distribution (no network / dataset dependency) that
reproduces the qualitative ShareGPT result: a right-skewed, heavy-tailed residual
law on which the Gaussian assumption mis-calibrates at tight delta.
"""

from __future__ import annotations

import numpy as np

from green_sarc.calibrator import SplitConformal
from green_sarc.gate import PreActionGate
from green_sarc.forecast import Forecast


def _skewed(rng: np.random.Generator, n: int) -> np.ndarray:
    # Strong right skew + a heavy upper tail of spikes: the Gaussian quantile sits
    # below the true (1-delta) quantile, so the Normal-sigma bound under-covers.
    base = rng.lognormal(mean=0.0, sigma=0.9, size=n) * 30 - 30
    spikes = rng.random(n) < 0.06
    return base + np.where(spikes, rng.normal(450, 90, n), 0.0)


def test_runtime_conformal_holds_where_normal_sigma_undercovers() -> None:
    rng = np.random.default_rng(7)
    cal, test = _skewed(rng, 12000), _skewed(rng, 12000)
    sigma = float(cal.std())
    level = 0.05  # tight operating point

    # Normal-sigma gate: bound = mu + z_{1-delta} * sigma.
    gate_norm = PreActionGate(estimator=None)
    fc = Forecast(cost_hat=0.0, carbon_hat=0.0, confidence=0.95, cost_std=sigma)
    norm_bound = gate_norm.cost_upper_bound(fc, level)
    norm_cov = float(np.mean(test <= norm_bound))

    # Runtime split-conformal gate fit on the residual log.
    sc = SplitConformal()
    sc.fit(cal)
    gate_conf = PreActionGate(estimator=None, calibrator=sc)
    conf_bound = gate_conf.cost_upper_bound(fc, level)
    conf_cov = float(np.mean(test <= conf_bound))

    assert norm_cov < 0.95  # Normal-sigma under-covers on heavy tails
    assert abs(conf_cov - 0.95) < 0.015  # conformal holds nominal coverage
    assert (0.95 - norm_cov) > (0.95 - conf_cov)  # conformal is strictly closer
