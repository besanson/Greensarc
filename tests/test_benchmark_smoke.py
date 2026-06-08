"""Smoke test for the IBP benchmark (working paper §8)."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo-root `benchmarks` package importable from the test.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.ibp import IBPConfig, run_condition, run_pair  # noqa: E402
from green_sarc.scoping import AdapterNode  # noqa: E402


def test_adapter_node_caps_the_snowball():
    node = AdapterNode(max_scope_tokens=400)
    assert node.scope(100) == 100.0  # below the cap -> unchanged
    assert node.scope(5000) == 400.0  # snowballed context capped
    # The cap is what turns Theta(n^2) cumulative cost into ~linear.
    capped = sum(node.scope(i * 120) for i in range(50))
    uncapped = sum(i * 120 for i in range(50))
    assert capped < uncapped


def test_treatment_is_cheaper_than_baseline():
    cfg = IBPConfig(n_skus=25, depth=6, runaway_fraction=0.2)
    baseline, treatment = run_pair(seed=0, cfg=cfg)
    # State-Snowball baseline pays Theta(depth^2); scoped treatment pays ~linear.
    assert treatment["tokens"] < baseline["tokens"]
    assert treatment["usd"] < baseline["usd"]
    assert treatment["carbon_fixed_g"] < baseline["carbon_fixed_g"]


def test_treatment_exercises_governance():
    cfg = IBPConfig(n_skus=40, depth=6, runaway_fraction=0.3)
    treatment = run_condition(seed=1, cfg=cfg, governed=True)
    assert treatment["admitted"] > 0
    assert treatment["breaker_trips"] > 0  # runaway SKUs are capped by the breaker
    assert "forecast_mae_tokens" in treatment  # the estimator learned and was audited


def test_carbon_reported_under_fixed_and_time_varying_kappa():
    cfg = IBPConfig(n_skus=10, depth=4)
    m = run_condition(seed=2, cfg=cfg, governed=True)
    assert m["carbon_fixed_g"] > 0.0
    assert m["carbon_tv_g"] > 0.0
    assert m["carbon_fixed_g"] != m["carbon_tv_g"]  # the daily curve matters
