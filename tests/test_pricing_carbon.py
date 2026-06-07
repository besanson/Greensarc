"""Tests for time-varying carbon intensity kappa(rho, t) (audit P0-3)."""

from __future__ import annotations

from green_sarc.pricing import TableCarbonModel, carbon_for_tokens, default_cost_model


def test_static_intensity_unchanged_without_time_or_series():
    cm = TableCarbonModel(intensities={"eu-west": 300.0}, default_intensity=400.0)
    assert cm.carbon_intensity("eu-west") == 300.0
    assert cm.carbon_intensity("unknown") == 400.0
    assert cm.carbon_intensity("eu-west", t=12345.0) == 300.0  # no series -> static


def test_time_series_interpolates_and_clamps():
    # Two points: 100 gCO2 at t=0, 300 gCO2 at t=100.
    cm = TableCarbonModel(time_series={"eu-west": [(0.0, 100.0), (100.0, 300.0)]})
    assert cm.carbon_intensity("eu-west", 0.0) == 100.0
    assert cm.carbon_intensity("eu-west", 50.0) == 200.0  # midpoint
    assert cm.carbon_intensity("eu-west", 100.0) == 300.0
    # Clamp outside the range.
    assert cm.carbon_intensity("eu-west", -10.0) == 100.0
    assert cm.carbon_intensity("eu-west", 999.0) == 300.0


def test_series_is_sorted_on_construction():
    cm = TableCarbonModel(time_series={"r": [(100.0, 300.0), (0.0, 100.0)]})
    assert cm.carbon_intensity("r", 50.0) == 200.0


def test_carbon_shifts_with_time_in_forecast():
    cost = default_cost_model()  # 3e-7 kWh/token
    cm = TableCarbonModel(time_series={"eu-west": [(0.0, 100.0), (100.0, 500.0)]})
    low = carbon_for_tokens(cost, cm, "m", 1_000_000.0, "eu-west", t=0.0)
    high = carbon_for_tokens(cost, cm, "m", 1_000_000.0, "eu-west", t=100.0)
    assert high > low  # curtailment-aware: same work, dirtier grid -> more carbon


def test_live_provider_takes_precedence():
    class _Provider:
        def get(self, region, when=None):
            return 42.0 if region == "eu-west" else None

    cm = TableCarbonModel(intensities={"eu-west": 300.0}, provider=_Provider())
    assert cm.carbon_intensity("eu-west") == 42.0  # provider wins
    assert cm.carbon_intensity("other") == 400.0  # provider returns None -> fallback
