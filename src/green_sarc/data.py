"""Reference pricing and carbon data so Green SARC is usable out of the box.

These are **approximate, editable defaults** — not authoritative. Override them
for your provider contract and your real grid data.

- USD per token: derived from public **list prices** (per 1M tokens) circa 2025.
  Your negotiated/committed-use price will differ — pass your own
  :class:`~green_sarc.pricing.TableCostModel`.
- Energy per token: a rough proxy (published inference-energy estimates vary by
  an order of magnitude); treat the carbon numbers as *relative* signals, not
  certified measurements.
- Carbon intensity: approximate annual-average grid intensity (gCO2e/kWh) by
  cloud region. For real GreenOps, wire a live ``IntensityProvider`` (Electricity
  Maps / WattTime) or a ``time_series``.

Nothing here is required: the core has zero data baked in. This module just
spares you hand-building tables to get started.
"""

from __future__ import annotations

from green_sarc.pricing import ModelProfile, TableCarbonModel, TableCostModel

__all__ = [
    "DEFAULT_REGION",
    "default_pricing",
    "default_carbon",
]

DEFAULT_REGION = "us-east-1"

# usd per token = list price per 1M tokens / 1e6.  energy_per_token_kwh is a rough
# size-tiered proxy (~0.3 Wh / 1k tokens for frontier models, less for small ones).
_PROFILES = {
    # OpenAI
    "gpt-4o": ModelProfile(3.0e-7, usd_per_prompt_token=2.5e-6, usd_per_completion_token=1.0e-5),
    "gpt-4o-mini": ModelProfile(
        8.0e-8, usd_per_prompt_token=1.5e-7, usd_per_completion_token=6.0e-7
    ),
    "gpt-4.1": ModelProfile(3.0e-7, usd_per_prompt_token=2.0e-6, usd_per_completion_token=8.0e-6),
    "gpt-4.1-mini": ModelProfile(
        8.0e-8, usd_per_prompt_token=4.0e-7, usd_per_completion_token=1.6e-6
    ),
    # Anthropic
    "claude-sonnet": ModelProfile(
        3.0e-7, usd_per_prompt_token=3.0e-6, usd_per_completion_token=1.5e-5
    ),
    "claude-haiku": ModelProfile(
        1.0e-7, usd_per_prompt_token=8.0e-7, usd_per_completion_token=4.0e-6
    ),
    "claude-opus": ModelProfile(
        5.0e-7, usd_per_prompt_token=1.5e-5, usd_per_completion_token=7.5e-5
    ),
    # Open-weights (hosted, representative)
    "llama-3.1-70b": ModelProfile(
        2.0e-7, usd_per_prompt_token=6.0e-7, usd_per_completion_token=6.0e-7
    ),
    "llama-3.1-8b": ModelProfile(
        6.0e-8, usd_per_prompt_token=6.0e-8, usd_per_completion_token=6.0e-8
    ),
}

# Approximate annual-average grid carbon intensity (gCO2e/kWh) by cloud region.
_INTENSITIES = {
    "us-east-1": 370.0,  # N. Virginia
    "us-west-2": 120.0,  # Oregon (hydro-heavy)
    "eu-west-1": 290.0,  # Ireland
    "eu-north-1": 30.0,  # Stockholm (very clean)
    "eu-central-1": 350.0,  # Frankfurt
    "ap-southeast-2": 520.0,  # Sydney (coal-heavy)
    "ap-northeast-1": 470.0,  # Tokyo
}


def default_pricing() -> TableCostModel:
    """A :class:`TableCostModel` seeded with approximate list prices (override me)."""
    return TableCostModel(
        profiles=dict(_PROFILES),
        default_profile=ModelProfile(
            3.0e-7, usd_per_prompt_token=2.0e-6, usd_per_completion_token=8.0e-6
        ),
    )


def default_carbon() -> TableCarbonModel:
    """A :class:`TableCarbonModel` seeded with approximate regional intensities."""
    return TableCarbonModel(intensities=dict(_INTENSITIES), default_intensity=400.0)
