"""Model-agnostic pricing and carbon tables.

Per the confirmed §6 decision, the estimator is *model-agnostic*: the caller
supplies a pricing + carbon table and the estimator predicts against any LLM.
This module defines the :class:`CostModel` and :class:`CarbonModel` protocols
and ships table-driven default implementations.

Carbon for an action is computed as::

    carbon_gco2e = energy_kwh(model, tokens) * kappa(region, t)

where ``kappa`` is the carbon intensity (gCO2e/kWh) of the compute region at
time ``t`` — the ``kappa(rho, t)`` term of the augmented state (§4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

__all__ = [
    "ModelProfile",
    "CostModel",
    "CarbonModel",
    "IntensityProvider",
    "TableCostModel",
    "TableCarbonModel",
    "carbon_for_tokens",
    "default_cost_model",
    "default_carbon_model",
]


@dataclass(frozen=True)
class ModelProfile:
    """Per-model cost and energy coefficients.

    ``energy_per_token_kwh`` is the marginal energy of one token (kWh).  The USD
    coefficients are optional and used only for monetary reporting, not for the
    token-budget gate.
    """

    energy_per_token_kwh: float
    usd_per_prompt_token: float = 0.0
    usd_per_completion_token: float = 0.0


@runtime_checkable
class CostModel(Protocol):
    """Maps token counts to energy (kWh) and, optionally, USD for a model."""

    def energy_kwh(self, model: str, tokens: float) -> float: ...

    def usd(self, model: str, prompt_tokens: float, completion_tokens: float) -> float: ...


@runtime_checkable
class CarbonModel(Protocol):
    """Carbon intensity ``kappa(rho, t)`` for a compute region."""

    def carbon_intensity(self, region: str, t: Optional[float] = None) -> float: ...


@dataclass
class TableCostModel:
    """Table-driven :class:`CostModel`.

    Unknown models fall back to ``default_profile`` so the layer keeps working
    against any LLM the caller has not catalogued.
    """

    profiles: Dict[str, ModelProfile] = field(default_factory=dict)
    default_profile: ModelProfile = field(
        default_factory=lambda: ModelProfile(energy_per_token_kwh=3.0e-7)
    )

    def _profile(self, model: str) -> ModelProfile:
        return self.profiles.get(model, self.default_profile)

    def energy_kwh(self, model: str, tokens: float) -> float:
        return self._profile(model).energy_per_token_kwh * tokens

    def usd(self, model: str, prompt_tokens: float, completion_tokens: float) -> float:
        p = self._profile(model)
        return (
            p.usd_per_prompt_token * prompt_tokens + p.usd_per_completion_token * completion_tokens
        )


@runtime_checkable
class IntensityProvider(Protocol):
    """Pluggable live carbon-intensity source (e.g. Electricity Maps / WattTime).

    Implement this to feed a real grid signal into :class:`TableCarbonModel`.
    """

    def get(self, region: str, when: Optional[float] = None) -> Optional[float]: ...


def _interpolate(series: List[Tuple[float, float]], t: float) -> float:
    """Linear interpolation over a time-sorted ``(timestamp, value)`` series.

    Values are clamped to the endpoints outside the series' range.
    """
    if t <= series[0][0]:
        return series[0][1]
    if t >= series[-1][0]:
        return series[-1][1]
    # Linear scan is fine for the short per-region series used here.
    for (t0, v0), (t1, v1) in zip(series, series[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return v0
            frac = (t - t0) / (t1 - t0)
            return v0 + frac * (v1 - v0)
    return series[-1][1]  # pragma: no cover - unreachable given the guards above


@dataclass
class TableCarbonModel:
    """Table-driven :class:`CarbonModel` keyed by region, optionally time-varying.

    Resolution order for ``carbon_intensity(region, t)``:

    1. a live :class:`IntensityProvider`, if one is configured and returns a value;
    2. a per-region time series interpolated at ``t`` (the ``kappa(rho, t)`` term);
    3. the static per-region ``intensities`` map;
    4. ``default_intensity``.

    ``time_series`` maps a region to a time-sorted list of ``(timestamp_seconds,
    gCO2e/kWh)`` points; it is sorted on construction.
    """

    intensities: Dict[str, float] = field(default_factory=dict)
    default_intensity: float = 400.0
    time_series: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    provider: Optional[IntensityProvider] = None

    def __post_init__(self) -> None:
        self.time_series = {r: sorted(pts) for r, pts in self.time_series.items()}

    def carbon_intensity(self, region: str, t: Optional[float] = None) -> float:
        if self.provider is not None:
            live = self.provider.get(region, t)
            if live is not None:
                return live
        if t is not None and self.time_series.get(region):
            return _interpolate(self.time_series[region], t)
        return self.intensities.get(region, self.default_intensity)


def carbon_for_tokens(
    cost_model: CostModel,
    carbon_model: CarbonModel,
    model: str,
    tokens: float,
    region: str,
    t: Optional[float] = None,
) -> float:
    """Predicted carbon (gCO2e) for ``tokens`` tokens of ``model`` in ``region``."""
    energy = cost_model.energy_kwh(model, tokens)
    return energy * carbon_model.carbon_intensity(region, t)


def default_cost_model() -> TableCostModel:
    """A conservative default cost model with a single fallback profile."""
    return TableCostModel()


def default_carbon_model() -> TableCarbonModel:
    """A default carbon model using a global-average grid intensity."""
    return TableCarbonModel()
