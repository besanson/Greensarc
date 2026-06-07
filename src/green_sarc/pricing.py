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
from typing import Dict, Optional, Protocol, runtime_checkable

__all__ = [
    "ModelProfile",
    "CostModel",
    "CarbonModel",
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


@dataclass
class TableCarbonModel:
    """Table-driven :class:`CarbonModel` keyed by region.

    The table is static (a simple ``region -> gCO2e/kWh`` map); the ``t``
    argument is accepted so callers may later supply a time-varying grid signal
    without changing the interface.
    """

    intensities: Dict[str, float] = field(default_factory=dict)
    default_intensity: float = 400.0

    def carbon_intensity(self, region: str, t: Optional[float] = None) -> float:
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
