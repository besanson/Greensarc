"""Build a :class:`TableCostModel` from LiteLLM's public price catalogue.

LiteLLM publishes ``model_prices_and_context_window.json`` (per-token input/output
USD for hundreds of models).  :func:`load_litellm_prices` maps the two fields the
gate needs --- ``input_cost_per_token`` / ``output_cost_per_token`` --- onto
:class:`ModelProfile` USD coefficients, so monetary budgets track real provider
prices without hand-maintaining a table.

LiteLLM carries no per-token *energy* figure, so each profile keeps a single
configurable ``energy_per_token_kwh`` (the carbon path stays table/feed-driven).
Unknown models still fall back to ``TableCostModel.default_profile`` with the
existing one-time warning.

Stdlib only (``urllib``); pass a local ``source`` path to load offline / in CI.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from green_sarc.pricing import ModelProfile, TableCostModel

__all__ = ["load_litellm_prices", "LITELLM_PRICES_URL"]

LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

Opener = Callable[[str], bytes]


def _urllib_opener(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=40) as resp:  # noqa: S310 - https raw file
        return bytes(resp.read())


def load_litellm_prices(
    source: Optional[str] = None,
    *,
    energy_per_token_kwh: float = 3.0e-7,
    opener: Optional[Opener] = None,
) -> TableCostModel:
    """Return a :class:`TableCostModel` populated from the LiteLLM catalogue.

    Parameters
    ----------
    source:
        A local path to a downloaded catalogue, or ``None`` to fetch the latest
        from ``LITELLM_PRICES_URL`` (``--refresh`` semantics live in the caller).
    energy_per_token_kwh:
        Per-token energy assigned to every loaded profile (LiteLLM has no energy
        figure); the carbon path remains feed/table-driven.
    opener:
        Injection seam for tests; defaults to a stdlib ``urllib`` fetch.
    """
    if source is not None:
        raw = Path(source).read_text(encoding="utf-8")
    else:
        raw = (opener or _urllib_opener)(LITELLM_PRICES_URL).decode("utf-8")
    catalogue: Dict[str, Any] = json.loads(raw)

    profiles: Dict[str, ModelProfile] = {}
    for model, spec in catalogue.items():
        if model == "sample_spec" or not isinstance(spec, dict):
            continue  # LiteLLM's documentation sentinel, not a real model
        in_cost = spec.get("input_cost_per_token")
        out_cost = spec.get("output_cost_per_token")
        if in_cost is None and out_cost is None:
            continue  # skips meta entries like "sample_spec" and free/unknown models
        profiles[model] = ModelProfile(
            energy_per_token_kwh=energy_per_token_kwh,
            usd_per_prompt_token=float(in_cost or 0.0),
            usd_per_completion_token=float(out_cost or 0.0),
        )
    return TableCostModel(
        profiles=profiles,
        default_profile=ModelProfile(energy_per_token_kwh=energy_per_token_kwh),
    )
