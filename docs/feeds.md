# Live pricing & carbon feeds

Two optional loaders turn stipulated tables into live signals. Both use **only
the standard library** (`urllib`), so there is **no extra to install** — the
zero-dependency core stays honest. (`pip install "green-sarc[feeds]"` is a no-op
alias kept for forward-compatibility if a `requests`-based fetcher is ever added.)

## LiteLLM price loader

```python
from green_sarc.pricing_loaders import load_litellm_prices

# Fetch the latest catalogue from GitHub (BerriAI/litellm):
cost_model = load_litellm_prices()
# ...or load a downloaded copy offline / in CI:
cost_model = load_litellm_prices("model_prices_and_context_window.json")
```

Maps `input_cost_per_token` / `output_cost_per_token` onto `ModelProfile` USD
coefficients for every catalogued model. LiteLLM carries no per-token *energy*,
so each profile keeps a single configurable `energy_per_token_kwh`
(default `3e-7`); the carbon path stays feed/table-driven. Unknown models fall
back to the default profile with the usual one-time warning.

## ElectricityMaps carbon feed

```python
import os
from green_sarc.carbon_feeds import ElectricityMapsKappa
from green_sarc.pricing import TableCarbonModel

# Key is read from the env only — never a constructor argument, never logged.
os.environ["ELECTRICITYMAPS_API_KEY"] = "..."
kappa = ElectricityMapsKappa("IT", cache_path="~/.cache/green-sarc/it.json", ttl_s=3600)
carbon_model = TableCarbonModel(intensities={"eu-south": 250.0}, provider=kappa)
```

`ElectricityMapsKappa` implements `IntensityProvider`, so a `TableCarbonModel`
prefers its live `gCO2e/kWh` over the static table. Resilience:

- a successful reading is cached to disk and served without a network call for
  `ttl_s` seconds;
- on any fetch failure (network, auth, parse) the **last cached value** is
  returned with a one-time warning;
- with neither a live response nor a cache, `get()` returns `None`, and
  `TableCarbonModel` falls back to its table.

### A `--refresh` path

`get()` refetches automatically once the cache is older than `ttl_s`. There is
no separate refresh CLI in the library (the paper's `fetch_grid.py` covers the
offline-snapshot workflow); CI never exercises the network path — tests inject a
recorded payload via the `opener=` seam against fixtures in `tests/fixtures/`.
