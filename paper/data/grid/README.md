# Grid intensity data (§11.5)

`load_grid.py` returns `kappa(t) -> gCO2eq/kWh` for a zone.

- **GB-north-scotland`, `GB-london`** — fetched automatically from the open UK
  Carbon Intensity API (https://carbonintensity.org.uk, no account) and cached
  here as `{zone}_hourly_2024.csv` (committed, ~30 KB each).
- **stipulated** — the benchmark's synthetic daily sine curve (no file).
- **IT, US-CAISO, or any other zone** — not reachable without an account in a
  sandbox. To use them, export an hourly 2024 CSV with columns `time_s,intensity`
  from ElectricityMaps (https://app.electricitymaps.com/datasets, free tier),
  ENTSO-E, or CAISO OASIS, and commit it as `{zone}_hourly_2024.csv`. It is then
  picked up automatically (`--grid-zone IT`).
