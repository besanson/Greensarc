"""Fetch real hourly carbon intensity per zone from the ElectricityMaps v3 API.

Reads the API key from the ``ELECTRICITYMAPS_API_KEY`` environment variable (never
hard-coded). Writes a committed CSV per zone to ``paper/data/grid/<NAME>_hourly_2024.csv``
with the schema ``load_grid.py`` consumes (``time_s,intensity``), so §11.5
reproduces from a clean clone without API access. Re-fetch with ``--refresh``.

The free tier exposes only the most recent ~24 hours of history
(``/carbon-intensity/history``); the ``past-range`` and ``past`` endpoints return
401 on the free plan. We therefore cache a representative 24-hour measured window
per zone, which is sufficient to expose the diurnal pattern (e.g. CAISO's daytime
solar trough vs. its gas-heavy evening peak).

    ELECTRICITYMAPS_API_KEY=... python paper/scripts/fetch_grid.py \
        --zones IT,US-CAL-CISO --year 2024 [--refresh]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
GRID_DIR = ROOT / "paper" / "data" / "grid"
API_BASE = "https://api.electricitymap.org/v3"

# API zone code -> committed CSV basename (the paper's zone label).
ZONE_OUTPUT = {"IT": "IT", "US-CAL-CISO": "US-CAISO"}


def _require_key() -> str:
    key = os.environ.get("ELECTRICITYMAPS_API_KEY")
    if not key:
        sys.stderr.write(
            "Set ELECTRICITYMAPS_API_KEY env var or commit the cached CSVs at "
            "paper/data/grid/*.csv to reproduce without API access.\n"
        )
        raise SystemExit(2)
    print(f"loaded API key (len={len(key)})")
    return key


def _fetch_history(zone: str, key: str) -> List[Tuple[float, float]]:
    """Return [(seconds_from_window_start, gCO2eq/kWh)] for the last ~24h of `zone`."""
    req = urllib.request.Request(
        f"{API_BASE}/carbon-intensity/history?zone={zone}",
        headers={"auth-token": key},
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        payload = json.loads(resp.read())
    hist = payload.get("history", [])
    rows: List[Tuple[datetime, float]] = []
    for h in hist:
        ci = h.get("carbonIntensity")
        if ci is None:
            continue
        ts = datetime.fromisoformat(h["datetime"].replace("Z", "+00:00"))
        rows.append((ts, float(ci)))
    rows.sort()
    if not rows:
        raise SystemExit(f"no carbon-intensity history returned for zone {zone}")
    t0 = rows[0][0]
    return [((ts - t0).total_seconds(), ci) for ts, ci in rows]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fetch_grid", description=__doc__)
    p.add_argument("--zones", default="IT,US-CAL-CISO",
                   help="comma-separated ElectricityMaps zone codes.")
    p.add_argument("--year", default="2024", help="label used in the cached CSV filename.")
    p.add_argument("--refresh", action="store_true", help="re-fetch even if the CSV exists.")
    args = p.parse_args(argv)

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    zones = [z.strip() for z in args.zones.split(",") if z.strip()]
    key = None
    for zone in zones:
        name = ZONE_OUTPUT.get(zone, zone)
        out = GRID_DIR / f"{name}_hourly_{args.year}.csv"
        if out.exists() and not args.refresh:
            print(f"  {name}: cached ({out.name}); pass --refresh to re-fetch")
            continue
        if key is None:
            key = _require_key()
        series = _fetch_history(zone, key)
        out.write_text("time_s,intensity\n" + "\n".join(f"{t},{k}" for t, k in series),
                       encoding="utf-8")
        mean = sum(k for _, k in series) / len(series)
        print(f"  {name}: wrote {len(series)} hourly points, mean {mean:.1f} gCO2eq/kWh -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
