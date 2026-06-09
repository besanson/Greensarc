"""Grid carbon-intensity loader for the §11.5 real-grid study.

Returns a callable ``kappa(t_seconds) -> gCO2eq/kWh`` for a named zone, by linear
interpolation over a real measured hourly/half-hourly series.

Sources, in priority order:
  1. A committed/cached CSV at ``paper/data/grid/{zone}_hourly_2024.csv`` with
     columns ``time_s,intensity`` (this is where an operator drops ElectricityMaps
     / ENTSO-E / CAISO exports for zones such as ``IT`` or ``US-CAISO``).
  2. For the ``GB-*`` zones used in the paper, real measured data fetched from the
     UK Carbon Intensity API (https://carbonintensity.org.uk, fully open, no
     account), then cached to the CSV path above.
  3. ``stipulated`` returns the benchmark's synthetic daily sine curve.

ElectricityMaps' historical portal, ENTSO-E, and CAISO OASIS all require an
account or are not reachable from a sandboxed agent; the UK Carbon Intensity API
is the one fully-open real source, so the paper's two real zones are GB regions
with contrasting generation mixes. To use IT/CAISO instead, export hourly 2024
CSVs to ``paper/data/grid/IT_hourly_2024.csv`` etc. and they will be picked up.
"""

from __future__ import annotations

import math
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
GRID_DIR = ROOT / "paper" / "data" / "grid"

# GB regions with deliberately contrasting mixes (region ids per the UK API).
GB_REGIONS = {"GB-north-scotland": 1, "GB-london": 13}
_WINDOW_DAYS = 14
_WINDOW_START = "2024-06-01T00:00Z"

_CACHE: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}


def _stipulated_series() -> List[Tuple[float, float]]:
    # The IBP daily sine curve (gCO2e/kWh): cleaner overnight, dirtier midday.
    return [(float(h * 3600), 250.0 + 180.0 * math.sin((h - 6) / 24.0 * 2 * math.pi))
            for h in range(24)]


def _fetch_gb(region_id: int) -> List[Tuple[float, float]]:
    """Real half-hourly regional intensity from the UK Carbon Intensity API."""
    start = datetime.strptime(_WINDOW_START, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    series: List[Tuple[float, float]] = []
    cur = start
    for _ in range((_WINDOW_DAYS + 1) // 2):  # API caps a request at ~2 days
        a = cur.strftime("%Y-%m-%dT%H:%MZ")
        b = (cur + timedelta(days=2)).strftime("%Y-%m-%dT%H:%MZ")
        url = f"https://api.carbonintensity.org.uk/regional/intensity/{a}/{b}/regionid/{region_id}"
        with urllib.request.urlopen(url, timeout=40) as resp:
            import json
            payload = json.loads(resp.read())
        rows = payload.get("data", {}).get("data", [])
        for r in rows:
            ts = datetime.strptime(r["from"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            inten = r["intensity"].get("actual") or r["intensity"].get("forecast")
            if inten is not None:
                series.append((ts.timestamp(), float(inten)))
        cur += timedelta(days=2)
    # de-dup and sort
    series = sorted(dict(series).items())
    return series


def _series(zone: str, year: int) -> List[Tuple[float, float]]:
    if (zone, year) in _CACHE:
        return _CACHE[(zone, year)]
    if zone == "stipulated":
        s = _stipulated_series()
        _CACHE[(zone, year)] = s
        return s
    csv = GRID_DIR / f"{zone}_hourly_{year}.csv"
    if csv.exists():
        s = []
        for line in csv.read_text().splitlines()[1:]:
            t, k = line.split(",")
            s.append((float(t), float(k)))
        _CACHE[(zone, year)] = sorted(s)
        return _CACHE[(zone, year)]
    if zone in GB_REGIONS:
        s = _fetch_gb(GB_REGIONS[zone])
        GRID_DIR.mkdir(parents=True, exist_ok=True)
        csv.write_text("time_s,intensity\n" + "\n".join(f"{t},{k}" for t, k in s), encoding="utf-8")
        _CACHE[(zone, year)] = s
        return s
    # Unknown zone (e.g. IT, US-CAISO) with no committed CSV: instruct and stop.
    sys.stderr.write(
        f"\nGrid zone '{zone}' has no data at {csv}.\n"
        "Export an hourly {year} CSV (columns time_s,intensity) from ElectricityMaps\n"
        "(https://app.electricitymaps.com/datasets, free tier), ENTSO-E, or CAISO,\n"
        "commit it there, and re-run. GB-* zones fetch automatically (no account).\n"
    )
    raise SystemExit(2)


def load_kappa(zone: str, year: int = 2024) -> Callable[[float], float]:
    """Return kappa(t_seconds) -> gCO2eq/kWh by interpolating the zone's series.

    Time is wrapped into the available window so an arbitrary workload timestamp
    maps onto a real intensity sample (the series is treated as periodic)."""
    s = _series(zone, year)
    times = [t for t, _ in s]
    vals = [v for _, v in s]
    t0, t1 = times[0], times[-1]
    span = (t1 - t0) or 1.0

    def kappa(t: float) -> float:
        x = t0 + ((t - t0) % span)
        # binary-free linear scan is fine for a ~700-point series
        lo = 0
        for i in range(1, len(times)):
            if times[i] >= x:
                lo = i - 1
                break
        else:
            return vals[-1]
        x0, x1 = times[lo], times[lo + 1]
        w = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        return vals[lo] * (1 - w) + vals[lo + 1] * w

    return kappa


def list_available_zones(data_dir: Path = GRID_DIR) -> List[str]:
    zones = ["stipulated", *GB_REGIONS.keys()]
    if data_dir.exists():
        for p in data_dir.glob("*_hourly_*.csv"):
            z = p.stem.rsplit("_hourly_", 1)[0]
            if z not in zones:
                zones.append(z)
    return zones


def mean_intensity(zone: str, year: int = 2024) -> float:
    s = _series(zone, year)
    return sum(v for _, v in s) / len(s)


if __name__ == "__main__":
    for z in ("stipulated", *GB_REGIONS.keys()):
        try:
            print(f"{z:20s} mean kappa = {mean_intensity(z):.1f} gCO2e/kWh "
                  f"({len(_series(z, 2024))} samples)")
        except SystemExit:
            pass
