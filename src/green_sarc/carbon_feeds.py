"""Live carbon-intensity feed: ElectricityMaps as an :class:`IntensityProvider`.

Generalizes the paper's ``fetch_grid.py`` into a runtime
:class:`~green_sarc.pricing.IntensityProvider` that :class:`TableCarbonModel`
can consume, so the carbon term ``kappa(rho, t)`` is a *live* grid signal rather
than a stipulated constant.

Stdlib only (``urllib``) — no new dependency, so no extra is required to use it
(the optional ``feeds`` story is documented in the README).  The API key is read
from the ``ELECTRICITYMAPS_API_KEY`` environment variable only; it is never a
constructor argument and is never logged.

Resilience: a successful fetch is cached to disk with a timestamp; within
``ttl_s`` the cache is served without a network call, and on any fetch failure
(network, auth, parse) the last cached value is returned with a one-time warning.
A provider with neither a live response nor a cache returns ``None`` so
:class:`TableCarbonModel` falls back to its table.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Callable, Optional

__all__ = ["ElectricityMapsKappa", "API_BASE"]

API_BASE = "https://api.electricitymap.org/v3"

# An opener maps (url, headers) -> raw response bytes; injectable for offline tests.
Opener = Callable[[str, dict], bytes]


def _urllib_opener(url: str, headers: dict) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=40) as resp:  # noqa: S310 - https API
        return bytes(resp.read())


class ElectricityMapsKappa:
    """Live ``gCO2e/kWh`` for one ElectricityMaps zone, with on-disk caching.

    Parameters
    ----------
    zone:
        ElectricityMaps zone code (e.g. ``"IT"``, ``"US-CAL-CISO"``).
    cache_path:
        Optional JSON file persisting the last good reading across restarts.
    ttl_s:
        Serve the cached reading without a network call for this many seconds.
    base_url, opener, clock:
        Injection seams for testing; defaults hit the live HTTPS API.
    """

    def __init__(
        self,
        zone: str,
        *,
        cache_path: Optional[str] = None,
        ttl_s: float = 3600.0,
        base_url: str = API_BASE,
        opener: Optional[Opener] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.zone = zone
        self.ttl_s = float(ttl_s)
        self.base_url = base_url.rstrip("/")
        self._opener = opener or _urllib_opener
        self._clock = clock
        self._cache_path = Path(cache_path) if cache_path else None
        self._mem: Optional[tuple] = None  # (fetched_at, intensity)
        self._warned = False
        if self._cache_path and self._cache_path.exists():
            try:
                d = json.loads(self._cache_path.read_text(encoding="utf-8"))
                self._mem = (float(d["fetched_at"]), float(d["intensity"]))
            except (ValueError, KeyError, OSError):
                self._mem = None

    def _api_key(self) -> Optional[str]:
        # Key is read from the environment only — never an argument, never logged.
        return os.environ.get("ELECTRICITYMAPS_API_KEY")

    def _fetch_latest(self) -> float:
        key = self._api_key()
        if not key:
            raise RuntimeError("ELECTRICITYMAPS_API_KEY not set")
        url = f"{self.base_url}/carbon-intensity/latest?zone={self.zone}"
        payload = json.loads(self._opener(url, {"auth-token": key}))
        ci = payload.get("carbonIntensity")
        if ci is None:
            raise ValueError(f"no carbonIntensity in response for zone {self.zone}")
        return float(ci)

    def _persist(self, fetched_at: float, intensity: float) -> None:
        self._mem = (fetched_at, intensity)
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps({"zone": self.zone, "fetched_at": fetched_at, "intensity": intensity}),
                encoding="utf-8",
            )

    def get(self, region: Optional[str] = None, when: Optional[float] = None) -> Optional[float]:
        """Current intensity for the zone (the ``region``/``when`` args are ignored).

        Serves a fresh cache within ``ttl_s``; otherwise fetches live and caches.
        On any failure, returns the last cached value (warning once) or ``None``.
        """
        now = self._clock()
        if self._mem is not None and (now - self._mem[0]) < self.ttl_s:
            return self._mem[1]
        try:
            intensity = self._fetch_latest()
            self._persist(now, intensity)
            return intensity
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on any feed error
            if not self._warned:
                self._warned = True
                stale = self._mem[1] if self._mem else None
                warnings.warn(
                    f"ElectricityMapsKappa({self.zone}): live fetch failed ({exc}); "
                    f"falling back to {'last cached value' if stale is not None else 'None'}.",
                    stacklevel=2,
                )
            return self._mem[1] if self._mem is not None else None
