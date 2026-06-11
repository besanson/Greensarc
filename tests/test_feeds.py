"""Live-feed loaders: LiteLLM prices + ElectricityMaps carbon, all offline.

No network: the LiteLLM loader reads a committed fixture path, and the carbon
provider is driven by an injected opener returning a recorded payload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from green_sarc.carbon_feeds import ElectricityMapsKappa
from green_sarc.pricing import IntensityProvider, TableCarbonModel
from green_sarc.pricing_loaders import load_litellm_prices

FIX = Path(__file__).parent / "fixtures"


# --- LiteLLM price loader ---------------------------------------------------

def test_litellm_loader_maps_usd_and_skips_meta():
    cm = load_litellm_prices(str(FIX / "litellm_prices_sample.json"))
    # Real models load with their per-token USD.
    assert cm.usd("gpt-4o", 1_000_000, 0) == pytest.approx(2.5)
    assert cm.usd("gpt-4o", 0, 1_000_000) == pytest.approx(10.0)
    assert cm.usd("claude-3-5-sonnet-20241022", 1_000_000, 0) == pytest.approx(3.0)
    # The "sample_spec" doc stub and the cost-free entry are skipped.
    assert "sample_spec" not in cm.profiles
    assert "some-free-router-model" not in cm.profiles


def test_litellm_loader_unknown_model_falls_back_to_default():
    cm = load_litellm_prices(str(FIX / "litellm_prices_sample.json"))
    with pytest.warns(UserWarning, match="unknown model"):
        # Unknown model: default profile (zero USD), preserving existing behaviour.
        assert cm.usd("no-such-model", 1000, 1000) == 0.0


def test_litellm_loader_via_opener():
    raw = (FIX / "litellm_prices_sample.json").read_bytes()
    cm = load_litellm_prices(opener=lambda url: raw)
    assert cm.usd("gpt-4o", 1_000_000, 0) == pytest.approx(2.5)


# --- ElectricityMaps carbon feed -------------------------------------------

def _fixture_opener():
    raw = (FIX / "electricitymaps_latest.json").read_bytes()
    calls = {"n": 0}

    def opener(url, headers):
        calls["n"] += 1
        assert "auth-token" in headers  # key forwarded via header, never in URL
        return raw

    return opener, calls


def test_em_provider_is_intensity_provider(monkeypatch):
    monkeypatch.setenv("ELECTRICITYMAPS_API_KEY", "test-key")
    opener, _ = _fixture_opener()
    p = ElectricityMapsKappa("IT", opener=opener)
    assert isinstance(p, IntensityProvider)
    assert p.get() == 247.0


def test_em_provider_caches_within_ttl(monkeypatch):
    monkeypatch.setenv("ELECTRICITYMAPS_API_KEY", "test-key")
    opener, calls = _fixture_opener()
    now = [1000.0]
    p = ElectricityMapsKappa("IT", opener=opener, ttl_s=100.0, clock=lambda: now[0])
    assert p.get() == 247.0
    now[0] = 1050.0  # within TTL -> served from cache, no second fetch
    assert p.get() == 247.0
    assert calls["n"] == 1


def test_em_provider_stale_fallback_on_failure(monkeypatch):
    monkeypatch.setenv("ELECTRICITYMAPS_API_KEY", "test-key")
    raw = (FIX / "electricitymaps_latest.json").read_bytes()
    state = {"fail": False}

    def opener(url, headers):
        if state["fail"]:
            raise OSError("network down")
        return raw

    now = [0.0]
    p = ElectricityMapsKappa("IT", opener=opener, ttl_s=10.0, clock=lambda: now[0])
    assert p.get() == 247.0  # primes the cache
    now[0] = 100.0           # past TTL, force a refetch
    state["fail"] = True
    with pytest.warns(UserWarning, match="falling back"):
        assert p.get() == 247.0  # serves the stale cached value


def test_em_provider_no_key_no_cache_returns_none(monkeypatch):
    monkeypatch.delenv("ELECTRICITYMAPS_API_KEY", raising=False)
    p = ElectricityMapsKappa("IT", opener=lambda u, h: b"{}")
    with pytest.warns(UserWarning):
        assert p.get() is None


def test_em_provider_feeds_table_carbon_model(monkeypatch, tmp_path):
    monkeypatch.setenv("ELECTRICITYMAPS_API_KEY", "test-key")
    opener, _ = _fixture_opener()
    p = ElectricityMapsKappa("IT", opener=opener, cache_path=str(tmp_path / "k.json"))
    carbon = TableCarbonModel(intensities={"it-north": 999.0}, provider=p)
    # Live provider value takes precedence over the static table.
    assert carbon.carbon_intensity("it-north") == 247.0
    # And it was persisted to the cache file.
    assert json.loads((tmp_path / "k.json").read_text())["intensity"] == 247.0
