"""Live-market data ingestion for the financial twin.

Provides 3 anchor values needed by FinancialTwin to start a sim from
*today's* market context instead of static defaults:
  - Euribor 3M  (Italian / euro-area short rate proxy)
  - BTP-Bund 10Y spread (sovereign risk premium IT)
  - ECB Deposit Facility Rate (policy rate)

Sources (in priority order, all free, no enterprise API tier needed):
  1. ECB Statistical Data Warehouse (SDW) — most authoritative for euro
     area; no API key required. Uses jsondata endpoint.
  2. yfinance (already a dependency) — broader coverage, used as fallback
     for BTP-Bund yield calc.
  3. Hardcoded defaults — last-resort fallback.

Caching: results stored in `outputs/market_data_cache.json` with a
24h TTL so we don't hammer external APIs in a tight retry loop. The
cache is process-shared (read on every call, written on successful
fetch).

This module is OPT-IN: FinancialTwin.refresh_market_anchors() must be
called explicitly to overwrite the params. Default behaviour is
unchanged (uses literature-based defaults).

Failure mode: every public function returns the cached/default value
on any error and logs a warning. No exception escapes to the caller.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# Cache lives next to the simulations.json so it survives Railway redeploys
# when the volume is mounted at /app/outputs.
_CACHE_PATH = os.path.join(
    os.environ.get("DTS_OUTPUTS_DIR")
    or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs"),
    "market_data_cache.json",
)
_CACHE_TTL_SECONDS = 24 * 3600  # 24h
_HTTP_TIMEOUT = 4.0

# Default values when network fails AND no cache (Italian commercial-bank
# benchmarks, late April 2026 — kept consistent with twin defaults).
_DEFAULTS = {
    "euribor_3m_pct": 2.40,
    "btp_bund_spread_bps": 95.0,
    "ecb_dfr_pct": 2.40,
}


# ── Cache helpers ───────────────────────────────────────────────────────────

def _read_cache() -> dict:
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f) or {}
    except Exception as exc:
        logger.warning(f"market_data cache read failed: {exc}")
        return {}


def _write_cache(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning(f"market_data cache write failed: {exc}")


def _is_fresh(entry: dict) -> bool:
    ts = entry.get("ts", 0)
    return (time.time() - ts) < _CACHE_TTL_SECONDS


def _cache_get(key: str) -> Optional[float]:
    cache = _read_cache()
    entry = cache.get(key)
    if entry and _is_fresh(entry):
        return float(entry.get("value", 0))
    return None


def _cache_set(key: str, value: float) -> None:
    cache = _read_cache()
    cache[key] = {"value": float(value), "ts": time.time()}
    _write_cache(cache)


# ── ECB Statistical Data Warehouse fetcher ─────────────────────────────────

_ECB_BASE = "https://data-api.ecb.europa.eu/service/data"


def _fetch_ecb_series(flow: str, key: str) -> Optional[float]:
    """Fetch the latest observation of an ECB series.

    Args:
        flow: dataflow id (e.g. 'FM' for financial markets, 'MIR' for
              monetary financial institutions interest rates)
        key: series key (dot-separated)

    Returns:
        Float = the last numeric observation, or None on failure.
    """
    url = f"{_ECB_BASE}/{flow}/{key}?lastNObservations=1&format=jsondata"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "DigitalTwinSim/0.6 (+contact via repo)"}
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        logger.warning(f"ECB SDW fetch failed for {flow}/{key}: {exc}")
        return None

    # ECB jsondata format: dataSets[0].series[<key>].observations["<idx>"] = [value, ...]
    try:
        ds = payload["dataSets"][0]["series"]
        for series_data in ds.values():
            obs = series_data.get("observations", {})
            for v in obs.values():
                if v and v[0] is not None:
                    return float(v[0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning(f"ECB SDW parse failed for {flow}/{key}: {exc}")
    return None


# ── yfinance fallback for BTP/Bund yields ──────────────────────────────────

def _fetch_yfinance_yield(ticker: str) -> Optional[float]:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        # Use 5-day history to be robust against bank-holiday gaps
        hist = t.history(period="5d", interval="1d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception as exc:
        logger.warning(f"yfinance fetch failed for {ticker}: {exc}")
    return None


# ── Public anchor getters ──────────────────────────────────────────────────

def get_euribor_3m_pct(use_cache: bool = True) -> float:
    """Latest Euribor 3M, percentage units (e.g. 2.40 = 2.40% annual).

    Source priority: ECB SDW (FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA) →
    cache → default 2.40.
    """
    if use_cache:
        v = _cache_get("euribor_3m_pct")
        if v is not None:
            return v
    val = _fetch_ecb_series("FM", "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA")
    if val is None:
        val = _DEFAULTS["euribor_3m_pct"]
    _cache_set("euribor_3m_pct", val)
    return val


def get_btp_bund_spread_bps(use_cache: bool = True) -> float:
    """Latest BTP-Bund 10Y spread, in basis points (e.g. 95 = 95 bps).

    Computed as IT 10Y yield - DE 10Y yield from yfinance tickers
    (BTPS.MI for BTP futures proxy is unstable; use direct yield series
    when available). Falls back to default 95bps.
    """
    if use_cache:
        v = _cache_get("btp_bund_spread_bps")
        if v is not None:
            return v
    # yfinance: ^BTP10Y / ^BUND10Y are unreliable; we use ECB SDW yield series
    # for both. Series IDs: IRS.M.IT.L.L40.CI.0000.EUR.N.Z (IT 10Y),
    # IRS.M.DE.L.L40.CI.0000.EUR.N.Z (DE 10Y). May not always be available.
    it_yield = _fetch_ecb_series("IRS", "M.IT.L.L40.CI.0000.EUR.N.Z")
    de_yield = _fetch_ecb_series("IRS", "M.DE.L.L40.CI.0000.EUR.N.Z")
    if it_yield is not None and de_yield is not None:
        val = (it_yield - de_yield) * 100.0  # pp → bps
    else:
        val = _DEFAULTS["btp_bund_spread_bps"]
    _cache_set("btp_bund_spread_bps", val)
    return val


def get_ecb_dfr_pct(use_cache: bool = True) -> float:
    """Latest ECB Deposit Facility Rate, percentage units."""
    if use_cache:
        v = _cache_get("ecb_dfr_pct")
        if v is not None:
            return v
    val = _fetch_ecb_series("FM", "B.U2.EUR.4F.KR.DFR.LEV")
    if val is None:
        val = _DEFAULTS["ecb_dfr_pct"]
    _cache_set("ecb_dfr_pct", val)
    return val


# ── Convenience: bundle for FinancialTwin.refresh_market_anchors() ────────

def fetch_all_anchors(use_cache: bool = True) -> dict:
    """Fetch all 3 anchor values. Returns a dict keyed by twin param name.

    Example:
        twin = FinancialTwin()
        twin.params.update(fetch_all_anchors())
        # now twin uses today's actual rates
    """
    return {
        "policy_rate_pct": get_ecb_dfr_pct(use_cache=use_cache),
        "btp_bund_spread_bps": get_btp_bund_spread_bps(use_cache=use_cache),
        # Euribor doesn't directly map onto twin params today, but expose it
        # for any downstream consumer (e.g. mortgage variable rate model).
        "_euribor_3m_pct": get_euribor_3m_pct(use_cache=use_cache),
        "_fetched_at": time.time(),
    }
