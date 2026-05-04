"""Real-price tracking for tickers mentioned in a scenario brief.

Replaces the old "ticker as 0-100 LLM score" anti-pattern (which produced
absurd outputs like "TLIT.MI Stock Price: 32" while TIM trades at €0.25)
with deterministic, anchor-based pricing:

  current_price[t] = anchor_price * (1 + cumulative_pct_move[t])
  cumulative_pct_move[t] = sum over rounds of predict_returns(cri, intensity)

The anchor is fetched once at simulation start via yfinance (24h-cached on
disk), and per-round shocks come from the empirical financial-impact
pieces already shipped: sector betas + impulse response + panic multiplier
(same path used by the nightly self-calibration loop).

Public surface:
  extract_tickers(text)         -> tickers found in `text` that exist in
                                    shared/stock_universe.json
  fetch_anchor_prices(tickers)  -> {ticker: {price, currency, name, country, sector}}
  step(state, cri, intensity)   -> mutate state with the round's price /
                                    pct move; returns serialisable dict
                                    suitable for round_result["ticker_prices"]
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UNIVERSE_PATH = REPO_ROOT / "shared" / "stock_universe.json"
ANCHOR_CACHE_PATH = REPO_ROOT / "outputs" / "ticker_anchor_cache.json"
ANCHOR_TTL_SEC = 12 * 3600  # 12h: spot can move intraday but not enough to invalidate a demo


# ── Universe loading ─────────────────────────────────────────────────────


_UNIVERSE_CACHE: Optional[dict] = None


def _universe() -> dict:
    """Load stock_universe.json once, cache in-process."""
    global _UNIVERSE_CACHE
    if _UNIVERSE_CACHE is not None:
        return _UNIVERSE_CACHE
    try:
        data = json.loads(UNIVERSE_PATH.read_text())
        meta: dict = {}
        for s in data.get("stocks", []):
            t = s.get("ticker")
            if t:
                meta[t] = s
        for idx in data.get("indices", []):
            t = idx.get("ticker")
            if t:
                meta[t] = {**idx, "sector": "index", "type": "index"}
        _UNIVERSE_CACHE = meta
        return meta
    except Exception as e:
        logger.warning("could not load stock universe: %s", e)
        _UNIVERSE_CACHE = {}
        return _UNIVERSE_CACHE


# ── Ticker extraction ────────────────────────────────────────────────────


# Allow MIC suffixes (.MI, .DE, .PA, .L, .AS, .MC, .SS, .HK, etc.) and pure
# US tickers up to 5 letters. Rejects all-digits and very common English
# words by intersecting with the loaded universe afterwards.
#
# Lookbehind/lookahead use [A-Za-z0-9] (NOT \w) so that underscores act as
# terminators — handles user-typed KPI labels like "TIT.MI_Prezzo_Azione"
# where \b would fail on the MI→_ transition because \w includes underscore.
_TICKER_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{0,5}(?:[.\-][A-Z]{1,4})?)(?![A-Za-z0-9])"
)


def extract_tickers(text: str, limit: int = 10) -> list[str]:
    """Find every ticker substring in `text` that exists in the universe.

    Order-preserving, deduplicated. `limit` caps how many tickers we
    return (a very long brief mentioning dozens of names would otherwise
    blow the per-round yfinance fan-out)."""
    if not text:
        return []
    universe = _universe()
    if not universe:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _TICKER_RE.finditer(text):
        candidate = m.group(1)
        if candidate in universe and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
            if len(out) >= limit:
                break
    return out


def is_ticker(metric_name: str) -> bool:
    """Is this user-provided metric label actually a ticker we can price?"""
    return bool(extract_tickers(metric_name, limit=1))


# ── Anchor cache ─────────────────────────────────────────────────────────


def _load_anchor_cache() -> dict:
    if not ANCHOR_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(ANCHOR_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_anchor_cache(cache: dict) -> None:
    try:
        ANCHOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        ANCHOR_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    except OSError as e:
        logger.warning("anchor cache write failed: %s", e)


def fetch_anchor_prices(tickers: list[str]) -> dict[str, dict]:
    """Fetch (or read from 12h cache) spot prices for `tickers`.

    Returns a dict ticker -> {price, currency, name, country, sector,
    fetched_at}. Tickers that yfinance can't resolve are simply omitted —
    no exception escapes to the caller, demo runs must not fail just
    because one ticker is unknown.
    """
    if not tickers:
        return {}
    cache = _load_anchor_cache()
    now = time.time()
    out: dict[str, dict] = {}
    to_fetch: list[str] = []
    universe = _universe()

    for tk in tickers:
        cached = cache.get(tk)
        if cached and (now - cached.get("fetched_at", 0)) < ANCHOR_TTL_SEC:
            out[tk] = cached
        else:
            to_fetch.append(tk)

    if to_fetch:
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed; ticker prices unavailable")
            return out
        for tk in to_fetch:
            try:
                t = yf.Ticker(tk)
                hist = t.history(period="5d")
                if hist is None or hist.empty or "Close" not in hist:
                    logger.debug("no recent history for %s", tk)
                    continue
                price = float(hist["Close"].iloc[-1])
                if not (price > 0):
                    continue
                meta = universe.get(tk, {})
                # yfinance currency: try fast_info, fallback to country guess
                currency = "USD"
                try:
                    fi = t.fast_info
                    currency = getattr(fi, "currency", None) or fi.get("currency", "USD")
                except Exception:
                    pass
                if not currency:
                    cc = meta.get("country") or "US"
                    currency = {
                        "IT": "EUR", "DE": "EUR", "FR": "EUR", "ES": "EUR",
                        "NL": "EUR", "FI": "EUR", "PT": "EUR", "AT": "EUR",
                        "GB": "GBP", "CH": "CHF", "JP": "JPY", "HK": "HKD",
                        "MX": "MXN", "BR": "BRL", "SE": "SEK",
                    }.get(cc, "USD")
                out[tk] = {
                    "price": price,
                    "currency": currency,
                    "name": meta.get("name", tk),
                    "country": meta.get("country", "US"),
                    "sector": meta.get("sector", "unknown"),
                    "fetched_at": now,
                }
                cache[tk] = out[tk]
            except Exception as e:
                logger.debug("anchor fetch failed for %s: %s", tk, e)
        if out:
            _save_anchor_cache(cache)
    return out


# ── Per-round simulation ─────────────────────────────────────────────────


class TickerPriceState:
    """Holds anchors + cumulative % move per ticker across rounds.

    Use as: state = TickerPriceState(brief_text); for round in 1..N:
    snapshot = state.step(cri, intensity)."""

    def __init__(self, brief_text: str, extra_metric_names: Optional[list[str]] = None):
        names_blob = brief_text or ""
        if extra_metric_names:
            names_blob += "\n" + "\n".join(extra_metric_names)
        tickers = extract_tickers(names_blob, limit=10)
        self.anchors: dict[str, dict] = fetch_anchor_prices(tickers)
        # Cumulative % move from anchor (in percent points, e.g. -2.4 = -2.4%)
        self.cumulative_pct: dict[str, float] = {tk: 0.0 for tk in self.anchors}
        self._round_idx = 0

    @property
    def tickers(self) -> list[str]:
        return list(self.anchors)

    def step(self, cri: float, intensity: float) -> dict[str, dict]:
        """Advance one round and return per-ticker snapshot.

        cri ∈ [0, 1], intensity ∈ ~[0, 5]. Both come from the round
        manager's contagion + escalation engines. We map them through
        ``predict_returns`` which already uses the empirical sector betas
        + impulse response, then accumulate."""
        self._round_idx += 1
        if not self.anchors:
            return {}
        try:
            from core.calibration.continuous import predict_returns
            forecasts = predict_returns(
                tickers=list(self.anchors),
                cri=float(cri),
                intensity=float(intensity),
                universe_meta=_universe(),
            )
        except Exception as e:
            logger.warning("predict_returns failed in round %d: %s", self._round_idx, e)
            forecasts = []

        # Single-day per-round increment; we use the t1_pct of each ticker
        # as the "this round" move and accumulate. Could be tuned to use
        # t3 over 3-round blocks if we want longer horizons.
        per_round_pct: dict[str, float] = {f.ticker: float(f.t1_pct) for f in forecasts}

        snapshot: dict[str, dict] = {}
        for tk, anchor in self.anchors.items():
            delta = per_round_pct.get(tk, 0.0)
            self.cumulative_pct[tk] += delta
            cum = self.cumulative_pct[tk]
            current = anchor["price"] * (1.0 + cum / 100.0)
            snapshot[tk] = {
                "ticker": tk,
                "name": anchor["name"],
                "anchor_price": round(anchor["price"], 4),
                "current_price": round(current, 4),
                "round_pct": round(delta, 3),
                "cum_pct": round(cum, 3),
                "currency": anchor["currency"],
                "country": anchor["country"],
                "sector": anchor["sector"],
            }
        return snapshot
