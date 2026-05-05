"""DORA economic-impact — Sprint B (live data ingestion).

STATUS: SCAFFOLDING ONLY.  All functions return cached/static data
from the existing JSON files. The interface is final; the
implementations are TODO and will land in Sprint B.

When implemented, this module will:
  1. Refresh ticker market caps daily from yfinance (shares × close)
     and overwrite shared/ticker_market_caps.json on disk.
  2. Pull realised cost annotations for new historical incidents
     (LLM-extracted from FT / Reuters / regulatory press) and propose
     additions to shared/dora_reference_incidents.json via PR.
  3. Track sovereign-spread snapshots (BTP-Bund, OAT-Bund) so the
     anchor α can be re-fit conditioned on regime.

Cron entry-point: scripts/refresh_dora_live_data.py (also TODO).
Wire as admin job "dora-refresh-live" so the operator can trigger
manually from /admin/jobs.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CACHE_TTL_SEC = 24 * 3600


def refresh_market_caps(force: bool = False, max_tickers: int = 100) -> dict:
    """Refresh shared/ticker_market_caps.json from yfinance.

    For each ticker already in the static caps file (curated, ~45
    rows): fetch shares_outstanding × close, convert to EUR via a
    USD/GBP/EUR fx snapshot, write back rounded to nearest 10M EUR.
    24h disk cache via _last_refreshed_at marker.

    Returns a structured summary the admin-jobs UI can render.
    """
    caps_path = REPO_ROOT / "shared" / "ticker_market_caps.json"
    if not caps_path.exists():
        return {"status": "error", "note": f"caps file missing: {caps_path}"}

    try:
        data = json.loads(caps_path.read_text())
    except Exception as e:
        return {"status": "error", "note": f"caps file unreadable: {e}"}

    last = data.get("_last_refreshed_at", 0)
    if not force and last and (time.time() - last) < CACHE_TTL_SEC:
        return {
            "status": "cached",
            "tickers_refreshed": 0,
            "note": f"cache valid (age {int(time.time() - last)}s); use force=True to bypass",
        }

    try:
        import yfinance as yf
    except ImportError:
        return {"status": "error", "note": "yfinance not installed"}

    # FX snapshot: prefer ECB SDW rates if available, fall back to yfinance
    # cross-rates against EUR. Only the 3 majors we care about.
    fx_to_eur: dict[str, float] = {"EUR": 1.0}
    for ccy, yf_pair in (("USD", "EURUSD=X"), ("GBP", "EURGBP=X"),
                         ("CHF", "EURCHF=X"), ("JPY", "EURJPY=X")):
        try:
            h = yf.Ticker(yf_pair).history(period="5d")
            if h is not None and not h.empty:
                eur_per_ccy = 1.0 / float(h["Close"].iloc[-1])
                fx_to_eur[ccy] = eur_per_ccy
        except Exception as e:
            logger.debug(f"FX fetch {yf_pair} failed: {e}")

    caps_eur_m = data.get("caps_eur_millions", {})
    tickers = list(caps_eur_m)[:max_tickers]
    refreshed: dict[str, dict] = {}
    skipped: list[str] = []

    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            fi = t.fast_info
            shares = float(getattr(fi, "shares", 0) or fi.get("shares", 0) or 0)
            price = float(getattr(fi, "last_price", 0) or fi.get("last_price", 0) or 0)
            ccy = (getattr(fi, "currency", None) or fi.get("currency") or "USD").upper()
            if shares <= 0 or price <= 0:
                skipped.append(tk)
                continue
            mcap_native = shares * price
            fx = fx_to_eur.get(ccy, 1.0)
            mcap_eur_m = round((mcap_native * fx) / 1_000_000.0, 0)
            # Round to nearest 10M for storage stability
            rounded = round(mcap_eur_m / 10) * 10
            refreshed[tk] = {
                "old": caps_eur_m.get(tk),
                "new": rounded,
                "currency": ccy,
                "shares_m": round(shares / 1e6, 1),
                "price_native": round(price, 4),
            }
            caps_eur_m[tk] = rounded
        except Exception as e:
            logger.debug(f"mcap fetch {tk} failed: {e}")
            skipped.append(tk)

    if not refreshed:
        return {
            "status": "error",
            "tickers_refreshed": 0,
            "skipped": skipped,
            "note": "no tickers refreshed — yfinance returned empty for all",
        }

    # Persist
    data["caps_eur_millions"] = caps_eur_m
    data["_last_refreshed_at"] = int(time.time())
    data["_last_refreshed_human"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data["_fx_to_eur"] = {k: round(v, 6) for k, v in fx_to_eur.items()}
    try:
        caps_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        return {"status": "error", "note": f"write failed: {e}", "tickers_refreshed": len(refreshed)}

    # Compute drift summary so the admin-jobs log is useful
    drifts = []
    for tk, info in refreshed.items():
        old = info.get("old")
        new = info.get("new")
        if old and new and old > 0:
            pct = (new - old) / old * 100
            drifts.append((tk, pct, old, new))
    drifts.sort(key=lambda r: -abs(r[1]))
    return {
        "status": "ok",
        "tickers_refreshed": len(refreshed),
        "skipped": skipped,
        "fx_to_eur": fx_to_eur,
        "top_drift": [
            {"ticker": tk, "pct": round(pct, 2), "old_m": old, "new_m": new}
            for tk, pct, old, new in drifts[:10]
        ],
        "fetched_at": data["_last_refreshed_human"],
    }


def fetch_recent_incidents_from_news(
    since_days: int = 30,
    llm_extract: bool = True,
) -> list[dict]:
    """TODO Sprint B — discover new historical incidents to annotate.

    Plan:
      • Query a news feed (Reuters / FT / Bloomberg) for terms like
        "bank resolution", "bail-in", "ransomware cost", "regulatory
        fine", "sovereign downgrade" in the last `since_days`.
      • For each candidate article, run an LLM (gemini-3.1-pro-preview
        per Sprint C — same model) to extract:
            { entity, date, cost_eur_m, sources, category }
      • Cross-check against existing incidents in
        shared/dora_reference_incidents.json (avoid duplicates).
      • Return the candidate list to a human reviewer who decides
        what to merge into the reference table.

    Operator workflow: list shown in /admin/jobs as "Pending DORA
    incident annotations (N)" — operator clicks to approve / edit /
    drop, and approved entries are appended to the JSON via a
    git-tracked commit.
    """
    return []


def sovereign_spread_snapshot(country: str = "IT") -> dict:
    """Current sovereign-spread snapshot tagged with a market regime.

    Pulls live BTP-Bund (or OAT-Bund / Bonos-Bund) spread via the
    existing core.financial.market_data anchors. Adds a regime tag
    based on historical bps thresholds:

        spread <= 100bp     → "calm"
        100 < spread <= 200 → "stressed"
        spread > 200        → "crisis"

    Calibration consumers can use this to slice α by regime — e.g.
    Italian banking_it incidents during sovereign-stress regimes
    behave very differently from calm-period ones (MPS 2016 hit
    when BTP-Bund was ~190bp; Italian budget crisis 2018 was ~325bp).
    """
    try:
        from core.financial.market_data import refresh_market_anchors
        anchors = refresh_market_anchors(use_cache=True, country=country)
    except Exception as e:
        logger.debug(f"market_data not available: {e}")
        return {"status": "stub", "country": country, "anchors": None, "regime": None}

    if not isinstance(anchors, dict):
        return {"status": "error", "country": country, "anchors": None, "regime": None}

    spread_bp = anchors.get("sovereign_spread_bp") or anchors.get("btp_bund_bp") or 0.0
    try:
        spread_bp = float(spread_bp)
    except (TypeError, ValueError):
        spread_bp = 0.0

    if spread_bp <= 100:
        regime = "calm"
    elif spread_bp <= 200:
        regime = "stressed"
    else:
        regime = "crisis"

    return {
        "status": "ok",
        "country": country,
        "anchors": anchors,
        "spread_bp": round(spread_bp, 1),
        "regime": regime,
        "regime_thresholds_bp": {"calm": 100, "stressed": 200},
    }
