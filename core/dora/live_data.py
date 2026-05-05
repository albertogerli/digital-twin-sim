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


def refresh_market_caps(force: bool = False) -> dict:
    """TODO Sprint B — refresh shared/ticker_market_caps.json from yfinance.

    Plan:
      • For each ticker in shared/stock_universe.json, call
        yf.Ticker(t).fast_info → shares_outstanding × close
      • Convert to EUR via fxrate snapshot (ECB SDW reference rate)
      • Write back rounded to nearest 10M EUR
      • Cache 24h on disk; respect `force` flag to bypass

    Until implemented: returns a no-op marker so callers can detect
    the stub and fall back to the static file.
    """
    return {
        "status": "stub",
        "fetched_at": None,
        "tickers_refreshed": 0,
        "note": "Sprint B not yet implemented — using static shared/ticker_market_caps.json",
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


def sovereign_spread_snapshot() -> dict:
    """TODO Sprint B — current BTP-Bund / OAT-Bund spreads for
    regime-conditioned α.

    Plan:
      • Read from existing core.financial.market_data
        (already pulls live ECB / yfinance values; just expose).
      • Tag the calibration snapshot with the regime: "calm" /
        "stressed" / "crisis" so α can be sliced accordingly.
    """
    try:
        from core.financial.market_data import refresh_market_anchors
        anchors = refresh_market_anchors(use_cache=True, country="IT")
        return {"status": "ok", "anchors": anchors}
    except Exception as e:
        logger.debug(f"market_data not available: {e}")
        return {"status": "stub", "anchors": None}
