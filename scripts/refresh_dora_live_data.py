"""DORA live-data refresh — Sprint B partial (market caps).

Refreshes shared/ticker_market_caps.json from yfinance + ECB/yfinance
FX rates. Run nightly via the GitHub Actions cron and on-demand
via the admin job "dora-refresh-live".

Returns 0 on success, 1 on hard failure. Soft failures (skipped
tickers) just print a warning and continue.

Usage:
  python -m scripts.refresh_dora_live_data
  python -m scripts.refresh_dora_live_data --force
  python -m scripts.refresh_dora_live_data --max 50

Future steps (not in this script yet, see core/dora/live_data.py
docstrings):
  • fetch_recent_incidents_from_news (LLM extractor for proposing
    new entries to dora_reference_incidents.json)
  • sovereign_spread_snapshot (regime tagging for α slicing)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.dora.live_data import refresh_market_caps  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="Bypass 24h cache")
    parser.add_argument("--max", type=int, default=100, help="Cap N tickers refreshed")
    parser.add_argument("--print", action="store_true", help="Echo full result JSON")
    args = parser.parse_args()

    result = refresh_market_caps(force=args.force, max_tickers=args.max)
    status = result.get("status")
    if status == "ok":
        logger.info(
            f"Refreshed {result['tickers_refreshed']} tickers "
            f"(skipped {len(result.get('skipped', []))}) at {result.get('fetched_at')}"
        )
        for d in result.get("top_drift", [])[:5]:
            logger.info(f"  drift {d['ticker']:10s} {d['pct']:+6.2f}%  {d['old_m']:>6.0f}M → {d['new_m']:>6.0f}M EUR")
    elif status == "cached":
        logger.info(result.get("note"))
    else:
        logger.warning(f"refresh result: {status} — {result.get('note')}")

    if args.print:
        print(json.dumps(result, indent=2))
    return 0 if status in ("ok", "cached") else 1


if __name__ == "__main__":
    sys.exit(main())
