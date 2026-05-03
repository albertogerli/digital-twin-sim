"""Empirical impulse-response calibration for the financial scorer.

Replaces the arbitrary post-shock ratios in ``financial_impact._compute_
ticker_impact``::

    if intensity < 2.0:
        recovery_factor = 0.5 + 0.1 * intensity
        t3_pct = t1_pct * recovery_factor
    else:
        escalation_factor = 1.0 + 0.1 * (intensity - 2.0)
        t3_pct = t1_pct * escalation_factor
    t7_pct = t3_pct * 1.3 + ...

with empirical coefficients estimated from the pooled backtest corpus
(SCENARIOS_EXTENDED + SCENARIOS_GLOBAL → ~85 events × ~4 tickers each).

For each (intensity_bin, sector) bucket we compute:

  - ``t3_over_t1``: weighted-mean of realized log-return ratio T+3/T+1
  - ``t7_over_t1``: weighted-mean of realized log-return ratio T+7/T+1
  - 95 % bootstrap CIs on both
  - n_obs

Weights are |t1| (so big-shock observations dominate small-noise ones).
A global (sector="ALL") row is also produced for fallback when a
sector-specific cell is too sparse.

The script prints a calibration table and writes
``shared/impulse_response_coefficients.json``. The financial scorer
loads this lazily; missing cells fall back to the legacy heuristic.

Usage::

    python scripts/calibrate_impulse_response.py
    python scripts/calibrate_impulse_response.py --rate-limit 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_PATH = REPO_ROOT / "shared" / "impulse_response_coefficients.json"
UNIVERSE_PATH = REPO_ROOT / "shared" / "stock_universe.json"


def load_corpora() -> list[tuple[str, float, list[str], str]]:
    """Return [(name, contagion_risk, verify_tickers, event_date)] from both
    backtest corpora. Event date is the scenario's start date.
    """
    out: list[tuple[str, float, list[str], str]] = []
    try:
        from backtest_scenarios import SCENARIOS_EXTENDED
        for s in SCENARIOS_EXTENDED:
            out.append((s.name, s.contagion_risk, s.verify_tickers, s.date_start))
    except ImportError as e:
        print(f"warning: backtest_scenarios import failed: {e}", file=sys.stderr)
    try:
        from backtest_scenarios_global import SCENARIOS_GLOBAL
        for s in SCENARIOS_GLOBAL:
            out.append((s.name, s.contagion_risk, s.verify_tickers, s.date_start))
    except ImportError as e:
        print(f"warning: backtest_scenarios_global import failed: {e}", file=sys.stderr)
    return out


def load_ticker_sector_map() -> dict[str, str]:
    universe = json.loads(UNIVERSE_PATH.read_text())
    return {s["ticker"]: s.get("sector", "other") for s in universe["stocks"]}


def fetch_returns(ticker: str, event_date: str) -> Optional[dict[str, float]]:
    """Pull realized log-returns at T+1/T+3/T+7 around an event date."""
    try:
        ed = datetime.strptime(event_date, "%Y-%m-%d").date()
        start = (ed - timedelta(days=20)).isoformat()
        end = (ed + timedelta(days=25)).isoformat()
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            return None
        before = close[close.index.date < ed]
        if before.empty:
            return None
        anchor = float(before.iloc[-1])
        after = close[close.index.date >= ed]
        if after.empty:
            return None
        out: dict[str, float] = {}
        for label, n in [("t1", 1), ("t3", 3), ("t7", 7)]:
            if len(after) >= n:
                out[label] = float(np.log(float(after.iloc[n - 1]) / anchor))
            else:
                out[label] = float("nan")
        return out
    except Exception as e:
        print(f"  fetch_returns failed for {ticker} @ {event_date}: {e}",
              file=sys.stderr)
        return None


def intensity_bin(cri: float) -> str:
    """Bucket a 0-1 contagion-risk into low/mid/high."""
    if cri < 0.4:
        return "low"
    if cri < 0.7:
        return "mid"
    return "high"


def bootstrap_mean_ci(
    values: np.ndarray, weights: np.ndarray, n_boot: int = 1000, seed: int = 42,
) -> tuple[float, float, float]:
    """Weighted-mean point estimate + 95 % bootstrap CI."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    w = weights / weights.sum() if weights.sum() > 0 else None
    point = float(np.average(values, weights=weights) if w is not None else values.mean())
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = values[idx]
        w_i = weights[idx]
        if w_i.sum() == 0:
            boots[i] = v.mean()
        else:
            boots[i] = float(np.average(v, weights=w_i))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return point, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-limit", type=float, default=0.3)
    ap.add_argument("--min-t1-bps", type=float, default=20.0,
                    help="drop observations whose |t1| is below this floor "
                         "(stabilizes the ratio against tiny-denominator noise)")
    args = ap.parse_args()

    print("Loading corpora...")
    corpus = load_corpora()
    print(f"  {len(corpus)} scenarios pooled")
    sector_map = load_ticker_sector_map()
    print(f"  {len(sector_map)} tickers in universe map")

    print("Fetching realized T+1/T+3/T+7 returns from yfinance...")
    observations: list[dict] = []  # {sector, intensity_bin, t1, t3, t7}
    for i, (name, cri, tickers, edate) in enumerate(corpus, 1):
        print(f"  [{i:3d}/{len(corpus)}] {name[:50]:<50} "
              f"({len(tickers)} tickers, cri={cri:.2f})",
              flush=True)
        for tk in tickers:
            ret = fetch_returns(tk, edate)
            time.sleep(args.rate_limit)
            if ret is None or not all(np.isfinite(ret.get(k, float("nan")))
                                       for k in ("t1", "t3", "t7")):
                continue
            sector = sector_map.get(tk, "other")
            observations.append({
                "ticker": tk,
                "sector": sector,
                "intensity_bin": intensity_bin(cri),
                "cri": cri,
                "t1": ret["t1"],
                "t3": ret["t3"],
                "t7": ret["t7"],
            })
    print(f"\n  collected {len(observations)} (event × ticker) observations")

    # Drop observations with tiny |t1| that destabilize the ratio
    floor = args.min_t1_bps / 10000.0  # bps → fractional
    valid = [o for o in observations if abs(o["t1"]) >= floor]
    dropped = len(observations) - len(valid)
    print(f"  dropped {dropped} observations with |t1| < {args.min_t1_bps:.0f}bps "
          f"(noise floor); {len(valid)} usable")
    if not valid:
        print("ERROR: no valid observations to calibrate.", file=sys.stderr)
        return 1

    # ── Bucket aggregation ────────────────────────────────────────────────
    by_bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_intensity: dict[str, list[dict]] = defaultdict(list)
    for o in valid:
        by_bucket[(o["intensity_bin"], o["sector"])].append(o)
        by_intensity[o["intensity_bin"]].append(o)

    print("\n──────────────────────────────────────────────────────────────────")
    print(f"{'INTENSITY':<10}{'SECTOR':<22}{'N':>5}{'T3/T1':>10}{'95% CI':>20}{'T7/T1':>10}{'95% CI':>20}")
    print("──────────────────────────────────────────────────────────────────")

    rows_out: dict[str, dict[str, dict]] = defaultdict(dict)

    # Per-(intensity, sector)
    for (intens, sector), obs in sorted(by_bucket.items()):
        if len(obs) < 4:  # too sparse
            continue
        t1 = np.array([o["t1"] for o in obs])
        t3 = np.array([o["t3"] for o in obs])
        t7 = np.array([o["t7"] for o in obs])
        weights = np.abs(t1)
        ratios_3 = t3 / t1
        ratios_7 = t7 / t1
        r3, r3_lo, r3_hi = bootstrap_mean_ci(ratios_3, weights)
        r7, r7_lo, r7_hi = bootstrap_mean_ci(ratios_7, weights)
        rows_out[intens][sector] = {
            "t3_over_t1": round(r3, 3),
            "t3_over_t1_ci95": [round(r3_lo, 3), round(r3_hi, 3)],
            "t7_over_t1": round(r7, 3),
            "t7_over_t1_ci95": [round(r7_lo, 3), round(r7_hi, 3)],
            "n_obs": len(obs),
        }
        print(f"{intens:<10}{sector:<22}{len(obs):>5}"
              f"{r3:>+10.3f}  [{r3_lo:+.2f},{r3_hi:+.2f}]   "
              f"{r7:>+10.3f}  [{r7_lo:+.2f},{r7_hi:+.2f}]")

    # Pooled (sector="ALL") fallback per intensity
    print()
    for intens, obs in sorted(by_intensity.items()):
        t1 = np.array([o["t1"] for o in obs])
        t3 = np.array([o["t3"] for o in obs])
        t7 = np.array([o["t7"] for o in obs])
        weights = np.abs(t1)
        r3, r3_lo, r3_hi = bootstrap_mean_ci(t3 / t1, weights)
        r7, r7_lo, r7_hi = bootstrap_mean_ci(t7 / t1, weights)
        rows_out[intens]["ALL"] = {
            "t3_over_t1": round(r3, 3),
            "t3_over_t1_ci95": [round(r3_lo, 3), round(r3_hi, 3)],
            "t7_over_t1": round(r7, 3),
            "t7_over_t1_ci95": [round(r7_lo, 3), round(r7_hi, 3)],
            "n_obs": len(obs),
        }
        print(f"{intens:<10}{'ALL (pooled)':<22}{len(obs):>5}"
              f"{r3:>+10.3f}  [{r3_lo:+.2f},{r3_hi:+.2f}]   "
              f"{r7:>+10.3f}  [{r7_lo:+.2f},{r7_hi:+.2f}]")

    payload = {
        "version": "1.0",
        "computed_at": date.today().isoformat(),
        "method": "weighted-mean ratios |t1|, 1000-boot 95% CI; corpus = SCENARIOS_EXTENDED + SCENARIOS_GLOBAL",
        "corpus_size": {"events": len(corpus), "observations_kept": len(valid),
                         "observations_dropped_t1_floor": dropped,
                         "min_t1_bps": args.min_t1_bps},
        "intensity_bins": {"low": "<0.4", "mid": "[0.4,0.7)", "high": ">=0.7"},
        "coefficients": dict(rows_out),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)}: "
          f"{OUTPUT_PATH.stat().st_size / 1024:.1f} KB · "
          f"{sum(len(v) for v in rows_out.values())} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
