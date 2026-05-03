"""Empirical per-(country, sector) sector-beta recalibration.

For each (country, sector) bucket in shared/stock_universe.json:

  1. Pool the daily log-returns of all member tickers (2018→today, yfinance).
  2. Compute the bucket's mean daily log-return series.
  3. OLS-regress that series on the country's main index returns:

         r_bucket(t) = α + β · r_index(t) + ε(t)

  4. Report β as the empirical political_beta. The intercept α (annualized)
     plays the role of crisis_alpha.

Country index map:

  US → ^GSPC (S&P 500)
  GB → ^FTSE
  DE → ^GDAXI
  FR → ^FCHI
  IT → FTSEMIB.MI
  JP → ^N225
  HK → ^HSI
  CH → ^SSMI
  ES → ^IBEX
  CN → 000001.SS

Tickers in countries without a configured index fall through to the
nearest regional index (e.g. KR → ^N225, IN → ^GSPC, BR → ^GSPC, AR → ^GSPC,
NL → ^GDAXI, AU → ^N225). This isn't perfect but it gives a defensible
empirical β where the universe lacks a local benchmark.

Output: shared/sector_betas_empirical.json with shape
    {"<country>": {"<sector>": {"political_beta": float, ...}}}

The script does NOT mutate shared/stock_universe.json — that file is the
declarative source of truth. The empirical betas are loaded by
MarketContext.get_beta() which can be wired to prefer them over the
hand-coded values in stock_universe.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = REPO_ROOT / "shared" / "stock_universe.json"
OUTPUT_PATH = REPO_ROOT / "shared" / "sector_betas_empirical.json"

COUNTRY_INDEX = {
    "US": "^GSPC",
    "GB": "^FTSE",
    "DE": "^GDAXI",
    "FR": "^FCHI",
    "IT": "FTSEMIB.MI",
    "JP": "^N225",
    "HK": "^HSI",
    "CH": "^SSMI",
    "NL": "^AEX",
    "ES": "^IBEX",
    "CN": "000001.SS",
}

# Fallback regional indices for countries without a local quote
COUNTRY_FALLBACK = {
    "KR": "^N225",
    "IN": "^GSPC",
    "BR": "^GSPC",
    "AR": "^GSPC",
    "AU": "^N225",
    "TW": "^N225",
    "MX": "^GSPC",
    "CL": "^GSPC",
    "DK": "^GDAXI",
    "FI": "^GDAXI",
    "SE": "^GDAXI",
}


def chunked(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def index_for(country: str) -> str | None:
    return COUNTRY_INDEX.get(country) or COUNTRY_FALLBACK.get(country)


def download_prices(tickers: list[str], start: str, end: str, batch: int = 30) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    n_batches = math.ceil(len(tickers) / batch)
    for i, group in enumerate(chunked(tickers, batch)):
        print(f"  batch {i + 1}/{n_batches} ({len(group)} tickers)", flush=True)
        try:
            df = yf.download(group, start=start, end=end, auto_adjust=True,
                             progress=False, threads=True, group_by="ticker")
        except Exception as e:
            print(f"    batch failed: {e}", flush=True)
            continue
        if df.empty:
            continue
        closes = {}
        for tk in group:
            try:
                col = df[tk]["Close"] if (tk, "Close") in df.columns else df["Close"][tk]
                if col.dropna().shape[0] > 100:
                    closes[tk] = col
            except (KeyError, TypeError):
                pass
        if closes:
            frames.append(pd.DataFrame(closes))
        time.sleep(0.5)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def ols_beta(y: pd.Series, x: pd.Series) -> tuple[float, float, float, int]:
    """Return (beta, alpha_daily, r2, n) from a univariate OLS y = α + β x.

    Uses pairwise-complete obs. Returns NaNs if fewer than 60 paired
    observations (less than ~3 months of trading).
    """
    aligned = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    if len(aligned) < 60:
        return float("nan"), float("nan"), float("nan"), len(aligned)
    yv = aligned["y"].values
    xv = aligned["x"].values
    x_mean = xv.mean()
    y_mean = yv.mean()
    cov = ((xv - x_mean) * (yv - y_mean)).sum()
    var = ((xv - x_mean) ** 2).sum()
    if var == 0:
        return float("nan"), float("nan"), float("nan"), len(aligned)
    beta = cov / var
    alpha = y_mean - beta * x_mean
    pred = alpha + beta * xv
    ss_res = ((yv - pred) ** 2).sum()
    ss_tot = ((yv - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(beta), float(alpha), float(r2), len(aligned)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()

    universe = json.loads(UNIVERSE_PATH.read_text())
    stocks = universe["stocks"]
    print(f"Universe: {len(stocks)} tickers, {args.start} → {args.end}")

    # Group tickers by (country, sector) and collect the full ticker list
    by_country_sector: dict[tuple[str, str], list[str]] = defaultdict(list)
    all_tickers = set()
    for s in stocks:
        c = s.get("country", "NA")
        sec = s.get("sector", "other")
        by_country_sector[(c, sec)].append(s["ticker"])
        all_tickers.add(s["ticker"])
    # Add country indices we'll need
    needed_indices = set()
    for (c, _) in by_country_sector.keys():
        idx = index_for(c)
        if idx:
            needed_indices.add(idx)
    print(f"Buckets (country × sector): {len(by_country_sector)}")
    print(f"Country indices needed: {sorted(needed_indices)}")

    download_set = sorted(all_tickers | needed_indices)
    print(f"Downloading {len(download_set)} symbols...")
    prices = download_prices(download_set, args.start, args.end)
    if prices.empty:
        print("ERROR: no prices downloaded.", file=sys.stderr)
        return 1
    print(f"  got {prices.shape[1]} symbols × {prices.shape[0]} days")

    # Compute log-returns
    rets = np.log(prices.ffill(limit=3) / prices.ffill(limit=3).shift(1)).dropna(how="all")

    # For each bucket, compute pooled mean return series, then OLS vs country index
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    rows_for_print: list[tuple[str, str, int, float, float, float, int]] = []

    for (country, sector), tickers in sorted(by_country_sector.items()):
        idx_ticker = index_for(country)
        if not idx_ticker or idx_ticker not in rets.columns:
            continue
        valid = [t for t in tickers if t in rets.columns]
        if not valid:
            continue
        pooled = rets[valid].mean(axis=1)
        beta, alpha_daily, r2, n = ols_beta(pooled, rets[idx_ticker])
        if not math.isfinite(beta):
            continue
        # Annualize alpha (252 trading days). Express as %.
        alpha_annual_pct = alpha_daily * 252 * 100
        out[country][sector] = {
            "political_beta": round(beta, 3),
            "crisis_alpha_pct": round(alpha_annual_pct, 2),
            "r2": round(r2, 3),
            "n_obs": n,
            "n_tickers": len(valid),
            "vs_index": idx_ticker,
            "method": "ols_pooled_log_returns_2018+",
        }
        rows_for_print.append((country, sector, len(valid), beta, alpha_annual_pct, r2, n))

    # Print sorted by country then descending β
    print()
    print(f"{'CTRY':<5}{'SECTOR':<22}{'TICKS':>6}{'BETA':>8}{'α/yr%':>9}{'R²':>8}{'N':>7}")
    print("─" * 65)
    for row in sorted(rows_for_print, key=lambda r: (r[0], -r[3])):
        c, sec, n_t, b, a, r2, n = row
        print(f"{c:<5}{sec:<22}{n_t:>6}{b:>+8.2f}{a:>+9.2f}{r2:>8.2f}{n:>7}")

    payload = {
        "version": "1.0",
        "computed_at": date.today().isoformat(),
        "start": args.start,
        "end": args.end,
        "method": "OLS pooled log-returns vs country index, 2018+; min 60 obs",
        "country_indices": COUNTRY_INDEX,
        "country_fallbacks": COUNTRY_FALLBACK,
        "betas": dict(out),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)}: "
          f"{OUTPUT_PATH.stat().st_size / 1024:.1f} KB, "
          f"{sum(len(v) for v in out.values())} (country, sector) cells "
          f"across {len(out)} countries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
