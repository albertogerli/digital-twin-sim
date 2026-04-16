"""Enrich empirical scenario JSONs with real market observations from Yahoo Finance.

For each FIN/CORP scenario, fetches historical prices for the relevant ticker(s),
computes per-round returns, and writes a 'market_observations' field into the JSON.

Usage:
    python scripts/enrich_market_data.py                    # dry-run (print only)
    python scripts/enrich_market_data.py --write            # update JSONs in-place
    python scripts/enrich_market_data.py --write --dir v2.1 # use scenarios_v2.1

Requires: pip install yfinance
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance")
    sys.exit(1)


# ── Ticker mapping per scenario ──────────────────────────────────────────────
# Each entry: scenario_id_prefix → (primary_ticker, sector_key, [extra_tickers])
# sector_key maps to SECTOR_BETAS in financial_impact.py

SCENARIO_TICKERS = {
    # Financial
    "FIN-2021-ARCHEGOS": ("XLF", "banking", ["GS"]),  # CS delisted, use XLF (Financials ETF)
    "FIN-2023-SVB": ("KRE", "banking", ["KBE", "XLF"]),
    "FIN-2021-GAMESTOP": ("GME", "banking", ["XRT"]),
    "FIN-2021-AMC": ("AMC", "banking", ["XRT"]),
    "FIN-2022-FTX": ("COIN", "tech", []),  # COIN as crypto proxy (FTT delisted)
    "FIN-2020-TESLA_STOCK": ("TSLA", "automotive", []),
    "FIN-2019-WEWORK": ("IWO", "real_estate", []),  # small growth ETF proxy
    # Corporate
    "CORP-2015-DIESELGATE": ("VOW3.DE", "automotive", ["STLA"]),
    "CORP-2017-UBER_LONDON": ("^GSPC", "tech", []),  # Uber pre-IPO, use S&P500
    "CORP-2017-UNITED": ("UAL", "infrastructure", []),
    "CORP-2018-AMAZON": ("AMZN", "tech", []),
    "CORP-2019-BOEING": ("BA", "defense", []),
    "CORP-2021-FACEBOOK": ("META", "tech", []),
    "CORP-2022-TWITTER": ("XLK", "tech", []),  # TWTR delisted, use XLK (Tech ETF)
}

BENCHMARK = "SPY"


def _match_scenario(scenario_id: str) -> tuple[str, str, list[str]] | None:
    """Find ticker config for a scenario by prefix match."""
    for prefix, config in SCENARIO_TICKERS.items():
        if scenario_id.startswith(prefix):
            return config
    return None


def _compute_round_dates(date_start: str, n_rounds: int, round_duration_days: int):
    """Compute (start, end) dates for each round."""
    start = datetime.strptime(date_start, "%Y-%m-%d")
    rounds = []
    for r in range(n_rounds):
        r_start = start + timedelta(days=r * round_duration_days)
        r_end = start + timedelta(days=(r + 1) * round_duration_days - 1)
        rounds.append((r_start, r_end))
    return rounds


def _fetch_returns(ticker: str, date_start: str, date_end: str,
                   round_dates: list) -> list[dict]:
    """Fetch prices and compute per-round returns."""
    # Add buffer around dates
    start_dt = datetime.strptime(date_start, "%Y-%m-%d") - timedelta(days=5)
    end_dt = datetime.strptime(date_end, "%Y-%m-%d") + timedelta(days=5)

    try:
        df = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        print(f"    WARNING: yfinance failed for {ticker}: {e}")
        return []

    if df.empty:
        print(f"    WARNING: No data for {ticker} ({date_start} → {date_end})")
        return []

    # Handle MultiIndex columns from yfinance
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]

    results = []
    for r, (r_start, r_end) in enumerate(round_dates):
        # Find closest trading days
        mask_start = close.index <= r_start.strftime("%Y-%m-%d")
        mask_end = close.index <= r_end.strftime("%Y-%m-%d")

        if mask_start.any() and mask_end.any():
            p_start = close[mask_start].iloc[-1]
            p_end = close[mask_end].iloc[-1]

            # Compute volume ratio (avg volume this round / overall avg)
            round_mask = (close.index >= r_start.strftime("%Y-%m-%d")) & \
                         (close.index <= r_end.strftime("%Y-%m-%d"))
            if "Volume" in df.columns and round_mask.any():
                round_vol = df.loc[round_mask, "Volume"].mean()
                total_vol = df["Volume"].mean()
                vol_ratio = float(round_vol / max(total_vol, 1))
            else:
                vol_ratio = 1.0

            ret = float((p_end - p_start) / p_start * 100)
            results.append({
                "round": r + 1,
                "return_pct": round(ret, 3),
                "volume_ratio": round(vol_ratio, 2),
            })
        else:
            results.append({
                "round": r + 1,
                "return_pct": None,
                "volume_ratio": None,
            })

    return results


def enrich_scenario(scenario_path: str, write: bool = False) -> dict | None:
    """Enrich a single scenario JSON with market observations."""
    with open(scenario_path) as f:
        scenario = json.load(f)

    sid = scenario["id"]
    config = _match_scenario(sid)
    if config is None:
        return None

    primary_ticker, sector_key, extra_tickers = config
    date_start = scenario["date_start"]
    date_end = scenario["date_end"]
    n_rounds = scenario["n_rounds"]
    round_dur = scenario.get("round_duration_days", 14)

    print(f"  {sid}: {primary_ticker} ({sector_key})")

    round_dates = _compute_round_dates(date_start, n_rounds, round_dur)

    # Fetch primary ticker
    primary_returns = _fetch_returns(primary_ticker, date_start, date_end, round_dates)
    if not primary_returns:
        print(f"    SKIP: no data for primary ticker {primary_ticker}")
        return None

    # Fetch benchmark
    benchmark_returns = _fetch_returns(BENCHMARK, date_start, date_end, round_dates)

    # Compute cumulative return
    valid_returns = [r["return_pct"] for r in primary_returns if r["return_pct"] is not None]
    if valid_returns:
        cum_factor = 1.0
        for ret in valid_returns:
            cum_factor *= (1 + ret / 100)
        cumulative_return = (cum_factor - 1) * 100
    else:
        cumulative_return = 0.0

    # Benchmark cumulative
    bench_valid = [r["return_pct"] for r in benchmark_returns if r["return_pct"] is not None]
    if bench_valid:
        bench_factor = 1.0
        for ret in bench_valid:
            bench_factor *= (1 + ret / 100)
        benchmark_cumulative = (bench_factor - 1) * 100
    else:
        benchmark_cumulative = 0.0

    # Realized volatility (annualized from per-round returns)
    if len(valid_returns) > 1:
        rounds_per_year = 365 / max(round_dur, 1)
        vol = float(np.std(valid_returns) * np.sqrt(rounds_per_year))
    else:
        vol = 0.0

    market_obs = {
        "primary_ticker": primary_ticker,
        "primary_sector": sector_key,
        "benchmark_ticker": BENCHMARK,
        "extra_tickers": extra_tickers,
        "per_round_returns": primary_returns,
        "benchmark_per_round": benchmark_returns,
        "cumulative_return_pct": round(cumulative_return, 3),
        "benchmark_return_pct": round(benchmark_cumulative, 3),
        "excess_return_pct": round(cumulative_return - benchmark_cumulative, 3),
        "volatility_realized": round(vol, 4),
        "data_source": "yfinance",
        "fetch_date": datetime.now().strftime("%Y-%m-%d"),
    }

    scenario["market_observations"] = market_obs

    if write:
        with open(scenario_path, "w") as f:
            json.dump(scenario, f, indent=2, ensure_ascii=False)
        print(f"    WRITTEN to {scenario_path}")

    return market_obs


def main():
    parser = argparse.ArgumentParser(description="Enrich scenarios with market data")
    parser.add_argument("--write", action="store_true", help="Write to JSON files")
    parser.add_argument("--dir", default="scenarios_v2.1", help="Scenarios subdirectory")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent / "calibration" / "empirical" / args.dir
    if not base.exists():
        # Try without subdir
        base = Path(__file__).resolve().parent.parent / "calibration" / "empirical" / "scenarios"
        if not base.exists():
            print(f"ERROR: directory not found: {base}")
            sys.exit(1)

    json_files = sorted(base.glob("*.json"))
    json_files = [f for f in json_files if not f.name.endswith(".meta.json")
                  and f.name != "manifest.json"]

    print(f"Scanning {len(json_files)} scenarios in {base}")
    enriched = 0
    skipped = 0

    for path in json_files:
        result = enrich_scenario(str(path), write=args.write)
        if result is not None:
            enriched += 1
            # Print summary
            cum = result["cumulative_return_pct"]
            excess = result["excess_return_pct"]
            n_valid = sum(1 for r in result["per_round_returns"]
                          if r["return_pct"] is not None)
            print(f"    → cum={cum:+.1f}%, excess={excess:+.1f}%, "
                  f"rounds_with_data={n_valid}")
        else:
            skipped += 1

    print(f"\nDone: {enriched} enriched, {skipped} skipped")
    if not args.write:
        print("(dry run — use --write to update JSONs)")


if __name__ == "__main__":
    main()
