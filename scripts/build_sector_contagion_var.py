"""Cross-sector VAR(1) contagion network.

Builds a sector-pair spillover matrix from daily returns 2018→today.
For each ordered (sector_i, sector_j) pair, we run::

    r_j(t)  =  α_ij  +  β_ij · r_i(t-1)  +  ε(t)

where r_s(t) is the equally-weighted basket return of all tickers in
sector s on day t. β_ij is the *next-day* spillover from sector i to
sector j (a "Granger-style" lagged effect, not a contemporaneous one
— contemporaneous correlation is already covered by
correlation_matrix.json).

The output network captures *dynamic* contagion: when banking moves
−1 % today, what does the model predict for energy / tech / industrials
tomorrow? This is the missing piece in a static correlation model.

Outputs
-------
shared/sector_contagion_var.json — full matrix + metadata.
frontend/public/data/sector_contagion.json — compact graph (edges with
|β| ≥ threshold) for the UI.

Usage::

    python scripts/build_sector_contagion_var.py
    python scripts/build_sector_contagion_var.py --start 2018-01-01 --min-beta 0.05
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
MATRIX_PATH = REPO_ROOT / "shared" / "sector_contagion_var.json"
GRAPH_PATH = REPO_ROOT / "frontend" / "public" / "data" / "sector_contagion.json"


def chunked(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


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
        time.sleep(0.4)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def ols_lag(y: pd.Series, x_lag: pd.Series) -> tuple[float, float, float, int]:
    """Univariate OLS y(t) = α + β x(t-1); return (β, t-stat, R², n)."""
    aligned = pd.concat([y, x_lag.shift(1)], axis=1, keys=["y", "xlag"]).dropna()
    if len(aligned) < 60:
        return float("nan"), float("nan"), float("nan"), len(aligned)
    yv = aligned["y"].values
    xv = aligned["xlag"].values
    xm = xv.mean()
    ym = yv.mean()
    cov = ((xv - xm) * (yv - ym)).sum()
    var = ((xv - xm) ** 2).sum()
    if var == 0:
        return float("nan"), float("nan"), float("nan"), len(aligned)
    beta = cov / var
    alpha = ym - beta * xm
    pred = alpha + beta * xv
    ss_res = ((yv - pred) ** 2).sum()
    n = len(aligned)
    sigma2 = ss_res / max(1, n - 2)
    se_beta = math.sqrt(max(0.0, sigma2 / var))
    t_stat = beta / se_beta if se_beta > 0 else float("nan")
    ss_tot = ((yv - ym) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(beta), float(t_stat), float(r2), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--min-beta", type=float, default=0.05,
                    help="absolute |β| floor for graph edges (full matrix written too)")
    ap.add_argument("--min-t-stat", type=float, default=2.0,
                    help="absolute t-stat floor for graph edges")
    args = ap.parse_args()

    universe = json.loads(UNIVERSE_PATH.read_text())
    stocks = universe["stocks"]
    by_sector: dict[str, list[str]] = defaultdict(list)
    for s in stocks:
        by_sector[s.get("sector", "other")].append(s["ticker"])
    sectors = sorted(by_sector.keys())
    print(f"Universe: {len(stocks)} tickers across {len(sectors)} sectors")
    for sec in sectors:
        print(f"  {sec:<22} {len(by_sector[sec]):>3} tickers")

    all_tickers = sorted({t for ts in by_sector.values() for t in ts})
    print(f"\nDownloading {len(all_tickers)} ticker prices...")
    prices = download_prices(all_tickers, args.start, args.end)
    if prices.empty:
        print("ERROR: no prices.", file=sys.stderr)
        return 1
    print(f"  got {prices.shape[1]} symbols × {prices.shape[0]} days")

    rets = np.log(prices.ffill(limit=3) / prices.ffill(limit=3).shift(1)).dropna(how="all")
    print(f"Returns matrix: {rets.shape}")

    # Sector basket = equal-weight mean of member tickers' returns
    sector_returns = pd.DataFrame()
    for sec, tickers in by_sector.items():
        valid = [t for t in tickers if t in rets.columns]
        if not valid:
            continue
        sector_returns[sec] = rets[valid].mean(axis=1)
    sector_returns = sector_returns.dropna(how="all")
    print(f"Sector baskets: {sector_returns.shape}")

    # ── VAR(1) cell-by-cell (univariate OLS lagged) ──
    print("\nFitting (sector_i → sector_j) lagged spillovers...")
    matrix: dict[str, dict[str, dict]] = defaultdict(dict)
    edges_payload = []
    rows_for_print = []
    for src in sector_returns.columns:
        for dst in sector_returns.columns:
            if src == dst:
                continue  # autoregressive own-lag is informative but not "contagion"
            beta, t_stat, r2, n = ols_lag(sector_returns[dst], sector_returns[src])
            if not math.isfinite(beta):
                continue
            cell = {
                "beta": round(beta, 4),
                "t_stat": round(t_stat, 2),
                "r2": round(r2, 4),
                "n": n,
            }
            matrix[src][dst] = cell
            rows_for_print.append((src, dst, beta, t_stat, r2, n))
            if abs(beta) >= args.min_beta and abs(t_stat) >= args.min_t_stat:
                edges_payload.append({
                    "source": src,
                    "target": dst,
                    "beta": round(beta, 3),
                    "t_stat": round(t_stat, 2),
                })

    # Print top spillovers
    rows_for_print.sort(key=lambda r: -abs(r[2]))
    print()
    print(f"Top 20 spillovers (|β| descending):")
    print(f"{'src':<22}{'→':<3}{'dst':<22}{'β':>10}{'t-stat':>10}{'n':>7}")
    print("─" * 75)
    for src, dst, beta, t_stat, r2, n in rows_for_print[:20]:
        print(f"{src:<22}{'→':<3}{dst:<22}{beta:>+10.3f}{t_stat:>+10.2f}{n:>7}")

    # ── Write outputs ──
    matrix_payload = {
        "version": "1.0",
        "computed_at": date.today().isoformat(),
        "start": args.start, "end": args.end,
        "n_sectors": len(sector_returns.columns),
        "n_observations": int(sector_returns.shape[0]),
        "method": "univariate OLS r_j(t) = α + β · r_i(t-1)",
        "sectors": list(sector_returns.columns),
        "matrix": dict(matrix),
    }
    MATRIX_PATH.write_text(json.dumps(matrix_payload, indent=None))
    print(f"\nWrote {MATRIX_PATH.relative_to(REPO_ROOT)}: {MATRIX_PATH.stat().st_size / 1024:.1f} KB")

    nodes_payload = [
        {
            "id": sec,
            "n_tickers": len(by_sector[sec]),
            "out_degree": sum(1 for e in edges_payload if e["source"] == sec),
            "in_degree": sum(1 for e in edges_payload if e["target"] == sec),
        }
        for sec in sector_returns.columns
    ]
    graph_payload = {
        "version": "1.0",
        "computed_at": matrix_payload["computed_at"],
        "start": args.start, "end": args.end,
        "min_beta": args.min_beta,
        "min_t_stat": args.min_t_stat,
        "n_observations": matrix_payload["n_observations"],
        "nodes": nodes_payload,
        "edges": edges_payload,
    }
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(json.dumps(graph_payload, separators=(",", ":")))
    print(f"Wrote {GRAPH_PATH.relative_to(REPO_ROOT)}: {GRAPH_PATH.stat().st_size / 1024:.1f} KB "
          f"({len(edges_payload)} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
