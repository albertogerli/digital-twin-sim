"""
Cross-market correlation matrix from yfinance for the 190-ticker universe.

Reads shared/stock_universe.json, downloads daily adjusted close prices
2015-01-01..today via yfinance in batches, computes log-return Pearson
correlations, runs Louvain community detection, and writes:

  shared/correlation_matrix.json         full 190x190 matrix + metadata
  frontend/public/data/contagion_graph.json   compact graph (top-K edges per
                                              node, communities, sectors)

Failed/illiquid tickers are dropped silently and reported in the JSON
metadata.

Usage:
  python scripts/compute_correlation_matrix.py
  python scripts/compute_correlation_matrix.py --start 2018-01-01 --top-k 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = REPO_ROOT / "shared" / "stock_universe.json"
MATRIX_PATH = REPO_ROOT / "shared" / "correlation_matrix.json"
GRAPH_PATH = REPO_ROOT / "frontend" / "public" / "data" / "contagion_graph.json"


def chunked(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def download_prices(tickers: list[str], start: str, end: str, batch: int = 30) -> pd.DataFrame:
    """Download adjusted close prices in batches; concatenate into a single
    DataFrame indexed by date with one column per ticker. Tickers with no data
    are dropped from the returned frame."""
    frames: list[pd.DataFrame] = []
    for i, group in enumerate(chunked(tickers, batch)):
        print(
            f"  batch {i + 1}/{math.ceil(len(tickers) / batch)} "
            f"({len(group)} tickers)",
            flush=True,
        )
        try:
            df = yf.download(
                tickers=group,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as e:
            print(f"    batch failed: {e}", flush=True)
            continue
        if df.empty:
            continue
        # When auto_adjust=True, yfinance returns a single multi-level frame
        # with (ticker, field) columns. Extract just Close per ticker.
        closes = {}
        for tk in group:
            try:
                col = df[tk]["Close"] if (tk, "Close") in df.columns else df["Close"][tk]
                if col.dropna().shape[0] > 100:  # need at least 100 obs
                    closes[tk] = col
            except (KeyError, TypeError):
                pass
        if closes:
            frames.append(pd.DataFrame(closes))
        time.sleep(0.5)  # courtesy gap to yfinance rate limiter
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, with ffill cap of 3 days for long weekends."""
    filled = prices.ffill(limit=3)
    return np.log(filled / filled.shift(1)).dropna(how="all")


def build_graph(corr: pd.DataFrame, top_k: int, min_corr: float) -> nx.Graph:
    """Symmetric undirected graph keeping the top-K edges per node above
    min_corr. We take the union across both endpoints' top-K lists, so a node
    can end up with degree > top_k."""
    g = nx.Graph()
    g.add_nodes_from(corr.columns)
    n = len(corr)
    arr = corr.values
    cols = list(corr.columns)
    for i in range(n):
        # Sort other tickers by absolute correlation descending
        scores = arr[i].copy()
        scores[i] = -np.inf  # exclude self
        order = np.argsort(-scores)
        kept = 0
        for j in order:
            if kept >= top_k:
                break
            c = arr[i, j]
            if not np.isfinite(c) or abs(c) < min_corr:
                continue
            g.add_edge(cols[i], cols[j], weight=float(c))
            kept += 1
    return g


def detect_communities(g: nx.Graph, seed: int = 42) -> dict[str, int]:
    """Louvain community detection on positive-weight version of g."""
    pos_g = nx.Graph()
    pos_g.add_nodes_from(g.nodes())
    for u, v, d in g.edges(data=True):
        w = abs(d["weight"])
        if w > 0:
            pos_g.add_edge(u, v, weight=w)
    if pos_g.number_of_edges() == 0:
        return {n: 0 for n in g.nodes()}
    communities = nx.community.louvain_communities(pos_g, seed=seed, resolution=1.1)
    out: dict[str, int] = {}
    for cid, comm in enumerate(sorted(communities, key=len, reverse=True)):
        for n in comm:
            out[n] = cid
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--top-k", type=int, default=8, help="top-K edges per node")
    ap.add_argument("--min-corr", type=float, default=0.30)
    ap.add_argument("--batch", type=int, default=30)
    args = ap.parse_args()

    universe = json.loads(UNIVERSE_PATH.read_text())
    stocks = universe["stocks"]
    tickers = [s["ticker"] for s in stocks]
    meta_by_ticker = {s["ticker"]: s for s in stocks}

    print(f"Universe: {len(tickers)} tickers, {args.start} → {args.end}")
    print(f"Downloading prices...")
    prices = download_prices(tickers, args.start, args.end, batch=args.batch)
    if prices.empty:
        print("ERROR: no prices downloaded.", file=sys.stderr)
        return 1
    coverage = prices.notna().sum() / len(prices)
    keep = coverage[coverage >= 0.40].index.tolist()  # need ≥40% non-NaN
    dropped = sorted(set(tickers) - set(keep))
    prices = prices[keep]
    print(f"  kept {len(keep)} / {len(tickers)} (dropped {len(dropped)})")
    if dropped:
        print(f"  dropped: {', '.join(dropped[:20])}{'...' if len(dropped) > 20 else ''}")

    print("Computing log returns...")
    rets = compute_log_returns(prices)
    print(f"  {rets.shape[0]} trading days × {rets.shape[1]} tickers")

    print("Computing Pearson correlation matrix (pairwise complete)...")
    corr = rets.corr(method="pearson", min_periods=250)  # need ≥1y overlap
    # Drop any rows/cols that are all-NaN
    valid = corr.notna().any(axis=1)
    corr = corr.loc[valid, valid]
    print(f"  final matrix: {corr.shape}")

    print(f"Building graph (top-{args.top_k} edges, |r| ≥ {args.min_corr})...")
    g = build_graph(corr, top_k=args.top_k, min_corr=args.min_corr)
    print(f"  nodes: {g.number_of_nodes()}, edges: {g.number_of_edges()}")

    print("Detecting communities (Louvain, resolution=1.1)...")
    communities = detect_communities(g)
    n_communities = len(set(communities.values()))
    print(f"  {n_communities} communities")

    # ── Write full matrix
    print(f"Writing {MATRIX_PATH.relative_to(REPO_ROOT)}...")
    matrix_payload = {
        "version": "1.0",
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "start": args.start,
        "end": args.end,
        "n_tickers": int(corr.shape[0]),
        "n_observations": int(rets.shape[0]),
        "method": "pearson_log_returns",
        "min_overlap_days": 250,
        "dropped_tickers": dropped,
        "tickers": list(corr.columns),
        # Round to 4 decimals to keep file size manageable
        "matrix": {
            t: {u: round(float(v), 4) for u, v in row.items() if np.isfinite(v)}
            for t, row in corr.round(4).iterrows()
        },
    }
    MATRIX_PATH.write_text(json.dumps(matrix_payload, indent=None))
    print(f"  {MATRIX_PATH.stat().st_size / 1024:.1f} KB")

    # ── Write compact graph for the frontend
    print(f"Writing {GRAPH_PATH.relative_to(REPO_ROOT)}...")
    nodes_payload = []
    for tk in g.nodes():
        m = meta_by_ticker.get(tk, {})
        nodes_payload.append(
            {
                "id": tk,
                "name": m.get("name", tk),
                "sector": m.get("sector", "other"),
                "country": m.get("country", "NA"),
                "region": m.get("region", "other"),
                "tier": m.get("market_cap_tier", "small"),
                "community": communities.get(tk, 0),
                "degree": g.degree(tk),
            }
        )
    edges_payload = [
        {"source": u, "target": v, "weight": round(d["weight"], 3)}
        for u, v, d in g.edges(data=True)
    ]
    graph_payload = {
        "version": "1.0",
        "computed_at": matrix_payload["computed_at"],
        "start": args.start,
        "end": args.end,
        "n_observations": matrix_payload["n_observations"],
        "n_communities": n_communities,
        "min_corr": args.min_corr,
        "top_k": args.top_k,
        "nodes": nodes_payload,
        "edges": edges_payload,
    }
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(json.dumps(graph_payload, separators=(",", ":")))
    print(f"  {GRAPH_PATH.stat().st_size / 1024:.1f} KB")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
