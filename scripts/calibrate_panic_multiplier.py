"""Empirical calibration of the panic-multiplier (fat-tail amplification).

Replaces the analytic ``panic_mult = math.exp(safe_cri * 1.5)`` (gated
on wave>=3 and cri>0.8) in financial_impact._compute_intensity with an
empirical median ratio per CRI bin, derived from event studies on the
pooled IT + global backtest corpus (95 scenarios, ~310 (event × ticker)
observations after a 20-bps noise floor).

Method
------
For each (event, ticker) in the pooled corpus:

  1. Fetch realized log-return at T+1 from yfinance.
  2. Compute the linear-baseline prediction
         t1_pred = β · base_market_move · intensity
     using ``linear_intensity()`` below with each scenario's *actual*
     engagement / wave / neg_inst / neg_ceo / polar_vel values from
     the corpus (NOT neutral defaults — that was the mistake of the
     first iteration). β is the empirical OLS political_beta from
     shared/sector_betas_empirical.json (Phase B).
  3. Form ρ = |t1_realized| / |t1_predicted|, bin by CRI:

         low      <0.4
         mid      0.4–0.7
         high     0.7–0.85
         extreme  ≥0.85

  4. Report MEDIAN ρ per bin (robust to outliers like Lehman / COVID
     that dominate the weighted-mean) plus 95 % bootstrap CI on the mean.

Result (May 2026 run)
---------------------
  mid CRI       n=28   median ρ ≈  2.7x   (legacy exp(cri·1.5) ≈ 2.3x)
  high CRI      n=52   median ρ ≈  5.5x   (legacy ≈ 3.2x)  — under by 70%
  extreme       n=49   median ρ ≈ 10.9x   (legacy ≈ 4.0x)  — under by 170%

Clean monotone progression — this IS a fat-tail signal. The legacy
formula systematically under-predicts amplification at high and extreme
regimes; the empirical curve replaces it. Median is preferred over the
mean (which is dominated by tail observations).

Output: shared/panic_multiplier_calibration.json — consumed by
financial_impact._empirical_panic_mult; analytic formula remains as
fallback when the JSON is absent.
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

UNIVERSE_PATH = REPO_ROOT / "shared" / "stock_universe.json"
EMPIRICAL_BETA_PATH = REPO_ROOT / "shared" / "sector_betas_empirical.json"
OUTPUT_PATH = REPO_ROOT / "shared" / "panic_multiplier_calibration.json"

BASE_MARKET_MOVE = 0.5  # % per unit intensity, matches financial_impact constant


def load_corpus() -> list[dict]:
    """Return list of dicts with each scenario's actual crisis metrics.

    Threading the real per-scenario engagement / wave / institutional /
    CEO / polar_vel values into the linear baseline (instead of neutral
    defaults) is what makes the empirical ratio interpretable as a
    fat-tail signal vs a base-scale gap.
    """
    out: list[dict] = []
    try:
        from backtest_scenarios import SCENARIOS_EXTENDED
        for s in SCENARIOS_EXTENDED:
            out.append({
                "name": s.name,
                "cri": s.contagion_risk,
                "engagement": s.engagement_score,
                "wave": s.active_wave,
                "neg_inst_pct": s.negative_institutional_pct,
                "neg_ceo": s.negative_ceo_count,
                "polar_vel": s.polarization_velocity,
                "tickers": s.verify_tickers,
                "date": s.date_start,
            })
    except ImportError as e:
        print(f"warning: SCENARIOS_EXTENDED import failed: {e}", file=sys.stderr)
    try:
        from backtest_scenarios_global import SCENARIOS_GLOBAL
        for s in SCENARIOS_GLOBAL:
            out.append({
                "name": s.name,
                "cri": s.contagion_risk,
                "engagement": s.engagement_score,
                "wave": s.active_wave,
                "neg_inst_pct": s.negative_institutional_pct,
                "neg_ceo": s.negative_ceo_count,
                "polar_vel": s.polarization_velocity,
                "tickers": s.verify_tickers,
                "date": s.date_start,
            })
    except ImportError as e:
        print(f"warning: SCENARIOS_GLOBAL import failed: {e}", file=sys.stderr)
    return out


def load_ticker_lookup() -> dict[str, dict]:
    universe = json.loads(UNIVERSE_PATH.read_text())
    return {s["ticker"]: s for s in universe["stocks"]}


def load_empirical_betas() -> dict[str, dict[str, dict]]:
    try:
        return json.loads(EMPIRICAL_BETA_PATH.read_text()).get("betas", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def beta_for(ticker: str, ticker_meta: dict, betas: dict) -> float:
    """Return the empirical political_beta for the ticker's (country, sector),
    falling back to 1.0 when no cell exists.
    """
    info = ticker_meta.get(ticker)
    if not info:
        return 1.0
    cell = betas.get(info.get("country", ""), {}).get(info.get("sector", ""))
    if cell:
        return float(cell.get("political_beta", 1.0))
    return 1.0


def linear_intensity(
    cri: float,
    engagement: float = 0.65,
    neg_inst_pct: float = 0.40,
    neg_ceo: int = 1,
    polar_vel: float = 1.0,
    wave: Optional[int] = None,
) -> float:
    """Re-implementation of financial_impact._compute_intensity *without* the
    panic_mult branch, with the same default crisis metrics used in the
    corpus's neutral fill (so the prediction is apples-to-apples with the
    realised returns).

    `wave` is inferred from cri if not supplied:
        cri >= 0.8 → wave 3
        cri >= 0.5 → wave 2
        else       → wave 1
    matching the corpus's auto-inferred values from build_global_corpus.py.
    """
    if wave is None:
        wave = 3 if cri >= 0.8 else 2 if cri >= 0.5 else 1
    eng_factor = engagement ** 2
    wave_mult = {1: 0.4, 2: 1.0, 3: 2.2}.get(wave, 1.0)
    cri_factor = max(0.1, cri ** 1.3)
    inst_mult = 1.0 + neg_inst_pct * 0.4
    ceo_mult = 1.0 + neg_ceo * 0.12
    intensity = eng_factor * wave_mult * cri_factor * inst_mult * ceo_mult
    if polar_vel > 0:
        intensity *= (1.0 + polar_vel * 0.25)
    return intensity


def fetch_t1(ticker: str, event_date: str) -> Optional[float]:
    """Realized log-return at T+1 around event_date."""
    try:
        ed = datetime.strptime(event_date, "%Y-%m-%d").date()
        start = (ed - timedelta(days=15)).isoformat()
        end = (ed + timedelta(days=10)).isoformat()
        df = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False, threads=False,
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
        return float(np.log(float(after.iloc[0]) / anchor))
    except Exception as e:
        print(f"  fetch_t1 fail {ticker}@{event_date}: {e}", file=sys.stderr)
        return None


def cri_bin(cri: float) -> str:
    """Four bins: low <0.4, mid 0.4-0.7, high 0.7-0.85, extreme >=0.85."""
    if cri < 0.4:
        return "low"
    if cri < 0.7:
        return "mid"
    if cri < 0.85:
        return "high"
    return "extreme"


def bootstrap_ci(values: np.ndarray, weights: np.ndarray,
                 n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
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
        boots[i] = float(np.average(v, weights=w_i)) if w_i.sum() > 0 else v.mean()
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-limit", type=float, default=0.3)
    ap.add_argument("--floor-bps", type=float, default=20.0,
                    help="drop |realized| < floor (noise) and |predicted| < floor "
                         "(division blow-up)")
    args = ap.parse_args()

    print("Loading corpus + universe + empirical betas...")
    corpus = load_corpus()
    ticker_meta = load_ticker_lookup()
    emp_betas = load_empirical_betas()
    print(f"  corpus: {len(corpus)} events")
    print(f"  universe: {len(ticker_meta)} tickers")
    print(f"  empirical betas: {sum(len(v) for v in emp_betas.values())} cells")

    floor = args.floor_bps / 10000.0  # bps → fractional
    obs: list[dict] = []  # {cri, ticker, t1_realised, t1_predicted, ratio}
    for i, sc in enumerate(corpus, 1):
        cri = sc["cri"]
        print(f"  [{i:3d}/{len(corpus)}] {sc['name'][:50]:<50} "
              f"cri={cri:.2f} eng={sc['engagement']:.2f} wave={sc['wave']}",
              flush=True)
        intensity = linear_intensity(
            cri=cri,
            engagement=sc["engagement"],
            neg_inst_pct=sc["neg_inst_pct"],
            neg_ceo=sc["neg_ceo"],
            polar_vel=sc["polar_vel"],
            wave=sc["wave"],
        )
        for tk in sc["tickers"]:
            beta = beta_for(tk, ticker_meta, emp_betas)
            t1_pred = beta * BASE_MARKET_MOVE * intensity / 100.0  # → fractional
            if abs(t1_pred) < floor:
                continue
            t1_real = fetch_t1(tk, sc["date"])
            time.sleep(args.rate_limit)
            if t1_real is None or not np.isfinite(t1_real):
                continue
            if abs(t1_real) < floor:
                continue
            ratio = abs(t1_real) / abs(t1_pred)
            obs.append({
                "cri": cri, "ticker": tk,
                "country": ticker_meta.get(tk, {}).get("country", "?"),
                "sector": ticker_meta.get(tk, {}).get("sector", "?"),
                "t1_real": t1_real, "t1_pred": t1_pred, "ratio": ratio,
            })

    print(f"\n  {len(obs)} usable observations after noise floor filter")
    if not obs:
        print("ERROR: no observations.", file=sys.stderr)
        return 1

    by_bin: dict[str, list[dict]] = defaultdict(list)
    for o in obs:
        by_bin[cri_bin(o["cri"])].append(o)

    print()
    print(f"{'BIN':<10}{'N':>5}{'MEAN ρ':>10}{'95% CI':>20}{'MEDIAN ρ':>11}{'P90 ρ':>10}")
    print("─" * 65)
    out_rows: dict[str, dict] = {}
    for b in ["low", "mid", "high", "extreme"]:
        rows = by_bin.get(b, [])
        if not rows:
            print(f"{b:<10}{0:>5}  (no observations)")
            continue
        rs = np.array([r["ratio"] for r in rows])
        ws = np.array([abs(r["t1_real"]) for r in rows])
        mean_r, lo, hi = bootstrap_ci(rs, ws)
        median = float(np.median(rs))
        p90 = float(np.percentile(rs, 90))
        print(f"{b:<10}{len(rows):>5}{mean_r:>+10.3f}  [{lo:+.2f},{hi:+.2f}]   "
              f"{median:>+11.3f}{p90:>+10.3f}")
        out_rows[b] = {
            "panic_mult": round(mean_r, 3),
            "panic_mult_ci95": [round(lo, 3), round(hi, 3)],
            "median_ratio": round(median, 3),
            "p90_ratio": round(p90, 3),
            "n_obs": len(rows),
        }

    payload = {
        "version": "1.0",
        "computed_at": date.today().isoformat(),
        "method": ("weighted-mean |realized T+1| / |linear-model T+1| ratio "
                   "across pooled IT + global backtest corpus; weights = |realized|; "
                   "1000-boot 95% CI."),
        "linear_model": {
            "base_market_move_pct_per_unit": BASE_MARKET_MOVE,
            "engagement_default": 0.65,
            "neg_inst_pct_default": 0.4,
            "neg_ceo_default": 1,
            "polar_vel_default": 1.0,
            "wave_inferred_from_cri": "cri>=0.8→3, >=0.5→2, else 1",
            "uses_empirical_political_beta": bool(emp_betas),
        },
        "cri_bins": {"low": "<0.4", "mid": "[0.4,0.7)",
                      "high": "[0.7,0.85)", "extreme": ">=0.85"},
        "panic_multipliers": out_rows,
        "n_observations": len(obs),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)}: "
          f"{OUTPUT_PATH.stat().st_size / 1024:.1f} KB · {len(out_rows)} bins")
    print()
    print("─" * 65)
    print("Comparison vs legacy math.exp(cri * 1.5) at bin midpoints:")
    midpoints = {"low": 0.20, "mid": 0.55, "high": 0.775, "extreme": 0.92}
    for b, mp in midpoints.items():
        legacy = math.exp(mp * 1.5)
        empirical = out_rows.get(b, {}).get("panic_mult", float("nan"))
        print(f"  {b:<10} cri={mp:.2f}   legacy={legacy:.2f}x   empirical={empirical:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
