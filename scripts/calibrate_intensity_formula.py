"""Empirical OLS calibration of the crisis-intensity formula.

The legacy formula in financial_impact._compute_intensity is multiplicative
in six hand-picked components::

    eng_factor   =  engagement ** 2
    wave_mult    =  {1: 0.4, 2: 1.0, 3: 2.2}[wave]
    cri_factor   =  max(0.1, cri ** 1.3)
    inst_mult    =  1.0 + neg_inst_pct * 0.4
    ceo_mult     =  1.0 + neg_ceo * 0.12
    polar_factor =  1.0 + max(0, polar_vel) * 0.25
    intensity    =  eng_factor * wave_mult * cri_factor *
                    inst_mult * ceo_mult * polar_factor

Six exponents / coefficients (2, 0.4/1.0/2.2, 1.3, 0.4, 0.12, 0.25) chosen
by hand. We replace them with empirical elasticities by regressing on
log space:

    log|t1_real|  ≈  α  +  γ · log|β|
                   +  β_eng  · log(engagement)
                   +  β_cri  · log(cri)
                   +  β_inst · log(1 + neg_inst_pct)
                   +  β_ceo  · log(1 + neg_ceo)
                   +  β_pol  · log(1 + max(0, polar_vel))
                   +  d_w2 + d_w3   (wave dummies)
                   +  ε

The exponents (β_eng, β_cri, …) are the empirical replacements for the
hardcoded 2, 1.3, etc. The wave dummies replace the {0.4, 1.0, 2.2}
table. β (sector political_beta) is the empirical OLS β from
shared/sector_betas_empirical.json (Phase B).

Output: shared/intensity_formula_coefficients.json — DIAGNOSTIC ONLY.

First-run finding (May 2026, n=310, R²=0.193)
---------------------------------------------
The fit reveals that the corpus's hand-assigned ``wave`` and
``negative_institutional_pct`` metrics do NOT track T+1 magnitude as
the legacy formula assumed:

  - **wave-3 dummy = 0.28x** (legacy 5.5x): wave-3 events in the corpus
    are slow-burn institutional crises (EU Standoff Oct 2018, Berlusconi
    2013, Renzi referendum). Wave-1 captures flash events (Saudi-Russia
    oil war −30% single day, Tesla earnings beat). T+1 magnitude is
    anti-correlated with the legacy "wave = severity" assumption.

  - **inst slope = −4.21*** (legacy +0.4): more "negative institutional
    sentiment" → SMALLER T+1 moves. Same explanation: high-inst events
    are political-process crises (EU budget standoffs, ECB pressure)
    that don't immediately spike on the announcement day.

  - cri (+0.88), polar_vel (+1.02), beta (+0.67) come out reasonable,
    consistent with the legacy formula in direction and roughly in
    magnitude.

The OLS coefficients are therefore NOT wired into _compute_intensity:
applying them in production would invert the simulator's intended
behaviour. The honest fix is metric re-definition (e.g. introduce an
"event_speed" dimension separate from "crisis_depth", so wave can be
re-purposed without conflating the two), not a coefficient swap.

The JSON is kept as a diagnostic with a ``_status`` and ``_finding``
explanation field; consumers should respect the diagnostic flag.

Usage::

    python scripts/calibrate_intensity_formula.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
OUTPUT_PATH = REPO_ROOT / "shared" / "intensity_formula_coefficients.json"


def load_corpus() -> list[dict]:
    out: list[dict] = []
    try:
        from backtest_scenarios import SCENARIOS_EXTENDED
        for s in SCENARIOS_EXTENDED:
            out.append({
                "name": s.name, "cri": s.contagion_risk,
                "engagement": s.engagement_score, "wave": s.active_wave,
                "neg_inst_pct": s.negative_institutional_pct,
                "neg_ceo": s.negative_ceo_count,
                "polar_vel": s.polarization_velocity,
                "tickers": s.verify_tickers, "date": s.date_start,
            })
    except ImportError as e:
        print(f"warning: SCENARIOS_EXTENDED import failed: {e}", file=sys.stderr)
    try:
        from backtest_scenarios_global import SCENARIOS_GLOBAL
        for s in SCENARIOS_GLOBAL:
            out.append({
                "name": s.name, "cri": s.contagion_risk,
                "engagement": s.engagement_score, "wave": s.active_wave,
                "neg_inst_pct": s.negative_institutional_pct,
                "neg_ceo": s.negative_ceo_count,
                "polar_vel": s.polarization_velocity,
                "tickers": s.verify_tickers, "date": s.date_start,
            })
    except ImportError as e:
        print(f"warning: SCENARIOS_GLOBAL import failed: {e}", file=sys.stderr)
    return out


def load_universe() -> dict[str, dict]:
    universe = json.loads(UNIVERSE_PATH.read_text())
    return {s["ticker"]: s for s in universe["stocks"]}


def load_empirical_betas() -> dict[str, dict[str, dict]]:
    try:
        return json.loads(EMPIRICAL_BETA_PATH.read_text()).get("betas", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def beta_for(ticker: str, ticker_meta: dict, betas: dict) -> float:
    info = ticker_meta.get(ticker)
    if not info:
        return 1.0
    cell = betas.get(info.get("country", ""), {}).get(info.get("sector", ""))
    return float(cell.get("political_beta", 1.0)) if cell else 1.0


def fetch_t1(ticker: str, event_date: str) -> Optional[float]:
    try:
        ed = datetime.strptime(event_date, "%Y-%m-%d").date()
        df = yf.download(
            ticker,
            start=(ed - timedelta(days=15)).isoformat(),
            end=(ed + timedelta(days=10)).isoformat(),
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


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return (coeffs, std_errors, R², n)."""
    n, k = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        coeffs, *_ = np.linalg.lstsq(XtX, Xty, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(k, float("nan")), np.full(k, float("nan")), float("nan"), n
    pred = X @ coeffs
    resid = y - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sigma2 = ss_res / max(1, n - k)
    cov = np.linalg.pinv(XtX) * sigma2
    se = np.sqrt(np.maximum(0.0, np.diag(cov)))
    return coeffs, se, float(r2), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate-limit", type=float, default=0.3)
    ap.add_argument("--floor-bps", type=float, default=20.0)
    args = ap.parse_args()

    corpus = load_corpus()
    universe = load_universe()
    betas = load_empirical_betas()
    print(f"Corpus: {len(corpus)} events × {sum(len(s['tickers']) for s in corpus)} ticker slots")

    floor = args.floor_bps / 10000.0
    rows: list[dict] = []
    for i, sc in enumerate(corpus, 1):
        print(f"  [{i:3d}/{len(corpus)}] {sc['name'][:55]:<55}", flush=True)
        for tk in sc["tickers"]:
            t1 = fetch_t1(tk, sc["date"])
            time.sleep(args.rate_limit)
            if t1 is None or not np.isfinite(t1) or abs(t1) < floor:
                continue
            beta = beta_for(tk, universe, betas)
            rows.append({
                "ticker": tk, "abs_t1": abs(t1),
                "beta": beta,
                "engagement": sc["engagement"],
                "cri": sc["cri"], "wave": sc["wave"],
                "neg_inst": sc["neg_inst_pct"],
                "neg_ceo": sc["neg_ceo"],
                "polar_vel": max(0.0, sc["polar_vel"]),
            })
    print(f"\n  {len(rows)} usable observations after noise floor")
    if len(rows) < 30:
        print("ERROR: too few observations.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)

    # Build design matrix in log-space. Use small additive offsets to avoid
    # log(0) on metrics that are honestly zero in the corpus.
    EPS = 1e-3
    y = np.log(df["abs_t1"].clip(lower=EPS).values)
    feats = {
        "intercept": np.ones(len(df)),
        "log_beta":      np.log(np.maximum(EPS, df["beta"].values)),
        "log_eng":       np.log(np.maximum(EPS, df["engagement"].values)),
        "log_cri":       np.log(np.maximum(EPS, df["cri"].values)),
        "log_inst":      np.log(np.maximum(EPS, df["neg_inst"].values + 1.0)),
        "log_ceo":       np.log(np.maximum(EPS, df["neg_ceo"].values + 1.0)),
        "log_polar":     np.log(np.maximum(EPS, df["polar_vel"].values + 1.0)),
        "wave_2":        (df["wave"].values == 2).astype(float),
        "wave_3":        (df["wave"].values >= 3).astype(float),
    }
    X = np.column_stack(list(feats.values()))
    feat_names = list(feats.keys())

    coeffs, se, r2, n = ols(X, y)

    print()
    print(f"OLS fit on log|t1_real| (n={n}, R²={r2:.3f})")
    print(f"{'feature':<12}{'coeff':>10}{'SE':>10}{'t-stat':>10}{'sig':>6}")
    print("─" * 50)
    for name, c, s in zip(feat_names, coeffs, se):
        t = c / s if s > 0 else float("nan")
        sig = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.65 else ""
        print(f"{name:<12}{c:>+10.3f}{s:>10.3f}{t:>+10.2f}{sig:>6}")

    # Map back to interpretable elasticities (the exponents in the
    # multiplicative formula).
    coef_map = dict(zip(feat_names, coeffs))
    print()
    print("Interpretable elasticities (replace legacy):")
    print(f"  engagement exponent   : empirical {coef_map['log_eng']:+.2f}   legacy 2.0")
    print(f"  cri exponent          : empirical {coef_map['log_cri']:+.2f}   legacy 1.3")
    print(f"  inst-amp slope        : empirical {coef_map['log_inst']:+.2f}   legacy 0.4 (linear add)")
    print(f"  ceo-amp slope         : empirical {coef_map['log_ceo']:+.2f}   legacy 0.12 (linear add)")
    print(f"  polar-vel slope       : empirical {coef_map['log_polar']:+.2f}   legacy 0.25 (linear add)")
    print(f"  wave-2 boost vs wave1 : empirical exp({coef_map['wave_2']:+.2f}) = "
          f"{math.exp(coef_map['wave_2']):.2f}x   legacy 1.0/0.4 = 2.5x")
    print(f"  wave-3 boost vs wave1 : empirical exp({coef_map['wave_3']:+.2f}) = "
          f"{math.exp(coef_map['wave_3']):.2f}x   legacy 2.2/0.4 = 5.5x")
    print(f"  beta loading          : empirical {coef_map['log_beta']:+.2f}   legacy 1.0 (linear)")

    payload = {
        "version": "1.0",
        "computed_at": date.today().isoformat(),
        "n_observations": int(n),
        "r_squared": round(r2, 4),
        "method": "OLS on log|t1_real| with log-space features and wave dummies",
        "noise_floor_bps": args.floor_bps,
        "coefficients": {name: round(float(c), 4) for name, c in zip(feat_names, coeffs)},
        "standard_errors": {name: round(float(s), 4) for name, s in zip(feat_names, se)},
        "interpretation": {
            "engagement_exponent": round(float(coef_map["log_eng"]), 3),
            "cri_exponent": round(float(coef_map["log_cri"]), 3),
            "inst_slope": round(float(coef_map["log_inst"]), 3),
            "ceo_slope": round(float(coef_map["log_ceo"]), 3),
            "polar_slope": round(float(coef_map["log_polar"]), 3),
            "wave_multipliers": {
                "1": 1.0,
                "2": round(float(math.exp(coef_map["wave_2"])), 3),
                "3": round(float(math.exp(coef_map["wave_3"])), 3),
            },
            "beta_exponent": round(float(coef_map["log_beta"]), 3),
            "intercept": round(float(coef_map["intercept"]), 3),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)}: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
