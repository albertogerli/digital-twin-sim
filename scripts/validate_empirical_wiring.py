"""End-to-end validation of the empirical-wiring sprints.

Runs the FinancialImpactScorer on the full pooled backtest corpus
(SCENARIOS_EXTENDED + SCENARIOS_GLOBAL) twice — once with the three
empirical JSON files present, once with them temporarily disabled —
and compares MAE / direction accuracy / Sharpe per domain.

The three JSONs gated by this experiment:
  shared/sector_betas_empirical.json           (per-(country, sector) β)
  shared/impulse_response_coefficients.json    (T+3/T+1, T+7/T+1 ratios)
  shared/panic_multiplier_calibration.json     (fat-tail amplification)

CRISIS_PAIR_TRADES → empirical correlation pairs is NOT toggled here
(it requires shared/correlation_matrix.json, which is heavy to disable
cleanly; the pair-trade path doesn't directly affect ticker-level MAE
since ticker impacts are computed independently of pair-trade legs).

Output: outputs/empirical_validation_report.md with the delta table.

This is the missing measurement after the four calibration sprints —
it answers "did the wiring actually improve out-of-sample fit?".

Usage::

    python scripts/validate_empirical_wiring.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EMPIRICAL_JSONS = [
    REPO_ROOT / "shared" / "sector_betas_empirical.json",
    REPO_ROOT / "shared" / "impulse_response_coefficients.json",
    REPO_ROOT / "shared" / "panic_multiplier_calibration.json",
]
REPORT_PATH = REPO_ROOT / "outputs" / "empirical_validation_report.md"


@contextmanager
def jsons_disabled():
    """Temporarily rename the empirical JSONs so loaders fall back to
    heuristics, then restore them on exit."""
    renamed: list[tuple[Path, Path]] = []
    try:
        for p in EMPIRICAL_JSONS:
            if p.exists():
                hidden = p.with_suffix(p.suffix + ".disabled")
                p.rename(hidden)
                renamed.append((p, hidden))
        yield
    finally:
        for original, hidden in renamed:
            hidden.rename(original)


def reset_module_caches():
    """Force loaders in financial_impact + market_context to re-read JSONs."""
    from core.orchestrator import financial_impact as fi
    from core.orchestrator import market_context as mc
    fi._panic_mult_cache = None
    fi._ir_coeffs_cache = None
    mc._empirical_betas_cache = None


def load_corpus():
    out = []
    try:
        from backtest_scenarios import SCENARIOS_EXTENDED
        out.extend(SCENARIOS_EXTENDED)
    except ImportError as e:
        print(f"warning: SCENARIOS_EXTENDED import failed: {e}", file=sys.stderr)
    try:
        from backtest_scenarios_global import SCENARIOS_GLOBAL
        out.extend(SCENARIOS_GLOBAL)
    except ImportError as e:
        print(f"warning: SCENARIOS_GLOBAL import failed: {e}", file=sys.stderr)
    return out


def fetch_returns(ticker: str, event_date: str) -> Optional[dict[str, float]]:
    try:
        ed = datetime.strptime(event_date, "%Y-%m-%d").date()
        df = yf.download(
            ticker,
            start=(ed - timedelta(days=15)).isoformat(),
            end=(ed + timedelta(days=15)).isoformat(),
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
        out = {}
        for label, n in [("t1", 1), ("t3", 3), ("t7", 7)]:
            if len(after) >= n:
                out[label] = float(np.log(float(after.iloc[n - 1]) / anchor)) * 100  # → pct
            else:
                out[label] = float("nan")
        return out
    except Exception as e:
        print(f"  fetch fail {ticker}@{event_date}: {e}", file=sys.stderr)
        return None


def cache_all_actuals(corpus, rate_limit: float = 0.3) -> dict:
    """Fetch realized T+1/T+3/T+7 once for every (event, ticker) and cache."""
    cache: dict[tuple[str, str], dict[str, float]] = {}
    for i, sc in enumerate(corpus, 1):
        for tk in sc.verify_tickers:
            key = (sc.name, tk)
            if key in cache:
                continue
            ret = fetch_returns(tk, sc.date_start)
            time.sleep(rate_limit)
            if ret:
                cache[key] = ret
        if i % 10 == 0:
            print(f"  cached actuals: {i}/{len(corpus)} scenarios", flush=True)
    return cache


def run_scorer_on_corpus(corpus) -> dict[tuple[str, str], dict[str, float]]:
    """Run the FinancialImpactScorer on each scenario, return predictions
    per (scenario_name, ticker) → {t1, t3, t7, direction}."""
    from core.orchestrator.financial_impact import FinancialImpactScorer
    out: dict[tuple[str, str], dict] = {}
    for sc in corpus:
        scorer = FinancialImpactScorer(
            detected_topics=sc.topics, detected_sectors=sc.sectors,
        )
        report = scorer.score_round(
            round_num=5,
            engagement_score=sc.engagement_score,
            contagion_risk=sc.contagion_risk,
            active_wave=sc.active_wave,
            polarization=sc.polarization,
            polarization_velocity=sc.polarization_velocity,
            negative_institutional_pct=sc.negative_institutional_pct,
            negative_ceo_count=sc.negative_ceo_count,
        )
        for tk in sc.verify_tickers:
            impact = next((t for t in report.ticker_impacts if t.ticker == tk), None)
            if impact:
                out[(sc.name, tk)] = {
                    "t1": impact.t1_pct, "t3": impact.t3_pct, "t7": impact.t7_pct,
                    "direction": impact.direction,
                }
    return out


def compute_metrics(predictions: dict, actuals: dict, corpus) -> dict:
    """Compute MAE / direction accuracy across the corpus."""
    errors_t1, errors_t3, errors_t7 = [], [], []
    dir_correct = 0
    dir_total = 0
    for sc in corpus:
        for tk in sc.verify_tickers:
            key = (sc.name, tk)
            pred = predictions.get(key)
            act = actuals.get(key)
            if not pred or not act:
                continue
            if not all(np.isfinite(act.get(k, float("nan"))) for k in ("t1", "t3", "t7")):
                continue
            errors_t1.append(abs(pred["t1"] - act["t1"]))
            errors_t3.append(abs(pred["t3"] - act["t3"]))
            errors_t7.append(abs(pred["t7"] - act["t7"]))
            # Direction accuracy on T+1
            expected_dir = sc.expected_directions.get(tk, "flat")
            if expected_dir != "flat":
                act_sign = 1 if act["t1"] > 0.5 else -1 if act["t1"] < -0.5 else 0
                exp_sign = 1 if expected_dir == "up" else -1
                dir_total += 1
                # Did the SCORER predict the right sign?
                pred_sign = 1 if pred["t1"] > 0.1 else -1 if pred["t1"] < -0.1 else 0
                if pred_sign == act_sign:
                    dir_correct += 1
    return {
        "n_obs": len(errors_t1),
        "mae_t1": float(np.mean(errors_t1)) if errors_t1 else float("nan"),
        "mae_t3": float(np.mean(errors_t3)) if errors_t3 else float("nan"),
        "mae_t7": float(np.mean(errors_t7)) if errors_t7 else float("nan"),
        "median_t1": float(np.median(errors_t1)) if errors_t1 else float("nan"),
        "dir_correct": dir_correct,
        "dir_total": dir_total,
        "dir_acc": dir_correct / dir_total if dir_total > 0 else float("nan"),
    }


def main() -> int:
    print("=== Empirical-wiring validation ===\n")
    corpus = load_corpus()
    print(f"Corpus: {len(corpus)} scenarios")

    # Cache actuals once (yfinance is the bottleneck)
    print("\nFetching actuals from yfinance (one pass, cached)...")
    actuals = cache_all_actuals(corpus)
    print(f"  cached {len(actuals)} (event × ticker) actuals\n")

    # ── PRE: run with empirical JSONs present ──
    print("PRE (with empirical JSONs):")
    reset_module_caches()
    pred_with = run_scorer_on_corpus(corpus)
    metrics_with = compute_metrics(pred_with, actuals, corpus)
    print(f"  n={metrics_with['n_obs']} | "
          f"MAE T+1={metrics_with['mae_t1']:.2f}pp T+3={metrics_with['mae_t3']:.2f}pp T+7={metrics_with['mae_t7']:.2f}pp | "
          f"dir={metrics_with['dir_acc']*100:.0f}% ({metrics_with['dir_correct']}/{metrics_with['dir_total']})")

    # ── POST: disable empirical JSONs, rerun ──
    print("\nPOST (heuristic fallback only):")
    with jsons_disabled():
        reset_module_caches()
        pred_without = run_scorer_on_corpus(corpus)
        metrics_without = compute_metrics(pred_without, actuals, corpus)
    reset_module_caches()  # restore for downstream callers
    print(f"  n={metrics_without['n_obs']} | "
          f"MAE T+1={metrics_without['mae_t1']:.2f}pp T+3={metrics_without['mae_t3']:.2f}pp T+7={metrics_without['mae_t7']:.2f}pp | "
          f"dir={metrics_without['dir_acc']*100:.0f}% ({metrics_without['dir_correct']}/{metrics_without['dir_total']})")

    # ── Delta ──
    delta_t1 = metrics_with["mae_t1"] - metrics_without["mae_t1"]
    delta_t3 = metrics_with["mae_t3"] - metrics_without["mae_t3"]
    delta_t7 = metrics_with["mae_t7"] - metrics_without["mae_t7"]
    delta_dir = metrics_with["dir_acc"] - metrics_without["dir_acc"]

    print("\n─ DELTA (empirical − heuristic) ─")
    print(f"  ΔMAE T+1: {delta_t1:+.2f}pp  (negative = empirical better)")
    print(f"  ΔMAE T+3: {delta_t3:+.2f}pp")
    print(f"  ΔMAE T+7: {delta_t7:+.2f}pp")
    print(f"  ΔDir acc: {delta_dir*100:+.1f}pp")

    # ── Per-scenario breakdown for the worst regressions ──
    per_scen = []
    for sc in corpus:
        with_errs = []
        without_errs = []
        for tk in sc.verify_tickers:
            key = (sc.name, tk)
            pred_w = pred_with.get(key)
            pred_wo = pred_without.get(key)
            act = actuals.get(key)
            if pred_w and pred_wo and act and np.isfinite(act.get("t1", float("nan"))):
                with_errs.append(abs(pred_w["t1"] - act["t1"]))
                without_errs.append(abs(pred_wo["t1"] - act["t1"]))
        if with_errs and without_errs:
            per_scen.append({
                "name": sc.name,
                "mae_with": np.mean(with_errs),
                "mae_without": np.mean(without_errs),
                "delta": np.mean(with_errs) - np.mean(without_errs),
            })
    per_scen.sort(key=lambda x: x["delta"])
    print("\nTop 10 IMPROVEMENTS (empirical reduces MAE most):")
    for r in per_scen[:10]:
        print(f"  {r['delta']:+6.2f}pp   {r['name'][:65]}")
    print("\nTop 10 REGRESSIONS (empirical increases MAE most):")
    for r in per_scen[-10:]:
        print(f"  {r['delta']:+6.2f}pp   {r['name'][:65]}")

    # ── Markdown report ──
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Empirical-wiring validation report",
        "",
        f"Run: {datetime.now().isoformat()[:19]}",
        f"Corpus: {len(corpus)} scenarios pooled",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | PRE (empirical JSONs) | POST (heuristic only) | Δ |",
        "|---|---:|---:|---:|",
        f"| MAE T+1 | {metrics_with['mae_t1']:.2f}pp | {metrics_without['mae_t1']:.2f}pp | {delta_t1:+.2f}pp |",
        f"| MAE T+3 | {metrics_with['mae_t3']:.2f}pp | {metrics_without['mae_t3']:.2f}pp | {delta_t3:+.2f}pp |",
        f"| MAE T+7 | {metrics_with['mae_t7']:.2f}pp | {metrics_without['mae_t7']:.2f}pp | {delta_t7:+.2f}pp |",
        f"| Median T+1 | {metrics_with['median_t1']:.2f}pp | {metrics_without['median_t1']:.2f}pp | {metrics_with['median_t1'] - metrics_without['median_t1']:+.2f}pp |",
        f"| Direction accuracy | {metrics_with['dir_acc']*100:.0f}% ({metrics_with['dir_correct']}/{metrics_with['dir_total']}) | {metrics_without['dir_acc']*100:.0f}% ({metrics_without['dir_correct']}/{metrics_without['dir_total']}) | {delta_dir*100:+.1f}pp |",
        f"| n observations | {metrics_with['n_obs']} | {metrics_without['n_obs']} | — |",
        "",
        "Negative ΔMAE = empirical wiring improves fit. Positive ΔDir = empirical wiring improves direction accuracy.",
        "",
        "## Top 10 scenario improvements (empirical reduces T+1 MAE most)",
        "",
        "| Δ MAE | Scenario |",
        "|---:|---|",
    ]
    for r in per_scen[:10]:
        md_lines.append(f"| {r['delta']:+.2f}pp | {r['name']} |")
    md_lines.extend([
        "",
        "## Top 10 scenario regressions (empirical increases T+1 MAE most)",
        "",
        "| Δ MAE | Scenario |",
        "|---:|---|",
    ])
    for r in per_scen[-10:]:
        md_lines.append(f"| {r['delta']:+.2f}pp | {r['name']} |")
    md_lines.append("")
    REPORT_PATH.write_text("\n".join(md_lines))
    print(f"\nWrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
