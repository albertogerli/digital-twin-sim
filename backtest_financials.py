#!/usr/bin/env python3
"""Backtest the Financial Impact Scorer against real market data.

Takes 3+ real Italian political/crisis events, runs them through the scorer,
and compares predicted sector impacts vs actual market moves from Yahoo Finance.

Usage:
    python backtest_financials.py [--plot] [--verbose]

Output:
    - Console table: Predicted vs Actual per event/ticker
    - Optional: matplotlib charts comparing DigitalTwin predictions vs real returns
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(__file__))

from core.orchestrator.financial_impact import FinancialImpactScorer
from core.orchestrator.market_context import MarketContext

# Historical backtest — pin to the IT static universe so runs are reproducible
# (live VIX/yields would make results non-deterministic across invocations).
_MARKET = MarketContext(geography="IT")


# ── Historical Crisis Scenarios ──────────────────────────────────────────────
# Each scenario encodes the crisis parameters as the DigitalTwin would
# estimate them at the peak of the crisis.

@dataclass
class BacktestScenario:
    """A real historical event for backtesting."""
    name: str
    date_start: str          # Event start (YYYY-MM-DD)
    date_end: str            # 1-week window end
    brief: str               # Crisis description
    topics: list[str]        # Detected topic_tags
    sectors: list[str]       # Detected sectors
    # Simulated crisis metrics at peak
    engagement_score: float
    contagion_risk: float
    active_wave: int
    polarization: float
    polarization_velocity: float
    negative_institutional_pct: float
    negative_ceo_count: int
    # Tickers to verify against
    verify_tickers: list[str]
    # Expected direction for each ticker (for accuracy check)
    expected_directions: dict[str, str] = field(default_factory=dict)  # ticker → "down" or "up"
    notes: str = ""


SCENARIOS = [
    # ─── 1. Draghi Resignation (July 14-21, 2022) ─────────────────────────
    # Mario Draghi resigned as PM after coalition partners withdrew support.
    # FTSE MIB -3.5% in 3 days, BTP spread widened ~25bps.
    # Banks hammered, utilities/healthcare defensive.
    BacktestScenario(
        name="Draghi Resignation (Jul 2022)",
        date_start="2022-07-14",
        date_end="2022-07-25",
        brief=(
            "Crisi di governo in Italia. Mario Draghi si dimette dopo che "
            "M5S, Lega e Forza Italia non votano la fiducia al Senato. "
            "Instabilità politica, rischio elezioni anticipate, incertezza "
            "su PNRR e rapporti con l'UE."
        ),
        topics=["premierato", "eu_integration", "fiscal_policy"],
        sectors=["banking"],
        engagement_score=0.82,
        contagion_risk=0.70,
        active_wave=3,
        polarization=7.5,
        polarization_velocity=1.5,
        negative_institutional_pct=0.55,
        negative_ceo_count=2,
        verify_tickers=["UCG.MI", "ISP.MI", "ENEL.MI", "G.MI", "LDO.MI", "ENI.MI"],
        expected_directions={
            "UCG.MI": "down", "ISP.MI": "down", "G.MI": "down",
            "ENEL.MI": "down", "ENI.MI": "down", "LDO.MI": "down",
        },
        notes="FTSE MIB -3.5% in 3 days, BTP 10Y spread +25bps",
    ),

    # ─── 2. Bank Windfall Tax (August 7-10, 2023) ─────────────────────────
    # Italian government announced surprise 40% windfall tax on bank profits.
    # UCG -7%, ISP -8% in one day. Reversed partially after PM softened stance.
    BacktestScenario(
        name="Bank Windfall Tax (Aug 2023)",
        date_start="2023-08-07",
        date_end="2023-08-14",
        brief=(
            "Il governo Meloni annuncia una tassa straordinaria del 40% sugli "
            "extraprofitti delle banche derivanti dal rialzo dei tassi BCE. "
            "Shock per il settore bancario, reazione violenta dei mercati."
        ),
        topics=["fiscal_policy"],
        sectors=["banking"],
        engagement_score=0.90,
        contagion_risk=0.65,
        active_wave=3,
        polarization=8.0,
        polarization_velocity=2.0,
        negative_institutional_pct=0.30,
        negative_ceo_count=4,
        verify_tickers=["UCG.MI", "ISP.MI", "BAMI.MI", "BMPS.MI", "ENEL.MI", "REC.MI"],
        expected_directions={
            "UCG.MI": "down", "ISP.MI": "down",
            "BAMI.MI": "down", "BMPS.MI": "down",
            "ENEL.MI": "flat", "REC.MI": "flat",
        },
        notes="UCG -7.2%, ISP -8.6% day 1. Reversed ~50% after Meloni softened.",
    ),

    # ─── 3. Stellantis/Melfi Layoffs (Oct-Nov 2024) ───────────────────────
    # Stellantis announced production cuts at Melfi, Mirafiori.
    # STLAM dropped ~15% over October, labor crisis intensified.
    BacktestScenario(
        name="Stellantis Layoffs Crisis (Oct 2024)",
        date_start="2024-10-14",
        date_end="2024-10-25",
        brief=(
            "Stellantis annuncia massicci tagli alla produzione a Melfi e Mirafiori. "
            "Cassa integrazione per migliaia di operai. Sciopero generale del settore "
            "automotive. Il governo convoca Tavares a Palazzo Chigi."
        ),
        topics=["labor_reform", "industrial_policy"],
        sectors=["automotive", "labor"],
        engagement_score=0.72,
        contagion_risk=0.50,
        active_wave=2,
        polarization=6.8,
        polarization_velocity=0.9,
        negative_institutional_pct=0.40,
        negative_ceo_count=1,
        verify_tickers=["STLAM.MI", "RACE.MI", "CNHI.MI", "UCG.MI", "ENEL.MI", "LDO.MI"],
        expected_directions={
            "STLAM.MI": "down", "CNHI.MI": "down",
            "RACE.MI": "flat", "UCG.MI": "flat",
            "ENEL.MI": "flat", "LDO.MI": "up",
        },
        notes="STLAM -15% over October. Tavares resigned in December.",
    ),

    # ─── 4. 2018 Budget Standoff (May-Jun 2018) ──────────────────────────
    # Conte I government formation. "Flat tax" and "Quota 100" proposals
    # spooked markets. BTP spread +180bps in 2 weeks. UCG -30%.
    BacktestScenario(
        name="Budget Standoff / Italexit Fear (May 2018)",
        date_start="2018-05-14",
        date_end="2018-06-01",
        brief=(
            "Formazione del governo Conte I (M5S-Lega). Il contratto di governo "
            "prevede flat tax, reddito di cittadinanza, quota 100. Paura di "
            "sforamento di bilancio e scontro con l'UE. Lo spread BTP-Bund "
            "schizza a 320 punti base."
        ),
        topics=["fiscal_policy", "eu_integration", "premierato"],
        sectors=["banking"],
        engagement_score=0.95,
        contagion_risk=0.85,
        active_wave=3,
        polarization=9.0,
        polarization_velocity=2.5,
        negative_institutional_pct=0.70,
        negative_ceo_count=3,
        verify_tickers=["UCG.MI", "ISP.MI", "G.MI", "ENEL.MI", "ENI.MI"],
        expected_directions={
            "UCG.MI": "down", "ISP.MI": "down", "G.MI": "down",
            "ENEL.MI": "down", "ENI.MI": "down",
        },
        notes="Worst Italian market crash since 2011. UCG -30%, ISP -25%, spread +180bps.",
    ),
]


# ── Yahoo Finance Data Fetcher ───────────────────────────────────────────────

def fetch_real_returns(tickers: list[str], start: str, end: str) -> dict[str, dict]:
    """Fetch actual stock returns from Yahoo Finance with T+1/T+3/T+7 windows.

    Returns {ticker: {return_pct, t1_pct, t3_pct, t7_pct, data_ok, ...}}.
    T+1 = Day 1 return, T+3 = Day 3 return, T+7 = full window return.
    """
    results = {}
    start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=3)
    end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=2)

    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            if data.empty or len(data) < 2:
                results[ticker] = {"return_pct": 0.0, "t1_pct": 0.0, "t3_pct": 0.0, "t7_pct": 0.0, "data_ok": False, "error": "No data"}
                continue

            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            close_col = data["Close"]
            valid_dates = close_col.dropna().index
            if len(valid_dates) < 2:
                results[ticker] = {"return_pct": 0.0, "t1_pct": 0.0, "t3_pct": 0.0, "t7_pct": 0.0, "data_ok": False, "error": "Not enough data"}
                continue

            # Find event start in the data (closest trading day to start date)
            start_target = pd.Timestamp(start)
            # Find first date >= start_target
            event_dates = valid_dates[valid_dates >= start_target]
            if len(event_dates) == 0:
                event_dates = valid_dates  # fallback to all
            event_start_idx = valid_dates.get_loc(event_dates[0])

            close_start = float(close_col.iloc[event_start_idx])

            # T+1: 1 trading day after event
            t1_idx = min(event_start_idx + 1, len(valid_dates) - 1)
            t1_ret = ((float(close_col.iloc[t1_idx]) - close_start) / close_start) * 100

            # T+3: 3 trading days after event
            t3_idx = min(event_start_idx + 3, len(valid_dates) - 1)
            t3_ret = ((float(close_col.iloc[t3_idx]) - close_start) / close_start) * 100

            # T+7: ~7 trading days (full window)
            t7_idx = min(event_start_idx + 5, len(valid_dates) - 1)  # 5 trading days ≈ 7 calendar days
            t7_ret = ((float(close_col.iloc[t7_idx]) - close_start) / close_start) * 100

            # Full window return (for backward compat)
            close_end = float(close_col.iloc[-1])
            full_ret = ((close_end - close_start) / close_start) * 100

            results[ticker] = {
                "return_pct": round(full_ret, 2),
                "t1_pct": round(t1_ret, 2),
                "t3_pct": round(t3_ret, 2),
                "t7_pct": round(t7_ret, 2),
                "close_start": round(close_start, 2),
                "close_end": round(close_end, 2),
                "n_days": len(data),
                "data_ok": True,
            }
        except Exception as e:
            results[ticker] = {"return_pct": 0.0, "t1_pct": 0.0, "t3_pct": 0.0, "t7_pct": 0.0, "data_ok": False, "error": str(e)}

    return results


# ── Scoring Engine ────────────────────────────────────────────────────────────

def run_scorer(scenario: BacktestScenario) -> dict:
    """Run the FinancialImpactScorer on a historical scenario."""
    scorer = FinancialImpactScorer(
        detected_topics=scenario.topics,
        detected_sectors=scenario.sectors,
    )

    report = scorer.score_round(
        round_num=5,  # Simulate at "peak crisis" round
        engagement_score=scenario.engagement_score,
        contagion_risk=scenario.contagion_risk,
        active_wave=scenario.active_wave,
        polarization=scenario.polarization,
        polarization_velocity=scenario.polarization_velocity,
        negative_institutional_pct=scenario.negative_institutional_pct,
        negative_ceo_count=scenario.negative_ceo_count,
    )

    # Extract predicted impacts per verify ticker (now with T+1/T+3/T+7)
    predicted = {}
    for ticker in scenario.verify_tickers:
        impact = next((t for t in report.ticker_impacts if t.ticker == ticker), None)
        if impact:
            predicted[ticker] = {
                "predicted_pct": impact.t1_pct,  # back-compat: T+1 as primary
                "predicted_t1": impact.t1_pct,
                "predicted_t3": impact.t3_pct,
                "predicted_t7": impact.t7_pct,
                "direction": impact.direction,
                "beta": impact.beta,
                "sector": impact.sector,
            }
        else:
            # Ticker not in direct impacts — estimate from FTSE MIB
            sector = _MARKET.get_ticker_sector(ticker) or "unknown"
            beta_data = _MARKET.get_beta(sector) if sector != "unknown" else None
            ftse_est = report.ftse_mib_impact_pct * (beta_data.political_beta if beta_data else 1.0)
            predicted[ticker] = {
                "predicted_pct": ftse_est,
                "predicted_t1": ftse_est,
                "predicted_t3": ftse_est * 0.6,
                "predicted_t7": ftse_est * 1.3,
                "direction": "short",
                "beta": beta_data.political_beta if beta_data else 1.0,
                "sector": sector,
            }

    return {
        "warning": report.market_volatility_warning,
        "crisis_scope": report.crisis_scope,
        "scope_confidence": report.scope_confidence,
        "scope_disclaimer": report.scope_disclaimer,
        "ftse_impact": report.ftse_mib_impact_pct,
        "btp_spread": report.btp_spread_impact_bps,
        "n_pair_trades": len(report.pair_trades),
        "headline": report.headline,
        "predicted": predicted,
    }


# ── Results Formatting ────────────────────────────────────────────────────────

def print_results(scenario: BacktestScenario, scorer_result: dict, actual: dict):
    """Print comparison table for a single scenario with T+1/T+3/T+7 temporal curve."""
    scope = scorer_result.get("crisis_scope", "?")
    scope_conf = scorer_result.get("scope_confidence", 0)
    print(f"\n{'='*120}")
    print(f"  {scenario.name}")
    print(f"  {scenario.date_start} → {scenario.date_end}")
    print(f"  Warning: {scorer_result['warning']} | Scope: {scope} ({scope_conf:.0%}) | FTSE: {scorer_result['ftse_impact']:+.2f}% | BTP: +{scorer_result['btp_spread']}bps")
    if scorer_result.get("scope_disclaimer"):
        print(f"  ⚠ {scorer_result['scope_disclaimer'][:100]}")
    print(f"  {scenario.notes}")
    print(f"{'='*120}")
    print(f"  {'Ticker':<10} {'Sector':<14} {'β':>4}  {'Pred T+1':>9} {'Act T+1':>8} {'Pred T+3':>9} {'Act T+3':>8} {'Pred T+7':>9} {'Act T+7':>8} {'Dir?':>4}")
    print(f"  {'-'*10} {'-'*14} {'-'*4}  {'-'*9} {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*8} {'-'*4}")

    direction_correct = 0
    direction_total = 0
    errors = []

    for ticker in scenario.verify_tickers:
        pred = scorer_result["predicted"].get(ticker, {})
        real = actual.get(ticker, {})

        pred_t1 = pred.get("predicted_t1", pred.get("predicted_pct", 0))
        pred_t3 = pred.get("predicted_t3", 0)
        pred_t7 = pred.get("predicted_t7", 0)
        real_t1 = real.get("t1_pct", 0) if real.get("data_ok") else float("nan")
        real_t3 = real.get("t3_pct", 0) if real.get("data_ok") else float("nan")
        real_t7 = real.get("t7_pct", 0) if real.get("data_ok") else float("nan")
        beta = pred.get("beta", 0)
        sector = pred.get("sector", "?")[:13]

        # Direction accuracy (based on T+1 actual)
        expected = scenario.expected_directions.get(ticker, "")
        if expected and not np.isnan(real_t1):
            direction_total += 1
            if (expected == "down" and real_t1 < -0.5) or \
               (expected == "up" and real_t1 > 0.5) or \
               (expected == "flat" and abs(real_t1) < 2.0):
                direction_correct += 1
                dir_check = "Y"
            else:
                dir_check = "N"
        else:
            dir_check = "-"

        def fmt(v):
            return f"{v:+.1f}%" if not np.isnan(v) else "N/A"

        print(f"  {ticker:<10} {sector:<14} {beta:>4.1f}  {pred_t1:>+8.1f}% {fmt(real_t1):>8} {pred_t3:>+8.1f}% {fmt(real_t3):>8} {pred_t7:>+8.1f}% {fmt(real_t7):>8} {dir_check:>4}")

        if not np.isnan(real_t1):
            errors.append(abs(real_t1 - pred_t1))

    print()
    if direction_total > 0:
        print(f"  Direction accuracy: {direction_correct}/{direction_total} ({direction_correct/direction_total*100:.0f}%)")
    if errors:
        mae = np.mean(errors)
        print(f"  MAE T+1 (predicted vs actual): {mae:.2f}pp")
    print()


def classify_scenario(scenario: BacktestScenario) -> str:
    """Classify scenario into a category for aggregation."""
    topics = set(scenario.topics)
    if topics & {"premierato", "judiciary_reform"} and "fiscal_policy" in topics:
        return "Political Crisis"
    if topics & {"premierato", "judiciary_reform"}:
        return "Political Crisis"
    if "fiscal_policy" in topics and "eu_integration" in topics:
        return "Fiscal / EU Standoff"
    if "fiscal_policy" in topics and "banking" in scenario.sectors:
        return "Banking / Financial"
    if topics & {"labor_reform"} or "labor" in scenario.sectors:
        return "Labor / Industrial"
    if topics & {"environment"} or "energy" in scenario.sectors:
        return "Energy / Environment"
    if topics & {"defense_spending"} or "defense" in scenario.sectors:
        return "Defense / Geopolitical"
    if topics & {"immigration"}:
        return "Immigration / Social"
    if topics & {"autonomia_differenziata"}:
        return "Constitutional Reform"
    if topics & {"media_freedom"}:
        return "Media / Tech"
    if "healthcare" in topics or "healthcare" in scenario.sectors:
        return "Healthcare / COVID"
    return "Corporate / Other"


def print_aggregate(all_results: list[dict]):
    """Print aggregate accuracy metrics across all scenarios with scope breakdown."""
    all_t1 = []
    all_t3 = []
    all_t7 = []
    all_dir_correct = 0
    all_dir_total = 0
    macro_results = [r for r in all_results if r.get("crisis_scope") == "macro_systematic"]
    idio_results = [r for r in all_results if r.get("crisis_scope") != "macro_systematic"]

    for r in all_results:
        all_t1.extend(r["errors"])
        all_t3.extend(r.get("errors_t3", []))
        all_t7.extend(r.get("errors_t7", []))
        all_dir_correct += r["dir_correct"]
        all_dir_total += r["dir_total"]

    print(f"\n{'='*80}")
    print(f"  AGGREGATE BACKTEST RESULTS ({len(all_results)} scenarios)")
    print(f"{'='*80}")

    print(f"\n  {'Window':<12} {'MAE':>8} {'Median':>8} {'Max':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for label, errs in [("T+1 (Day 1)", all_t1), ("T+3 (Day 3)", all_t3), ("T+7 (Day 7)", all_t7)]:
        if errs:
            print(f"  {label:<12} {np.mean(errs):>7.2f}pp {np.median(errs):>7.2f}pp {np.max(errs):>7.2f}pp")

    if all_dir_total > 0:
        print(f"\n  Direction accuracy: {all_dir_correct}/{all_dir_total} ({all_dir_correct/all_dir_total*100:.0f}%)")

    # Scope breakdown
    print(f"\n  {'Scope':<25} {'N':>4} {'Dir%':>7} {'MAE T+1':>9}")
    print(f"  {'-'*25} {'-'*4} {'-'*7} {'-'*9}")
    for label, group in [("Macro/Systematic", macro_results), ("Micro/Idiosyncratic", idio_results)]:
        g_errs = [e for r in group for e in r["errors"]]
        g_dc = sum(r["dir_correct"] for r in group)
        g_dt = sum(r["dir_total"] for r in group)
        dir_pct = f"{g_dc/g_dt*100:.0f}%" if g_dt > 0 else "N/A"
        mae = f"{np.mean(g_errs):.2f}pp" if g_errs else "N/A"
        print(f"  {label:<25} {len(group):>4} {dir_pct:>7} {mae:>9}")

    print()


def print_category_table(all_results: list[dict]):
    """Print breakdown by crisis category."""
    from collections import defaultdict
    cats = defaultdict(lambda: {"errors": [], "dir_c": 0, "dir_t": 0, "count": 0, "warnings": defaultdict(int)})

    for r in all_results:
        cat = r["category"]
        c = cats[cat]
        c["count"] += 1
        c["errors"].extend(r["errors"])
        c["dir_c"] += r["dir_correct"]
        c["dir_t"] += r["dir_total"]
        c["warnings"][r["warning"]] += 1

    print(f"\n{'='*90}")
    print(f"  BREAKDOWN BY CRISIS CATEGORY")
    print(f"{'='*90}")
    print(f"  {'Category':<25} {'N':>3} {'Dir%':>6} {'MAE':>7} {'MedAE':>7} {'MaxAE':>7}  Warning Distribution")
    print(f"  {'-'*25} {'-'*3} {'-'*6} {'-'*7} {'-'*7} {'-'*7}  {'-'*25}")

    for cat in sorted(cats.keys()):
        c = cats[cat]
        dir_pct = f"{c['dir_c']/c['dir_t']*100:.0f}%" if c["dir_t"] > 0 else "N/A"
        mae = f"{np.mean(c['errors']):.2f}" if c["errors"] else "N/A"
        med = f"{np.median(c['errors']):.2f}" if c["errors"] else "N/A"
        mx = f"{np.max(c['errors']):.2f}" if c["errors"] else "N/A"
        wdist = " ".join(f"{k}:{v}" for k, v in sorted(c["warnings"].items()))
        print(f"  {cat:<25} {c['count']:>3} {dir_pct:>6} {mae:>7} {med:>7} {mx:>7}  {wdist}")

    print()


def print_wave_table(all_results: list[dict]):
    """Print accuracy by crisis severity (wave)."""
    from collections import defaultdict
    waves = defaultdict(lambda: {"errors": [], "dir_c": 0, "dir_t": 0, "count": 0})

    for r in all_results:
        w = r["wave"]
        waves[w]["count"] += 1
        waves[w]["errors"].extend(r["errors"])
        waves[w]["dir_c"] += r["dir_correct"]
        waves[w]["dir_t"] += r["dir_total"]

    print(f"  {'Wave':<20} {'N':>3} {'Dir%':>8} {'MAE':>8} {'MedAE':>8}")
    print(f"  {'-'*20} {'-'*3} {'-'*8} {'-'*8} {'-'*8}")
    for w in sorted(waves.keys()):
        c = waves[w]
        dir_pct = f"{c['dir_c']/c['dir_t']*100:.0f}%" if c["dir_t"] > 0 else "N/A"
        mae = f"{np.mean(c['errors']):.2f}" if c["errors"] else "N/A"
        med = f"{np.median(c['errors']):.2f}" if c["errors"] else "N/A"
        label = {1: "Wave 1 (Local)", 2: "Wave 2 (National)", 3: "Wave 3 (Institutional)"}
        print(f"  {label.get(w, f'Wave {w}'):<20} {c['count']:>3} {dir_pct:>8} {mae:>8} {med:>8}")
    print()


def print_ticker_leaderboard(all_results: list[dict], top_n: int = 15):
    """Print per-ticker accuracy ranked by frequency."""
    from collections import defaultdict
    tickers = defaultdict(lambda: {"errors": [], "dir_c": 0, "dir_t": 0})

    for r in all_results:
        for t, err, dir_ok, dir_counted in r.get("per_ticker", []):
            tickers[t]["errors"].append(err)
            tickers[t]["dir_c"] += dir_ok
            tickers[t]["dir_t"] += dir_counted

    print(f"\n  TOP {top_n} MOST-TESTED TICKERS")
    print(f"  {'Ticker':<12} {'Tests':>5} {'Dir%':>7} {'MAE':>8} {'MedAE':>8} {'Sector':<18}")
    print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*18}")

    sorted_tickers = sorted(tickers.items(), key=lambda x: len(x[1]["errors"]), reverse=True)
    for ticker, data in sorted_tickers[:top_n]:
        n = len(data["errors"])
        valid_errors = [e for e in data["errors"] if not np.isnan(e)]
        dir_pct = f"{data['dir_c']/data['dir_t']*100:.0f}%" if data["dir_t"] > 0 else "N/A"
        mae = f"{np.mean(valid_errors):.2f}" if valid_errors else "N/A"
        med = f"{np.median(valid_errors):.2f}" if valid_errors else "N/A"
        sector = _MARKET.get_ticker_sector(ticker) or "?"
        print(f"  {ticker:<12} {n:>5} {dir_pct:>7} {mae:>8} {med:>8} {sector:<18}")
    print()


def plot_results(all_results: list[dict], output_path: str = "backtest_results.png"):
    """Generate comparison chart: predicted vs actual returns."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping chart generation")
        return

    fig, axes = plt.subplots(len(all_results), 1, figsize=(14, 5 * len(all_results)))
    if len(all_results) == 1:
        axes = [axes]

    for idx, r in enumerate(all_results):
        ax = axes[idx]
        tickers = r["tickers"]
        predicted = r["predicted_pcts"]
        actual = r["actual_pcts"]

        x = np.arange(len(tickers))
        width = 0.35

        bars_pred = ax.bar(x - width/2, predicted, width, label="DigitalTwin Predicted",
                           color="#2563eb", alpha=0.85)
        bars_actual = ax.bar(x + width/2, actual, width, label="Yahoo Finance Actual",
                             color="#dc2626", alpha=0.85)

        ax.set_ylabel("Return (%)")
        ax.set_title(r["name"], fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, ha="right")
        ax.legend()
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars_pred:
            height = bar.get_height()
            ax.annotate(f"{height:+.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
        for bar in bars_actual:
            height = bar.get_height()
            ax.annotate(f"{height:+.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backtest(scenarios: list[BacktestScenario], verbose: bool = True) -> list[dict]:
    """Run backtest on a list of scenarios. Returns per-scenario results."""
    all_results = []

    for i, scenario in enumerate(scenarios, 1):
        # 1. Run scorer
        scorer_result = run_scorer(scenario)

        # 2. Fetch real data
        real_data = fetch_real_returns(
            scenario.verify_tickers,
            scenario.date_start,
            scenario.date_end,
        )

        # 3. Print comparison (if verbose)
        if verbose:
            print_results(scenario, scorer_result, real_data)

        # 4. Collect per-scenario metrics (T+1 based)
        errors_t1 = []
        errors_t3 = []
        errors_t7 = []
        dir_correct = 0
        dir_total = 0
        predicted_pcts = []
        actual_pcts = []
        per_ticker = []

        for ticker in scenario.verify_tickers:
            pred = scorer_result["predicted"].get(ticker, {})
            real = real_data.get(ticker, {})
            pred_t1 = pred.get("predicted_t1", pred.get("predicted_pct", 0))
            pred_t3 = pred.get("predicted_t3", 0)
            pred_t7 = pred.get("predicted_t7", 0)
            real_t1 = real.get("t1_pct", 0) if real.get("data_ok") else 0
            real_t3 = real.get("t3_pct", 0) if real.get("data_ok") else 0
            real_t7 = real.get("t7_pct", 0) if real.get("data_ok") else 0

            predicted_pcts.append(pred_t1)
            actual_pcts.append(real_t1)

            if real.get("data_ok"):
                err_t1 = abs(real_t1 - pred_t1)
                err_t3 = abs(real_t3 - pred_t3)
                err_t7 = abs(real_t7 - pred_t7)
                errors_t1.append(err_t1)
                errors_t3.append(err_t3)
                errors_t7.append(err_t7)
                expected = scenario.expected_directions.get(ticker, "")
                dir_ok = 0
                dir_counted = 0
                if expected:
                    dir_counted = 1
                    dir_total += 1
                    if (expected == "down" and real_t1 < -0.5) or \
                       (expected == "up" and real_t1 > 0.5) or \
                       (expected == "flat" and abs(real_t1) < 2.0):
                        dir_correct += 1
                        dir_ok = 1
                per_ticker.append((ticker, err_t1, dir_ok, dir_counted))

        category = classify_scenario(scenario)
        crisis_scope = scorer_result.get("crisis_scope", "?")

        all_results.append({
            "name": scenario.name,
            "category": category,
            "crisis_scope": crisis_scope,
            "wave": scenario.active_wave,
            "warning": scorer_result["warning"],
            "errors": errors_t1,  # back-compat: T+1 errors as primary
            "errors_t3": errors_t3,
            "errors_t7": errors_t7,
            "dir_correct": dir_correct,
            "dir_total": dir_total,
            "predicted_pcts": predicted_pcts,
            "actual_pcts": actual_pcts,
            "tickers": scenario.verify_tickers,
            "per_ticker": per_ticker,
            "ftse_impact": scorer_result["ftse_impact"],
            "btp_spread": scorer_result["btp_spread"],
            "n_pair_trades": scorer_result["n_pair_trades"],
        })

        if not verbose:
            # Compact progress
            dir_str = f"{dir_correct}/{dir_total}" if dir_total > 0 else "-"
            mae_str = f"{np.mean(errors_t1):.1f}" if errors_t1 else "-"
            scope_tag = "M" if crisis_scope == "macro_systematic" else "I"
            print(f"  [{i:>3}/{len(scenarios)}] {scenario.name[:45]:<45} [{scope_tag}] Dir:{dir_str:>5}  MAE:{mae_str:>5}  [{scorer_result['warning']}]")

    return all_results


def print_scenario_summary_table(all_results: list[dict]):
    """Print compact summary table of all scenarios with scope and T+1/T+3/T+7 MAE."""
    print(f"\n{'='*130}")
    print(f"  SCENARIO SUMMARY TABLE ({len(all_results)} scenarios)")
    print(f"{'='*130}")
    print(f"  {'#':>3}  {'Scenario':<40} {'Cat':<14} {'Scp':>3} {'W':>2} {'Warn':>8} {'Dir':>6} {'MAE₁':>6} {'MAE₃':>6} {'MAE₇':>6} {'BTP':>5} {'P':>2}")
    print(f"  {'-'*3}  {'-'*40} {'-'*14} {'-'*3} {'-'*2} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*2}")

    for i, r in enumerate(all_results, 1):
        dir_str = f"{r['dir_correct']}/{r['dir_total']}" if r['dir_total'] > 0 else "-"
        mae1 = f"{np.mean(r['errors']):.1f}" if r["errors"] else "-"
        mae3 = f"{np.mean(r.get('errors_t3', [])):.1f}" if r.get("errors_t3") else "-"
        mae7 = f"{np.mean(r.get('errors_t7', [])):.1f}" if r.get("errors_t7") else "-"
        btp = f"+{r['btp_spread']}" if r["btp_spread"] > 0 else "0"
        cat = r["category"][:13]
        scope = "M" if r.get("crisis_scope") == "macro_systematic" else "I"
        print(f"  {i:>3}  {r['name'][:39]:<40} {cat:<14} {scope:>3} {r['wave']:>2} {r['warning']:>8} {dir_str:>6} {mae1:>6} {mae3:>6} {mae7:>6} {btp:>5} {r['n_pair_trades']:>2}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Backtest Financial Impact Scorer vs real market data")
    parser.add_argument("--plot", action="store_true", help="Generate comparison charts (top 8 scenarios)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-scenario detail tables")
    parser.add_argument("--base-only", action="store_true", help="Only run the 4 base scenarios")
    parser.add_argument("--output", default="backtest_results.png", help="Chart output path")
    args = parser.parse_args()

    # Combine base + extended scenarios
    if args.base_only:
        all_scenarios = list(SCENARIOS)
    else:
        try:
            from backtest_scenarios import SCENARIOS_EXTENDED
            all_scenarios = list(SCENARIOS) + SCENARIOS_EXTENDED
        except ImportError:
            print("  Warning: backtest_scenarios.py not found. Running base scenarios only.")
            all_scenarios = list(SCENARIOS)

    print(f"\n  {'='*60}")
    print(f"  DIGITAL TWIN — Financial Impact Backtest")
    print(f"  {'='*60}")
    print(f"  Scenarios: {len(all_scenarios)}")
    print(f"  Data source: Yahoo Finance (yfinance)")
    print(f"  Model: FinancialImpactScorer v2 (sector betas + pair trades)")
    print(f"  {'='*60}")
    print(f"\n  Fetching real market data...\n")

    # Run backtest
    all_results = run_backtest(all_scenarios, verbose=args.verbose)

    # ── Output Tables ──────────────────────────────────────────────────────

    # 1. Compact scenario summary
    print_scenario_summary_table(all_results)

    # 2. Aggregate metrics
    print_aggregate(all_results)

    # 3. Category breakdown
    print_category_table(all_results)

    # 4. Wave breakdown
    print(f"  ACCURACY BY CRISIS SEVERITY (Wave)")
    print(f"  {'='*60}")
    print_wave_table(all_results)

    # 5. Per-ticker leaderboard
    print_ticker_leaderboard(all_results, top_n=15)

    # ── Plot (top 8 most interesting) ──────────────────────────────────────
    if args.plot:
        # Pick top 8 scenarios by error magnitude (most interesting)
        scored = [(r, np.mean(r["errors"]) if r["errors"] else 0) for r in all_results]
        scored.sort(key=lambda x: -x[1])
        top_plot = [r for r, _ in scored[:8]]
        plot_results(top_plot, args.output)

    # ── Save JSON ──────────────────────────────────────────────────────────
    output_json = os.path.splitext(args.output)[0] + ".json"
    json_data = []
    for r in all_results:
        json_data.append({
            "scenario": r["name"],
            "category": r["category"],
            "crisis_scope": r.get("crisis_scope", "?"),
            "wave": r["wave"],
            "warning": r["warning"],
            "tickers": {
                t: {"predicted_t1": p, "actual_t1": a}
                for t, p, a in zip(r["tickers"], r["predicted_pcts"], r["actual_pcts"])
            },
            "mae_t1": round(np.mean(r["errors"]), 2) if r["errors"] else None,
            "mae_t3": round(np.mean(r.get("errors_t3", [])), 2) if r.get("errors_t3") else None,
            "mae_t7": round(np.mean(r.get("errors_t7", [])), 2) if r.get("errors_t7") else None,
            "direction_accuracy": f"{r['dir_correct']}/{r['dir_total']}",
            "ftse_impact": r["ftse_impact"],
            "btp_spread": r["btp_spread"],
        })
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  Raw results saved to: {output_json}")


if __name__ == "__main__":
    main()
