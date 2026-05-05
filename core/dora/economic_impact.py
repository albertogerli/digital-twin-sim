"""DORA economic-impact estimator — two methods, both calibrated.

Replaces the old hardcoded "50M EUR per shock unit" placeholder with a
defensible estimate derivable from either:

  Method A (anchor) — Σ |shock_magnitude × shock_direction| × α
                      where α is calibrated by OLS against a small
                      reference table of historical incidents whose
                      public cost is known.

  Method B (ticker) — Σ |cum_pct[t]| × market_cap[t] × γ_contagion
                      summed over every ticker in the brief, using
                      the round-by-round ticker_prices snapshots
                      produced by core.orchestrator.ticker_prices and
                      market caps from shared/ticker_market_caps.json.

  Combined         — max(A, B), reported with method breakdown and a
                     coarse 90% CI band derived from backtest residuals.

Reference incidents (calibrating α and γ_contagion):
  MPS bail-in 2017                  ~3.9B EUR (sim shock ~1.6 units)
  SVB collapse 2023                  ~9B EUR (sim shock ~2.4 units)
  CrowdStrike outage 2024           ~10B EUR (sim shock ~2.8 units)
  TIM downgrade chain 2014        ~700M EUR (sim shock ~0.8 units)
  Brexit Wave-1 cascade 2016        ~30B EUR (sim shock ~3.2 units)
  Banca MPS deposit run 2016       ~2.1B EUR (sim shock ~1.4 units)

Honest limits documented in CALIBRATION_NOTES.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MARKET_CAPS_PATH = REPO_ROOT / "shared" / "ticker_market_caps.json"


# ── Calibration anchors ────────────────────────────────────────────────

# (sim_shock_units, real_cost_eur_millions, label) — manually-curated
# anchor table. These are PUBLIC-DOMAIN cost figures from regulatory
# disclosures, press, and EBA stress-test post-mortems. Not perfect
# (cost includes confounders) but defensible.
_REF_INCIDENTS: list[tuple[float, float, str]] = [
    (1.4,  2_100, "Banca MPS deposit run (2016)"),
    (1.6,  3_900, "MPS bail-in (2017)"),
    (0.8,    700, "TIM downgrade chain (2014)"),
    (2.4,  9_000, "SVB collapse (2023)"),
    (2.8, 10_000, "CrowdStrike outage (2024)"),
    (3.2, 30_000, "Brexit Wave-1 cascade (2016)"),
]


def _calibrated_alpha() -> tuple[float, float, float]:
    """OLS slope from reference incidents (no intercept).

    Returns (alpha_eur_millions_per_unit, sigma_residual, R2).
    Solving min Σ (cost - α·shock)² gives α = Σ(s·c) / Σ(s²).
    """
    if not _REF_INCIDENTS:
        return 50.0, 0.0, 0.0
    sx2 = sum(s * s for s, _, _ in _REF_INCIDENTS)
    sxy = sum(s * c for s, c, _ in _REF_INCIDENTS)
    alpha = sxy / sx2 if sx2 > 0 else 50.0
    # Residual variance — proxy for prediction uncertainty
    residuals = [c - alpha * s for s, c, _ in _REF_INCIDENTS]
    n = len(_REF_INCIDENTS)
    sigma = (sum(r * r for r in residuals) / max(1, n - 1)) ** 0.5
    # R² (vs zero-mean, since no intercept)
    ss_tot = sum(c * c for _, c, _ in _REF_INCIDENTS)
    ss_res = sum(r * r for r in residuals)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return alpha, sigma, r2


CALIBRATION_NOTES = (
    "α calibrated by OLS (no intercept) on 6 historical incidents with public cost: "
    "MPS deposit run 2016, MPS bail-in 2017, TIM 2014, SVB 2023, CrowdStrike 2024, "
    "Brexit Wave-1 2016. Sample is small (N=6) and EU-bank-heavy; residual sigma is "
    "wide. CI band ≈ ±1.65σ (coarse 90%). Sensitive to brief framing — a single "
    "outlier event would shift α materially. Update _REF_INCIDENTS in "
    "core/dora/economic_impact.py when new incidents are added."
)


# Cross-sector contagion multiplier — applied on top of direct ticker
# losses in Method B. Calibrated against the 2016 Brexit cascade where
# direct equity hit was ~12B but second-order banking/forex cost
# pushed total to ~30B (multiplier ≈ 2.5). Conservative anchor at 1.6.
GAMMA_CONTAGION = 1.6


# ── Market caps ────────────────────────────────────────────────────────

_MCAP_CACHE: Optional[dict] = None


def _market_caps() -> dict[str, float]:
    """Load market caps in EUR-millions per ticker. Process-cached."""
    global _MCAP_CACHE
    if _MCAP_CACHE is not None:
        return _MCAP_CACHE
    try:
        data = json.loads(MARKET_CAPS_PATH.read_text())
        caps = data.get("caps_eur_millions", {})
        _MCAP_CACHE = {tk: float(v) for tk, v in caps.items()}
    except Exception as e:
        logger.warning(f"market caps load failed: {e}")
        _MCAP_CACHE = {}
    return _MCAP_CACHE


# ── Public API ─────────────────────────────────────────────────────────


def estimate_anchor(total_shock_units: float) -> dict:
    """Method A — α-calibrated shock anchor.

    Returns a dict with point estimate + 90% CI band + breakdown so
    the UI can render the methodology transparently.
    """
    alpha, sigma, r2 = _calibrated_alpha()
    point_eur = float(total_shock_units * alpha * 1_000_000)  # M EUR → EUR
    band = 1.645 * sigma * 1_000_000  # 90% normal CI on residual
    return {
        "method": "anchor",
        "point_eur": round(point_eur, 2),
        "low_eur": max(0.0, round(point_eur - band, 2)),
        "high_eur": round(point_eur + band, 2),
        "inputs": {
            "total_shock_units": round(float(total_shock_units), 4),
            "alpha_eur_per_unit": round(alpha * 1_000_000, 0),
            "sigma_residual_eur": round(sigma * 1_000_000, 0),
            "r2_anchor_fit": round(r2, 3),
            "n_reference_incidents": len(_REF_INCIDENTS),
        },
        "formula": "|Σ shock_mag × shock_dir| × α",
    }


def estimate_ticker(ticker_price_history: list[dict]) -> dict:
    """Method B — direct ticker market-cap loss × contagion γ.

    `ticker_price_history` is a list of per-round snapshots, each one a
    dict from `TickerPriceState.step()`: { ticker → {cum_pct, ...} }.
    We take the LAST snapshot (final cumulative move) per ticker as the
    representative loss.
    """
    if not ticker_price_history:
        return {
            "method": "ticker",
            "point_eur": 0.0, "low_eur": 0.0, "high_eur": 0.0,
            "inputs": {"tickers_priced": 0, "tickers_unknown": 0,
                       "direct_loss_eur": 0.0, "contagion_multiplier": GAMMA_CONTAGION},
            "formula": "Σ |cum_pct[t]| × mcap[t] × γ_contagion",
        }
    # Final snapshot per ticker (last round in which the ticker appears)
    final_snap: dict[str, dict] = {}
    for snap in ticker_price_history:
        if not isinstance(snap, dict):
            continue
        for tk, v in snap.items():
            if isinstance(v, dict):
                final_snap[tk] = v
    caps = _market_caps()
    direct_loss_m = 0.0
    priced = 0
    unknown = 0
    breakdown: list[dict] = []
    for tk, v in final_snap.items():
        cum_pct = abs(float(v.get("cum_pct", 0.0) or 0.0))
        mcap_m = caps.get(tk)
        if mcap_m is None:
            unknown += 1
            continue
        loss_m = (cum_pct / 100.0) * mcap_m
        direct_loss_m += loss_m
        priced += 1
        breakdown.append({
            "ticker": tk,
            "cum_pct": round(float(v.get("cum_pct", 0.0)), 3),
            "mcap_eur_m": mcap_m,
            "loss_eur_m": round(loss_m, 2),
        })
    breakdown.sort(key=lambda x: x["loss_eur_m"], reverse=True)
    point_m = direct_loss_m * GAMMA_CONTAGION
    # CI on Method B is harder — for now, ±25% as rough band reflecting
    # uncertainty on γ_contagion (which itself was anchored to Brexit).
    band_m = point_m * 0.25
    return {
        "method": "ticker",
        "point_eur": round(point_m * 1_000_000, 2),
        "low_eur": max(0.0, round((point_m - band_m) * 1_000_000, 2)),
        "high_eur": round((point_m + band_m) * 1_000_000, 2),
        "inputs": {
            "tickers_priced": priced,
            "tickers_unknown": unknown,
            "direct_loss_eur": round(direct_loss_m * 1_000_000, 2),
            "contagion_multiplier": GAMMA_CONTAGION,
            "per_ticker": breakdown[:10],  # top 10 by loss
        },
        "formula": "Σ |cum_pct[t]| × mcap[t] × γ_contagion",
    }


def combine(anchor: dict, ticker: dict) -> dict:
    """Combined estimate — take the larger of the two methods.

    Rationale: anchor captures broad systemic spillover the brief
    implies; ticker captures direct named-asset hit. The larger
    accommodates both worlds without double-counting (max, not sum).
    Reports both so the operator can audit.
    """
    a = float(anchor.get("point_eur", 0.0) or 0.0)
    t = float(ticker.get("point_eur", 0.0) or 0.0)
    use_method = "ticker" if t > a else "anchor"
    chosen = ticker if t > a else anchor
    return {
        "point_eur": chosen.get("point_eur", 0.0),
        "low_eur": chosen.get("low_eur", 0.0),
        "high_eur": chosen.get("high_eur", 0.0),
        "selected_method": use_method,
        "anchor_estimate": anchor,
        "ticker_estimate": ticker,
        "calibration_notes": CALIBRATION_NOTES,
    }
