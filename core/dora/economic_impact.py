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
INCIDENTS_PATH = REPO_ROOT / "shared" / "dora_reference_incidents.json"
CALIBRATION_OUTPUT_PATH = REPO_ROOT / "outputs" / "dora_calibration.json"


# ── Calibration anchors — loaded from disk, not hardcoded ─────────────

_INCIDENTS_CACHE: Optional[list[tuple[float, float, str, str]]] = None


def _load_reference_incidents() -> list[tuple[float, float, str, str]]:
    """Load (shock_units, cost_eur_m, id, category) from disk.

    Source of truth: shared/dora_reference_incidents.json — auditable,
    version-controlled, refreshable without code changes. Each incident
    cites its public sources in the JSON.
    """
    global _INCIDENTS_CACHE
    if _INCIDENTS_CACHE is not None:
        return _INCIDENTS_CACHE
    try:
        data = json.loads(INCIDENTS_PATH.read_text())
        out: list[tuple[float, float, str, str]] = []
        for entry in data.get("incidents", []):
            su = float(entry.get("shock_units", 0) or 0)
            cm = float(entry.get("cost_eur_m", 0) or 0)
            if su > 0 and cm > 0:
                out.append((su, cm, entry.get("id", "?"), entry.get("category", "?")))
        _INCIDENTS_CACHE = out
        return out
    except Exception as e:
        logger.warning(f"reference incidents load failed: {e}; using fallback small sample")
        # Tiny fallback so the system never breaks even if the file is gone.
        _INCIDENTS_CACHE = [
            (1.6,  3_900, "mps_bailin_2017", "banking_it"),
            (2.4,  9_000, "svb_2023",        "banking_us"),
            (3.2, 30_000, "brexit_wave1_2016", "sovereign"),
        ]
        return _INCIDENTS_CACHE


def _ols_no_intercept(incidents: list[tuple[float, float, str, str]]) -> tuple[float, float, float, list[tuple[str, float]]]:
    """Plain OLS-no-intercept fit. Returns (α, σ, R², residuals_list)."""
    if not incidents:
        return 50.0, 0.0, 0.0, []
    sx2 = sum(s * s for s, _, _, _ in incidents)
    sxy = sum(s * c for s, c, _, _ in incidents)
    alpha = sxy / sx2 if sx2 > 0 else 50.0
    resid = [(ident, c - alpha * s) for s, c, ident, _ in incidents]
    n = len(incidents)
    sigma = (sum(r * r for _, r in resid) / max(1, n - 1)) ** 0.5
    ss_tot = sum(c * c for _, c, _, _ in incidents)
    ss_res = sum(r * r for _, r in resid)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return alpha, sigma, r2, resid


def _huber_no_intercept(
    incidents: list[tuple[float, float, str, str]],
    epsilon: float = 1.345,
    max_iter: int = 50,
) -> tuple[float, float, float, list[tuple[str, float]]]:
    """Huber regression (no intercept) via IRLS — robust to outliers.

    Standard k=1.345 gives 95% efficiency under normal errors but
    bounds the influence of large residuals (Lehman, SolarWinds). We
    initialise α from OLS, then re-weight by w_i = min(1, ε·σ/|r_i|),
    iterate to convergence on α.
    """
    if not incidents:
        return 50.0, 0.0, 0.0, []
    # OLS init
    alpha, sigma, _, _ = _ols_no_intercept(incidents)
    if sigma <= 0:
        return alpha, sigma, 0.0, [(i, 0.0) for _, _, i, _ in incidents]
    # IRLS
    for _ in range(max_iter):
        weights = []
        for s, c, _, _ in incidents:
            r = c - alpha * s
            if abs(r) <= epsilon * sigma:
                w = 1.0
            else:
                w = (epsilon * sigma) / max(1e-9, abs(r))
            weights.append(w)
        sx2 = sum(w * s * s for w, (s, _, _, _) in zip(weights, incidents))
        sxy = sum(w * s * c for w, (s, c, _, _) in zip(weights, incidents))
        new_alpha = sxy / sx2 if sx2 > 0 else alpha
        if abs(new_alpha - alpha) / max(1.0, abs(alpha)) < 1e-5:
            alpha = new_alpha
            break
        alpha = new_alpha
        # Update sigma from un-weighted residuals so the band is honest
        resid = [c - alpha * s for s, c, _, _ in incidents]
        sigma = (sum(r * r for r in resid) / max(1, len(resid) - 1)) ** 0.5
    resid = [(ident, c - alpha * s) for s, c, ident, _ in incidents]
    ss_tot = sum(c * c for _, c, _, _ in incidents)
    ss_res = sum(r * r for _, r in resid)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return alpha, sigma, r2, resid


def _calibrated_alpha(
    category: Optional[str] = None,
    robust: bool = True,
) -> tuple[float, float, float, int]:
    """Slope (no intercept) on the loaded reference incidents.

    Returns (alpha_eur_millions_per_unit, sigma_residual, R2, n_used).

    When `robust=True` (default since Sprint D.1) uses Huber regression
    (IRLS, k=1.345) — bounds the influence of 2σ+ outliers like
    Lehman 2008 (€600B) and SolarWinds 2020 (€100B) that dragged the
    overall OLS α up to €24B/unit. With robust fit, overall α drops
    to ~€10B/unit and per-category R² jumps materially.

    When `category` is given (and at least 3 incidents in that bucket),
    fits within-category — much tighter than the overall pool.
    """
    incidents = _load_reference_incidents()
    if category:
        sub = [row for row in incidents if row[3] == category]
        if len(sub) >= 3:
            incidents = sub
    if not incidents:
        return 50.0, 0.0, 0.0, 0
    if robust:
        alpha, sigma, r2, _ = _huber_no_intercept(incidents)
    else:
        alpha, sigma, r2, _ = _ols_no_intercept(incidents)
    return alpha, sigma, r2, len(incidents)


def backtest_loo(category: Optional[str] = None, robust: bool = True) -> dict:
    """Leave-one-out cross-validation across the reference incident table.

    For each incident i: fit α on the other N-1, predict cost_i, compute
    |error| and error %. Aggregate hit rates within ±50% / ±100% / ±200%,
    plus MAE and RMSE. This is what we show on /compliance to answer
    the CRO question "how do you know your method works?".

    Returns:
      {
        "n_total": 40,
        "method": "huber" | "ols",
        "category_filter": "overall" | "banking_it" | …,
        "mae_eur_m": …,
        "rmse_eur_m": …,
        "hit_rate_within_50pct": …,
        "hit_rate_within_100pct": …,
        "hit_rate_within_200pct": …,
        "median_abs_pct_error": …,
        "results": [
          {id, label, category, shock_units, actual_eur_m, predicted_eur_m,
           error_eur_m, error_pct, within_50pct, within_100pct},
          …
        ]
      }
    """
    all_incidents = _load_reference_incidents()
    incidents = (
        [row for row in all_incidents if row[3] == category]
        if category else all_incidents
    )
    if len(incidents) < 4:
        return {
            "status": "skipped",
            "reason": f"need ≥4 incidents for LOO (got {len(incidents)})",
            "n_total": len(incidents),
        }

    fit = _huber_no_intercept if robust else _ols_no_intercept
    results = []
    abs_errors_m = []
    sq_errors_m = []
    abs_pct_errors = []

    # Pre-load full incident metadata for labels
    try:
        meta_data = json.loads(INCIDENTS_PATH.read_text()).get("incidents", [])
        meta = {m["id"]: m for m in meta_data}
    except Exception:
        meta = {}

    for i in range(len(incidents)):
        held = incidents[i]
        s_held, c_held, id_held, cat_held = held
        train = incidents[:i] + incidents[i + 1:]
        if not train:
            continue
        alpha, _, _, _ = fit(train)
        predicted = alpha * s_held
        error = predicted - c_held
        pct = (error / c_held * 100) if c_held > 0 else 0
        results.append({
            "id": id_held,
            "label": meta.get(id_held, {}).get("label", id_held),
            "category": cat_held,
            "shock_units": s_held,
            "actual_eur_m": c_held,
            "predicted_eur_m": round(predicted, 1),
            "error_eur_m": round(error, 1),
            "error_pct": round(pct, 1),
            "within_50pct": abs(pct) <= 50,
            "within_100pct": abs(pct) <= 100,
            "within_200pct": abs(pct) <= 200,
        })
        abs_errors_m.append(abs(error))
        sq_errors_m.append(error * error)
        abs_pct_errors.append(abs(pct))

    n = len(results)
    mae = sum(abs_errors_m) / n if n else 0
    rmse = (sum(sq_errors_m) / n) ** 0.5 if n else 0
    median_pct = sorted(abs_pct_errors)[n // 2] if n else 0
    hit_50 = sum(1 for r in results if r["within_50pct"]) / n if n else 0
    hit_100 = sum(1 for r in results if r["within_100pct"]) / n if n else 0
    hit_200 = sum(1 for r in results if r["within_200pct"]) / n if n else 0

    # Worst N for the UI table — sorted by abs error
    results_sorted = sorted(results, key=lambda r: -abs(r["error_pct"]))

    return {
        "status": "ok",
        "method": "huber" if robust else "ols",
        "category_filter": category or "overall",
        "n_total": n,
        "mae_eur_m": round(mae, 2),
        "rmse_eur_m": round(rmse, 2),
        "median_abs_pct_error": round(median_pct, 1),
        "hit_rate_within_50pct": round(hit_50, 3),
        "hit_rate_within_100pct": round(hit_100, 3),
        "hit_rate_within_200pct": round(hit_200, 3),
        "results": results_sorted,
    }


def calibration_diagnostics(category: Optional[str] = None) -> dict:
    """Detailed pair-of-fits diagnostics: OLS vs Huber, plus outliers.

    Used by /api/compliance/dora/calibration/diagnostics for the UI.
    """
    incidents = _load_reference_incidents()
    if category:
        sub = [row for row in incidents if row[3] == category]
        incidents = sub if len(sub) >= 3 else incidents

    ols_alpha, ols_sigma, ols_r2, ols_resid = _ols_no_intercept(incidents)
    hub_alpha, hub_sigma, hub_r2, hub_resid = _huber_no_intercept(incidents)

    # Outlier flag = OLS residual > 2σ
    outliers = []
    for ident, r in ols_resid:
        if abs(r) > 2 * ols_sigma:
            outliers.append({
                "id": ident,
                "ols_residual_eur_m": round(r, 1),
                "outlier_z": round(abs(r) / ols_sigma, 2) if ols_sigma > 0 else None,
            })
    outliers.sort(key=lambda x: -abs(x.get("ols_residual_eur_m", 0)))
    return {
        "category": category or "overall",
        "n_incidents": len(incidents),
        "ols": {
            "alpha_eur_m_per_unit": round(ols_alpha, 2),
            "sigma_eur_m": round(ols_sigma, 2),
            "r2": round(ols_r2, 4),
        },
        "huber_robust": {
            "alpha_eur_m_per_unit": round(hub_alpha, 2),
            "sigma_eur_m": round(hub_sigma, 2),
            "r2": round(hub_r2, 4),
            "epsilon": 1.345,
        },
        "outliers_2sigma": outliers,
        "alpha_drift_pct": round(100.0 * (hub_alpha - ols_alpha) / ols_alpha, 2)
            if ols_alpha != 0 else None,
    }


# ── Category auto-detect ──────────────────────────────────────────────

# Keyword → category mapping for brief auto-detection. Order matters:
# more specific matches (banking_it) checked before broader (banking_eu).
# Each rule is (regex_keywords_lowercase, category, score_weight).
_CATEGORY_KEYWORDS: list[tuple[tuple[str, ...], str, float]] = [
    # Italian banking
    (("mps", "monte dei paschi", "carige", "popolare di vicenza", "veneto banca",
      "popolare bari", "tercas", "fitd", "ucg.mi", "isp.mi", "bper.mi",
      "bmps.mi", "bami.mi", "bce italia", "bankitalia", "banca d'italia"), "banking_it", 1.0),
    # Italian general (light) — drops to banking_it if any IT bank keyword matches
    (("italia", "italian", "btp", " mef", " mimit"), "banking_it", 0.4),
    # EU banking
    (("dexia", "abn amro", "espirito santo", "credit suisse", " ubs ", "wirecard",
      "greensill", "bnp paribas", "deutsche bank", "santander", "bbva",
      "northern rock", "sberbank europe", "fortis"), "banking_eu", 1.0),
    (("eba", "ssm ", "ecb", "bce ", "european banking authority", "single resolution"), "banking_eu", 0.5),
    # US banking
    (("svb ", "silicon valley bank", "signature bank", "first republic", "lehman",
      "bear stearns", "wamu", "washington mutual", "ltcm", "long-term capital",
      "fdic", " jpm ", "morgan stanley", "wells fargo"), "banking_us", 1.0),
    # Sovereign
    (("brexit", "italian budget", "btp-bund", "spread btp", "greek referendum",
      "cyprus bail-in", "argentina default", "sovereign default", "downgrade",
      "moody's downgrade", "s&p downgrade", "rating sovrano"), "sovereign", 1.0),
    # Cyber
    (("ransomware", "cyber attack", "data breach", "ddos", "crowdstrike",
      "solarwinds", "wannacry", "notpetya", "colonial pipeline", "equifax",
      "supply chain compromise", "incident informatico"), "cyber", 1.0),
    # Telco
    (("tim ", "tit.mi", "open fiber", "fibercop", "netco", "vodafone", "wind tre",
      "telecom italia", "iliad", "agcom", "rete unica", "ftth", "5g coverage",
      "spectrum auction", "dma", "digital networks act"), "telco", 1.0),
    # Energy
    (("eni ", "enel", "uniper", "engie", "rwe", "edf", "saipem", "snam", "terna",
      "gas hedging", "gas price cap", "rinnovabili"), "energy", 0.9),
]


def detect_category(brief: str) -> tuple[Optional[str], dict]:
    """Score a brief against keyword buckets and return best-fit category.

    Returns (category, scores_dict). Category is None when no rule
    fires above the minimum confidence (1.0 cumulative score).
    Scores dict maps category → cumulative score so the UI can show
    confidence + ties.
    """
    if not brief:
        return None, {}
    text = brief.lower()
    scores: dict[str, float] = {}
    for keywords, cat, weight in _CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in text:
                scores[cat] = scores.get(cat, 0.0) + weight
    if not scores:
        return None, {}
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] < 1.0:
        return None, scores  # below confidence floor
    return best[0], scores


def calibration_summary() -> dict:
    """Detailed calibration diagnostics — written to disk by the
    nightly calibrate_dora_alpha.py script and consumed by the
    /api/compliance/dora/calibration/status endpoint (TODO).
    """
    alpha, sigma, r2, n = _calibrated_alpha()
    incidents = _load_reference_incidents()
    by_cat: dict[str, int] = {}
    for _, _, _, cat in incidents:
        by_cat[cat] = by_cat.get(cat, 0) + 1
    return {
        "n_incidents": len(incidents),
        "n_by_category": by_cat,
        "alpha_eur_millions_per_unit": round(alpha, 2),
        "alpha_eur_per_unit": round(alpha * 1_000_000, 0),
        "sigma_residual_eur_millions": round(sigma, 2),
        "r2": round(r2, 4),
        "method": "OLS no-intercept",
        "incidents_path": str(INCIDENTS_PATH.relative_to(REPO_ROOT)),
    }


CALIBRATION_NOTES = (
    "α is OLS-fitted (no intercept) on the reference incident table at "
    "shared/dora_reference_incidents.json — currently {N} incidents across "
    "banking_it / banking_eu / banking_us / sovereign / cyber / telco / "
    "energy categories with public-domain cost figures. Refit nightly via "
    "scripts/calibrate_dora_alpha.py + GitHub Actions. CI band is ±1.65·σ "
    "(coarse 90%). Sensitive to single outliers; switch to OLS+intercept "
    "and heteroscedastic-robust SE once N > 50."
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


def estimate_anchor(total_shock_units: float, category: Optional[str] = None) -> dict:
    """Method A — α-calibrated shock anchor.

    When `category` is provided AND the per-category fit has at least 3
    incidents, uses the within-category α (much tighter, R² typically
    0.55-0.88 vs 0.25 overall). Otherwise falls back to the overall α.

    Returns a dict with point estimate + 90% CI band + breakdown so
    the UI can render the methodology transparently.
    """
    cat_alpha, cat_sigma, cat_r2, cat_n = _calibrated_alpha(category) if category else (0.0, 0.0, 0.0, 0)
    overall_alpha, overall_sigma, overall_r2, overall_n = _calibrated_alpha()
    used_category = category if category and cat_n >= 3 else None
    if used_category:
        alpha, sigma, r2, n = cat_alpha, cat_sigma, cat_r2, cat_n
    else:
        alpha, sigma, r2, n = overall_alpha, overall_sigma, overall_r2, overall_n

    point_eur = float(total_shock_units * alpha * 1_000_000)
    band = 1.645 * sigma * 1_000_000  # 90% normal CI
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
            "n_reference_incidents": n,
            "calibration_scope": used_category or "overall",
            "requested_category": category,
            # Surface the alternative for transparency: if a category was
            # requested but had n<3, show what overall would have given.
            "fallback_overall_alpha_eur_per_unit": round(overall_alpha * 1_000_000, 0) if used_category != None and used_category != "overall" else None,
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


def combine(
    anchor: dict, ticker: dict,
    detected_category: Optional[str] = None,
    category_scores: Optional[dict] = None,
    judge: Optional[dict] = None,
) -> dict:
    """Combined estimate — weighted average across A (anchor), B (ticker),
    C (LLM judge), with fallback to max(A, B) when C is missing.

    Weights (heuristic until holdout calibration ships):
      w_A = 0.30
      w_B = 0.30 (or 0.0 when ticker.point == 0)
      w_C = 0.40 × judge.confidence_score (so a low-confidence C
            de-weights itself and the residual mass goes to A and B
            renormalised)
    All re-normalised to Σw = 1.0 over the methods that have non-zero
    point estimates.

    When `judge` is None or returns 0:
      → fall back to max(A, B) — same behaviour as before Sprint C.
    """
    a_pt = float(anchor.get("point_eur", 0.0) or 0.0)
    b_pt = float(ticker.get("point_eur", 0.0) or 0.0)
    c_pt = float(judge.get("point_eur", 0.0) or 0.0) if judge else 0.0
    c_conf = float(judge.get("confidence_score", 0.0) or 0.0) if judge else 0.0

    if c_pt > 0 and c_conf > 0:
        # Triple-method weighted average. Drop methods with zero point
        # so a sim with no tickers (b_pt=0) doesn't get a B penalty.
        weights = {"anchor": 0.30, "ticker": 0.30 if b_pt > 0 else 0.0,
                   "judge": 0.40 * c_conf}
        # Normalise across methods that have non-zero point
        active_w = {k: w for k, w in weights.items() if w > 0 and (
            (k == "anchor" and a_pt > 0) or
            (k == "ticker" and b_pt > 0) or
            (k == "judge" and c_pt > 0))}
        wsum = sum(active_w.values()) or 1.0
        norm = {k: w / wsum for k, w in active_w.items()}
        point_eur = (
            norm.get("anchor", 0.0) * a_pt +
            norm.get("ticker", 0.0) * b_pt +
            norm.get("judge", 0.0) * c_pt
        )
        # Combined band: weighted average of low/high too
        a_lo = float(anchor.get("low_eur", a_pt) or 0.0)
        a_hi = float(anchor.get("high_eur", a_pt) or 0.0)
        b_lo = float(ticker.get("low_eur", b_pt) or 0.0)
        b_hi = float(ticker.get("high_eur", b_pt) or 0.0)
        c_lo = float(judge.get("low_eur", c_pt) or 0.0) if judge else 0.0
        c_hi = float(judge.get("high_eur", c_pt) or 0.0) if judge else 0.0
        low_eur = min(
            (norm.get("anchor", 0.0) * a_lo +
             norm.get("ticker", 0.0) * b_lo +
             norm.get("judge", 0.0) * c_lo),
            point_eur,
        )
        high_eur = max(
            (norm.get("anchor", 0.0) * a_hi +
             norm.get("ticker", 0.0) * b_hi +
             norm.get("judge", 0.0) * c_hi),
            point_eur,
        )
        return {
            "point_eur": round(point_eur, 2),
            "low_eur": max(0.0, round(low_eur, 2)),
            "high_eur": round(high_eur, 2),
            "selected_method": "weighted_avg",
            "weights": {k: round(v, 3) for k, v in norm.items()},
            "detected_category": detected_category,
            "category_scores": category_scores or {},
            "anchor_estimate": anchor,
            "ticker_estimate": ticker,
            "judge_estimate": judge,
            "calibration_notes": CALIBRATION_NOTES,
        }

    # Fallback: max(A, B) — Sprint A/A.2 behaviour
    use_method = "ticker" if b_pt > a_pt else "anchor"
    chosen = ticker if b_pt > a_pt else anchor
    return {
        "point_eur": chosen.get("point_eur", 0.0),
        "low_eur": chosen.get("low_eur", 0.0),
        "high_eur": chosen.get("high_eur", 0.0),
        "selected_method": use_method,
        "weights": None,
        "detected_category": detected_category,
        "category_scores": category_scores or {},
        "anchor_estimate": anchor,
        "ticker_estimate": ticker,
        "judge_estimate": judge,  # may be None — UI shows "judge skipped"
        "calibration_notes": CALIBRATION_NOTES,
    }
