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
VAR_CONTAGION_PATH = REPO_ROOT / "shared" / "sector_contagion_var.json"
STOCK_UNIVERSE_PATH = REPO_ROOT / "shared" / "stock_universe.json"


# ── Calibration anchors — loaded from disk, not hardcoded ─────────────

_INCIDENTS_CACHE: Optional[list[tuple[float, float, str, str, str]]] = None


def _load_reference_incidents() -> list[tuple[float, float, str, str, str]]:
    """Load (shock_units, cost_eur_m, id, category, regime) from disk.

    Source of truth: shared/dora_reference_incidents.json — auditable,
    version-controlled, refreshable without code changes. Each incident
    cites its public sources in the JSON.

    `regime` is "calm" / "stressed" / "crisis" — added in Sprint D.3 to
    let α be sliced by market-stress regime as well as category.
    Defaults to "stressed" if a row has no regime label.
    """
    global _INCIDENTS_CACHE
    if _INCIDENTS_CACHE is not None:
        return _INCIDENTS_CACHE
    try:
        data = json.loads(INCIDENTS_PATH.read_text())
        out: list[tuple[float, float, str, str, str]] = []
        for entry in data.get("incidents", []):
            su = float(entry.get("shock_units", 0) or 0)
            cm = float(entry.get("cost_eur_m", 0) or 0)
            if su > 0 and cm > 0:
                out.append((
                    su, cm,
                    entry.get("id", "?"),
                    entry.get("category", "?"),
                    entry.get("regime", "stressed"),
                ))
        _INCIDENTS_CACHE = out
        return out
    except Exception as e:
        logger.warning(f"reference incidents load failed: {e}; using fallback small sample")
        _INCIDENTS_CACHE = [
            (1.6,  3_900, "mps_bailin_2017", "banking_it", "stressed"),
            (2.4,  9_000, "svb_2023",        "banking_us", "stressed"),
            (3.2, 30_000, "brexit_wave1_2016", "sovereign", "stressed"),
        ]
        return _INCIDENTS_CACHE


def _ols_no_intercept(incidents: list[tuple]) -> tuple[float, float, float, list[tuple[str, float]]]:
    """Plain OLS-no-intercept fit. Returns (α, σ, R², residuals_list).

    Accepts incidents as either 4-tuples (s, c, id, cat) or 5-tuples
    (s, c, id, cat, regime) — only s and c are used.
    """
    if not incidents:
        return 50.0, 0.0, 0.0, []
    sx2 = sum(row[0] * row[0] for row in incidents)
    sxy = sum(row[0] * row[1] for row in incidents)
    alpha = sxy / sx2 if sx2 > 0 else 50.0
    resid = [(row[2], row[1] - alpha * row[0]) for row in incidents]
    n = len(incidents)
    sigma = (sum(r * r for _, r in resid) / max(1, n - 1)) ** 0.5
    ss_tot = sum(row[1] * row[1] for row in incidents)
    ss_res = sum(r * r for _, r in resid)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return alpha, sigma, r2, resid


def _huber_no_intercept(
    incidents: list[tuple],
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
        return alpha, sigma, 0.0, [(row[2], 0.0) for row in incidents]
    # IRLS — accept 4- or 5-tuple rows; only s and c used
    for _ in range(max_iter):
        weights = []
        for row in incidents:
            s, c = row[0], row[1]
            r = c - alpha * s
            if abs(r) <= epsilon * sigma:
                w = 1.0
            else:
                w = (epsilon * sigma) / max(1e-9, abs(r))
            weights.append(w)
        sx2 = sum(w * row[0] * row[0] for w, row in zip(weights, incidents))
        sxy = sum(w * row[0] * row[1] for w, row in zip(weights, incidents))
        new_alpha = sxy / sx2 if sx2 > 0 else alpha
        if abs(new_alpha - alpha) / max(1.0, abs(alpha)) < 1e-5:
            alpha = new_alpha
            break
        alpha = new_alpha
        resid = [row[1] - alpha * row[0] for row in incidents]
        sigma = (sum(r * r for r in resid) / max(1, len(resid) - 1)) ** 0.5
    resid = [(row[2], row[1] - alpha * row[0]) for row in incidents]
    ss_tot = sum(row[1] * row[1] for row in incidents)
    ss_res = sum(r * r for _, r in resid)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return alpha, sigma, r2, resid


def _calibrated_alpha(
    category: Optional[str] = None,
    regime: Optional[str] = None,
    robust: bool = True,
) -> tuple[float, float, float, int]:
    """Slope (no intercept) on the loaded reference incidents.

    Returns (alpha_eur_millions_per_unit, sigma_residual, R2, n_used).

    Filtering precedence:
      • If category AND regime given AND the bucket has ≥3 incidents,
        slice on both (Sprint D.3 behaviour).
      • Else if category alone has ≥3 incidents, slice on category.
      • Else if regime alone has ≥3 incidents, slice on regime.
      • Else fall back to the overall pool.

    Regime = "calm" / "stressed" / "crisis" — same incident behaves
    very differently across regimes (Italian-bank cost in 2017 stressed
    BTP-Bund regime ≠ same in 2008 crisis regime).

    `robust=True` (default since D.1) uses Huber regression (IRLS,
    k=1.345) bounding the influence of 2σ+ outliers like Lehman 2008.
    """
    incidents = _load_reference_incidents()
    # Try most-specific filter first; widen progressively until ≥3 rows
    candidates = []
    if category and regime:
        candidates.append([r for r in incidents if r[3] == category and r[4] == regime])
    if category:
        candidates.append([r for r in incidents if r[3] == category])
    if regime:
        candidates.append([r for r in incidents if r[4] == regime])
    candidates.append(incidents)  # fallback overall
    chosen = next((c for c in candidates if len(c) >= 3), incidents)
    if not chosen:
        return 50.0, 0.0, 0.0, 0
    if robust:
        alpha, sigma, r2, _ = _huber_no_intercept(chosen)
    else:
        alpha, sigma, r2, _ = _ols_no_intercept(chosen)
    return alpha, sigma, r2, len(chosen)


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


# Cross-sector contagion multiplier — Sprint D.5: derive per-brief
# from the empirical VAR(1) network in shared/sector_contagion_var.json
# instead of a hardcoded 1.6. Fall back to 1.6 only if the VAR file is
# missing or no significant edges fire.
GAMMA_CONTAGION_DEFAULT = 1.6


_VAR_CACHE: Optional[dict] = None
_TICKER_TO_SECTOR_CACHE: Optional[dict[str, str]] = None


def _load_var_matrix() -> dict:
    """Load and cache the cross-sector VAR(1) matrix (Sprint 82)."""
    global _VAR_CACHE
    if _VAR_CACHE is not None:
        return _VAR_CACHE
    try:
        _VAR_CACHE = json.loads(VAR_CONTAGION_PATH.read_text()) or {}
    except Exception as e:
        logger.warning(f"VAR matrix load failed: {e}")
        _VAR_CACHE = {}
    return _VAR_CACHE


def _ticker_to_sector(tk: str) -> Optional[str]:
    """Cached ticker → sector lookup from stock_universe.json."""
    global _TICKER_TO_SECTOR_CACHE
    if _TICKER_TO_SECTOR_CACHE is None:
        try:
            data = json.loads(STOCK_UNIVERSE_PATH.read_text())
            _TICKER_TO_SECTOR_CACHE = {}
            for s in data.get("stocks", []):
                t = s.get("ticker")
                if t:
                    _TICKER_TO_SECTOR_CACHE[t] = s.get("sector", "unknown")
        except Exception as e:
            logger.warning(f"stock universe load failed: {e}")
            _TICKER_TO_SECTOR_CACHE = {}
    return _TICKER_TO_SECTOR_CACHE.get(tk)


def _gamma_from_var(source_sectors: set[str], t_threshold: float = 1.96) -> tuple[float, list[dict]]:
    """Compute γ_contagion from VAR(1) network for the given source sectors.

    For each source sector mentioned in the brief (via priced tickers),
    sum the absolute β of all statistically-significant outgoing edges
    (|t-stat| > 1.96) to OTHER sectors. γ = 1 + mean(per-source sum).

    Returns (γ, evidence_list) where evidence_list documents each
    significant edge — auditable by the CRO.
    """
    var = _load_var_matrix()
    matrix = var.get("matrix") or {}
    if not matrix or not source_sectors:
        return GAMMA_CONTAGION_DEFAULT, []

    per_source_sums: list[float] = []
    evidence: list[dict] = []
    for src in source_sectors:
        row = matrix.get(src) or {}
        if not isinstance(row, dict):
            continue
        edge_sum = 0.0
        for dst, info in row.items():
            if dst == src or not isinstance(info, dict):
                continue
            try:
                beta = float(info.get("beta", 0) or 0)
                t_stat = float(info.get("t_stat", 0) or 0)
            except (TypeError, ValueError):
                continue
            if abs(t_stat) >= t_threshold:
                edge_sum += abs(beta)
                evidence.append({
                    "from": src, "to": dst,
                    "beta": round(beta, 4), "t_stat": round(t_stat, 2),
                })
        per_source_sums.append(edge_sum)

    if not per_source_sums:
        return GAMMA_CONTAGION_DEFAULT, evidence
    mean_edge_sum = sum(per_source_sums) / len(per_source_sums)
    gamma = round(1.0 + mean_edge_sum, 3)
    # Bound to a sane range: 1.0 (no contagion) up to 4.0 (extreme)
    gamma = max(1.0, min(4.0, gamma))
    return gamma, sorted(evidence, key=lambda e: -abs(e["beta"]))[:20]


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


def estimate_anchor(
    total_shock_units: float,
    category: Optional[str] = None,
    regime: Optional[str] = None,
) -> dict:
    """Method A — α-calibrated shock anchor.

    Slices α on (category, regime) when both have ≥3 incidents in the
    intersection — banking_it×stressed gives a much tighter fit than
    banking_it alone or stressed alone, when both signals are present.

    Returns a dict with point estimate + 90% CI band + breakdown so
    the UI can render the methodology transparently.
    """
    overall_alpha, overall_sigma, overall_r2, overall_n = _calibrated_alpha()
    alpha, sigma, r2, n = _calibrated_alpha(category=category, regime=regime)

    # Determine actual scope used (the most-specific filter that hit ≥3 rows)
    incidents = _load_reference_incidents()
    if category and regime and len([r for r in incidents if r[3] == category and r[4] == regime]) >= 3:
        scope = f"{category} × {regime}"
    elif category and len([r for r in incidents if r[3] == category]) >= 3:
        scope = category
    elif regime and len([r for r in incidents if r[4] == regime]) >= 3:
        scope = f"{regime} regime"
    else:
        scope = "overall"

    point_eur = float(total_shock_units * alpha * 1_000_000)
    band = 1.645 * sigma * 1_000_000
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
            "calibration_scope": scope,
            "requested_category": category,
            "requested_regime": regime,
            "fallback_overall_alpha_eur_per_unit":
                round(overall_alpha * 1_000_000, 0) if scope != "overall" else None,
        },
        "formula": "|Σ shock_mag × shock_dir| × α",
    }


def estimate_ticker(ticker_price_history: list[dict]) -> dict:
    """Method B — direct ticker market-cap loss × VAR(1)-derived γ.

    `ticker_price_history` is a list of per-round snapshots, each one a
    dict from `TickerPriceState.step()`: { ticker → {cum_pct, ...} }.
    We take the LAST snapshot (final cumulative move) per ticker as the
    representative loss.

    γ_contagion (Sprint D.5): derived from the empirical cross-sector
    VAR(1) coefficients (shared/sector_contagion_var.json) by summing
    the |β| of statistically-significant outgoing edges (|t|>1.96) from
    each source sector touched by the brief. Replaces the hardcoded
    γ=1.6 anchored to Brexit alone.
    """
    if not ticker_price_history:
        return {
            "method": "ticker",
            "point_eur": 0.0, "low_eur": 0.0, "high_eur": 0.0,
            "inputs": {"tickers_priced": 0, "tickers_unknown": 0,
                       "direct_loss_eur": 0.0,
                       "contagion_multiplier": GAMMA_CONTAGION_DEFAULT,
                       "contagion_source": "default (no tickers)"},
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
    source_sectors: set[str] = set()
    for tk, v in final_snap.items():
        cum_pct = abs(float(v.get("cum_pct", 0.0) or 0.0))
        mcap_m = caps.get(tk)
        sect = _ticker_to_sector(tk)
        if sect:
            source_sectors.add(sect)
        if mcap_m is None:
            unknown += 1
            continue
        loss_m = (cum_pct / 100.0) * mcap_m
        direct_loss_m += loss_m
        priced += 1
        breakdown.append({
            "ticker": tk,
            "sector": sect or "unknown",
            "cum_pct": round(float(v.get("cum_pct", 0.0)), 3),
            "mcap_eur_m": mcap_m,
            "loss_eur_m": round(loss_m, 2),
        })
    breakdown.sort(key=lambda x: x["loss_eur_m"], reverse=True)

    gamma, var_evidence = _gamma_from_var(source_sectors)
    contagion_source = "VAR(1) empirical" if var_evidence else "default fallback (no significant edges)"

    point_m = direct_loss_m * gamma
    band_m = point_m * 0.25  # ±25% on γ uncertainty
    return {
        "method": "ticker",
        "point_eur": round(point_m * 1_000_000, 2),
        "low_eur": max(0.0, round((point_m - band_m) * 1_000_000, 2)),
        "high_eur": round((point_m + band_m) * 1_000_000, 2),
        "inputs": {
            "tickers_priced": priced,
            "tickers_unknown": unknown,
            "direct_loss_eur": round(direct_loss_m * 1_000_000, 2),
            "contagion_multiplier": gamma,
            "contagion_source": contagion_source,
            "source_sectors": sorted(source_sectors),
            "var_evidence": var_evidence[:8],  # top 8 strongest edges
            "per_ticker": breakdown[:10],
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
