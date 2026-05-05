"""DORA economic-impact α — refit script (Sprint A).

Reads the curated reference incidents from
shared/dora_reference_incidents.json, fits the OLS-no-intercept slope
α (EUR per simulated shock-unit), computes residual diagnostics, and
writes a structured snapshot to outputs/dora_calibration.json so the
backend / UI can show "α was last refit on YYYY-MM-DD with N incidents,
R² = X" without recomputing on every API request.

Run:
  python -m scripts.calibrate_dora_alpha
  python -m scripts.calibrate_dora_alpha --by-category   # also fit per-category α
  python -m scripts.calibrate_dora_alpha --print

Wire this into the nightly admin-jobs cron once the Sprint B live-data
pipeline is in (so α tracks fresh historical-cost annotations).

Future (Sprint A.2):
  • Switch to OLS+intercept once N > 50.
  • Heteroscedastic-robust SE (HC3) on the residual band.
  • Bootstrap 1000-sample CI on α.
  • Drop top/bottom 5% as outlier sensitivity check.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.dora.economic_impact import (  # noqa: E402
    INCIDENTS_PATH, CALIBRATION_OUTPUT_PATH,
    _load_reference_incidents, _calibrated_alpha, calibration_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def fit_overall() -> dict:
    alpha, sigma, r2 = _calibrated_alpha()
    incidents = _load_reference_incidents()
    residuals = []
    for s, c, ident, cat in incidents:
        pred = alpha * s
        res = c - pred
        residuals.append({
            "id": ident,
            "category": cat,
            "shock_units": s,
            "actual_eur_m": c,
            "predicted_eur_m": round(pred, 1),
            "residual_eur_m": round(res, 1),
            "residual_pct": round(100.0 * res / c, 1) if c > 0 else None,
        })
    # Outlier flags: residual > 2σ
    for r in residuals:
        if abs(r["residual_eur_m"]) > 2 * sigma:
            r["outlier_2sigma"] = True
    return {
        "alpha_eur_m_per_unit": round(alpha, 2),
        "sigma_residual_eur_m": round(sigma, 2),
        "r2": round(r2, 4),
        "n_incidents": len(incidents),
        "residuals": sorted(residuals, key=lambda r: -abs(r["residual_eur_m"])),
    }


def fit_by_category() -> dict:
    incidents = _load_reference_incidents()
    by_cat: dict[str, list] = {}
    for s, c, ident, cat in incidents:
        by_cat.setdefault(cat, []).append((s, c, ident))
    out: dict[str, dict] = {}
    for cat, rows in by_cat.items():
        if len(rows) < 2:
            out[cat] = {"n": len(rows), "alpha_eur_m_per_unit": None, "note": "n<2 — skipped"}
            continue
        sx2 = sum(s * s for s, _, _ in rows)
        sxy = sum(s * c for s, c, _ in rows)
        alpha = sxy / sx2 if sx2 > 0 else 0.0
        residuals = [c - alpha * s for s, c, _ in rows]
        sigma = (sum(r * r for r in residuals) / max(1, len(rows) - 1)) ** 0.5
        ss_tot = sum(c * c for _, c, _ in rows)
        ss_res = sum(r * r for r in residuals)
        r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        out[cat] = {
            "n": len(rows),
            "alpha_eur_m_per_unit": round(alpha, 2),
            "sigma_residual_eur_m": round(sigma, 2),
            "r2": round(r2, 4),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--by-category", action="store_true", help="Also fit α per-category")
    parser.add_argument("--print", action="store_true", help="Echo result to stdout")
    parser.add_argument("--output", type=Path, default=CALIBRATION_OUTPUT_PATH)
    args = parser.parse_args()

    overall = fit_overall()
    summary = calibration_summary()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "incidents_path": str(INCIDENTS_PATH.relative_to(REPO_ROOT)),
        "overall": overall,
        "summary": summary,
    }
    if args.by_category:
        payload["by_category"] = fit_by_category()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    logger.info(
        f"Refit DORA α from {summary['n_incidents']} incidents → "
        f"€{summary['alpha_eur_per_unit']:,.0f}/unit (R²={summary['r2']:.3f}). "
        f"Wrote {args.output.relative_to(REPO_ROOT)}"
    )
    if args.print:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
