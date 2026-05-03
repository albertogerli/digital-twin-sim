"""Compare Sprint 15 re-calibration vs the v2_discrepancy baseline.

Reads both validation_results_v2.json files, computes per-group (train/test)
metrics, and writes a markdown comparison report into the sprint15 dir.

Metrics:
- MAE   — mean absolute error in percentage points
- cov90 — fraction of held-out scenarios whose 90% credible interval covers gt
- cov50 — same, 50% interval
- CRPS  — continuous ranked probability score (lower is better)

Per-domain breakdown also included.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES_ROOT = PROJECT_ROOT / "calibration" / "results" / "hierarchical_calibration"

BASELINE_DIR = RES_ROOT / "v2_discrepancy"
NEW_DIR      = RES_ROOT / "sprint15"


def _summary(group: list[dict]) -> dict:
    if not group:
        return {"n": 0}
    abs_errs = [s["abs_error"] for s in group]
    cov90    = [s["in_90"] for s in group]
    cov50    = [s["in_50"] for s in group]
    crps     = [s["crps"] for s in group]
    return {
        "n":     len(group),
        "mae":   statistics.mean(abs_errs),
        "cov90": sum(cov90) / len(cov90),
        "cov50": sum(cov50) / len(cov50),
        "crps":  statistics.mean(crps),
    }


def _group_by(results: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for s in results:
        out[s.get(key, "?")].append(s)
    return dict(out)


def _delta_str(new: float, old: float, *, lower_is_better: bool, fmt: str = ".2f") -> str:
    d = new - old
    if abs(d) < 0.005:
        return f" ({d:+{fmt}})"
    arrow = ""
    if lower_is_better:
        arrow = "✓" if d < 0 else "✗"
    else:
        arrow = "✓" if d > 0 else "✗"
    return f" ({d:+{fmt}} {arrow})"


def _fmt_block(label: str, new: dict, old: dict) -> list[str]:
    if not new and not old:
        return []
    if not new:
        return [f"- **{label}** N={old['n']:3d}  (no new run)"]
    if not old:
        return [f"- **{label}** N={new['n']:3d}  MAE={new['mae']:.2f}pp  cov90={new['cov90']*100:.1f}%  cov50={new['cov50']*100:.1f}%  CRPS={new['crps']:.2f}"]

    return [(
        f"- **{label}**  N={new['n']:3d}  "
        f"MAE={new['mae']:.2f}pp{_delta_str(new['mae'], old['mae'], lower_is_better=True)}  "
        f"cov90={new['cov90']*100:.1f}%{_delta_str(new['cov90']*100, old['cov90']*100, lower_is_better=False, fmt='.1f')}  "
        f"cov50={new['cov50']*100:.1f}%{_delta_str(new['cov50']*100, old['cov50']*100, lower_is_better=False, fmt='.1f')}  "
        f"CRPS={new['crps']:.2f}{_delta_str(new['crps'], old['crps'], lower_is_better=True)}"
    )]


def _load(p: Path) -> list[dict]:
    if not p.exists():
        raise FileNotFoundError(f"validation results not found: {p}")
    with open(p) as f:
        return json.load(f)


def _losses(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    baseline = _load(BASELINE_DIR / "validation_results_v2.json")
    new      = _load(NEW_DIR      / "validation_results_v2.json")

    base_train = _summary([s for s in baseline if s.get("group") == "train"])
    base_test  = _summary([s for s in baseline if s.get("group") == "test"])
    new_train  = _summary([s for s in new      if s.get("group") == "train"])
    new_test   = _summary([s for s in new      if s.get("group") == "test"])
    base_all   = _summary(baseline)
    new_all    = _summary(new)

    base_loss = _losses(BASELINE_DIR / "loss_history_v2.json")
    new_loss  = _losses(NEW_DIR      / "loss_history_v2.json")

    base_final_loss = base_loss["phase_b"][-1] if base_loss else None
    new_final_loss  = new_loss["phase_b"][-1]  if new_loss  else None

    # Per-domain (test set is too small to slice; use full)
    base_by_dom = {d: _summary(g) for d, g in _group_by(baseline, "domain").items()}
    new_by_dom  = {d: _summary(g) for d, g in _group_by(new,      "domain").items()}

    lines = [
        "# Sprint 15 — Re-calibration vs v2_discrepancy baseline",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        "## Setup",
        "- **Baseline**: `calibration/results/hierarchical_calibration/v2_discrepancy/`",
        "  (last run before Sprint 1-13 simulator fixes)",
        "- **Sprint 15**: `calibration/results/hierarchical_calibration/sprint15/`",
        "  (re-run after country-alias fix, realism gate fix, agent prompt + engine improvements)",
        "- Same 42 empirical scenarios, same SVI hyperparameters (3000 steps, lr=0.005, seed=42)",
        "",
        "## Final SVI loss",
    ]
    if base_final_loss is not None and new_final_loss is not None:
        d = new_final_loss - base_final_loss
        lines.append(f"- baseline: {base_final_loss:.2f}")
        lines.append(f"- sprint15: {new_final_loss:.2f}{_delta_str(new_final_loss, base_final_loss, lower_is_better=True)}")
    else:
        lines.append("- (loss histories not both available)")

    lines.extend([
        "",
        "## Aggregate metrics",
        "Format: `MAE` ± delta (✓ improved, ✗ regressed) | `cov` lift to 90%/50% | `CRPS`",
        "",
        *_fmt_block("OVERALL", new_all,   base_all),
        *_fmt_block("TRAIN",   new_train, base_train),
        *_fmt_block("TEST",    new_test,  base_test),
        "",
        "## Per-domain MAE (full corpus)",
        "",
        "| Domain         | N |  baseline MAE | sprint15 MAE | Δ            | cov90 base→new |",
        "|----------------|---|---------------|--------------|--------------|----------------|",
    ])

    domains = sorted(set(base_by_dom) | set(new_by_dom))
    for d in domains:
        b = base_by_dom.get(d)
        n = new_by_dom.get(d)
        if not n or not b:
            continue
        d_mae = n["mae"] - b["mae"]
        arrow = "✓" if d_mae < 0 else "✗" if d_mae > 0 else "·"
        lines.append(
            f"| {d:14s} | {n['n']:1d} | {b['mae']:11.2f}pp | {n['mae']:10.2f}pp | "
            f"{d_mae:+5.2f}pp {arrow} | {b['cov90']*100:5.1f}% → {n['cov90']*100:5.1f}% |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if new_test["n"] > 0 and base_test["n"] > 0:
        mae_d = new_test["mae"] - base_test["mae"]
        if mae_d < -1.0:
            lines.append(f"**TEST MAE improved by {abs(mae_d):.2f}pp** "
                         "— Sprint 1-13 simulator changes meaningfully tightened predictions.")
        elif mae_d < 0:
            lines.append(f"**TEST MAE marginally improved ({mae_d:+.2f}pp)** "
                         "— no regression; calibration remains stable.")
        elif mae_d < 1.0:
            lines.append(f"**TEST MAE essentially unchanged ({mae_d:+.2f}pp)** "
                         "— Sprint 1-13 fixes didn't move calibration; expected since SVI uses a forward "
                         "model decoupled from the live LLM simulator.")
        else:
            lines.append(f"**TEST MAE regressed by {mae_d:+.2f}pp** "
                         "— investigate which scenarios shifted (per-scenario diff below).")

    # Per-scenario diff for the test set (smallest set, easiest to scan)
    base_by_id = {s["id"]: s for s in baseline if s.get("group") == "test"}
    new_by_id  = {s["id"]: s for s in new      if s.get("group") == "test"}
    common = sorted(set(base_by_id) & set(new_by_id))
    if common:
        lines.extend([
            "",
            "### Per-scenario TEST diff",
            "",
            "| Scenario | gt | base pred | new pred | base |err| | new |err| | Δ |",
            "|----------|----|-----------|----------|------------|-----------|---|",
        ])
        for sid in common:
            b = base_by_id[sid]
            n = new_by_id[sid]
            d = n["abs_error"] - b["abs_error"]
            arrow = "✓" if d < 0 else "✗" if d > 0 else "·"
            lines.append(
                f"| {sid[:38]:38s} | {b['gt']:5.1f} | "
                f"{b['sim_mean']:7.2f} | {n['sim_mean']:7.2f} | "
                f"{b['abs_error']:6.2f} | {n['abs_error']:6.2f} | "
                f"{d:+5.2f} {arrow} |"
            )

    out = NEW_DIR / "sprint15_vs_baseline.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved comparison report → {out}")
    print()
    # Echo the headline numbers
    for label, new, base in [("OVERALL", new_all, base_all), ("TRAIN", new_train, base_train), ("TEST", new_test, base_test)]:
        if new and base:
            print(f"  {label:7s} N={new['n']:3d}  MAE base→new = {base['mae']:5.2f} → {new['mae']:5.2f}pp  "
                  f"({new['mae']-base['mae']:+.2f})")


if __name__ == "__main__":
    main()
