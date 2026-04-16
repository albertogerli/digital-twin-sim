#!/usr/bin/env python3
"""Grounding v2.2: Hybrid grounding (events only) + discrepancy model.

Strategy:
  - Keep original agent positions (calibrated by human expert)
  - Replace only events with grounded verified events from v2.1
  - Use hierarchical_model_v2 (discrepancy) instead of transfer model
  - This should give us: better events + calibrated coverage via δ_s

Usage:
    .venv_cal/bin/python calibration/run_grounding_v2_2.py
"""

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("grounding_v2.2")

# Paths
SCENARIOS_ORIG_DIR = ROOT / "calibration" / "empirical" / "scenarios"
SCENARIOS_V21_DIR = ROOT / "calibration" / "empirical" / "scenarios_v2.1"
SCENARIOS_V22_DIR = ROOT / "calibration" / "empirical" / "scenarios_v2.2"
GROUNDING_V21_DIR = ROOT / "calibration" / "results" / "grounding_v2.1"
POSTERIORS_V2_PATH = ROOT / "calibration" / "results" / "hierarchical_calibration" / "v2_discrepancy" / "posteriors_v2.json"
SYNTHETIC_PRIOR_PATH = ROOT / "calibration" / "results" / "hierarchical_calibration" / "synthetic_prior.json"
V22_OUTPUT_DIR = ROOT / "calibration" / "results" / "hierarchical_calibration" / "v2.2_hybrid"

# Same domains as v2.1
GROUND_DOMAINS = {"financial", "corporate", "energy", "public_health"}


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Build hybrid dataset (original agents + grounded events)
# ══════════════════════════════════════════════════════════════════════════

def step1_build_hybrid_dataset():
    """For each GROUND scenario, keep original agents, inject grounded events."""
    print("=" * 70)
    print("STEP 1: Build hybrid dataset (original agents + grounded events)")
    print("=" * 70)

    SCENARIOS_V22_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest to know all scenarios
    manifest_path = SCENARIOS_ORIG_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    grounded_count = 0
    kept_count = 0
    hybrid_scenarios = []

    for entry in manifest["scenarios"]:
        sid = entry["id"]
        orig_path = SCENARIOS_ORIG_DIR / f"{sid}.json"
        dest_path = SCENARIOS_V22_DIR / f"{sid}.json"

        if not orig_path.exists():
            continue

        with open(orig_path) as f:
            orig = json.load(f)

        domain = orig.get("domain", "")

        if domain in GROUND_DOMAINS:
            # Check if we have grounded events from v2.1
            events_path = GROUNDING_V21_DIR / sid / "events.json"
            if events_path.exists():
                with open(events_path) as f:
                    grounded_events = json.load(f)

                if grounded_events:
                    # HYBRID: original agents + grounded events
                    hybrid = dict(orig)
                    hybrid["events"] = grounded_events
                    hybrid["_grounding_v2.2"] = {
                        "strategy": "hybrid_events_only",
                        "agents": "original",
                        "events": "grounded_v2.1",
                        "n_grounded_events": len(grounded_events),
                        "n_original_events": len(orig.get("events", [])),
                    }

                    with open(dest_path, "w") as f:
                        json.dump(hybrid, f, indent=2)

                    grounded_count += 1
                    hybrid_scenarios.append(sid)
                    print(f"  HYBRID {sid}: {len(orig.get('agents', []))} orig agents, "
                          f"{len(orig.get('events', []))} → {len(grounded_events)} events")
                    continue

            # Fallback: copy original
            print(f"  KEEP (no grounded events) {sid}")

        # KEEP: copy original as-is
        with open(dest_path, "w") as f:
            json.dump(orig, f, indent=2)
        kept_count += 1

    # Copy manifest
    new_manifest = dict(manifest)
    with open(SCENARIOS_V22_DIR / "manifest.json", "w") as f:
        json.dump(new_manifest, f, indent=2)

    # Copy meta files
    for meta in SCENARIOS_ORIG_DIR.glob("*.meta.json"):
        with open(meta) as f:
            meta_data = json.load(f)
        with open(SCENARIOS_V22_DIR / meta.name, "w") as f:
            json.dump(meta_data, f, indent=2)

    print(f"\n  Total: {grounded_count + kept_count} scenarios")
    print(f"  Hybrid (events grounded): {grounded_count}")
    print(f"  Kept unchanged: {kept_count}")

    return hybrid_scenarios


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Run Phase B with discrepancy model on v2.2 dataset
# ══════════════════════════════════════════════════════════════════════════

def step2_recalibrate_discrepancy():
    """Run Phase B+C using hierarchical_model_v2 (with discrepancy δ_s)."""
    print("\n" + "=" * 70)
    print("STEP 2: Phase B+C with discrepancy model on v2.2 dataset")
    print("=" * 70)

    # Monkey-patch EMPIRICAL_DIR to point to v2.2
    from src.inference import hierarchical_model_v2 as hm_v2
    original_dir = hm_v2.EMPIRICAL_DIR
    hm_v2.EMPIRICAL_DIR = SCENARIOS_V22_DIR

    V22_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Also monkey-patch V2_RESULTS_DIR so it saves to our output
    original_results_dir = hm_v2.V2_RESULTS_DIR
    hm_v2.V2_RESULTS_DIR = V22_OUTPUT_DIR

    try:
        result = hm_v2.run_phase_bc_v2(
            n_svi_steps=3000,
            n_pp_samples=200,
            lr=0.005,
            seed=42,
        )
    finally:
        hm_v2.EMPIRICAL_DIR = original_dir
        hm_v2.V2_RESULTS_DIR = original_results_dir

    return result


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Compare v2 vs v2.1 vs v2.2
# ══════════════════════════════════════════════════════════════════════════

def step3_compare(result_v22, hybrid_scenarios):
    """Compare all three versions."""
    print("\n" + "=" * 70)
    print("STEP 3: Compare v2 vs v2.1 vs v2.2")
    print("=" * 70)

    # Load v2 posteriors
    with open(POSTERIORS_V2_PATH) as f:
        posteriors_v2 = json.load(f)

    # v2.2 posteriors
    posteriors_v22 = result_v22.get("posteriors", {})

    # v2.1 Phase C results (from previous run)
    v21_mae_test = 18.9
    v21_mae_train = 15.1
    v21_rmse_test = 21.6
    v21_cov90_train = 2.9

    # v2.2 Phase C results
    phase_c = result_v22.get("phase_c", {})
    train_results = phase_c.get("train_results", [])
    test_results = phase_c.get("test_results", [])

    # Compute v2.2 metrics from result
    per_scenario = result_v22.get("per_scenario", {})

    # ── Table A: Headline Metrics ──
    print("\n── Table A: Headline Metrics ──")
    print(f"{'Metric':<30} {'v2':>10} {'v2.1':>10} {'v2.2':>10}")
    print("-" * 65)

    # Extract v2.2 metrics from the result
    v22_train_agg = result_v22.get("train_aggregate", {})
    v22_test_agg = result_v22.get("test_aggregate", {})

    rows = [
        ("MAE test",        19.2, v21_mae_test,   v22_test_agg.get("mae")),
        ("MAE train",       14.3, v21_mae_train,  v22_train_agg.get("mae")),
        ("RMSE test",       26.6, v21_rmse_test,  v22_test_agg.get("rmse")),
        ("Coverage 90% tr", 79.4, v21_cov90_train, (v22_train_agg.get("coverage_90", 0) or 0) * 100),
    ]

    for name, v2_val, v21_val, v22_val in rows:
        v22_str = f"{v22_val:>8.1f}pp" if v22_val is not None else f"{'?':>10}"
        print(f"  {name:<28} {v2_val:>8.1f}pp {v21_val:>8.1f}pp {v22_str}")

    # ── Table B: Discrepancy ──
    v2_disc = posteriors_v2.get("discrepancy", {})
    v22_disc = posteriors_v22.get("discrepancy", {})

    print("\n── Table B: Discrepancy ──")
    print(f"{'Metric':<25} {'v2':>12} {'v2.2':>12} {'Δ':>10}")
    print("-" * 62)

    for label, key in [("σ_b,between", "sigma_delta_between"), ("σ_b,within", "sigma_delta_within")]:
        v2_val = v2_disc.get(key, {}).get("mean", 0)
        v22_val = v22_disc.get(key, {}).get("mean", None)
        if v22_val is not None:
            delta = v22_val - v2_val
            print(f"  {label:<23} {v2_val:>10.3f} {v22_val:>10.3f} {delta:>+10.3f}")
        else:
            print(f"  {label:<23} {v2_val:>10.3f} {'?':>12} {'?':>10}")

    # ── Table C: Per-scenario δ_s comparison ──
    v2_scenarios = posteriors_v2.get("scenarios", {})
    v22_scenarios = posteriors_v22.get("scenarios", {})

    print("\n── Table C: Per-Scenario δ_s (Grounded, sorted by |Δ|) ──")
    print(f"{'Scenario':<45} {'GT%':>5} {'v2 δ_s':>8} {'v2.2 δ_s':>9} {'Δ|δ|':>7}")
    print("-" * 80)

    deltas = []
    for sid in sorted(hybrid_scenarios):
        v2_ds = v2_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
        v22_ds = v22_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
        delta_abs = abs(v2_ds) - abs(v22_ds)
        deltas.append((sid, v2_ds, v22_ds, delta_abs))

    # Sort by improvement (positive = v2.2 better)
    deltas.sort(key=lambda x: -x[3])
    for sid, v2_ds, v22_ds, delta_abs in deltas:
        # Find GT
        orig_path = SCENARIOS_ORIG_DIR / f"{sid}.json"
        gt = 0
        if orig_path.exists():
            with open(orig_path) as f:
                gt = json.load(f).get("ground_truth_outcome", {}).get("final_pro_pct", 0)
        print(f"{sid:<45} {gt:>5.1f} {v2_ds:>+8.3f} {v22_ds:>+9.3f} {delta_abs:>+7.3f}")

    # Summary
    improved = sum(1 for _, v2_ds, v22_ds, _ in deltas if abs(v22_ds) < abs(v2_ds))
    total = len(deltas)
    mean_v2 = sum(abs(d[1]) for d in deltas) / total if total else 0
    mean_v22 = sum(abs(d[2]) for d in deltas) / total if total else 0
    print(f"\n  Improved: {improved}/{total} scenarios")
    print(f"  Mean |δ_s| grounded: v2={mean_v2:.3f} → v2.2={mean_v22:.3f} ({mean_v22-mean_v2:+.3f})")

    # ── Table D: Kept scenarios sanity check ──
    print("\n── Table D: Sanity Check (Kept Scenarios) ──")
    manifest_path = SCENARIOS_ORIG_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    kept = [(e["id"], e["domain"]) for e in manifest["scenarios"]
            if e["domain"] not in GROUND_DOMAINS]
    print(f"{'Scenario':<45} {'v2 δ_s':>8} {'v2.2 δ_s':>9} {'Δ|δ|':>7}")
    print("-" * 75)
    for sid, domain in sorted(kept)[:8]:
        v2_ds = v2_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
        v22_ds = v22_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
        delta_abs = abs(v2_ds) - abs(v22_ds)
        print(f"{sid:<45} {v2_ds:>+8.3f} {v22_ds:>+9.3f} {delta_abs:>+7.3f}")

    return {
        "train_agg": v22_train_agg,
        "test_agg": v22_test_agg,
        "posteriors_v22": posteriors_v22,
    }


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Generate report
# ══════════════════════════════════════════════════════════════════════════

def step4_report(hybrid_scenarios, comparison):
    """Generate markdown report."""
    print("\n" + "=" * 70)
    print("STEP 4: Generate Report")
    print("=" * 70)

    posteriors_v22 = comparison.get("posteriors_v22", {})
    train_agg = comparison.get("train_agg", {})
    test_agg = comparison.get("test_agg", {})

    with open(POSTERIORS_V2_PATH) as f:
        posteriors_v2 = json.load(f)

    v2_disc = posteriors_v2.get("discrepancy", {})
    v22_disc = posteriors_v22.get("discrepancy", {})
    v2_scenarios = posteriors_v2.get("scenarios", {})
    v22_scenarios = posteriors_v22.get("scenarios", {})

    lines = [
        "# Grounding v2.2: Hybrid Grounding + Discrepancy Model",
        "",
        "## Strategy",
        "",
        "- **Agents**: Keep original (human-calibrated positions, influence, rigidity)",
        "- **Events**: Replace with grounded verified events from Google Search",
        "- **Model**: hierarchical_model_v2 with discrepancy δ_s (not transfer model)",
        "",
        f"- Hybrid scenarios: {len(hybrid_scenarios)}",
        "",
        "## Headline Metrics",
        "",
        "| Metric | v2 | v2.1 | v2.2 |",
        "|---|---|---|---|",
    ]

    v22_mae_test = test_agg.get("mae", None)
    v22_mae_train = train_agg.get("mae", None)
    v22_rmse_test = test_agg.get("rmse", None)
    v22_cov90 = (train_agg.get("coverage_90", 0) or 0) * 100

    lines.append(f"| MAE test | 19.2pp | 18.9pp | {v22_mae_test:.1f}pp |" if v22_mae_test else "| MAE test | 19.2pp | 18.9pp | ? |")
    lines.append(f"| MAE train | 14.3pp | 15.1pp | {v22_mae_train:.1f}pp |" if v22_mae_train else "| MAE train | 14.3pp | 15.1pp | ? |")
    lines.append(f"| RMSE test | 26.6pp | 21.6pp | {v22_rmse_test:.1f}pp |" if v22_rmse_test else "| RMSE test | 26.6pp | 21.6pp | ? |")
    lines.append(f"| Coverage 90% train | 79.4% | 2.9% | {v22_cov90:.1f}% |")

    # Discrepancy
    sigma_bw_v2 = v2_disc.get("sigma_delta_within", {}).get("mean", 0)
    sigma_bw_v22 = v22_disc.get("sigma_delta_within", {}).get("mean", None)
    sigma_bb_v2 = v2_disc.get("sigma_delta_between", {}).get("mean", 0)
    sigma_bb_v22 = v22_disc.get("sigma_delta_between", {}).get("mean", None)

    lines.extend(["", "## Discrepancy", ""])
    if sigma_bw_v22 is not None:
        lines.extend([
            "| Metric | v2 | v2.2 | Δ |",
            "|---|---|---|---|",
            f"| σ_b,between | {sigma_bb_v2:.3f} | {sigma_bb_v22:.3f} | {sigma_bb_v22-sigma_bb_v2:+.3f} |",
            f"| σ_b,within | {sigma_bw_v2:.3f} | {sigma_bw_v22:.3f} | {sigma_bw_v22-sigma_bw_v2:+.3f} |",
        ])

    # Per-scenario
    lines.extend(["", "## Per-Scenario δ_s (Grounded)", ""])
    lines.append("| Scenario | GT% | v2 δ_s | v2.2 δ_s | Improved? |")
    lines.append("|---|---|---|---|---|")

    improved_count = 0
    for sid in sorted(hybrid_scenarios):
        v2_ds = v2_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
        v22_ds = v22_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
        better = abs(v22_ds) < abs(v2_ds)
        if better:
            improved_count += 1
        orig_path = SCENARIOS_ORIG_DIR / f"{sid}.json"
        gt = 0
        if orig_path.exists():
            with open(orig_path) as f:
                gt = json.load(f).get("ground_truth_outcome", {}).get("final_pro_pct", 0)
        emoji = "YES" if better else "no"
        lines.append(f"| {sid[:40]} | {gt:.1f} | {v2_ds:+.3f} | {v22_ds:+.3f} | {emoji} |")

    # Verdict
    lines.extend(["", "## Verdict", ""])

    verdict_parts = []
    if v22_mae_test is not None:
        if v22_mae_test < 18.9:
            verdict_parts.append(f"MAE test improved: 19.2→{v22_mae_test:.1f}pp (better than v2.1's 18.9)")
        elif v22_mae_test < 19.2:
            verdict_parts.append(f"MAE test: {v22_mae_test:.1f}pp (between v2 and v2.1)")
        else:
            verdict_parts.append(f"MAE test: {v22_mae_test:.1f}pp (no improvement)")

    if v22_cov90 > 50:
        verdict_parts.append(f"Coverage 90% restored: {v22_cov90:.1f}% (v2.1 was 2.9%)")
    elif v22_cov90 > 10:
        verdict_parts.append(f"Coverage 90% partially recovered: {v22_cov90:.1f}%")

    if sigma_bw_v22 is not None and sigma_bw_v22 < sigma_bw_v2:
        verdict_parts.append(f"σ_b,within reduced: {sigma_bw_v2:.3f}→{sigma_bw_v22:.3f} ({sigma_bw_v22-sigma_bw_v2:+.3f})")

    verdict_parts.append(f"Improved δ_s: {improved_count}/{len(hybrid_scenarios)} grounded scenarios")

    for v in verdict_parts:
        lines.append(f"- {v}")

    lines.extend(["", "---", f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*"])

    report_path = V22_OUTPUT_DIR / "calibration_report_v2.2.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Report: {report_path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("╔" + "═" * 68 + "╗")
    print("║    GROUNDING v2.2: Hybrid (events only) + Discrepancy Model       ║")
    print("╚" + "═" * 68 + "╝")

    # Step 1: Build hybrid dataset
    hybrid_scenarios = step1_build_hybrid_dataset()

    t1 = time.time()
    print(f"\n⏱ Dataset built in {t1 - t_start:.0f}s")

    # Step 2: Recalibrate with discrepancy model
    result_v22 = step2_recalibrate_discrepancy()

    t2 = time.time()
    print(f"\n⏱ Recalibration: {(t2 - t1)/60:.1f} min")

    # Step 3: Compare
    comparison = step3_compare(result_v22, hybrid_scenarios)

    # Step 4: Report
    step4_report(hybrid_scenarios, comparison)

    print(f"\n{'=' * 70}")
    print(f"TOTAL TIME: {(time.time() - t_start)/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
