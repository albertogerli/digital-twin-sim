#!/usr/bin/env python3
"""Grounding v2.3: Same as v2.2 (hybrid events) + Public Opinion agent.

The Public Opinion agent is added in build_scenario_data_from_json(),
so all we need to do is re-run Phase B+C on the v2.2 dataset.
The agent will be automatically injected by the data loader.

Usage:
    .venv_cal/bin/python calibration/run_grounding_v2_3.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Paths
SCENARIOS_V22_DIR = ROOT / "calibration" / "empirical" / "scenarios_v2.2"
V23_OUTPUT_DIR = ROOT / "calibration" / "results" / "hierarchical_calibration" / "v2.3_pubop"

def main():
    t_start = time.time()
    print("=" * 70)
    print("GROUNDING v2.3: Hybrid events + Public Opinion agent")
    print("=" * 70)

    # Monkey-patch to use v2.2 dataset (which has grounded events)
    # The PubOp agent is injected automatically by build_scenario_data_from_json
    from src.inference import hierarchical_model_v2 as hm_v2

    original_dir = hm_v2.EMPIRICAL_DIR
    original_results_dir = hm_v2.V2_RESULTS_DIR

    hm_v2.EMPIRICAL_DIR = SCENARIOS_V22_DIR
    V23_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hm_v2.V2_RESULTS_DIR = V23_OUTPUT_DIR

    try:
        posteriors, results_per_scenario = hm_v2.run_phase_bc_v2(
            n_svi_steps=3000,
            n_pp_samples=200,
            lr=0.005,
            seed=42,
        )
    finally:
        hm_v2.EMPIRICAL_DIR = original_dir
        hm_v2.V2_RESULTS_DIR = original_results_dir

    t_cal = time.time()
    print(f"\n⏱ Calibration: {(t_cal - t_start)/60:.1f} min")

    # Generate comparison report
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    import numpy as np

    train_results = [r for r in results_per_scenario if r["group"] == "train"]
    test_results = [r for r in results_per_scenario if r["group"] == "test"]

    def agg(results):
        if not results:
            return {}
        maes = [r["abs_error"] for r in results]
        errors = [r["error"] for r in results]
        return {
            "mae": np.mean(maes),
            "rmse": np.sqrt(np.mean(np.array(errors)**2)),
            "cov90": np.mean([r["in_90"] for r in results]) * 100,
            "cov50": np.mean([r["in_50"] for r in results]) * 100,
        }

    train_agg = agg(train_results)
    test_agg = agg(test_results)

    print(f"\n{'Metric':<25} {'v2':>8} {'v2.1':>8} {'v2.2':>8} {'v2.3':>8}")
    print("-" * 62)
    rows = [
        ("MAE test",        19.2, 18.9, 11.4, test_agg.get("mae")),
        ("MAE train",       14.3, 15.1, 15.3, train_agg.get("mae")),
        ("RMSE test",       26.6, 21.6, 16.8, test_agg.get("rmse")),
        ("Coverage 90% tr", 79.4,  2.9, 73.3, train_agg.get("cov90")),
    ]
    for name, v2, v21, v22, v23 in rows:
        v23_str = f"{v23:>6.1f}pp" if v23 is not None else f"{'?':>8}"
        print(f"  {name:<23} {v2:>6.1f}pp {v21:>6.1f}pp {v22:>6.1f}pp {v23_str}")

    # Per-scenario
    print(f"\n{'Scenario':<45} {'GT':>5} {'Sim':>6} {'Err':>7} {'90CI':>5}")
    print("-" * 72)
    for r in sorted(results_per_scenario, key=lambda x: -abs(x["error"])):
        ci = "Y" if r["in_90"] else "N"
        print(f"  {r['id'][:43]:<43} {r['gt']:>5.1f} {r['sim_mean']:>6.1f} {r['error']:>+7.1f} {ci:>5}")

    # δ_s comparison
    v2_post_path = ROOT / "calibration" / "results" / "hierarchical_calibration" / "v2_discrepancy" / "posteriors_v2.json"
    if v2_post_path.exists():
        with open(v2_post_path) as f:
            v2_post = json.load(f)
        v2_scenarios = v2_post.get("scenarios", {})

        print(f"\n{'Scenario':<45} {'δ_s v2':>8} {'δ_s v2.3':>9}")
        print("-" * 65)
        for sid, sdata in sorted(posteriors.get("scenarios", {}).items()):
            v2_ds = v2_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
            v23_ds = sdata.get("delta_s", {}).get("mean", 0)
            print(f"  {sid[:43]:<43} {v2_ds:>+8.3f} {v23_ds:>+9.3f}")

    # Save report
    report_lines = [
        "# Grounding v2.3: Hybrid Events + Public Opinion Agent",
        "",
        "## Strategy",
        "- v2.2 hybrid dataset (original agents + grounded events)",
        "- Added implicit Public Opinion agent from polling trajectory",
        "  - Position: first polling pro_pct mapped to [-1, +1]",
        "  - Type: citizen (rigidity=0.1, tolerance=0.9, influence=0.7)",
        "  - Very event-reactive, broad social tolerance",
        "- Discrepancy model with δ_s",
        "",
        "## Headline Metrics",
        "",
        "| Metric | v2 | v2.1 | v2.2 | v2.3 |",
        "|---|---|---|---|---|",
    ]
    for name, v2, v21, v22, v23 in rows:
        v23_str = f"{v23:.1f}pp" if v23 is not None else "?"
        report_lines.append(f"| {name} | {v2:.1f}pp | {v21:.1f}pp | {v22:.1f}pp | {v23_str} |")

    report_lines.extend(["", "## Per-Scenario", ""])
    report_lines.append("| Scenario | GT | Sim | Error | in 90% CI |")
    report_lines.append("|---|---|---|---|---|")
    for r in sorted(results_per_scenario, key=lambda x: x["id"]):
        ci = "YES" if r["in_90"] else "no"
        report_lines.append(f"| {r['id'][:40]} | {r['gt']:.1f} | {r['sim_mean']:.1f} | {r['error']:+.1f} | {ci} |")

    report_lines.extend(["", "---", f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*"])

    report_path = V23_OUTPUT_DIR / "calibration_report_v2.3.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport: {report_path}")
    print(f"\nTOTAL: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
