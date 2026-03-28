#!/usr/bin/env python3
"""Calibration CLI — grid search over 52 historical scenarios.

Usage:
    # Calibrate all scenarios (finds best params per domain)
    python -m calibration.run_calibration

    # Calibrate a specific domain only
    python -m calibration.run_calibration --domain political

    # Calibrate a single scenario
    python -m calibration.run_calibration --scenario italy_referendum_2016

    # Use a custom parameter grid
    python -m calibration.run_calibration --fine-grid

    # Show results only (no recalculation)
    python -m calibration.run_calibration --show-results

Cost: $0 (pure math, no LLM calls)
Runtime: ~30-60 seconds for all 52 scenarios × 720 param combos
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration.historical_scenario import GroundTruth, PollingDataPoint
from calibration.parameter_tuner import (
    CalibrationResult,
    find_best_params,
    generate_parameter_combinations,
    save_calibrated_params,
)

# Extended grid including synthetic sim params
# ~972 combinations — runs in ~1 min per scenario at 100 agents
DEFAULT_GRID = {
    "anchor_weight": [0.05, 0.12, 0.20],
    "social_weight": [0.08, 0.15, 0.25],
    "event_weight": [0.04, 0.08, 0.12],
    "herd_weight": [0.03, 0.06],
    "herd_threshold": [0.15, 0.25],
    "direct_shift_weight": [0.3, 0.5, 0.7],
    "anchor_drift_rate": [0.12, 0.25, 0.40],
}
from calibration.trajectory_comparator import compute_calibration_score
from calibration.synthetic_sim import run_synthetic_simulation

logger = logging.getLogger(__name__)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
OUTPUT_DIR = Path(__file__).parent / "results"

# Finer grid for second-pass optimization
FINE_GRID = {
    "anchor_weight": [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.22],
    "social_weight": [0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30],
    "event_weight": [0.02, 0.04, 0.06, 0.08, 0.10, 0.14],
    "herd_weight": [0.01, 0.03, 0.05, 0.07, 0.10],
    "herd_threshold": [0.10, 0.15, 0.20, 0.25, 0.35],
    "direct_shift_weight": [0.2, 0.35, 0.5, 0.65, 0.8],
    "anchor_drift_rate": [0.08, 0.15, 0.22, 0.30, 0.40],
}


def load_scenario(path: Path) -> GroundTruth:
    """Load a scenario JSON as GroundTruth."""
    with open(path) as f:
        data = json.load(f)

    polling = [
        PollingDataPoint(**p) for p in data.get("polling_trajectory", [])
    ]

    return GroundTruth(
        scenario_name=data["scenario_name"],
        description=data.get("description", ""),
        final_outcome_pro_pct=data["final_outcome_pro_pct"],
        final_outcome_against_pct=data["final_outcome_against_pct"],
        final_turnout_pct=data.get("final_turnout_pct", 0),
        polling_trajectory=polling,
        key_events=data.get("key_events", []),
        calibration_notes=data.get("calibration_notes", ""),
    )


def load_all_scenarios(
    domain_filter: str = None,
    scenario_filter: str = None,
) -> list[tuple[str, str, GroundTruth]]:
    """Load all scenario files, optionally filtered.

    Returns list of (filename, domain, GroundTruth).
    """
    scenarios = []
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        if scenario_filter and scenario_filter not in path.stem:
            continue

        with open(path) as f:
            raw = json.load(f)

        domain = raw.get("domain", "unknown")

        if domain_filter and domain_filter not in domain:
            continue

        gt = load_scenario(path)
        scenarios.append((path.stem, domain, gt))

    return scenarios


def calibrate_scenario(
    ground_truth: GroundTruth,
    grid: dict = None,
    n_agents: int = 100,
) -> list[CalibrationResult]:
    """Run grid search for a single scenario.

    Returns all results sorted by composite score.
    """
    combos = generate_parameter_combinations(grid or DEFAULT_GRID)
    results = []

    for params in combos:
        sim_pro_pct, sim_positions = run_synthetic_simulation(
            ground_truth, params, n_agents=n_agents
        )
        metrics = compute_calibration_score(
            sim_pro_pct, sim_positions, ground_truth
        )
        results.append(CalibrationResult(params, metrics, sim_pro_pct))

    results.sort(key=lambda r: r.composite_score)
    return results


def aggregate_domain_params(
    domain_results: dict[str, CalibrationResult],
) -> dict:
    """Average the best params across all scenarios in a domain.

    Uses inverse-composite-score weighting: scenarios where we fit better
    contribute more to the average.
    """
    if not domain_results:
        return {}

    param_keys = list(next(iter(domain_results.values())).params.keys())
    weighted_sums = {k: 0.0 for k in param_keys}
    total_weight = 0.0

    for scenario_name, result in domain_results.items():
        # Weight: better fitting scenarios count more
        weight = 1.0 / max(0.01, result.composite_score)
        total_weight += weight
        for k in param_keys:
            weighted_sums[k] += result.params[k] * weight

    return {
        k: round(weighted_sums[k] / total_weight, 4)
        for k in param_keys
    }


def print_scenario_result(name: str, domain: str, result: CalibrationResult, gt: GroundTruth):
    """Pretty-print a single scenario result."""
    m = result.metrics
    print(f"  {name:<45} {domain:<25} "
          f"err={m['outcome_error_pct']:5.1f}%  "
          f"MAE={m['position_mae'] or 0:.4f}  "
          f"DTW={m['trajectory_dtw'] or 0:.4f}  "
          f"score={m['composite_score']:5.2f}  "
          f"sim={result.simulated_pro_pct:5.1f}% vs real={gt.final_outcome_pro_pct:5.1f}%")


def run_calibration(
    domain_filter: str = None,
    scenario_filter: str = None,
    use_fine_grid: bool = False,
    n_agents: int = 100,
):
    """Main calibration pipeline."""
    grid = FINE_GRID if use_fine_grid else DEFAULT_GRID
    combos = generate_parameter_combinations(grid)
    print(f"\n{'='*120}")
    print(f"CALIBRATION RUN — {len(combos)} parameter combinations per scenario")
    print(f"Grid: {'FINE' if use_fine_grid else 'DEFAULT'} | Agents: {n_agents}")
    print(f"{'='*120}\n")

    scenarios = load_all_scenarios(domain_filter, scenario_filter)
    if not scenarios:
        print("No scenarios found!")
        return

    print(f"Loaded {len(scenarios)} scenarios\n")

    # Per-domain tracking
    domain_best: dict[str, dict[str, CalibrationResult]] = defaultdict(dict)
    all_results: list[tuple[str, str, CalibrationResult, GroundTruth]] = []

    t0 = time.time()

    for i, (name, domain, gt) in enumerate(scenarios):
        t1 = time.time()
        results = calibrate_scenario(gt, grid, n_agents)
        best = find_best_params(results)
        elapsed = time.time() - t1

        if best:
            # Normalize domain to primary category
            primary_domain = domain.split("/")[0].strip()
            domain_best[primary_domain][name] = best
            all_results.append((name, domain, best, gt))
            print_scenario_result(name, domain, best, gt)
            if elapsed > 1.0:
                print(f"    ({elapsed:.1f}s)")

    total_time = time.time() - t0
    print(f"\n{'─'*120}")
    print(f"Total: {len(scenarios)} scenarios in {total_time:.1f}s "
          f"({total_time/len(scenarios):.2f}s/scenario)")

    # Aggregate per domain
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*120}")
    print("CALIBRATED PARAMETERS PER DOMAIN")
    print(f"{'='*120}\n")

    domain_params = {}
    for domain, scenario_results in sorted(domain_best.items()):
        avg_params = aggregate_domain_params(scenario_results)
        domain_params[domain] = avg_params

        # Compute average metrics
        avg_score = sum(r.composite_score for r in scenario_results.values()) / len(scenario_results)
        avg_error = sum(r.metrics["outcome_error_pct"] for r in scenario_results.values()) / len(scenario_results)

        print(f"  {domain:<20} ({len(scenario_results)} scenarios, avg_error={avg_error:.1f}%, avg_score={avg_score:.2f})")
        for k, v in avg_params.items():
            print(f"    {k:<20} = {v:.4f}")
        print()

        # Save per-domain params
        output_path = OUTPUT_DIR / f"calibrated_params_{domain}.json"
        save_calibrated_params(
            avg_params,
            {"avg_composite_score": round(avg_score, 4),
             "avg_outcome_error": round(avg_error, 2),
             "n_scenarios": len(scenario_results)},
            str(output_path),
            domain=domain,
        )
        print(f"    → Saved: {output_path}")

    # Save master file with all domains
    master_output = {
        "calibration_date": time.strftime("%Y-%m-%d %H:%M"),
        "n_scenarios": len(scenarios),
        "grid_type": "fine" if use_fine_grid else "default",
        "n_combinations": len(combos),
        "n_agents": n_agents,
        "domains": {},
    }
    for domain, params in domain_params.items():
        results_for_domain = domain_best[domain]
        avg_score = sum(r.composite_score for r in results_for_domain.values()) / len(results_for_domain)
        master_output["domains"][domain] = {
            "calibrated_params": params,
            "n_scenarios": len(results_for_domain),
            "avg_composite_score": round(avg_score, 4),
            "scenarios": {
                name: {
                    "outcome_error": r.metrics["outcome_error_pct"],
                    "composite_score": r.composite_score,
                    "best_params": r.params,
                }
                for name, r in results_for_domain.items()
            },
        }

    master_path = OUTPUT_DIR / "calibration_master.json"
    with open(master_path, "w") as f:
        json.dump(master_output, f, indent=2)
    print(f"\n  → Master file: {master_path}")

    # Summary
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    all_errors = [r.metrics["outcome_error_pct"] for _, _, r, _ in all_results]
    all_scores = [r.composite_score for _, _, r, _ in all_results]
    print(f"  Scenarios: {len(all_results)}")
    print(f"  Domains:   {len(domain_params)}")
    print(f"  Avg outcome error: {sum(all_errors)/len(all_errors):.1f}%")
    print(f"  Avg composite score: {sum(all_scores)/len(all_scores):.2f}")
    print(f"  Best:  {min(all_scores):.2f} ({all_results[all_scores.index(min(all_scores))][0]})")
    print(f"  Worst: {max(all_scores):.2f} ({all_results[all_scores.index(max(all_scores))][0]})")

    # Show how to use results
    print(f"\n{'='*120}")
    print("USAGE")
    print(f"{'='*120}")
    print("""
  I parametri calibrati vengono caricati automaticamente dal SimulationEngine.
  Basta copiare il file del dominio nella directory di output della simulazione:

    cp calibration/results/calibrated_params_political.json outputs/calibrated_params.json

  Oppure specificare seed_data_path nel config YAML per auto-discovery.

  Per calibrazione fine su un singolo scenario:
    python -m calibration.run_calibration --scenario italy_referendum_2016 --fine-grid
""")


def show_results():
    """Show existing calibration results without recalculating."""
    results_dir = OUTPUT_DIR
    if not results_dir.exists():
        print("No results found. Run calibration first.")
        return

    master_path = results_dir / "calibration_master.json"
    if not master_path.exists():
        print("No master file found. Run calibration first.")
        return

    with open(master_path) as f:
        data = json.load(f)

    print(f"\nCalibration from {data['calibration_date']}")
    print(f"Grid: {data['grid_type']}, {data['n_combinations']} combos, {data['n_agents']} agents")
    print(f"Scenarios: {data['n_scenarios']}\n")

    for domain, info in data["domains"].items():
        print(f"  {domain} ({info['n_scenarios']} scenarios, avg_score={info['avg_composite_score']:.2f})")
        for k, v in info["calibrated_params"].items():
            print(f"    {k:<20} = {v}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Calibrate OpinionDynamics parameters")
    parser.add_argument("--domain", help="Filter by domain (e.g., political, corporate)")
    parser.add_argument("--scenario", help="Filter by scenario name")
    parser.add_argument("--fine-grid", action="store_true", help="Use finer parameter grid")
    parser.add_argument("--agents", type=int, default=100, help="Number of synthetic agents")
    parser.add_argument("--show-results", action="store_true", help="Show existing results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.show_results:
        show_results()
    else:
        run_calibration(
            domain_filter=args.domain,
            scenario_filter=args.scenario,
            use_fine_grid=args.fine_grid,
            n_agents=args.agents,
        )


if __name__ == "__main__":
    main()
