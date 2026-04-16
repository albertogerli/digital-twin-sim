"""Confronto v2 (no regime) vs v3 (regime switching) sui financial scenarios.

For each financial scenario in the empirical dataset:
1. Load v2 calibrated posterior (global means)
2. Run v2 prediction (standard simulator)
3. Run v3 prediction (regime switching with default crisis params)
4. Compare errors and regime activation patterns

Usage:
    cd digital-twin-sim
    source .venv_cal/bin/activate
    python -m calibration.run_regime_comparison
"""

import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dynamics.opinion_dynamics_jax import simulate_scenario
from src.dynamics.regime_switching import RegimeSwitchingSimulator
from src.dynamics.param_utils import get_default_frozen_params
from src.observation.observation_model import (
    build_scenario_data_from_json,
    load_scenario_observations,
)

BASE = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = BASE / "calibration" / "empirical" / "scenarios"
POSTERIORS_PATH = (
    BASE / "calibration" / "results" / "hierarchical_calibration"
    / "v2_discrepancy" / "posteriors_v2.json"
)
VALIDATION_PATH = (
    BASE / "calibration" / "results" / "hierarchical_calibration"
    / "v2_discrepancy" / "validation_results_v2.json"
)


def load_v2_posterior():
    """Load calibrated posterior means."""
    with open(POSTERIORS_PATH) as f:
        post = json.load(f)
    mu = post["global"]["mu_global"]["mean"]
    params = {
        "alpha_herd": mu[0],
        "alpha_anchor": mu[1],
        "alpha_social": mu[2],
        "alpha_event": mu[3],
    }
    frozen = get_default_frozen_params()
    return {k: float(v) if hasattr(v, "item") else v
            for k, v in {**params, **frozen}.items()}


def load_validation_results():
    """Load per-scenario validation results from v2."""
    with open(VALIDATION_PATH) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("v2 vs v3 (Regime Switching) Comparison — Financial Scenarios")
    print("=" * 70)

    params = load_v2_posterior()
    v2_results = load_validation_results()

    # Build lookup for v2 validation results
    v2_lookup = {r["id"]: r for r in v2_results}

    # Find all financial scenarios
    scenario_files = sorted(SCENARIOS_DIR.glob("FIN-*.json"))
    # Also include corporate crisis scenarios
    scenario_files += sorted(SCENARIOS_DIR.glob("CORP-*.json"))

    sim_rs = RegimeSwitchingSimulator()

    rows = []

    for path in scenario_files:
        sid = path.stem
        if sid not in v2_lookup:
            continue

        v2_entry = v2_lookup[sid]
        gt = v2_entry["gt"]
        v2_error = v2_entry["error"]

        try:
            scenario_dict, obs = load_scenario_observations(str(path))
            scenario_data = build_scenario_data_from_json(scenario_dict)
        except Exception as e:
            print(f"  Error loading {sid}: {e}")
            continue

        covs = scenario_dict.get("covariates", {})
        inst_trust = covs.get("institutional_trust", 0.3)

        # v2: standard simulation
        result_v2 = simulate_scenario(params, scenario_data)
        pred_v2 = float(result_v2["final_pro_pct"])

        # v3: regime switching
        result_v3 = sim_rs.simulate(params, scenario_data, institutional_trust=inst_trust)
        pred_v3 = float(result_v3["final_pro_pct"])
        regime_probs = np.array(result_v3["regime_probs"])
        crisis_rounds = int(np.sum(np.array(result_v3["regime_sequence"])))
        max_rp = float(np.max(regime_probs))

        err_v2 = abs(pred_v2 - gt)
        err_v3 = abs(pred_v3 - gt)
        improvement = err_v2 - err_v3

        rows.append({
            "id": sid,
            "domain": v2_entry["domain"],
            "group": v2_entry["group"],
            "gt": gt,
            "pred_v2": pred_v2,
            "pred_v3": pred_v3,
            "err_v2": err_v2,
            "err_v3": err_v3,
            "improvement": improvement,
            "max_regime_prob": max_rp,
            "crisis_rounds": crisis_rounds,
            "inst_trust": inst_trust,
        })

    # Sort by v2 error (worst first)
    rows.sort(key=lambda x: x["err_v2"], reverse=True)

    # Print table
    print(f"\n{'Scenario':<45} {'GT':>5} {'v2':>6} {'v3':>6} "
          f"{'|Δv2|':>6} {'|Δv3|':>6} {'Δerr':>6} {'MaxRP':>6} {'CrR':>4}")
    print("-" * 130)

    total_v2 = 0.0
    total_v3 = 0.0
    improved = 0
    n = len(rows)

    for r in rows:
        total_v2 += r["err_v2"]
        total_v3 += r["err_v3"]
        if r["improvement"] > 0:
            improved += 1

        marker = "✓" if r["improvement"] > 1.0 else ("~" if r["improvement"] > -1.0 else "✗")

        print(
            f"{r['id'][:44]:<45} {r['gt']:5.1f} {r['pred_v2']:6.1f} {r['pred_v3']:6.1f} "
            f"{r['err_v2']:6.1f} {r['err_v3']:6.1f} {r['improvement']:+6.1f} "
            f"{r['max_regime_prob']:6.3f} {r['crisis_rounds']:4d} {marker}"
        )

    print("-" * 130)

    if n > 0:
        mae_v2 = total_v2 / n
        mae_v3 = total_v3 / n
        print(f"\n{'MAE v2:':<20} {mae_v2:.2f}pp")
        print(f"{'MAE v3:':<20} {mae_v3:.2f}pp")
        print(f"{'Improvement:':<20} {mae_v2 - mae_v3:+.2f}pp ({improved}/{n} scenarios improved)")
        print(f"{'Mean max regime_p:':<20} {np.mean([r['max_regime_prob'] for r in rows]):.3f}")

        # Financial-only subset
        fin_rows = [r for r in rows if r["domain"] == "financial"]
        if fin_rows:
            mae_v2_fin = np.mean([r["err_v2"] for r in fin_rows])
            mae_v3_fin = np.mean([r["err_v3"] for r in fin_rows])
            print(f"\nFinancial-only (N={len(fin_rows)}):")
            print(f"  MAE v2: {mae_v2_fin:.2f}pp → MAE v3: {mae_v3_fin:.2f}pp "
                  f"(Δ={mae_v2_fin - mae_v3_fin:+.2f}pp)")

    # Save results
    output_path = BASE / "calibration" / "results" / "hierarchical_calibration" / "v2_vs_v3_comparison.json"
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
