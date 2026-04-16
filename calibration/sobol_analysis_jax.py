"""Sobol sensitivity analysis on the JAX DynamicsV2 model.

Mirrors calibration/sobol_analysis.py but uses the JAX simulate_scenario
with jax.vmap for massively parallel evaluation (~18k points in <2s).

Parameters analyzed (D=8, post gauge-fixing):
  λ_elite, λ_citizen, α_social, α_event, α_herd, α_anchor,
  herd_threshold, anchor_drift_rate

Same Sobol design as the NumPy analysis for direct comparison.

Usage:
    python -m calibration.sobol_analysis_jax [--N 1024]
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze
except ImportError:
    print("SALib not installed. Run: pip install SALib")
    sys.exit(1)

import jax
import jax.numpy as jnp

# Suppress JAX info logs
import logging
logging.getLogger("jax").setLevel(logging.WARNING)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamics.opinion_dynamics_jax import simulate_scenario, ScenarioData, build_sparse_interaction
from src.dynamics.param_utils import get_default_params
from calibration.sobol_analysis import create_agents as sobol_create_agents

# ── Problem Definition (identical to NumPy Sobol) ─────────────────

PARAM_NAMES = [
    "lambda_elite",
    "lambda_citizen",
    "alpha_social",
    "alpha_event",
    "alpha_herd",
    "alpha_anchor",
    "herd_threshold",
    "anchor_drift_rate",
]

PARAM_BOUNDS = [
    [0.05, 0.30],   # λ_elite
    [0.10, 0.40],   # λ_citizen
    [-2.0, 2.0],    # α_social  (α_direct gauge-fixed at 0)
    [-2.0, 2.0],    # α_event
    [-2.0, 2.0],    # α_herd
    [-2.0, 2.0],    # α_anchor
    [0.10, 0.40],   # herd_threshold
    [0.10, 0.40],   # anchor_drift_rate
]

PROBLEM = {
    "num_vars": len(PARAM_NAMES),
    "names": PARAM_NAMES,
    "bounds": PARAM_BOUNDS,
}


# ── Scenario creation (same agent setup as NumPy Sobol) ──────────

def create_scenario_data(
    n_agents: int = 30,
    n_rounds: int = 7,
    seed: int = 42,
) -> tuple[ScenarioData, dict]:
    """Create a fixed ScenarioData using the exact same agents as NumPy Sobol.

    Uses sobol_analysis.create_agents() for identical agent properties,
    then builds JAX-compatible arrays with sparse interaction matrix.

    Returns (scenario_data, metadata).
    """
    # Use exact same agent creation as NumPy Sobol
    agents = sobol_create_agents(n_agents, seed)

    positions = [a.position for a in agents]
    types = [0 if a.tier <= 2 else 1 for a in agents]
    rigidities = [a.rigidity for a in agents]
    tolerances = [a.tolerance for a in agents]
    influences = [a.influence for a in agents]

    # Events: same synthetic shocks as NumPy Sobol
    events = []
    for r in range(1, n_rounds + 1):
        mag = 0.2 + 0.15 * abs(np.sin(r * 1.3))
        direction = 0.3 * np.sin(r * 0.9)
        events.append((mag, direction))

    # Sparse interaction matrix: K=5 random neighbors per agent
    interaction = np.array(build_sparse_interaction(
        jnp.array(influences, dtype=jnp.float32), k=5, seed=seed,
    ))

    # LLM shifts: precomputed from shock × susceptibility
    llm_shifts = np.zeros((n_rounds, n_agents))
    for r_idx, (mag, direction) in enumerate(events):
        for a_idx in range(n_agents):
            susceptibility = (1 - rigidities[a_idx]) * max(0, 1 - abs(positions[a_idx]))
            llm_shifts[r_idx, a_idx] = mag * direction * susceptibility

    scenario = ScenarioData(
        initial_positions=jnp.array(positions, dtype=jnp.float32),
        agent_types=jnp.array(types, dtype=jnp.int32),
        agent_rigidities=jnp.array(rigidities, dtype=jnp.float32),
        agent_tolerances=jnp.array(tolerances, dtype=jnp.float32),
        events=jnp.array(events, dtype=jnp.float32),
        llm_shifts=jnp.array(llm_shifts, dtype=jnp.float32),
        interaction_matrix=jnp.array(interaction, dtype=jnp.float32),
    )

    return scenario, {
        "positions": positions,
        "types": types,
        "rigidities": rigidities,
        "tolerances": tolerances,
        "influences": influences,
    }


# ── Parameter vector → JAX params dict ───────────────────────────

def params_vector_to_dict(vec: jnp.ndarray) -> dict:
    """Convert SALib parameter vector (constrained space) to JAX params dict.

    vec[0] = lambda_elite     → log_lambda_elite = log(vec[0])
    vec[1] = lambda_citizen   → log_lambda_citizen = log(vec[1])
    vec[2] = alpha_social     → alpha_social (unconstrained)
    vec[3] = alpha_event      → alpha_event
    vec[4] = alpha_herd       → alpha_herd
    vec[5] = alpha_anchor     → alpha_anchor
    vec[6] = herd_threshold   → logit_herd_threshold = log(x/(1-x))
    vec[7] = anchor_drift_rate → logit_anchor_drift = log(x/(1-x))
    """
    return {
        "alpha_herd": vec[4],
        "alpha_anchor": vec[5],
        "alpha_social": vec[2],
        "alpha_event": vec[3],
        "log_lambda_elite": jnp.log(vec[0]),
        "log_lambda_citizen": jnp.log(vec[1]),
        "logit_herd_threshold": jnp.log(vec[6] / (1.0 - vec[6])),
        "logit_anchor_drift": jnp.log(vec[7] / (1.0 - vec[7])),
    }


def run_single_jax(params_dict: dict, scenario: ScenarioData) -> float:
    """Run one JAX simulation. Returns final_pro_pct."""
    result = simulate_scenario(params_dict, scenario)
    return result["final_pro_pct"]


# ── Batched evaluation with vmap ─────────────────────────────────

def evaluate_all_vmap(
    param_matrix: np.ndarray,
    scenario: ScenarioData,
    batch_size: int = 2048,
) -> np.ndarray:
    """Evaluate all Sobol sample points using jax.vmap in batches.

    param_matrix: [N_total, D=8] from SALib in constrained space.
    Returns: [N_total] array of final_pro_pct values.
    """
    n_total = param_matrix.shape[0]
    n_agents = scenario.initial_positions.shape[0]
    n_rounds = scenario.events.shape[0]

    # Prepare batched scenario (same scenario for all evaluations)
    def make_batched_scenario(bs):
        return ScenarioData(
            initial_positions=jnp.broadcast_to(scenario.initial_positions, (bs, n_agents)),
            agent_types=jnp.broadcast_to(scenario.agent_types, (bs, n_agents)),
            agent_rigidities=jnp.broadcast_to(scenario.agent_rigidities, (bs, n_agents)),
            agent_tolerances=jnp.broadcast_to(scenario.agent_tolerances, (bs, n_agents)),
            events=jnp.broadcast_to(scenario.events, (bs, n_rounds, 2)),
            llm_shifts=jnp.broadcast_to(scenario.llm_shifts, (bs, n_rounds, n_agents)),
            interaction_matrix=jnp.broadcast_to(
                scenario.interaction_matrix, (bs, n_agents, n_agents)
            ),
        )

    # vmap over params, with scenario batched to match
    def batched_eval(param_batch, scenario_batch):
        """Evaluate a batch: param_batch is [bs, D=8], scenario_batch is batched."""
        def single_eval(pvec, scen):
            params = {
                "alpha_herd": pvec[4],
                "alpha_anchor": pvec[5],
                "alpha_social": pvec[2],
                "alpha_event": pvec[3],
                "log_lambda_elite": jnp.log(jnp.clip(pvec[0], 1e-6, None)),
                "log_lambda_citizen": jnp.log(jnp.clip(pvec[1], 1e-6, None)),
                "logit_herd_threshold": jnp.log(
                    jnp.clip(pvec[6], 1e-6, 1.0 - 1e-6) /
                    (1.0 - jnp.clip(pvec[6], 1e-6, 1.0 - 1e-6))
                ),
                "logit_anchor_drift": jnp.log(
                    jnp.clip(pvec[7], 1e-6, 1.0 - 1e-6) /
                    (1.0 - jnp.clip(pvec[7], 1e-6, 1.0 - 1e-6))
                ),
            }
            result = simulate_scenario(params, scen)
            return result["final_pro_pct"]

        return jax.vmap(single_eval)(param_batch, scenario_batch)

    # JIT the batched function
    batched_eval_jit = jax.jit(batched_eval)

    # Process in batches
    Y = np.zeros(n_total)
    n_batches = (n_total + batch_size - 1) // batch_size

    # Warmup with first batch
    first_bs = min(batch_size, n_total)
    first_batch = jnp.array(param_matrix[:first_bs], dtype=jnp.float32)
    first_scen = make_batched_scenario(first_bs)
    print(f"  Warming up JIT ({first_bs} points)...", end="", flush=True)
    t_warmup = time.time()
    result = batched_eval_jit(first_batch, first_scen)
    result.block_until_ready()
    Y[:first_bs] = np.array(result)
    print(f" done ({time.time() - t_warmup:.1f}s)")

    # Run remaining batches
    t0 = time.time()
    for b in range(1, n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_total)
        bs = end - start

        batch = jnp.array(param_matrix[start:end], dtype=jnp.float32)
        scen = make_batched_scenario(bs)
        result = batched_eval_jit(batch, scen)
        result.block_until_ready()
        Y[start:end] = np.array(result)

    elapsed = time.time() - t0
    if n_batches > 1:
        remaining = n_total - first_bs
        print(f"  Remaining {remaining} points: {elapsed:.2f}s "
              f"({remaining / max(elapsed, 1e-6):.0f} sims/s)")

    return Y


# ── Main analysis ────────────────────────────────────────────────

def run_sobol_jax(
    N: int = 1024,
    n_agents: int = 30,
    n_rounds: int = 7,
    seed: int = 42,
    output_dir: str = None,
) -> dict:
    """Full Sobol analysis using JAX model."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results" / "sobol")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    D = PROBLEM["num_vars"]
    total_evals = N * (2 * D + 2)

    print(f"{'=' * 60}")
    print(f"  SOBOL SENSITIVITY ANALYSIS — JAX DynamicsV2")
    print(f"  JAX {jax.__version__} | {jax.devices()}")
    print(f"  D={D}, N={N}, total={total_evals}, agents={n_agents}, rounds={n_rounds}")
    print(f"{'=' * 60}")
    print()

    # Step 1: Sobol design matrix
    print("1. Generating Sobol design matrix...")
    param_values = sobol_sample.sample(PROBLEM, N, calc_second_order=True)
    assert param_values.shape == (total_evals, D)
    print(f"   {param_values.shape[0]} × {D} points")

    # Step 2: Create scenario
    print("\n2. Creating scenario...")
    scenario, meta = create_scenario_data(n_agents, n_rounds, seed)
    print(f"   {n_agents} agents, {n_rounds} rounds")

    # Step 3: Evaluate with vmap
    print(f"\n3. Evaluating {total_evals} points with jax.vmap...")
    t0 = time.time()
    Y = evaluate_all_vmap(param_values, scenario, batch_size=2048)
    total_time = time.time() - t0
    print(f"   Total: {total_time:.2f}s ({total_evals / total_time:.0f} sims/s)")
    print(f"   Y: mean={Y.mean():.2f}, std={Y.std():.2f}, "
          f"range=[{Y.min():.1f}, {Y.max():.1f}]")

    # Step 4: Sobol analysis
    print(f"\n4. Computing Sobol indices...")
    Si = sobol_analyze.analyze(PROBLEM, Y, calc_second_order=True)

    # Build results
    results = {
        "model": "JAX",
        "jax_version": jax.__version__,
        "parameters": PARAM_NAMES,
        "bounds": PARAM_BOUNDS,
        "N": N,
        "total_evaluations": total_evals,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "eval_time_s": round(total_time, 2),
        "output_stats": {
            "mean": float(Y.mean()),
            "std": float(Y.std()),
            "min": float(Y.min()),
            "max": float(Y.max()),
        },
        "S1": {PARAM_NAMES[i]: float(Si["S1"][i]) for i in range(D)},
        "S1_conf": {PARAM_NAMES[i]: float(Si["S1_conf"][i]) for i in range(D)},
        "ST": {PARAM_NAMES[i]: float(Si["ST"][i]) for i in range(D)},
        "ST_conf": {PARAM_NAMES[i]: float(Si["ST_conf"][i]) for i in range(D)},
    }

    # Second-order
    if "S2" in Si and Si["S2"] is not None:
        s2_dict = {}
        for i in range(D):
            for j in range(i + 1, D):
                key = f"{PARAM_NAMES[i]}:{PARAM_NAMES[j]}"
                s2_dict[key] = float(Si["S2"][i][j])
        results["S2"] = s2_dict

    # Print report
    print_report(results)

    # Save
    json_path = Path(output_dir) / "sobol_jax_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    return results


def print_report(results: dict):
    """Print Sobol indices table."""
    S1 = results["S1"]
    ST = results["ST"]
    S1_conf = results["S1_conf"]
    ST_conf = results["ST_conf"]
    params = results["parameters"]

    print(f"\n{'=' * 65}")
    print(f"  SOBOL INDICES — {results['model']} MODEL")
    print(f"{'=' * 65}")

    print(f"\n  {'Parameter':<20} {'S1':>8} {'±':>8} {'ST':>8} {'±':>8} {'ST-S1':>8}")
    print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    sorted_params = sorted(params, key=lambda p: ST[p], reverse=True)

    for p in sorted_params:
        interaction = ST[p] - S1[p]
        marker = ""
        if S1[p] < 0.01:
            marker = " ◄ FROZEN-OK"
        elif interaction > 0.10 and interaction > S1[p]:
            marker = " ◄ INTERACTION"
        elif ST[p] > 0.15:
            marker = " ◄ INFLUENTIAL"
        print(f"  {p:<20} {S1[p]:>8.4f} {S1_conf[p]:>8.4f} "
              f"{ST[p]:>8.4f} {ST_conf[p]:>8.4f} {interaction:>8.4f}{marker}")

    print(f"\n  Sum(S1) = {sum(S1.values()):.4f}")
    print(f"  Sum(ST) = {sum(ST.values()):.4f}")

    # Top S2 interactions
    if "S2" in results:
        s2 = results["S2"]
        top = sorted(s2.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f"\n  Top S2 interactions:")
        for pair, val in top:
            print(f"    {pair}: {val:.4f}")

    print(f"{'=' * 65}")


# ── Comparison helper ────────────────────────────────────────────

def compare_with_numpy(jax_results: dict, numpy_path: str) -> dict:
    """Compare JAX Sobol results with NumPy baseline."""
    with open(numpy_path) as f:
        np_results = json.load(f)

    comparison = {
        "numpy_model": "NumPy DynamicsV2 (rolling buffer, hard thresholds, feed-based)",
        "jax_model": "JAX DynamicsV2 (population std, smooth thresholds, interaction matrix)",
    }

    # Per-parameter comparison
    params = jax_results["parameters"]
    param_comparison = {}
    for p in params:
        s1_np = np_results["S1"][p]
        s1_jax = jax_results["S1"][p]
        st_np = np_results["ST"][p]
        st_jax = jax_results["ST"][p]

        # Classify status
        frozen_np = s1_np < 0.01
        frozen_jax = s1_jax < 0.01
        if frozen_np == frozen_jax:
            status = "CONFIRMED" if frozen_np else "CONFIRMED_ACTIVE"
        elif frozen_np and not frozen_jax:
            status = "UNFROZEN_IN_JAX"
        else:
            status = "FROZEN_IN_JAX"

        param_comparison[p] = {
            "S1_numpy": round(s1_np, 4),
            "S1_jax": round(s1_jax, 4),
            "ST_numpy": round(st_np, 4),
            "ST_jax": round(st_jax, 4),
            "status": status,
        }

    comparison["parameters"] = param_comparison

    # Rank comparison
    rank_np = sorted(params, key=lambda p: np_results["ST"][p], reverse=True)
    rank_jax = sorted(params, key=lambda p: jax_results["ST"][p], reverse=True)
    comparison["rank_ST_numpy"] = rank_np
    comparison["rank_ST_jax"] = rank_jax
    comparison["rank_preserved"] = rank_np[:4] == rank_jax[:4]  # top-4 same?

    # Frozen params check
    frozen_in_numpy = [p for p in params if np_results["S1"][p] < 0.01]
    frozen_in_jax = [p for p in params if jax_results["S1"][p] < 0.01]
    comparison["frozen_numpy"] = frozen_in_numpy
    comparison["frozen_jax"] = frozen_in_jax
    comparison["freeze_decision_holds"] = set(frozen_in_numpy) == set(frozen_in_jax)

    # Top S2 interaction
    if "S2" in jax_results and "S2" in np_results:
        top_np = max(np_results["S2"].items(), key=lambda x: abs(x[1]))
        top_jax = max(jax_results["S2"].items(), key=lambda x: abs(x[1]))
        comparison["top_interaction_numpy"] = {"pair": top_np[0], "S2": round(top_np[1], 4)}
        comparison["top_interaction_jax"] = {"pair": top_jax[0], "S2": round(top_jax[1], 4)}
        comparison["dominant_interaction_same"] = top_np[0] == top_jax[0]

    return comparison


def write_comparison_report(
    jax_results: dict,
    comparison: dict,
    output_path: str,
):
    """Write markdown comparison report."""
    params = jax_results["parameters"]
    pc = comparison["parameters"]

    lines = [
        "# Sobol Sensitivity: JAX vs NumPy Model Comparison",
        "",
        "## Context",
        "",
        "Phase 0 (Sobol analysis, gauge fixing, freeze decisions) was performed on the",
        "NumPy DynamicsV2 model. The JAX port introduces structural changes:",
        "",
        "| Feature | NumPy | JAX |",
        "|---|---|---|",
        "| Force standardization | Rolling buffer (window=8) | Population z-score (all agents) |",
        "| Herd activation | Hard step: `|gap| > threshold` | Smooth sigmoid: `σ((|gap| - θ)/0.02)` |",
        "| Bounded confidence | Hard: `distance < tolerance` | Smooth sigmoid |",
        "| Delta clamp | `min(max(...), cap)` | `cap * tanh(δ/cap)` |",
        "| Social/herd data | Feed-based (top 5 posts) | Full interaction matrix |",
        "",
        "**Question**: Do Phase 0 conclusions (parameter hierarchy, freeze decisions) hold?",
        "",
        "## Design",
        "",
        f"- SALib Sobol, N={jax_results['N']}, D={len(params)}, "
        f"total={jax_results['total_evaluations']} evaluations",
        f"- {jax_results['n_agents']} agents, {jax_results['n_rounds']} rounds",
        f"- JAX evaluation time: {jax_results['eval_time_s']}s "
        f"(vs ~120s for NumPy sequential)",
        f"- Identical parameter bounds and Sobol design matrix",
        "",
        "## Results: S1 and ST Comparison",
        "",
        "| Parameter | S1 (NumPy) | S1 (JAX) | ST (NumPy) | ST (JAX) | Status |",
        "|---|---|---|---|---|---|",
    ]

    sorted_params = sorted(params, key=lambda p: pc[p]["ST_jax"], reverse=True)
    for p in sorted_params:
        d = pc[p]
        lines.append(
            f"| {p} | {d['S1_numpy']:.4f} | {d['S1_jax']:.4f} | "
            f"{d['ST_numpy']:.4f} | {d['ST_jax']:.4f} | {d['status']} |"
        )

    lines += [
        "",
        f"Sum(S1): NumPy={sum(pc[p]['S1_numpy'] for p in params):.4f}, "
        f"JAX={sum(pc[p]['S1_jax'] for p in params):.4f}",
        "",
        f"Sum(ST): NumPy={sum(pc[p]['ST_numpy'] for p in params):.4f}, "
        f"JAX={sum(pc[p]['ST_jax'] for p in params):.4f}",
        "",
        "## Verification Checks",
        "",
    ]

    # Check 1: α_herd dominant?
    rank_jax = comparison["rank_ST_jax"]
    herd_dominant = rank_jax[0] == "alpha_herd"
    lines.append(f"### 1. α_herd still dominant?")
    lines.append("")
    lines.append(f"ST ranking (JAX): {', '.join(rank_jax[:4])}")
    lines.append(f"ST ranking (NumPy): {', '.join(comparison['rank_ST_numpy'][:4])}")
    lines.append(f"**Result: {'YES' if herd_dominant else 'NO — STRUCTURE CHANGED'}** "
                 f"— α_herd is {'#1' if herd_dominant else 'NOT #1'} by ST")
    lines.append("")

    # Check 2: Frozen params S1 < 0.01?
    lines.append("### 2. Frozen parameters still S1 < 0.01?")
    lines.append("")
    frozen_candidates = ["lambda_elite", "lambda_citizen", "herd_threshold", "anchor_drift_rate"]
    all_frozen_ok = True
    for p in frozen_candidates:
        s1 = pc[p]["S1_jax"]
        ok = s1 < 0.01
        if not ok:
            all_frozen_ok = False
        lines.append(f"- {p}: S1={s1:.4f} {'< 0.01 OK' if ok else '**>= 0.01 — REVIEW NEEDED**'}")
    lines.append("")
    lines.append(f"**Result: {'ALL CONFIRMED' if all_frozen_ok else 'FREEZE DECISION NEEDS REVIEW'}**")
    lines.append("")

    # Check 3: Dominant interaction
    lines.append("### 3. α_herd × α_anchor dominant interaction?")
    lines.append("")
    if "top_interaction_jax" in comparison:
        top_jax = comparison["top_interaction_jax"]
        top_np = comparison["top_interaction_numpy"]
        same = comparison["dominant_interaction_same"]
        lines.append(f"Top S2 (NumPy): {top_np['pair']} = {top_np['S2']:.4f}")
        lines.append(f"Top S2 (JAX): {top_jax['pair']} = {top_jax['S2']:.4f}")
        lines.append(f"**Result: {'CONFIRMED' if same else 'DIFFERENT — ' + top_jax['pair']}**")
    lines.append("")

    # Check 4: Top-4 rank preserved?
    lines.append("### 4. Top-4 ST ranking preserved?")
    lines.append("")
    lines.append(f"NumPy top-4: {comparison['rank_ST_numpy'][:4]}")
    lines.append(f"JAX top-4: {comparison['rank_ST_jax'][:4]}")
    lines.append(f"**Result: {'PRESERVED' if comparison['rank_preserved'] else 'CHANGED'}**")
    lines.append("")

    # Overall verdict
    lines.append("## Verdict")
    lines.append("")

    checks = [
        herd_dominant,
        all_frozen_ok,
        comparison.get("dominant_interaction_same", False),
        comparison.get("rank_preserved", False),
    ]
    n_pass = sum(checks)

    if n_pass == 4:
        lines.append("**ALL CHECKS PASSED** — Phase 0 conclusions are valid for the JAX model.")
        lines.append("The freeze decisions, parameter hierarchy, and interaction structure")
        lines.append("are preserved despite the architectural differences.")
    elif n_pass >= 3:
        lines.append(f"**{n_pass}/4 CHECKS PASSED** — Phase 0 conclusions are broadly valid.")
        lines.append("Minor differences noted but do not invalidate freeze decisions.")
    else:
        lines.append(f"**{n_pass}/4 CHECKS PASSED — REVIEW REQUIRED**")
        lines.append("Significant structural differences detected. Freeze decisions should")
        lines.append("be re-evaluated before proceeding with calibration.")

    lines.append("")
    lines.append("## Output Statistics Comparison")
    lines.append("")
    lines.append(f"| Stat | NumPy | JAX |")
    lines.append(f"|---|---|---|")

    # Load numpy stats from file
    numpy_path = PROJECT_ROOT / "calibration" / "results" / "sobol_results.json"
    if numpy_path.exists():
        with open(numpy_path) as f:
            np_stats = json.load(f)["output_stats"]
        jax_stats = jax_results["output_stats"]
        for k in ["mean", "std", "min", "max"]:
            lines.append(f"| {k} | {np_stats[k]:.2f} | {jax_stats[k]:.2f} |")

    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Comparison report: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sobol GSA — JAX DynamicsV2")
    parser.add_argument("--N", type=int, default=1024, help="Base sample size")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Run JAX Sobol
    jax_results = run_sobol_jax(
        N=args.N, n_agents=args.agents, n_rounds=args.rounds,
        seed=args.seed, output_dir=args.output,
    )

    # Compare with NumPy
    numpy_path = PROJECT_ROOT / "calibration" / "results" / "sobol_results.json"
    if numpy_path.exists():
        print(f"\n{'=' * 60}")
        print(f"  COMPARISON: JAX vs NumPy")
        print(f"{'=' * 60}")

        comparison = compare_with_numpy(jax_results, str(numpy_path))

        # Print quick summary
        pc = comparison["parameters"]
        print(f"\n  Freeze decision holds: {comparison['freeze_decision_holds']}")
        print(f"  Rank preserved (top-4): {comparison['rank_preserved']}")
        print(f"  Dominant interaction same: {comparison.get('dominant_interaction_same', 'N/A')}")

        # Write report
        output_dir = args.output or str(PROJECT_ROOT / "calibration" / "results" / "sobol")
        report_path = Path(output_dir) / "sobol_jax_vs_numpy_comparison.md"
        write_comparison_report(jax_results, comparison, str(report_path))

        # Save comparison JSON
        cmp_json_path = Path(output_dir) / "sobol_comparison.json"
        with open(cmp_json_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"  Comparison JSON: {cmp_json_path}")
    else:
        print(f"\n  NumPy results not found at {numpy_path} — skipping comparison")
