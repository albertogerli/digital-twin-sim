"""Global Sensitivity Analysis with Sobol indices for DynamicsV2.

Uses SALib to quantify how each calibratable parameter influences the
final pro-percentage output. Identifies influential vs. redundant parameters
and detects strong interaction effects (ST >> S1).

Parameters analyzed (D=8, post gauge-fixing):
  λ_elite, λ_citizen, α_social, α_event, α_herd, α_anchor,
  herd_threshold, anchor_drift_rate

Note: α_direct is gauge-fixed at 0.0 (reference level for softmax).
See calibration/fix_alpha_direct_test.py for the proof that this is
a lossless reparametrization (shift invariance of softmax).

Usage:
    python -m calibration.sobol_analysis [--N 1024] [--rounds 7] [--agents 30]
"""

import json
import logging
import random
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np

try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze
except ImportError:
    print("SALib not installed. Run: pip install SALib")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — skipping plot generation")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.simulation.opinion_dynamics_v2 import DynamicsV2, TIER_MAP

logger = logging.getLogger(__name__)

# ── Problem Definition ──────────────────────────────────────────────

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


# ── Synthetic Agent ─────────────────────────────────────────────────

class SobolAgent:
    """Lightweight agent for Sobol evaluation — no LLM."""
    __slots__ = ("id", "position", "original_position", "rigidity",
                 "tolerance", "influence", "tier")

    def __init__(self, id, position, tier, rigidity, tolerance, influence):
        self.id = id
        self.position = position
        self.original_position = position
        self.tier = tier
        self.rigidity = rigidity
        self.tolerance = tolerance
        self.influence = influence


class SobolPlatform:
    """Minimal platform stub — returns synthetic posts from agents."""

    def __init__(self, agents, rng):
        self._agents = agents
        self._rng = rng
        self.conn = None  # triggers get_top_posts fallback in DynamicsV2

    def get_top_posts(self, round_num, top_n=5, platform=None):
        """Return synthetic posts from random agents."""
        sample = self._rng.sample(
            self._agents, min(top_n, len(self._agents))
        )
        return [
            {
                "id": i + 1,
                "author_id": a.id,
                "likes": self._rng.randint(5, 100),
                "reposts": self._rng.randint(0, 30),
            }
            for i, a in enumerate(sample)
        ]


# ── Simulation Runner ───────────────────────────────────────────────

def create_agents(n_agents: int = 30, seed: int = 42) -> list[SobolAgent]:
    """Create 30 agents: 50% pro, 50% against, positions ~ U[-0.8, 0.8].

    5% elite (tier 1), 10% institutional (tier 2), 85% citizen (tier 3).
    """
    rng = random.Random(seed)
    agents = []
    n_pro = n_agents // 2
    n_against = n_agents - n_pro

    for i in range(n_agents):
        # Position: first half pro, second half against
        if i < n_pro:
            pos = rng.uniform(0.05, 0.8)
        else:
            pos = rng.uniform(-0.8, -0.05)

        # Tier assignment
        r = i / n_agents
        if r < 0.05:
            tier, influence = 1, rng.uniform(0.7, 1.0)
            rigidity = rng.uniform(0.4, 0.7)
        elif r < 0.15:
            tier, influence = 2, rng.uniform(0.4, 0.7)
            rigidity = rng.uniform(0.3, 0.5)
        else:
            tier, influence = 3, rng.uniform(0.1, 0.4)
            rigidity = rng.uniform(0.15, 0.40)

        agents.append(SobolAgent(
            id=f"agent_{i}",
            position=pos,
            tier=tier,
            rigidity=rigidity,
            tolerance=rng.uniform(0.3, 0.6),
            influence=influence,
        ))

    return agents


def run_single_evaluation(
    params_vector: np.ndarray,
    n_agents: int = 30,
    n_rounds: int = 7,
    seed: int = 42,
) -> float:
    """Run one synthetic simulation with given parameter vector.

    Returns final pro-percentage ∈ [0, 100].
    """
    # Unpack parameters (D=8, α_direct gauge-fixed at 0.0)
    lam_elite = params_vector[0]
    lam_citizen = params_vector[1]
    alpha = {
        "direct": 0.0,  # gauge-fixed reference level
        "social": params_vector[2],
        "event": params_vector[3],
        "herd": params_vector[4],
        "anchor": params_vector[5],
    }
    herd_threshold = params_vector[6]
    anchor_drift_rate = params_vector[7]

    # Institutional λ = midpoint of elite and citizen
    lam_institutional = (lam_elite + lam_citizen) / 2.0

    step_sizes = {
        "elite": lam_elite,
        "institutional": lam_institutional,
        "citizen": lam_citizen,
    }

    model = DynamicsV2(
        alpha=alpha,
        step_sizes=step_sizes,
        herd_threshold=herd_threshold,
        anchor_drift_rate=anchor_drift_rate,
    )

    agents = create_agents(n_agents, seed)
    rng = random.Random(seed + 100)
    platform = SobolPlatform(agents, rng)

    # Run rounds with synthetic events
    for r in range(1, n_rounds + 1):
        # Synthetic event: alternating mild shocks
        shock_mag = 0.2 + 0.15 * abs(np.sin(r * 1.3))
        shock_dir = 0.3 * np.sin(r * 0.9)  # oscillating direction

        event = {
            "round": r,
            "shock_magnitude": shock_mag,
            "shock_direction": shock_dir,
        }

        model.step(agents, platform, event)

    # Compute final pro-percentage
    pro = sum(1 for a in agents if a.position > 0.05)
    against = sum(1 for a in agents if a.position < -0.05)
    total_decided = pro + against
    if total_decided == 0:
        return 50.0
    return (pro / total_decided) * 100.0


# ── Sobol Analysis Pipeline ────────────────────────────────────────

def run_sobol_analysis(
    N: int = 1024,
    n_agents: int = 30,
    n_rounds: int = 7,
    seed: int = 42,
    output_dir: str = None,
) -> dict:
    """Full Sobol sensitivity analysis pipeline.

    Args:
        N: Base sample size. Total evaluations = N * (2D + 2).
        n_agents: Agents per simulation.
        n_rounds: Rounds per simulation.
        seed: Random seed for agent creation.
        output_dir: Directory for outputs (defaults to calibration/results/).

    Returns:
        Dict with S1, ST, S2, parameter names, and diagnostics.
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    D = PROBLEM["num_vars"]
    total_evals = N * (2 * D + 2)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Sobol Global Sensitivity Analysis — DynamicsV2 ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Parameters (D):     {D:>5}                        ║")
    print(f"║  Base samples (N):   {N:>5}                        ║")
    print(f"║  Total evaluations:  {total_evals:>5}                        ║")
    print(f"║  Agents per sim:     {n_agents:>5}                        ║")
    print(f"║  Rounds per sim:     {n_rounds:>5}                        ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # Step 1: Generate Sobol sample
    print("▸ Generating Sobol design matrix...")
    param_values = sobol_sample.sample(PROBLEM, N, calc_second_order=True)
    assert param_values.shape == (total_evals, D), \
        f"Expected ({total_evals}, {D}), got {param_values.shape}"
    print(f"  {param_values.shape[0]} sample points × {D} parameters")

    # Step 2: Evaluate model at all sample points
    print(f"\n▸ Running {total_evals} simulations...")
    Y = np.zeros(total_evals)
    t0 = time.time()
    report_every = max(1, total_evals // 20)

    for i in range(total_evals):
        Y[i] = run_single_evaluation(
            param_values[i], n_agents=n_agents, n_rounds=n_rounds, seed=seed,
        )
        if (i + 1) % report_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total_evals - i - 1) / rate
            print(f"  [{i+1:>5}/{total_evals}]  "
                  f"{rate:.0f} sims/s  "
                  f"ETA {eta:.0f}s  "
                  f"Y range [{Y[:i+1].min():.1f}, {Y[:i+1].max():.1f}]")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({total_evals/elapsed:.0f} sims/s)")
    print(f"  Output Y: mean={Y.mean():.2f}, std={Y.std():.2f}, "
          f"range=[{Y.min():.1f}, {Y.max():.1f}]")

    # Step 3: Sobol analysis
    print(f"\n▸ Computing Sobol indices...")
    Si = sobol_analyze.analyze(PROBLEM, Y, calc_second_order=True)

    # Step 4: Build results
    results = {
        "parameters": PARAM_NAMES,
        "bounds": PARAM_BOUNDS,
        "N": N,
        "total_evaluations": total_evals,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
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

    # Second-order indices (interaction matrix)
    if "S2" in Si and Si["S2"] is not None:
        s2_dict = {}
        for i in range(D):
            for j in range(i + 1, D):
                key = f"{PARAM_NAMES[i]}:{PARAM_NAMES[j]}"
                s2_dict[key] = float(Si["S2"][i][j])
        results["S2"] = s2_dict

    # Step 5: Print report
    print_report(results)

    # Step 6: Save JSON
    json_path = Path(output_dir) / "sobol_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n▸ Results saved to {json_path}")

    # Step 7: Plot
    if HAS_MPL:
        plot_path = Path(output_dir) / "sobol_indices.png"
        plot_sobol(results, str(plot_path))
        print(f"▸ Plot saved to {plot_path}")

    return results


# ── Report ──────────────────────────────────────────────────────────

def print_report(results: dict):
    """Print textual analysis of Sobol indices."""
    S1 = results["S1"]
    ST = results["ST"]
    S1_conf = results["S1_conf"]
    ST_conf = results["ST_conf"]
    params = results["parameters"]

    print("\n" + "=" * 65)
    print("  SOBOL SENSITIVITY ANALYSIS — RESULTS")
    print("=" * 65)

    # Table header
    print(f"\n  {'Parameter':<20} {'S1':>8} {'±conf':>8} {'ST':>8} {'±conf':>8} {'ST-S1':>8}")
    print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    # Sort by ST descending
    sorted_params = sorted(params, key=lambda p: ST[p], reverse=True)

    for p in sorted_params:
        interaction = ST[p] - S1[p]
        marker = ""
        if S1[p] < 0.01:
            marker = " ◄ REDUNDANT"
        elif interaction > 0.10 and interaction > S1[p]:
            marker = " ◄ STRONG INTERACTION"
        elif ST[p] > 0.15:
            marker = " ◄ INFLUENTIAL"

        print(f"  {p:<20} {S1[p]:>8.4f} {S1_conf[p]:>8.4f} "
              f"{ST[p]:>8.4f} {ST_conf[p]:>8.4f} {interaction:>8.4f}{marker}")

    # Summary
    print(f"\n  Sum(S1) = {sum(S1.values()):.4f}  "
          f"(should be ~1.0 for additive model)")
    print(f"  Sum(ST) = {sum(ST.values()):.4f}  "
          f"(>1.0 indicates interactions)")

    # Redundant parameters
    redundant = [p for p in params if S1[p] < 0.01]
    if redundant:
        print(f"\n  ⚠ REDUNDANT PARAMETERS (S1 < 0.01) — candidates for fixing:")
        for p in redundant:
            bounds = results["bounds"][params.index(p)]
            midpoint = (bounds[0] + bounds[1]) / 2
            print(f"    • {p}: S1={S1[p]:.4f}, ST={ST[p]:.4f} "
                  f"→ fix at default or midpoint ({midpoint:.2f})")
    else:
        print(f"\n  ✓ All parameters have S1 ≥ 0.01 — none clearly redundant")

    # Strong interactions
    if "S2" in results:
        s2 = results["S2"]
        top_interactions = sorted(s2.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        if top_interactions and abs(top_interactions[0][1]) > 0.02:
            print(f"\n  Top parameter interactions (S2):")
            for pair, val in top_interactions:
                if abs(val) > 0.01:
                    print(f"    • {pair}: S2 = {val:.4f}")

    # Influential parameters
    influential = [p for p in params if ST[p] > 0.10]
    if influential:
        print(f"\n  ★ MOST INFLUENTIAL (ST > 0.10):")
        for p in sorted(influential, key=lambda x: ST[x], reverse=True):
            print(f"    • {p}: ST={ST[p]:.4f}")

    print("\n" + "=" * 65)


# ── Plot ────────────────────────────────────────────────────────────

def plot_sobol(results: dict, output_path: str):
    """Horizontal bar chart of S1 and ST indices."""
    params = results["parameters"]
    S1 = results["S1"]
    ST = results["ST"]
    S1_conf = results["S1_conf"]
    ST_conf = results["ST_conf"]

    # Sort by ST
    sorted_params = sorted(params, key=lambda p: ST[p], reverse=True)

    y_pos = np.arange(len(sorted_params))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))

    s1_vals = [S1[p] for p in sorted_params]
    st_vals = [ST[p] for p in sorted_params]
    s1_err = [S1_conf[p] for p in sorted_params]
    st_err = [ST_conf[p] for p in sorted_params]

    ax.barh(y_pos - bar_height / 2, st_vals, bar_height,
            xerr=st_err, label="ST (Total)", color="#3b82f6",
            alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)
    ax.barh(y_pos + bar_height / 2, s1_vals, bar_height,
            xerr=s1_err, label="S1 (First-order)", color="#f97316",
            alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_pretty_name(p) for p in sorted_params], fontsize=11)
    ax.set_xlabel("Sobol Index", fontsize=12)
    ax.set_title("Global Sensitivity Analysis — DynamicsV2\n"
                 f"N={results['N']}, {results['total_evaluations']} evaluations, "
                 f"{results['n_agents']} agents × {results['n_rounds']} rounds",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.axvline(x=0.01, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(0.015, len(sorted_params) - 0.5, "S1 < 0.01\n(redundant)",
            fontsize=8, color="red", alpha=0.6)
    ax.set_xlim(left=-0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _pretty_name(param: str) -> str:
    """Human-readable parameter names for plots."""
    names = {
        "lambda_elite": "λ_elite (step size)",
        "lambda_citizen": "λ_citizen (step size)",
        "alpha_social": "α_social (peer influence)",
        "alpha_event": "α_event (shock weight)",
        "alpha_herd": "α_herd (bandwagon)",
        "alpha_anchor": "α_anchor (inertia)",
        "herd_threshold": "herd_threshold",
        "anchor_drift_rate": "anchor_drift_rate",
    }
    return names.get(param, param)


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sobol GSA for DynamicsV2")
    parser.add_argument("--N", type=int, default=1024, help="Base sample size")
    parser.add_argument("--rounds", type=int, default=7, help="Rounds per sim")
    parser.add_argument("--agents", type=int, default=30, help="Agents per sim")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_sobol_analysis(
        N=args.N,
        n_agents=args.agents,
        n_rounds=args.rounds,
        seed=args.seed,
        output_dir=args.output,
    )
