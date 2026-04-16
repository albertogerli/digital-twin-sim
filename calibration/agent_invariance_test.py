"""Agent count invariance test for DynamicsV2 calibration.

Verifies that calibrated parameters are not artifacts of the agent
discretization (N=30). Runs the same scenario at N=30, 100, 300, 1000
with 50 seeds each, then checks whether:
  1. Final outcome distributions are stable across N
  2. Optimal parameters shift when N increases (mini grid-search)

If results change drastically with N, the calibration is unreliable
and N must be fixed at a higher value before proceeding.

Usage:
    python -m calibration.agent_invariance_test [--seeds 50]
"""

import itertools
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.simulation.opinion_dynamics_v2 import DynamicsV2

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Configuration ───────────────────────────────────────────────────

# Political domain calibrated params (v1)
V1_POLITICAL = {
    "anchor_weight": 0.130,
    "social_weight": 0.125,
    "event_weight": 0.069,
    "herd_weight": 0.05,
    "direct_shift_weight": 0.526,
    "herd_threshold": 0.21,
    "anchor_drift_rate": 0.25,
}

AGENT_COUNTS = [30, 100, 300, 1000]
N_SEEDS = 50
N_ROUNDS = 7

# Event sequence: fixed across all runs for comparability
# Alternating shocks simulating a political campaign
EVENT_SEQUENCE = [
    {"round": 1, "shock_magnitude": 0.15, "shock_direction": 0.2},
    {"round": 2, "shock_magnitude": 0.35, "shock_direction": -0.4},
    {"round": 3, "shock_magnitude": 0.20, "shock_direction": 0.1},
    {"round": 4, "shock_magnitude": 0.50, "shock_direction": 0.6},   # major event
    {"round": 5, "shock_magnitude": 0.25, "shock_direction": -0.2},
    {"round": 6, "shock_magnitude": 0.30, "shock_direction": 0.3},
    {"round": 7, "shock_magnitude": 0.40, "shock_direction": -0.3},
]


# ── Lightweight Agent ───────────────────────────────────────────────

class InvAgent:
    """Minimal agent for invariance testing."""
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


class InvPlatform:
    """Minimal platform — returns posts from sampled agents."""

    def __init__(self, agents, rng):
        self._agents = agents
        self._rng = rng
        self.conn = None

    def get_top_posts(self, round_num, top_n=5, platform=None):
        sample = self._rng.sample(self._agents, min(top_n, len(self._agents)))
        return [
            {"id": i + 1, "author_id": a.id,
             "likes": self._rng.randint(5, 80),
             "reposts": self._rng.randint(0, 20)}
            for i, a in enumerate(sample)
        ]


# ── Agent Creation ──────────────────────────────────────────────────

def create_agents(n: int, seed: int) -> list[InvAgent]:
    """Create N agents: 50/50 pro/against, positions ~ U[-0.8, 0.8].

    Tier split: 5% elite, 10% institutional, 85% citizen.
    Agent properties are drawn from the same distributions regardless of N,
    so increasing N adds resolution, not different behavior.
    """
    rng = random.Random(seed)
    agents = []
    n_pro = n // 2

    for i in range(n):
        # Position
        if i < n_pro:
            pos = rng.uniform(0.05, 0.8)
        else:
            pos = rng.uniform(-0.8, -0.05)

        # Tier
        r = i / n
        if r < 0.05:
            tier, influence = 1, rng.uniform(0.7, 1.0)
            rigidity = rng.uniform(0.4, 0.7)
        elif r < 0.15:
            tier, influence = 2, rng.uniform(0.4, 0.7)
            rigidity = rng.uniform(0.3, 0.5)
        else:
            tier, influence = 3, rng.uniform(0.1, 0.4)
            rigidity = rng.uniform(0.15, 0.40)

        agents.append(InvAgent(
            id=f"a{i}", position=pos, tier=tier,
            rigidity=rigidity, tolerance=rng.uniform(0.3, 0.6),
            influence=influence,
        ))

    return agents


# ── Simulation Runner ───────────────────────────────────────────────

def run_sim(model: DynamicsV2, n_agents: int, seed: int) -> float:
    """Run one simulation, return final pro-percentage."""
    agents = create_agents(n_agents, seed)
    rng = random.Random(seed + 7777)
    platform = InvPlatform(agents, rng)

    for event in EVENT_SEQUENCE:
        model.step(agents, platform, event)

    pro = sum(1 for a in agents if a.position > 0.05)
    against = sum(1 for a in agents if a.position < -0.05)
    decided = pro + against
    return (pro / decided * 100.0) if decided > 0 else 50.0


def run_batch(model_factory, n_agents: int, seeds: list[int]) -> list[float]:
    """Run multiple seeds, each with a fresh model instance."""
    results = []
    for s in seeds:
        model = model_factory()
        results.append(run_sim(model, n_agents, s))
    return results


# ── Mini Grid Search ────────────────────────────────────────────────

def mini_grid_search(
    base_v2: DynamicsV2,
    n_agents: int,
    seeds: list[int],
    perturbation: float = 0.20,
    grid_steps: int = 5,
) -> dict:
    """Search ±perturbation around current params to find local optimum.

    For each parameter, try grid_steps values in [param*(1-pert), param*(1+pert)].
    Optimize for minimal std (most stable outcome) and report the optimal value.

    Returns dict with optimal params and their shift from baseline.
    """
    base_alpha = dict(base_v2.alpha)
    base_steps = dict(base_v2.step_sizes)
    base_herd = base_v2.herd_threshold
    base_drift = base_v2.anchor_drift_rate

    # Use fewer seeds for grid search (speed)
    grid_seeds = seeds[:15]

    # Baseline score: mean pro% with base params
    def make_base():
        return DynamicsV2(
            alpha=dict(base_alpha), step_sizes=dict(base_steps),
            herd_threshold=base_herd, anchor_drift_rate=base_drift,
        )

    baseline_results = run_batch(make_base, n_agents, grid_seeds)
    baseline_mean = mean(baseline_results)

    # Test each parameter independently
    param_optima = {}

    # Test alpha logits
    for key in ["direct", "social", "event", "herd", "anchor"]:
        base_val = base_alpha[key]
        lo = base_val - 2.0 * perturbation  # logits: additive perturbation
        hi = base_val + 2.0 * perturbation
        test_vals = np.linspace(lo, hi, grid_steps)

        best_val, best_score = base_val, abs(mean(baseline_results) - baseline_mean) + stdev(baseline_results)

        for v in test_vals:
            alpha_test = dict(base_alpha)
            alpha_test[key] = float(v)

            def make_model(a=dict(alpha_test)):
                return DynamicsV2(
                    alpha=a, step_sizes=dict(base_steps),
                    herd_threshold=base_herd, anchor_drift_rate=base_drift,
                )

            res = run_batch(make_model, n_agents, grid_seeds)
            score = stdev(res)  # lower variance = more stable
            if score < best_score:
                best_score = score
                best_val = float(v)

        param_optima[f"alpha_{key}"] = {
            "baseline": base_val,
            "optimal": best_val,
            "shift": best_val - base_val,
            "shift_pct": ((best_val - base_val) / max(abs(base_val), 0.01)) * 100,
        }

    # Test step sizes
    for tier_key in ["elite", "citizen"]:
        base_val = base_steps[tier_key]
        lo = max(0.01, base_val * (1 - perturbation))
        hi = base_val * (1 + perturbation)
        test_vals = np.linspace(lo, hi, grid_steps)

        best_val, best_score = base_val, stdev(baseline_results)

        for v in test_vals:
            steps_test = dict(base_steps)
            steps_test[tier_key] = float(v)

            def make_model(s=dict(steps_test)):
                return DynamicsV2(
                    alpha=dict(base_alpha), step_sizes=s,
                    herd_threshold=base_herd, anchor_drift_rate=base_drift,
                )

            res = run_batch(make_model, n_agents, grid_seeds)
            score = stdev(res)
            if score < best_score:
                best_score = score
                best_val = float(v)

        param_optima[f"lambda_{tier_key}"] = {
            "baseline": base_val,
            "optimal": best_val,
            "shift": best_val - base_val,
            "shift_pct": ((best_val - base_val) / base_val) * 100,
        }

    return param_optima


# ── Main Pipeline ───────────────────────────────────────────────────

def run_invariance_test(
    n_seeds: int = N_SEEDS,
    output_dir: str = None,
) -> dict:
    """Full agent count invariance test."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Agent Count Invariance Test — DynamicsV2           ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Agent counts:    {AGENT_COUNTS}            ║")
    print(f"║  Seeds per N:     {n_seeds:<5}                            ║")
    print(f"║  Rounds:          {N_ROUNDS:<5}                            ║")
    print(f"║  Total sims:      {len(AGENT_COUNTS) * n_seeds:<5}                            ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Convert v1 → v2
    model_v2 = DynamicsV2.from_v1_params(**V1_POLITICAL)
    print(f"\n▸ V1 params converted to V2:")
    pi = model_v2.get_mix_weights()
    print(f"  α = {model_v2.alpha}")
    print(f"  π = {{{', '.join(f'{k}: {v:.3f}' for k, v in pi.items())}}}")
    print(f"  λ = {model_v2.step_sizes}")
    print(f"  herd_threshold = {model_v2.herd_threshold}")
    print(f"  anchor_drift_rate = {model_v2.anchor_drift_rate}")

    seeds = list(range(1000, 1000 + n_seeds))
    all_results = {}
    t0_total = time.time()

    # ── Part 1: Outcome distributions per N ──
    print(f"\n{'─' * 55}")
    print("  PART 1: Outcome Distributions")
    print(f"{'─' * 55}")

    for n_agents in AGENT_COUNTS:
        t0 = time.time()

        def make_model():
            return DynamicsV2.from_v1_params(**V1_POLITICAL)

        outcomes = run_batch(make_model, n_agents, seeds)
        elapsed = time.time() - t0

        m = mean(outcomes)
        s = stdev(outcomes) if len(outcomes) > 1 else 0.0
        ci95 = 1.96 * s / (len(outcomes) ** 0.5)

        all_results[n_agents] = {
            "outcomes": outcomes,
            "mean": m,
            "std": s,
            "ci95": ci95,
            "min": min(outcomes),
            "max": max(outcomes),
            "median": sorted(outcomes)[len(outcomes) // 2],
            "elapsed_s": elapsed,
        }

        print(f"\n  N={n_agents:>4}:  mean={m:5.1f}%  std={s:5.2f}  "
              f"CI95=[{m-ci95:.1f}, {m+ci95:.1f}]  "
              f"range=[{min(outcomes):.1f}, {max(outcomes):.1f}]  "
              f"({elapsed:.1f}s)")

    # ── Part 2: Parameter drift (mini grid-search) ──
    print(f"\n{'─' * 55}")
    print("  PART 2: Parameter Drift (mini grid-search ±20%)")
    print(f"{'─' * 55}")

    drift_results = {}
    for n_agents in AGENT_COUNTS:
        t0 = time.time()
        print(f"\n  N={n_agents}: searching...", end="", flush=True)

        base_model = DynamicsV2.from_v1_params(**V1_POLITICAL)
        optima = mini_grid_search(base_model, n_agents, seeds)
        elapsed = time.time() - t0

        drift_results[n_agents] = optima
        print(f" done ({elapsed:.1f}s)")

        max_shift = max(abs(v["shift"]) for v in optima.values())
        drifted = [k for k, v in optima.items() if abs(v["shift_pct"]) > 15]
        if drifted:
            print(f"    ⚠ Parameters drifted >15%: {', '.join(drifted)}")
        else:
            print(f"    ✓ All parameters stable (max shift: {max_shift:.4f})")

    # ── Part 3: Convergence analysis ──
    print(f"\n{'─' * 55}")
    print("  PART 3: Convergence Analysis")
    print(f"{'─' * 55}")

    # Check if means converge
    means = [all_results[n]["mean"] for n in AGENT_COUNTS]
    stds = [all_results[n]["std"] for n in AGENT_COUNTS]

    # Mean drift from largest N (reference)
    ref_mean = all_results[AGENT_COUNTS[-1]]["mean"]
    ref_std = all_results[AGENT_COUNTS[-1]]["std"]

    print(f"\n  Reference (N={AGENT_COUNTS[-1]}): mean={ref_mean:.2f}%, std={ref_std:.2f}")
    print(f"\n  {'N':>6}  {'Mean':>8}  {'Δ from ref':>10}  {'Std':>8}  {'Std ratio':>10}  {'Verdict'}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*20}")

    convergence_ok = True
    min_acceptable_n = AGENT_COUNTS[0]

    for n in AGENT_COUNTS:
        r = all_results[n]
        delta = r["mean"] - ref_mean
        std_ratio = r["std"] / ref_std if ref_std > 0 else float("inf")

        # Verdict: |Δmean| < 3% AND std_ratio < 2.0
        ok = abs(delta) < 3.0 and std_ratio < 2.0
        verdict = "✓ STABLE" if ok else "⚠ UNSTABLE"
        if not ok and n == min_acceptable_n:
            min_acceptable_n = AGENT_COUNTS[AGENT_COUNTS.index(n) + 1] if n != AGENT_COUNTS[-1] else n
            convergence_ok = False

        print(f"  {n:>6}  {r['mean']:>7.1f}%  {delta:>+9.2f}%  {r['std']:>7.2f}  {std_ratio:>9.2f}x  {verdict}")

    # ── Conclusion ──
    print(f"\n{'═' * 55}")
    print("  CONCLUSION")
    print(f"{'═' * 55}")

    if convergence_ok:
        print(f"\n  ✓ Calibration is STABLE across agent counts.")
        print(f"    N=30 is acceptable: mean drift < 3%, std ratio < 2x.")
        print(f"    Parameters calibrated on N=30 generalize to larger populations.")
    else:
        print(f"\n  ⚠ Calibration shows N-DEPENDENCE.")
        print(f"    Minimum acceptable N: {min_acceptable_n}")
        print(f"    Parameters calibrated on N=30 may not generalize.")
        print(f"    Recommendation: recalibrate with N≥{min_acceptable_n}.")

    # Check parameter drift across N
    max_drifts = {}
    for param in drift_results[AGENT_COUNTS[0]].keys():
        vals = [drift_results[n][param]["optimal"] for n in AGENT_COUNTS]
        spread = max(vals) - min(vals)
        baseline = abs(drift_results[AGENT_COUNTS[0]][param]["baseline"])
        max_drifts[param] = {
            "spread": spread,
            "relative": (spread / max(baseline, 0.01)) * 100,
        }

    unstable_params = {k: v for k, v in max_drifts.items() if v["relative"] > 25}
    if unstable_params:
        print(f"\n  ⚠ Parameters with >25% spread across N:")
        for p, v in sorted(unstable_params.items(), key=lambda x: x[1]["relative"], reverse=True):
            print(f"    • {p}: spread={v['spread']:.4f} ({v['relative']:.0f}%)")
    else:
        print(f"\n  ✓ All parameters have <25% spread across agent counts.")

    total_time = time.time() - t0_total
    print(f"\n  Total time: {total_time:.0f}s")

    # ── Save outputs ──
    output = {
        "v1_params": V1_POLITICAL,
        "v2_alpha": model_v2.alpha,
        "v2_step_sizes": model_v2.step_sizes,
        "agent_counts": AGENT_COUNTS,
        "n_seeds": n_seeds,
        "n_rounds": N_ROUNDS,
        "distributions": {
            str(n): {k: v for k, v in r.items() if k != "outcomes"}
            for n, r in all_results.items()
        },
        "outcomes_per_n": {str(n): r["outcomes"] for n, r in all_results.items()},
        "parameter_drift": {
            str(n): optima for n, optima in drift_results.items()
        },
        "convergence": {
            "reference_n": AGENT_COUNTS[-1],
            "reference_mean": ref_mean,
            "stable": convergence_ok,
            "min_acceptable_n": min_acceptable_n,
        },
        "parameter_spread": max_drifts,
    }

    json_path = Path(output_dir) / "agent_invariance.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n▸ Results saved to {json_path}")

    # ── Plot ──
    if HAS_MPL:
        plot_path = Path(output_dir) / "agent_invariance.png"
        _plot_results(all_results, drift_results, str(plot_path))
        print(f"▸ Plot saved to {plot_path}")

    return output


# ── Plot ────────────────────────────────────────────────────────────

def _plot_results(all_results: dict, drift_results: dict, output_path: str):
    """Boxplot of outcome distributions + parameter drift heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: Boxplots
    ax = axes[0]
    data = [all_results[n]["outcomes"] for n in AGENT_COUNTS]
    labels = [f"N={n}" for n in AGENT_COUNTS]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=2))

    colors = ["#fbbf24", "#f97316", "#3b82f6", "#10b981"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add mean markers
    for i, n in enumerate(AGENT_COUNTS):
        ax.plot(i + 1, all_results[n]["mean"], "D", color="red",
                markersize=8, zorder=5, label="Mean" if i == 0 else None)

    # Reference line
    ref = all_results[AGENT_COUNTS[-1]]["mean"]
    ax.axhline(y=ref, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(0.6, ref + 0.5, f"ref={ref:.1f}%", fontsize=8, color="gray")

    ax.set_ylabel("Final Pro %", fontsize=12)
    ax.set_title("Outcome Distribution by Agent Count\n(50 seeds each)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: Parameter drift
    ax2 = axes[1]
    params = list(drift_results[AGENT_COUNTS[0]].keys())
    n_params = len(params)

    shifts = np.zeros((n_params, len(AGENT_COUNTS)))
    for j, n in enumerate(AGENT_COUNTS):
        for i, p in enumerate(params):
            shifts[i, j] = drift_results[n][p]["shift"]

    y_pos = np.arange(n_params)
    bar_h = 0.18
    colors_drift = ["#fbbf24", "#f97316", "#3b82f6", "#10b981"]

    for j, n in enumerate(AGENT_COUNTS):
        ax2.barh(y_pos + j * bar_h - 0.27, shifts[:, j], bar_h,
                 label=f"N={n}", color=colors_drift[j], alpha=0.8,
                 edgecolor="white", linewidth=0.5)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([p.replace("alpha_", "α_").replace("lambda_", "λ_") for p in params], fontsize=10)
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.set_xlabel("Optimal Shift from Baseline", fontsize=11)
    ax2.set_title("Parameter Drift by Agent Count\n(mini grid-search ±20%)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent count invariance test")
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="Seeds per N")
    parser.add_argument("--output", type=str, default=None, help="Output dir")
    args = parser.parse_args()

    run_invariance_test(n_seeds=args.seeds, output_dir=args.output)
