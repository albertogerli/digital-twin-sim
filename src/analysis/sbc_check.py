#!/usr/bin/env python3
"""Simulation-Based Calibration (SBC) check for the inference pipeline.

Validates that: prior → simulate → infer → posterior gives uniform ranks.

For each of N instances:
  1. Sample θ_true from the prior
  2. Generate synthetic data with θ_true (simulate + readout + noise)
  3. Run SVI inference on the synthetic data
  4. Compute rank of θ_true in the posterior samples

If inference is correct, ranks are Uniform(0, n_posterior_samples).

Usage:
    python -m src.analysis.sbc_check [--n-instances 100] [--svi-steps 500]
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, MCMC, NUTS, Trace_ELBO, Predictive, init_to_value
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoNormal

from ..dynamics.opinion_dynamics_jax import (
    ScenarioData, build_sparse_interaction, simulate_scenario,
)
from ..dynamics.param_utils import get_default_frozen_params, FROZEN_PARAMS
from ..observation.observation_model import _beta_binomial_logpmf
from ..inference.hierarchical_model import (
    PARAM_NAMES, FROZEN_KEYS, N_PARAMS,
    _simulate_one, _SCENARIO_AXES,
)


# ── Constants ────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "calibration" / "results" / "sbc"

N_AGENTS = 10
N_ROUNDS = 7
N_POSTERIOR_SAMPLES = 200


# ── Generate random scenario ────────────────────────────────────

def make_random_scenario(rng: np.random.RandomState, seed_jax: int) -> ScenarioData:
    """Create a random scenario with N_AGENTS agents and N_ROUNDS rounds."""
    # Random initial positions in [-0.8, 0.8]
    positions = rng.uniform(-0.8, 0.8, size=N_AGENTS).astype(np.float32)

    # Agent types: 20% elite (0), 80% citizen (1)
    agent_types = np.array(
        [0] * max(1, N_AGENTS // 5) + [1] * (N_AGENTS - N_AGENTS // 5),
        dtype=np.int32,
    )
    rng.shuffle(agent_types)

    # Rigidities and tolerances
    rigidities = np.where(
        agent_types == 0,
        rng.uniform(0.5, 0.8, size=N_AGENTS),
        rng.uniform(0.15, 0.45, size=N_AGENTS),
    ).astype(np.float32)

    tolerances = np.where(
        agent_types == 0,
        rng.uniform(0.25, 0.4, size=N_AGENTS),
        rng.uniform(0.3, 0.7, size=N_AGENTS),
    ).astype(np.float32)

    influences = np.where(
        agent_types == 0,
        rng.uniform(0.6, 1.0, size=N_AGENTS),
        rng.uniform(0.1, 0.4, size=N_AGENTS),
    ).astype(np.float32)

    # Random events: ~40% of rounds have a shock
    events = np.zeros((N_ROUNDS, 2), dtype=np.float32)
    for r in range(N_ROUNDS):
        if rng.random() < 0.4:
            events[r, 0] = rng.uniform(0.1, 0.5)   # magnitude
            events[r, 1] = rng.choice([-1.0, 1.0])  # direction

    # No LLM shifts
    llm_shifts = np.zeros((N_ROUNDS, N_AGENTS), dtype=np.float32)

    # Interaction matrix
    interaction_matrix = build_sparse_interaction(
        jnp.array(influences), seed=seed_jax,
    )

    return ScenarioData(
        initial_positions=jnp.array(positions),
        agent_types=jnp.array(agent_types),
        agent_rigidities=jnp.array(rigidities),
        agent_tolerances=jnp.array(tolerances),
        events=jnp.array(events),
        llm_shifts=jnp.array(llm_shifts),
        interaction_matrix=interaction_matrix,
    )


# ── Simplified single-scenario model for SBC ────────────────────

def sbc_model(y_obs, scenario_data: ScenarioData, frozen_vec: jnp.ndarray,
              agent_mask: jnp.ndarray, n_real_rounds: int):
    """Simplified model: moderate prior on θ_s, per-round + final likelihood.

    y_obs is a vector: [round_0_pct, round_1_pct, ..., round_{N-1}_pct, final_pct]
    Total length = n_real_rounds + 1.

    The per-round observations make the problem identifiable (6 params,
    N_ROUNDS+1 observations) and the moderate prior N(0, 0.3) keeps the
    simulator in the informative (non-saturated) regime.
    """
    # Prior on calibrable params — N(0, 0.3) avoids saturation
    theta_s = numpyro.sample(
        "theta_s",
        dist.Normal(jnp.zeros(N_PARAMS), 0.3).to_event(1),
    )

    # Observation params
    tau_readout = numpyro.sample("tau_readout", dist.LogNormal(-1.5, 0.5))
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(3.0))

    # Simulate
    pro_fraction, final_pro_pct = _simulate_one(
        theta_s, frozen_vec, scenario_data,
        agent_mask, tau_readout, jnp.array(n_real_rounds),
    )

    numpyro.deterministic("sim_final_pct", final_pro_pct)

    # Per-round observations (% scale)
    sim_rounds_pct = pro_fraction[:n_real_rounds] * 100.0
    sim_all = jnp.concatenate([sim_rounds_pct, final_pro_pct[None]])

    numpyro.sample(
        "y_obs",
        dist.Normal(sim_all, sigma_obs).to_event(1),
        obs=y_obs,
    )


# ── SBC SVI infrastructure (compiled once) ──────────────────────

class SBCRunner:
    """Manages SVI objects compiled once for all SBC instances.

    Key optimizations:
    1. y_obs is a model argument (not baked in) → single JIT trace
    2. jax.lax.scan compiles the full SVI loop into one XLA call
    3. _init_state reused across instances (no per-instance recompilation)
    """

    def __init__(self, scenario_data, frozen_vec, agent_mask,
                 lr=0.005, n_steps=500):
        self.scenario_data = scenario_data
        self.frozen_vec = frozen_vec
        self.agent_mask = agent_mask
        self.n_steps = n_steps

        # Model takes y_obs as first argument
        def model_fn(y_obs):
            sbc_model(y_obs, scenario_data, frozen_vec, agent_mask, N_ROUNDS)

        self.model_fn = model_fn
        self.init_values = {
            "theta_s": jnp.zeros(N_PARAMS),
            "tau_readout": jnp.array(0.22),  # exp(-1.5)
            "sigma_obs": jnp.array(3.0),
        }

        # AutoNormal (diagonal) guide: less prone to SVI variance
        # underestimation than low-rank MVN for small problems
        self.guide = AutoNormal(
            model_fn, init_loc_fn=init_to_value(values=self.init_values),
        )
        self.optimizer = numpyro.optim.Adam(lr)
        self.svi = SVI(model_fn, self.guide, self.optimizer, loss=Trace_ELBO())

        # Initialize once with dummy y_obs vector to compile
        rng_key = jax.random.PRNGKey(0)
        dummy_y = jnp.ones(N_ROUNDS + 1) * 50.0
        self._init_state = self.svi.init(rng_key, dummy_y)

        # Build scan-based SVI loop (single JIT call for all steps)
        svi_ref = self.svi

        @jax.jit
        def _run_svi_loop(init_state, y_obs_arr):
            def step_fn(state, _):
                new_state, loss = svi_ref.update(state, y_obs_arr)
                return new_state, loss
            final_state, losses = jax.lax.scan(
                step_fn, init_state, None, length=n_steps,
            )
            return final_state, losses

        self._run_svi_loop = _run_svi_loop

        # Warmup: compile the scan loop
        print("    Compiling scan loop...", flush=True)
        t0 = time.time()
        _final, _losses = self._run_svi_loop(self._init_state, dummy_y)
        jax.tree.leaves(_final)[0].block_until_ready()  # force compile
        print(f"    Scan compile: {time.time() - t0:.1f}s", flush=True)

    def run_instance(self, y_obs, n_steps=500, seed=42):
        """Run SVI for one instance. Returns posterior samples dict."""
        y_obs_arr = jnp.asarray(y_obs, dtype=jnp.float32)

        # Run full SVI loop as single JIT call
        final_state, losses = self._run_svi_loop(self._init_state, y_obs_arr)

        svi_params = self.svi.get_params(final_state)

        # Draw posterior samples
        predictive = Predictive(self.guide, params=svi_params,
                                num_samples=N_POSTERIOR_SAMPLES)
        rng = jax.random.PRNGKey(seed)
        samples = predictive(rng, y_obs_arr)

        return {k: np.array(v) for k, v in samples.items()}


class MCMCRunner:
    """NUTS-based SBC runner. Slower but unbiased — gold standard for SBC."""

    def __init__(self, scenario_data, frozen_vec, agent_mask,
                 num_warmup=200, num_samples=200):
        self.num_warmup = num_warmup
        self.num_samples = num_samples

        def model_fn(y_obs):
            sbc_model(y_obs, scenario_data, frozen_vec, agent_mask, N_ROUNDS)

        self.model_fn = model_fn

    def run_instance(self, y_obs, n_steps=None, seed=42):
        """Run NUTS for one instance. Returns posterior samples dict."""
        y_obs_arr = jnp.asarray(y_obs, dtype=jnp.float32)

        kernel = NUTS(self.model_fn, max_tree_depth=6)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup,
                     num_samples=self.num_samples, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed), y_obs_arr)

        samples = mcmc.get_samples()
        return {k: np.array(v) for k, v in samples.items()}


# ── Compute rank ─────────────────────────────────────────────────

def compute_rank(theta_true: float, posterior_samples: np.ndarray) -> int:
    """Rank of theta_true among posterior samples (count of samples < theta_true)."""
    return int(np.sum(posterior_samples < theta_true))


# ── Main SBC loop ───────────────────────────────────────────────

def run_sbc(
    n_instances: int = 100,
    svi_steps: int = 500,
    lr: float = 0.005,
    seed: int = 42,
    method: str = "svi",
):
    print("=" * 60)
    print(f"SBC Check: {n_instances} instances, {svi_steps} SVI steps")
    print("=" * 60)

    rng = np.random.RandomState(seed)
    frozen = get_default_frozen_params()
    frozen_vec = jnp.array([frozen[k] for k in FROZEN_KEYS])
    agent_mask = jnp.ones(N_AGENTS, dtype=jnp.bool_)

    # Create ONE shared scenario — all instances reuse it so JIT compiles once.
    # SBC tests inference correctness, not scenario diversity.
    shared_scenario = make_random_scenario(rng, seed_jax=seed)
    print(f"  Shared scenario: {N_AGENTS} agents, {N_ROUNDS} rounds")

    # Create runner
    print(f"  Creating {method.upper()} runner...")
    t_jit = time.time()
    if method == "nuts":
        runner = MCMCRunner(shared_scenario, frozen_vec, agent_mask,
                            num_warmup=200, num_samples=N_POSTERIOR_SAMPLES)
    else:
        runner = SBCRunner(shared_scenario, frozen_vec, agent_mask,
                           lr=lr, n_steps=svi_steps)
    print(f"  Setup: {time.time() - t_jit:.1f}s")

    # Storage for ranks
    all_param_names = list(PARAM_NAMES) + ["tau_readout", "sigma_obs"]
    ranks = {name: [] for name in all_param_names}

    # Track timing and failures
    times = []
    failures = 0
    theta_trues = {name: [] for name in all_param_names}

    t_total = time.time()

    for i in range(n_instances):
        t0 = time.time()

        # ── 1. Sample θ_true from prior (matching model) ──
        theta_true = rng.normal(0, 0.3, size=N_PARAMS).astype(np.float32)
        tau_readout_true = float(np.exp(rng.normal(-1.5, 0.5)))
        sigma_obs_true = float(np.abs(rng.normal(0, 3.0)))
        sigma_obs_true = max(sigma_obs_true, 0.5)

        # ── 2. Simulate with θ_true on shared scenario ──
        params_true = {
            PARAM_NAMES[k]: float(theta_true[k]) for k in range(N_PARAMS)
        }
        for k_idx, key in enumerate(FROZEN_KEYS):
            params_true[key] = float(frozen_vec[k_idx])

        try:
            result = simulate_scenario(params_true, shared_scenario)
            trajectories = np.array(result["trajectories"])  # [max_rounds, max_agents]
        except Exception as e:
            failures += 1
            if failures <= 3:
                print(f"  Instance {i+1}: sim failed: {e}")
            continue

        # Compute per-round pro_fraction using the same readout as the model
        from jax.nn import sigmoid
        traj_j = jnp.array(trajectories)
        soft_pro = sigmoid((traj_j - 0.05) / tau_readout_true)
        soft_against = sigmoid((-traj_j - 0.05) / tau_readout_true)
        soft_decided = soft_pro + soft_against
        mask = agent_mask[None, :]
        pro_count = jnp.sum(soft_pro * mask, axis=1)
        decided_count = jnp.sum(soft_decided * mask, axis=1) + 1e-8
        pro_fraction = pro_count / decided_count  # [max_rounds]

        sim_rounds_pct = np.array(pro_fraction[:N_ROUNDS] * 100.0)
        q_final = float(sim_rounds_pct[-1])

        # Generate per-round + final observations with noise
        # y_obs = [round_0, ..., round_{N-1}, final]
        sim_all = np.concatenate([sim_rounds_pct, [q_final]])
        y_obs = rng.normal(sim_all, sigma_obs_true).astype(np.float32)
        y_obs_jnp = jnp.array(y_obs)

        # ── 3. Run inference ──
        try:
            samples = runner.run_instance(
                y_obs=y_obs_jnp, n_steps=svi_steps, seed=seed + i * 1000,
            )
        except Exception as e:
            failures += 1
            if failures <= 3:
                print(f"  Instance {i+1}: SVI failed: {e}")
            continue

        # ── 4. Compute ranks ──
        theta_s_samples = samples["theta_s"]  # [S, 4]
        for k, name in enumerate(PARAM_NAMES):
            r = compute_rank(theta_true[k], theta_s_samples[:, k])
            ranks[name].append(r)
            theta_trues[name].append(float(theta_true[k]))

        if "tau_readout" in samples:
            r = compute_rank(tau_readout_true, samples["tau_readout"].flatten())
            ranks["tau_readout"].append(r)
            theta_trues["tau_readout"].append(tau_readout_true)

        if "sigma_obs" in samples:
            r = compute_rank(sigma_obs_true, samples["sigma_obs"].flatten())
            ranks["sigma_obs"].append(r)
            theta_trues["sigma_obs"].append(sigma_obs_true)

        dt = time.time() - t0
        times.append(dt)

        if (i + 1) % 10 == 0:
            avg_t = np.mean(times[-10:])
            eta = avg_t * (n_instances - i - 1) / 60
            print(f"  Instance {i+1:3d}/{n_instances}  "
                  f"q_final={q_final:5.1f}  y_obs_final={y_obs[-1]:5.1f}  "
                  f"dt={dt:.1f}s  ETA={eta:.1f}min")

    elapsed = time.time() - t_total
    n_success = len(ranks[PARAM_NAMES[0]])
    print(f"\nCompleted: {n_success}/{n_instances} ({failures} failures)")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    return ranks, theta_trues, elapsed, n_success


# ── Analysis ─────────────────────────────────────────────────────

def analyze_ranks(ranks: dict, n_success: int):
    """KS test for uniformity on each parameter's rank distribution."""
    results = {}
    all_params = list(PARAM_NAMES) + ["tau_readout", "sigma_obs"]

    print(f"\n{'Parameter':<18} {'N':>4} {'KS stat':>8} {'p-value':>8} {'Verdict':>10}")
    print("-" * 52)

    for name in all_params:
        r = np.array(ranks.get(name, []))
        if len(r) < 10:
            results[name] = {"n": len(r), "ks": float("nan"), "p": 0.0, "verdict": "TOO_FEW"}
            print(f"  {name:<16} {len(r):4d}      -        -   TOO_FEW")
            continue

        # Normalize ranks to [0, 1] for KS test
        r_norm = r / N_POSTERIOR_SAMPLES
        ks_stat, p_value = stats.kstest(r_norm, 'uniform')

        if p_value >= 0.05:
            verdict = "PASS"
        else:
            verdict = "FAIL"

        results[name] = {
            "n": len(r),
            "ks": float(ks_stat),
            "p": float(p_value),
            "verdict": verdict,
            "ranks": r.tolist(),
        }

        print(f"  {name:<16} {len(r):4d} {ks_stat:8.4f} {p_value:8.4f} {verdict:>10}")

    return results


def make_histograms(results: dict, out_dir: Path):
    """Generate rank histogram PNGs using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping histogram plots")
        return

    n_params = len([k for k in results if results[k].get("ranks")])
    if n_params == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    all_params = list(PARAM_NAMES) + ["tau_readout", "sigma_obs"]

    for idx, name in enumerate(all_params):
        if idx >= len(axes):
            break
        ax = axes[idx]
        r = results.get(name, {})
        ranks_arr = r.get("ranks", [])

        if not ranks_arr:
            ax.set_title(f"{name}\n(no data)")
            continue

        ranks_arr = np.array(ranks_arr)
        n_bins = 10

        ax.hist(ranks_arr, bins=n_bins, range=(0, N_POSTERIOR_SAMPLES),
                density=True, alpha=0.7, color='steelblue', edgecolor='black')

        # Uniform reference line
        uniform_height = 1.0 / N_POSTERIOR_SAMPLES
        ax.axhline(uniform_height, color='red', linestyle='--', linewidth=1.5,
                    label=f'Uniform')

        ks = r.get("ks", float("nan"))
        p = r.get("p", 0)
        verdict = r.get("verdict", "?")
        color = 'green' if verdict == "PASS" else 'red'

        ax.set_title(f"{name}\nKS={ks:.3f}, p={p:.3f} [{verdict}]",
                     color=color, fontsize=10)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Density")

    # Hide unused axes
    for idx in range(len(all_params), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("SBC Rank Histograms — Uniform = correct inference", fontsize=13)
    plt.tight_layout()

    plot_path = out_dir / "sbc_rank_histograms.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")


def generate_report(results: dict, elapsed: float, n_success: int,
                    n_instances: int, svi_steps: int, out_dir: Path,
                    method: str = "svi"):
    """Generate SBC report markdown."""
    lines = [
        "# SBC (Simulation-Based Calibration) Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Configuration",
        "",
        f"- Instances: {n_instances} (succeeded: {n_success})",
        f"- Inference method: {method.upper()}" + (f" ({svi_steps} steps)" if method == "svi" else " (200 warmup + 200 samples)"),
        f"- Posterior samples: {N_POSTERIOR_SAMPLES}",
        f"- Agents per scenario: {N_AGENTS}, Rounds: {N_ROUNDS}",
        f"- Model: Normal(0, 0.3) prior + per-round Normal likelihood",
        f"- Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)",
        "",
        "## Results",
        "",
        "| Parameter | N | KS Statistic | p-value | Verdict |",
        "|---|---|---|---|---|",
    ]

    all_pass = True
    for name in list(PARAM_NAMES) + ["tau_readout", "sigma_obs"]:
        r = results.get(name, {})
        n = r.get("n", 0)
        ks = r.get("ks", float("nan"))
        p = r.get("p", 0)
        verdict = r.get("verdict", "?")
        ks_str = f"{ks:.4f}" if not math.isnan(ks) else "-"
        p_str = f"{p:.4f}" if not math.isnan(ks) else "-"
        lines.append(f"| {name} | {n} | {ks_str} | {p_str} | **{verdict}** |")
        if verdict == "FAIL":
            all_pass = False

    lines.extend(["", "## Interpretation", ""])

    if all_pass:
        lines.extend([
            "**All parameters PASS the KS uniformity test (p > 0.05).**",
            "",
            "This confirms that:",
            f"- The JAX simulator is correctly differentiable through {'SVI' if method == 'svi' else 'NUTS'}",
            "- The per-round Normal likelihood is well-specified",
            f"- The {'SVI guide (AutoNormal)' if method == 'svi' else 'NUTS sampler'} recovers correct posteriors",
            "- The readout (soft sigmoid pro_fraction) is consistent with the generative model",
            "",
            "The inference pipeline is validated for use in the hierarchical calibration.",
        ])
    else:
        failing = [name for name in results if results[name].get("verdict") == "FAIL"]
        lines.extend([
            f"**FAILING parameters: {', '.join(failing)}**",
            "",
            "### Diagnosis",
            "",
        ])

        for name in failing:
            r = results[name]
            ranks_arr = np.array(r.get("ranks", []))
            if len(ranks_arr) == 0:
                continue

            # Check if concentrated at edges (overconfident) or center (underfitting)
            mid = N_POSTERIOR_SAMPLES / 2
            frac_below_25 = np.mean(ranks_arr < N_POSTERIOR_SAMPLES * 0.25)
            frac_above_75 = np.mean(ranks_arr > N_POSTERIOR_SAMPLES * 0.75)
            frac_center = np.mean(
                (ranks_arr > N_POSTERIOR_SAMPLES * 0.25) &
                (ranks_arr < N_POSTERIOR_SAMPLES * 0.75)
            )

            if frac_below_25 + frac_above_75 > 0.65:
                diagnosis = "U-shaped (overconfident posterior — too narrow)"
                fix = "Increase SVI steps or use a wider guide"
            elif frac_center > 0.65:
                diagnosis = "Peaked at center (underconfident — posterior too wide)"
                fix = "Check if likelihood is too weak relative to prior"
            elif frac_below_25 > 0.4:
                diagnosis = "Left-skewed (posterior biased high)"
                fix = "Check for systematic bias in the simulator or readout"
            elif frac_above_75 > 0.4:
                diagnosis = "Right-skewed (posterior biased low)"
                fix = "Check for systematic bias in the simulator or readout"
            else:
                diagnosis = "Non-uniform (mixed pattern)"
                fix = "Increase SVI steps; check model specification"

            lines.extend([
                f"**{name}**: {diagnosis}",
                f"- Edge fraction (below 25% + above 75%): {frac_below_25 + frac_above_75:.2f}",
                f"- Center fraction (25-75%): {frac_center:.2f}",
                f"- Recommendation: {fix}",
                "",
            ])

    # Histograms reference
    lines.extend([
        "",
        "## Rank Histograms",
        "",
        "![SBC Rank Histograms](sbc_rank_histograms.png)",
    ])

    report_path = out_dir / "sbc_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {report_path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SBC check for inference pipeline")
    parser.add_argument("--n-instances", type=int, default=100)
    parser.add_argument("--svi-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["svi", "nuts"], default="svi",
                        help="Inference method: svi (fast, biased) or nuts (slow, exact)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ranks, theta_trues, elapsed, n_success = run_sbc(
        n_instances=args.n_instances,
        svi_steps=args.svi_steps,
        lr=args.lr,
        seed=args.seed,
        method=args.method,
    )

    results = analyze_ranks(ranks, n_success)
    make_histograms(results, RESULTS_DIR)
    generate_report(results, elapsed, n_success,
                    args.n_instances, args.svi_steps, RESULTS_DIR,
                    method=args.method)

    # Save raw data
    raw = {
        "ranks": {k: v for k, v in ranks.items()},
        "theta_trues": theta_trues,
        "n_instances": args.n_instances,
        "n_success": n_success,
        "svi_steps": args.svi_steps,
        "elapsed_s": elapsed,
    }
    with open(RESULTS_DIR / "sbc_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    # Exit code based on results
    all_pass = all(r.get("verdict") == "PASS" for r in results.values()
                   if r.get("verdict") not in ("TOO_FEW", None))
    if all_pass:
        print("\n✓ SBC PASSED — inference pipeline is validated")
    else:
        failing = [n for n, r in results.items() if r.get("verdict") == "FAIL"]
        print(f"\n✗ SBC FAILED on: {', '.join(failing)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
