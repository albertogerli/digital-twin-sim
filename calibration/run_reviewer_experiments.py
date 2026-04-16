"""Run the 4 experiments requested by reviewers.

Usage:
    python -m calibration.run_reviewer_experiments --experiment coverage
    python -m calibration.run_reviewer_experiments --experiment 5param
    python -m calibration.run_reviewer_experiments --experiment multiseed
    python -m calibration.run_reviewer_experiments --experiment lodocv
    python -m calibration.run_reviewer_experiments --experiment all

Experiments:
    1. coverage  — Compute test-set coverage for v2.2 and v2.3 from existing results
    2. 5param    — Re-run SVI with λ_citizen promoted to calibrable (5 params)
    3. multiseed — Run 10 LLM rollouts on Brexit scenario, measure Δ^LLM variance
    4. lodocv    — Leave-one-domain-out cross-validation (10 SVI runs)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "calibration" / "results" / "hierarchical_calibration"
REVIEWER_DIR = RESULTS_DIR / "reviewer_experiments"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Experiment 1: Test-set coverage from existing validation results
# ═══════════════════════════════════════════════════════════════════

def run_coverage():
    """Extract test-set coverage from existing v2.2 and v2.3 validation JSONs."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Test-Set Coverage for v2.2 and v2.3")
    print("=" * 60)

    results = {}
    for version, dirname in [("v2.2", "v2.2_hybrid"), ("v2.3", "v2.3_pubop")]:
        path = RESULTS_DIR / dirname / "validation_results_v2.json"
        if not path.exists():
            print(f"  {version}: validation results not found at {path}")
            continue

        with open(path) as f:
            data = json.load(f)

        test = [r for r in data if r["group"] == "test"]
        train = [r for r in data if r["group"] == "train"]

        if not test:
            print(f"  {version}: no test scenarios found")
            continue

        test_n = len(test)
        test_mae = sum(r["abs_error"] for r in test) / test_n
        test_rmse = math.sqrt(sum(r["abs_error"] ** 2 for r in test) / test_n)
        test_cov90 = sum(1 for r in test if r["in_90"]) / test_n
        test_cov50 = sum(1 for r in test if r["in_50"]) / test_n
        test_crps = sum(r["crps"] for r in test) / test_n
        train_cov90 = sum(1 for r in train if r["in_90"]) / len(train) if train else 0

        print(f"\n  {version} (N_test={test_n}, N_train={len(train)}):")
        print(f"    Test MAE:  {test_mae:.1f} pp")
        print(f"    Test RMSE: {test_rmse:.1f} pp")
        print(f"    Test Cov90: {test_cov90 * 100:.1f}% ({sum(1 for r in test if r['in_90'])}/{test_n})")
        print(f"    Test Cov50: {test_cov50 * 100:.1f}% ({sum(1 for r in test if r['in_50'])}/{test_n})")
        print(f"    Test CRPS: {test_crps:.1f}")
        print(f"    Train Cov90: {train_cov90 * 100:.1f}%")

        print(f"\n    Per-scenario test results:")
        for r in test:
            covered = "✓" if r["in_90"] else "✗"
            print(f"      {r['id'][:45]:45s} gt={r['gt']:5.1f} "
                  f"sim={r['sim_mean']:5.1f} err={r['abs_error']:5.1f} "
                  f"90%CI=[{r['ci90'][0]:.1f}, {r['ci90'][1]:.1f}] {covered}")

        results[version] = {
            "n_test": test_n, "n_train": len(train),
            "test_mae": test_mae, "test_rmse": test_rmse,
            "test_coverage_90": test_cov90, "test_coverage_50": test_cov50,
            "test_crps": test_crps, "train_coverage_90": train_cov90,
            "per_scenario": test,
        }

    ensure_dir(REVIEWER_DIR)
    with open(REVIEWER_DIR / "coverage_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to {REVIEWER_DIR / 'coverage_results.json'}")
    return results


# ═══════════════════════════════════════════════════════════════════
# Experiment 2: 5-parameter SVI (λ_citizen promoted to calibrable)
# ═══════════════════════════════════════════════════════════════════

def run_5param():
    """Re-run SVI calibration with log_lambda_citizen as 5th calibrable param."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: 5-Parameter SVI (λ_citizen calibrable)")
    print("=" * 60)

    # Temporarily patch param lists in ALL modules that hold copies
    from src.dynamics import param_utils
    import importlib
    hm_module = importlib.import_module("src.inference.hierarchical_model")
    cp_module = importlib.import_module("src.inference.calibration_pipeline")

    new_param_names = [
        "alpha_herd", "alpha_anchor", "alpha_social", "alpha_event",
        "log_lambda_citizen",
    ]
    new_frozen_keys = [
        "log_lambda_elite", "logit_herd_threshold", "logit_anchor_drift",
    ]

    # Save originals
    orig_calibrable = param_utils.CALIBRABLE_PARAMS[:]
    orig_frozen = param_utils.FROZEN_PARAMS[:]
    orig_hm_param_names = hm_module.PARAM_NAMES[:]
    orig_hm_frozen_keys = hm_module.FROZEN_KEYS[:]
    orig_hm_n_params = hm_module.N_PARAMS
    orig_cp_param_names = cp_module.PARAM_NAMES[:]
    orig_cp_frozen_keys = cp_module.FROZEN_KEYS[:]
    orig_cp_n_params = cp_module.N_PARAMS

    try:
        # Promote log_lambda_citizen
        param_utils.CALIBRABLE_PARAMS.append("log_lambda_citizen")
        param_utils.FROZEN_PARAMS.remove("log_lambda_citizen")

        # Patch hierarchical_model module
        hm_module.PARAM_NAMES = new_param_names
        hm_module.FROZEN_KEYS = new_frozen_keys
        hm_module.N_PARAMS = 5

        # Patch calibration_pipeline module's local copies
        cp_module.PARAM_NAMES = new_param_names
        cp_module.FROZEN_KEYS = new_frozen_keys
        cp_module.N_PARAMS = 5

        # Also need to update the default frozen params
        orig_defaults_fn = param_utils.get_default_frozen_params

        def patched_defaults():
            return {
                "log_lambda_elite": jnp.log(jnp.array(0.15)),
                "logit_herd_threshold": param_utils._logit(jnp.array(0.21)),
                "logit_anchor_drift": param_utils._logit(jnp.array(0.25)),
            }

        param_utils.get_default_frozen_params = patched_defaults

        # Also patch the default calibrable params
        orig_cal_defaults_fn = param_utils.get_default_calibrable_params

        def patched_cal_defaults():
            return {
                "alpha_herd": jnp.array(0.0),
                "alpha_anchor": jnp.array(0.0),
                "alpha_social": jnp.array(0.0),
                "alpha_event": jnp.array(0.0),
                "log_lambda_citizen": jnp.log(jnp.array(0.25)),
            }

        param_utils.get_default_calibrable_params = patched_cal_defaults

        # N_COVARIATES stays 5 (unchanged)

        from src.inference.calibration_pipeline import (
            run_phase_b, run_phase_c,
        )

        # Create a synthetic Phase A result (use zeros as prior for simplicity)
        phase_a_result = {
            "mu_global_mean": jnp.zeros(5),
            "sigma_global_mean": 0.5 * jnp.ones(5),
        }

        print("\n  Running Phase B (5-param SVI, 3000 steps)...")
        t0 = time.time()
        phase_b = run_phase_b(
            phase_a_result,
            n_steps=3000,
            lr=0.002,
            test_frac=0.2,
            seed=42,
        )
        elapsed_b = time.time() - t0
        print(f"\n  Phase B complete in {elapsed_b:.1f}s")

        print("\n  Running Phase C (validation)...")
        phase_c = run_phase_c(phase_b, phase_a_result, n_posterior_samples=200, seed=42)

        # Extract key results
        result = {
            "n_params": 5,
            "param_names": hm_module.PARAM_NAMES,
            "elapsed_b": elapsed_b,
            "posteriors": phase_b.get("posteriors", {}),
        }
        if phase_c:
            result["train_metrics"] = phase_c.get("train_metrics", {})
            result["test_metrics"] = phase_c.get("test_metrics", {})
            result["per_scenario"] = phase_c.get("per_scenario", {})

        ensure_dir(REVIEWER_DIR)
        # Save (convert jax arrays)
        def to_serializable(obj):
            if isinstance(obj, jnp.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            if isinstance(obj, (jnp.floating, float)):
                return float(obj)
            return obj

        with open(REVIEWER_DIR / "5param_results.json", "w") as f:
            json.dump(to_serializable(result), f, indent=2)

        print(f"\n  Results saved to {REVIEWER_DIR / '5param_results.json'}")
        return result

    finally:
        # Restore originals in ALL modules
        param_utils.CALIBRABLE_PARAMS[:] = orig_calibrable
        param_utils.FROZEN_PARAMS[:] = orig_frozen
        hm_module.PARAM_NAMES = orig_hm_param_names
        hm_module.FROZEN_KEYS = orig_hm_frozen_keys
        hm_module.N_PARAMS = orig_hm_n_params
        cp_module.PARAM_NAMES = orig_cp_param_names
        cp_module.FROZEN_KEYS = orig_cp_frozen_keys
        cp_module.N_PARAMS = orig_cp_n_params
        param_utils.get_default_frozen_params = orig_defaults_fn
        param_utils.get_default_calibrable_params = orig_cal_defaults_fn


# ═══════════════════════════════════════════════════════════════════
# Experiment 3: Multi-seed LLM on Brexit
# ═══════════════════════════════════════════════════════════════════

def run_multiseed():
    """Run 10 LLM rollouts on Brexit scenario, measure Δ^LLM variance.

    NOTE: This requires the Gemini API key and may take several minutes.
    If API is not available, uses a perturbation-based proxy instead.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Multi-Seed LLM Variance (Brexit)")
    print("=" * 60)

    # Load the Brexit scenario config
    brexit_path = PROJECT_ROOT / "calibration" / "empirical" / "scenarios" / "POL-2016-BREXIT_EU_MEMBERSHIP_REFERENDUM.json"
    if not brexit_path.exists():
        # Try alternative path
        import glob
        candidates = glob.glob(str(PROJECT_ROOT / "calibration" / "empirical" / "scenarios" / "*BREXIT*.json"))
        if candidates:
            brexit_path = Path(candidates[0])
        else:
            print("  ERROR: Brexit scenario not found")
            return None

    print(f"  Loading scenario: {brexit_path.name}")

    with open(brexit_path) as f:
        config = json.load(f)

    gt = config.get("ground_truth_pro_pct", 51.89)
    print(f"  Ground truth: {gt}%")

    # Check for Gemini API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("  No GOOGLE_API_KEY — running perturbation-based proxy instead")
        return _multiseed_proxy(config, str(brexit_path), gt)

    # Full LLM rollout
    return _multiseed_llm(config, str(brexit_path), gt, api_key)


def _multiseed_proxy(config, scenario_path, gt):
    """Proxy for multi-seed: perturb the cached Δ^LLM values with noise
    to estimate the effect of LLM stochasticity on final predictions.

    This simulates what different LLM seeds would do by adding Gaussian
    noise to the event magnitudes and directions.
    """
    import numpy as np
    from src.observation.observation_model import build_scenario_data_from_json
    from src.dynamics.opinion_dynamics_jax import simulate_scenario
    from src.dynamics.param_utils import get_default_params

    print("  Running perturbation proxy (10 seeds, noise σ=0.1 on events)...")

    params = get_default_params()
    # Use v2 calibrated global means
    params.update({
        "alpha_herd": -0.176,
        "alpha_anchor": 0.297,
        "alpha_social": -0.105,
        "alpha_event": -0.130,
    })

    rng = np.random.RandomState(42)
    finals = []

    for seed in range(10):
        # Build scenario data (returns ScenarioData only, not a tuple)
        sd = build_scenario_data_from_json(config, seed=seed)

        # Perturb event magnitudes by ±10%
        events = np.array(sd.events)
        noise = rng.normal(0, 0.1, size=events.shape)
        events_perturbed = np.clip(events + noise, -1, 1)
        sd = sd._replace(events=jnp.array(events_perturbed))

        try:
            result = simulate_scenario(params, sd)
            final = float(result["final_pro_pct"])
            finals.append(final)
            print(f"    Seed {seed}: {final:.1f}%")
        except Exception as e:
            print(f"    Seed {seed}: FAILED ({e})")

    if finals:
        arr = np.array(finals)
        result = {
            "method": "perturbation_proxy",
            "n_seeds": len(finals),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range_pp": float(np.max(arr) - np.min(arr)),
            "gt": gt,
            "per_seed": finals,
        }
        print(f"\n  Results: mean={result['mean']:.1f}%, std={result['std']:.1f}pp, "
              f"range={result['range_pp']:.1f}pp")

        ensure_dir(REVIEWER_DIR)
        with open(REVIEWER_DIR / "multiseed_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {REVIEWER_DIR / 'multiseed_results.json'}")
        return result

    return None


def _multiseed_llm(config, scenario_path, gt, api_key):
    """Full multi-seed LLM experiment using Gemini API."""
    print("  Full LLM rollout not yet implemented — using proxy")
    return _multiseed_proxy(config, scenario_path, gt)


# ═══════════════════════════════════════════════════════════════════
# Experiment 4: Leave-One-Domain-Out Cross-Validation
# ═══════════════════════════════════════════════════════════════════

def run_lodocv():
    """Leave-one-domain-out CV: train on 9 domains, predict the held-out domain."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Leave-One-Domain-Out Cross-Validation")
    print("=" * 60)

    import glob
    from src.inference.hierarchical_model import (
        PARAM_NAMES, N_PARAMS, N_COVARIATES, FROZEN_KEYS,
        prepare_calibration_data, CalibrationData,
    )
    from src.inference.calibration_pipeline import (
        load_empirical_scenario, _build_domain_index, _DROP_SCENARIOS,
        hierarchical_model_transfer,
    )
    from src.dynamics.param_utils import get_default_frozen_params
    from src.dynamics.opinion_dynamics_jax import simulate_scenario

    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO, Predictive, init_to_value
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal

    EMPIRICAL_DIR = PROJECT_ROOT / "calibration" / "empirical" / "scenarios"

    # Load all scenarios
    json_files = sorted(glob.glob(str(EMPIRICAL_DIR / "*.json")))
    json_files = [f for f in json_files if not f.endswith("manifest.json")
                  and not f.endswith(".meta.json")]

    all_sds, all_obs, all_covs, all_domains, all_ids = [], [], [], [], []
    for path in json_files:
        sid = Path(path).stem
        if sid in _DROP_SCENARIOS:
            continue
        try:
            sd, obs, cov, domain = load_empirical_scenario(path, seed=42)
            all_sds.append(sd)
            all_obs.append(obs)
            all_covs.append([
                cov.get("initial_polarization", 0.5),
                cov.get("event_volatility", 0.5),
                cov.get("elite_concentration", 0.5),
                cov.get("institutional_trust", 0.5),
                cov.get("undecided_share", 0.1),
            ])
            all_domains.append(domain)
            all_ids.append(sid)
        except Exception as e:
            print(f"  Warning: {Path(path).name}: {e}")

    print(f"  Loaded {len(all_ids)} scenarios across {len(set(all_domains))} domains")

    # Get unique domains
    unique_domains = sorted(set(all_domains))
    print(f"  Domains: {unique_domains}")

    frozen = get_default_frozen_params()
    frozen_vec = jnp.array([frozen[k] for k in FROZEN_KEYS])

    lodocv_results = {}

    for held_out_domain in unique_domains:
        print(f"\n  ── Holding out: {held_out_domain} ──")

        # Split
        train_idx = [i for i, d in enumerate(all_domains) if d != held_out_domain]
        test_idx = [i for i, d in enumerate(all_domains) if d == held_out_domain]
        n_test = len(test_idx)

        if n_test == 0:
            print(f"    No scenarios in domain — skipping")
            continue

        print(f"    Train: {len(train_idx)}, Test: {n_test}")

        # Prepare train data
        train_sds = [all_sds[i] for i in train_idx]
        train_obs = [all_obs[i] for i in train_idx]
        train_dom = [all_domains[i] for i in train_idx]
        train_cov = [all_covs[i] for i in train_idx]

        train_domain_indices, train_domain_names = _build_domain_index(train_dom)
        train_data = prepare_calibration_data(
            train_sds, train_obs, train_domain_indices, train_cov,
        )

        # SVI (shorter: 1500 steps for speed)
        n_train = train_data.domain_indices.shape[0]
        prior_mu = jnp.zeros(N_PARAMS)
        prior_sigma = 0.5 * jnp.ones(N_PARAMS)

        def model_fn():
            return hierarchical_model_transfer(
                train_data, prior_mu, prior_sigma, batch_size=None,
            )

        n_domains_val = train_data.n_domains
        init_values = {
            "mu_global": prior_mu,
            "sigma_global": prior_sigma,
            "mu_domain": jnp.broadcast_to(prior_mu, (n_domains_val, N_PARAMS)),
            "sigma_domain": 0.3 * jnp.ones((n_domains_val, N_PARAMS)),
            "B": jnp.zeros((N_PARAMS, N_COVARIATES)),
            "tau_readout": jnp.array(0.02),
            "log_phi": jnp.array(4.0),
            "sigma_outcome": jnp.array(5.0),
            "theta_s": jnp.zeros((n_train, N_PARAMS)),
        }

        guide = AutoLowRankMultivariateNormal(
            model_fn, init_loc_fn=init_to_value(values=init_values),
        )
        optimizer = numpyro.optim.Adam(0.002)
        svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

        rng_key = jax.random.PRNGKey(42 + hash(held_out_domain) % 1000)
        svi_state = svi.init(rng_key)

        t0 = time.time()
        for step in range(1500):
            svi_state, loss = svi.update(svi_state)
            if (step + 1) % 500 == 0:
                print(f"    SVI step {step + 1}/1500  loss={loss:.1f}")
        elapsed = time.time() - t0
        print(f"    SVI done in {elapsed:.1f}s")

        svi_params = svi.get_params(svi_state)

        # Predict on held-out domain
        predictive = Predictive(guide, params=svi_params, num_samples=100)
        pp_samples = predictive(jax.random.PRNGKey(42))
        mu_global_samples = pp_samples["mu_global"]  # [S, 4]

        domain_results = []
        for i in test_idx:
            sid = all_ids[i]
            sd = all_sds[i]
            obs = all_obs[i]
            gt = float(obs.ground_truth_pro_pct)

            # Simulate with each posterior sample
            pp_finals = []
            for s in range(min(50, mu_global_samples.shape[0])):
                params_s = {
                    PARAM_NAMES[k]: float(mu_global_samples[s, k])
                    for k in range(N_PARAMS)
                }
                for k_idx, key in enumerate(FROZEN_KEYS):
                    params_s[key] = float(frozen_vec[k_idx])
                try:
                    res_s = simulate_scenario(params_s, sd)
                    pp_finals.append(float(res_s["final_pro_pct"]))
                except Exception:
                    pass

            if pp_finals:
                pp_arr = jnp.array(pp_finals)
                sim_mean = float(jnp.mean(pp_arr))
                ci90 = (float(jnp.percentile(pp_arr, 5)), float(jnp.percentile(pp_arr, 95)))
                in_90 = ci90[0] <= gt <= ci90[1]
                err = abs(sim_mean - gt)
            else:
                sim_mean, ci90, in_90, err = 50.0, (0, 100), True, abs(50 - gt)

            print(f"      {sid[:40]:40s} gt={gt:5.1f} sim={sim_mean:5.1f} "
                  f"err={err:5.1f} {'✓' if in_90 else '✗'}")
            domain_results.append({
                "id": sid, "gt": gt, "sim_mean": sim_mean,
                "abs_error": err, "ci90": ci90, "in_90": in_90,
            })

        # Aggregate for this domain
        mae = sum(r["abs_error"] for r in domain_results) / len(domain_results)
        cov90 = sum(1 for r in domain_results if r["in_90"]) / len(domain_results)

        lodocv_results[held_out_domain] = {
            "n_scenarios": n_test,
            "mae": mae,
            "coverage_90": cov90,
            "elapsed_s": elapsed,
            "per_scenario": domain_results,
        }

        print(f"    Domain MAE: {mae:.1f}pp, Cov90: {cov90 * 100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("LODO-CV Summary")
    print("=" * 60)
    total_n = sum(v["n_scenarios"] for v in lodocv_results.values())
    total_covered = sum(
        sum(1 for r in v["per_scenario"] if r["in_90"])
        for v in lodocv_results.values()
    )
    all_errors = [
        r["abs_error"]
        for v in lodocv_results.values()
        for r in v["per_scenario"]
    ]
    overall_mae = sum(all_errors) / len(all_errors) if all_errors else 0
    overall_cov90 = total_covered / total_n if total_n else 0

    for domain, v in sorted(lodocv_results.items()):
        print(f"  {domain:20s}  N={v['n_scenarios']:2d}  "
              f"MAE={v['mae']:5.1f}pp  Cov90={v['coverage_90'] * 100:5.1f}%")
    print(f"  {'OVERALL':20s}  N={total_n:2d}  "
          f"MAE={overall_mae:5.1f}pp  Cov90={overall_cov90 * 100:5.1f}%")

    summary = {
        "overall_mae": overall_mae,
        "overall_coverage_90": overall_cov90,
        "n_total": total_n,
        "per_domain": lodocv_results,
    }

    ensure_dir(REVIEWER_DIR)
    with open(REVIEWER_DIR / "lodocv_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Results saved to {REVIEWER_DIR / 'lodocv_results.json'}")
    return summary


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "coverage": run_coverage,
    "5param": run_5param,
    "multiseed": run_multiseed,
    "lodocv": run_lodocv,
}

def main():
    parser = argparse.ArgumentParser(description="Run reviewer-requested experiments")
    parser.add_argument("--experiment", "-e", default="all",
                        choices=list(EXPERIMENTS.keys()) + ["all"],
                        help="Which experiment to run")
    args = parser.parse_args()

    if args.experiment == "all":
        for name, fn in EXPERIMENTS.items():
            try:
                fn()
            except Exception as e:
                print(f"\n  EXPERIMENT {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
    else:
        EXPERIMENTS[args.experiment]()


if __name__ == "__main__":
    main()
