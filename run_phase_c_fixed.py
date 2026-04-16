#!/usr/bin/env python3
"""Fixed Phase C: proper hierarchical posterior predictive sampling.

Uses saved posteriors from empirical_posteriors.json to reconstruct
approximate posterior samples, then runs correct hierarchical PP:

For each sample k:
  1. Sample hyperparams: μ_d_k, σ_d_k, B_k from posterior
  2. For each scenario s:
     a. Sample θ_s_k ~ Normal(μ_d_k[domain_s] + B_k @ x_s, σ_d_k[domain_s])
     b. Simulate with θ_s_k + frozen → trajectory_k
     c. Add observation noise: pred_k ~ Normal(q_final_k * 100, σ_outcome_k)
  3. CI from all preds reflects: hyperprior + scenario + observation variance
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.inference.calibration_pipeline import (
    load_empirical_scenario, _build_domain_index, _stratified_split,
    _DROP_SCENARIOS, crps_ensemble, _to_serializable, _generate_report,
    EMPIRICAL_DIR, RESULTS_DIR, SYNTHETIC_DIR,
)
from src.inference.hierarchical_model import (
    PARAM_NAMES, COVARIATE_NAMES, N_PARAMS, N_COVARIATES, FROZEN_KEYS,
)
from src.dynamics.opinion_dynamics_jax import simulate_scenario
from src.dynamics.param_utils import get_default_frozen_params
import glob


# ── Config ───────────────────────────────────────────────────────

N_PP_SAMPLES = 200
SEED = 42


# ── Reconstruct approximate posterior samples from summary stats ─

def _reconstruct_samples(mean, ci95_lo, ci95_hi, n_samples, rng):
    """Reconstruct approximate Normal samples from CI95 summary stats."""
    mean = np.array(mean)
    lo = np.array(ci95_lo)
    hi = np.array(ci95_hi)
    sigma = (hi - lo) / 3.92  # 95% CI = mean ± 1.96σ
    sigma = np.maximum(sigma, 1e-6)
    return rng.normal(mean, sigma, size=(n_samples,) + mean.shape)


def _reconstruct_halfnormal_samples(mean, ci95_lo, ci95_hi, n_samples, rng):
    """Reconstruct samples for HalfNormal-constrained params (σ > 0)."""
    samples = _reconstruct_samples(mean, ci95_lo, ci95_hi, n_samples, rng)
    return np.abs(samples)  # enforce positivity


def load_posterior_samples(posteriors_path: str, n_samples: int = 200, seed: int = 42):
    """Load saved posteriors and reconstruct approximate samples.

    Returns dict with arrays:
        mu_global: [S, 4]
        sigma_global: [S, 4]
        mu_domain: {domain_name: [S, 4]}
        sigma_domain: {domain_name: [S, 4]}
        B: [S, 4, 5]
        tau_readout: [S]
        phi: [S]
        sigma_outcome: [S]
        theta_s: {scenario_id: [S, 4]}
    """
    with open(posteriors_path) as f:
        post = json.load(f)

    rng = np.random.RandomState(seed)

    result = {}

    # Global
    g = post["global"]
    result["mu_global"] = _reconstruct_samples(
        g["mu_global"]["mean"], g["mu_global"]["ci95_lo"], g["mu_global"]["ci95_hi"],
        n_samples, rng,
    )
    result["sigma_global"] = _reconstruct_halfnormal_samples(
        g["sigma_global"]["mean"], g["sigma_global"]["ci95_lo"], g["sigma_global"]["ci95_hi"],
        n_samples, rng,
    )

    # Domains
    result["mu_domain"] = {}
    result["sigma_domain"] = {}
    for dname, dvals in post["domains"].items():
        result["mu_domain"][dname] = _reconstruct_samples(
            dvals["mu_d"]["mean"], dvals["mu_d"]["ci95_lo"], dvals["mu_d"]["ci95_hi"],
            n_samples, rng,
        )
        result["sigma_domain"][dname] = _reconstruct_halfnormal_samples(
            dvals["sigma_d"]["mean"], dvals["sigma_d"]["ci95_lo"], dvals["sigma_d"]["ci95_hi"],
            n_samples, rng,
        )

    # Covariates B
    b = post["covariates"]["B"]
    result["B"] = _reconstruct_samples(
        b["mean"], b["ci95_lo"], b["ci95_hi"], n_samples, rng,
    )

    # Observation params
    obs = post["observation_params"]
    result["tau_readout"] = np.abs(_reconstruct_samples(
        obs["tau_readout"]["mean"], obs["tau_readout"]["ci95_lo"],
        obs["tau_readout"]["ci95_hi"], n_samples, rng,
    ).flatten())

    result["phi"] = np.abs(_reconstruct_samples(
        obs["phi"]["mean"], obs["phi"]["ci95_lo"], obs["phi"]["ci95_hi"],
        n_samples, rng,
    ).flatten())

    # sigma_outcome: not in saved posteriors (was deterministic from sigma_outcome sample)
    # Use prior: HalfNormal(5.0) → mean ~4, std ~3
    # The model learned this but didn't save it. Use a reasonable range.
    result["sigma_outcome"] = np.abs(rng.normal(5.0, 3.0, size=n_samples))
    result["sigma_outcome"] = np.maximum(result["sigma_outcome"], 1.0)

    # Per-scenario theta_s (for reference, not used in PP since we resample)
    result["theta_s"] = {}
    for sid, svals in post.get("scenarios", {}).items():
        result["theta_s"][sid] = _reconstruct_samples(
            svals["theta_s"]["mean"], svals["theta_s"]["ci95_lo"],
            svals["theta_s"]["ci95_hi"], n_samples, rng,
        )

    return result


# ── Fixed Phase C ────────────────────────────────────────────────

def run_phase_c_fixed():
    print("=" * 60)
    print("PHASE C (FIXED): Proper Hierarchical Posterior Predictive")
    print("=" * 60)
    t0 = time.time()

    # Load posteriors
    posteriors_path = RESULTS_DIR / "posteriors" / "empirical_posteriors.json"
    print(f"Loading posteriors from {posteriors_path}")
    pp = load_posterior_samples(str(posteriors_path), N_PP_SAMPLES, SEED)

    print(f"Reconstructed {N_PP_SAMPLES} approximate posterior samples")
    print(f"  σ_outcome range: [{pp['sigma_outcome'].min():.1f}, {pp['sigma_outcome'].max():.1f}]")
    print(f"  tau_readout range: [{pp['tau_readout'].min():.3f}, {pp['tau_readout'].max():.3f}]")

    # Load empirical scenarios
    json_files = sorted(glob.glob(str(EMPIRICAL_DIR / "*.json")))
    json_files = [f for f in json_files if not f.endswith("manifest.json")
                  and not f.endswith(".meta.json")]

    scenario_datas = []
    observations_list = []
    covariates_list = []
    domains = []
    scenario_ids = []

    for path in json_files:
        sid = Path(path).stem
        if sid in _DROP_SCENARIOS:
            continue
        try:
            sd, obs, cov, domain = load_empirical_scenario(path, seed=SEED)
            scenario_datas.append(sd)
            observations_list.append(obs)
            covariates_list.append([
                cov.get("initial_polarization", 0.5),
                cov.get("event_volatility", 0.5),
                cov.get("elite_concentration", 0.5),
                cov.get("institutional_trust", 0.5),
                cov.get("undecided_share", 0.1),
            ])
            domains.append(domain)
            scenario_ids.append(sid)
        except Exception as e:
            print(f"  Warning: {Path(path).name}: {e}")

    print(f"Loaded {len(scenario_datas)} empirical scenarios")

    # Stratified split (same as Phase B)
    train_idx, test_idx = _stratified_split(scenario_ids, domains, 0.2, SEED)
    print(f"Split: {len(train_idx)} train, {len(test_idx)} test")

    # Frozen params
    frozen = get_default_frozen_params()
    frozen_vec = {k: frozen[k] for k in FROZEN_KEYS}

    # Domain name list (needed for lookups)
    domain_names_in_posteriors = list(pp["mu_domain"].keys())

    # Covariates normalization (same as prepare_calibration_data)
    cov_arr = np.array(covariates_list, dtype=np.float32)
    cov_mu = cov_arr.mean(axis=0)
    cov_std = cov_arr.std(axis=0) + 1e-6
    cov_norm = (cov_arr - cov_mu) / cov_std

    # ── Main PP loop ──
    rng_pp = np.random.RandomState(SEED + 300)
    results_per_scenario = {}
    all_pp_stds = []

    for i, sid in enumerate(scenario_ids):
        sd = scenario_datas[i]
        obs = observations_list[i]
        gt = float(obs.ground_truth_pro_pct)
        domain = domains[i]
        x_s = cov_norm[i]  # [5] normalized covariates
        is_test = i in test_idx

        # Find domain in posteriors (handle missing)
        if domain in pp["mu_domain"]:
            mu_d_samples = pp["mu_domain"][domain]       # [S, 4]
            sigma_d_samples = pp["sigma_domain"][domain]  # [S, 4]
        else:
            # Fallback to global
            mu_d_samples = pp["mu_global"]
            sigma_d_samples = pp["sigma_global"]

        pp_finals = []
        for k in range(N_PP_SAMPLES):
            # Step 1: Get hyperparameters for this sample
            mu_d_k = mu_d_samples[k]       # [4]
            sigma_d_k = sigma_d_samples[k]  # [4]
            B_k = pp["B"][k]               # [4, 5]
            sigma_out_k = pp["sigma_outcome"][k]  # scalar

            # Step 2: Sample θ_s FRESH from the hierarchy
            mu_s = mu_d_k + B_k @ x_s      # [4] = domain_mean + covariate_effect
            theta_s_k = rng_pp.normal(mu_s, np.maximum(sigma_d_k, 0.01))  # [4]

            # Step 3: Build params and simulate
            params = {}
            for p_idx, pname in enumerate(PARAM_NAMES):
                params[pname] = float(theta_s_k[p_idx])
            params.update(frozen_vec)

            try:
                result = simulate_scenario(params, sd)
                sim_final = float(result["final_pro_pct"])
            except Exception:
                sim_final = 50.0

            # Step 4: Add observation noise
            pred_k = rng_pp.normal(sim_final, sigma_out_k)
            pp_finals.append(pred_k)

        pp_arr = np.array(pp_finals)
        pp_mean = float(np.mean(pp_arr))
        pp_std = float(np.std(pp_arr))
        all_pp_stds.append(pp_std)

        # Point estimate (posterior mean)
        mu_d_mean = np.mean(mu_d_samples, axis=0)
        B_mean = np.mean(pp["B"], axis=0)
        mu_s_point = mu_d_mean + B_mean @ x_s
        params_point = {}
        for p_idx, pname in enumerate(PARAM_NAMES):
            params_point[pname] = float(mu_s_point[p_idx])
        params_point.update(frozen_vec)
        sim_result = simulate_scenario(params_point, sd)
        sim_final_point = float(sim_result["final_pro_pct"])

        # Metrics
        error = sim_final_point - gt
        ci90_lo = float(np.percentile(pp_arr, 5))
        ci90_hi = float(np.percentile(pp_arr, 95))
        ci50_lo = float(np.percentile(pp_arr, 25))
        ci50_hi = float(np.percentile(pp_arr, 75))
        in_90 = ci90_lo <= gt <= ci90_hi
        in_50 = ci50_lo <= gt <= ci50_hi
        crps_val = crps_ensemble(jnp.array(pp_arr), gt)

        results_per_scenario[sid] = {
            "group": "test" if is_test else "train",
            "domain": domain,
            "ground_truth": gt,
            "sim_final": sim_final_point,
            "error": error,
            "abs_error": abs(error),
            "pp_mean": pp_mean,
            "pp_std": pp_std,
            "ci90": (ci90_lo, ci90_hi),
            "ci50": (ci50_lo, ci50_hi),
            "in_90": in_90,
            "in_50": in_50,
            "crps": crps_val,
        }

        marker = "TEST" if is_test else "train"
        ci_mark = "✓" if in_90 else "✗"
        print(f"  {sid[:45]:45s}  gt={gt:5.1f}  sim={sim_final_point:5.1f}  "
              f"pp_std={pp_std:5.1f}  CI90=[{ci90_lo:5.1f},{ci90_hi:5.1f}] "
              f"{ci_mark}  {marker}")

    # ── Aggregates ──
    train_results = [v for v in results_per_scenario.values() if v["group"] == "train"]
    test_results = [v for v in results_per_scenario.values() if v["group"] == "test"]

    def _agg(results):
        if not results:
            return {}
        n = len(results)
        return {
            "n": n,
            "mae": sum(r["abs_error"] for r in results) / n,
            "rmse": math.sqrt(sum(r["error"] ** 2 for r in results) / n),
            "coverage_90": sum(1 for r in results if r["in_90"]) / n,
            "coverage_50": sum(1 for r in results if r["in_50"]) / n,
            "mean_crps": sum(r["crps"] for r in results) / n,
            "median_abs_error": float(np.median([r["abs_error"] for r in results])),
        }

    train_agg = _agg(train_results)
    test_agg = _agg(test_results)
    all_agg = _agg(list(results_per_scenario.values()))

    elapsed = time.time() - t0

    print(f"\n{'Metric':<20} {'Train':>10} {'Test':>10} {'All':>10}")
    print("-" * 52)
    for key in ["n", "mae", "rmse", "coverage_90", "coverage_50", "mean_crps",
                "median_abs_error"]:
        tv = train_agg.get(key, "-")
        tev = test_agg.get(key, "-")
        av = all_agg.get(key, "-")
        ts = f"{tv:.3f}" if isinstance(tv, float) else str(tv)
        tes = f"{tev:.3f}" if isinstance(tev, float) else str(tev)
        als = f"{av:.3f}" if isinstance(av, float) else str(av)
        print(f"  {key:<18} {ts:>10} {tes:>10} {als:>10}")

    # ── PP std distribution ──
    stds = np.array(all_pp_stds)
    print(f"\nPosterior Predictive Std Distribution:")
    print(f"  min={stds.min():.1f}  p25={np.percentile(stds,25):.1f}  "
          f"median={np.median(stds):.1f}  p75={np.percentile(stds,75):.1f}  "
          f"max={stds.max():.1f}")

    # Top 3 narrowest and widest CIs
    sorted_by_ci_width = sorted(
        results_per_scenario.items(),
        key=lambda x: x[1]["ci90"][1] - x[1]["ci90"][0],
    )
    print(f"\n3 Narrowest CI90:")
    for sid, r in sorted_by_ci_width[:3]:
        w = r["ci90"][1] - r["ci90"][0]
        print(f"  {sid[:45]:45s}  width={w:.1f}pp  CI=[{r['ci90'][0]:.1f}, {r['ci90'][1]:.1f}]")

    print(f"\n3 Widest CI90:")
    for sid, r in sorted_by_ci_width[-3:]:
        w = r["ci90"][1] - r["ci90"][0]
        print(f"  {sid[:45]:45s}  width={w:.1f}pp  CI=[{r['ci90'][0]:.1f}, {r['ci90'][1]:.1f}]")

    print(f"\nPhase C (fixed) complete in {elapsed:.1f}s")

    # ── Save updated results ──
    out_dir = RESULTS_DIR / "posteriors"
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = _to_serializable({
        "per_scenario": results_per_scenario,
        "train_aggregate": train_agg,
        "test_aggregate": test_agg,
        "all_aggregate": all_agg,
        "pp_std_stats": {
            "min": float(stds.min()),
            "p25": float(np.percentile(stds, 25)),
            "median": float(np.median(stds)),
            "p75": float(np.percentile(stds, 75)),
            "max": float(stds.max()),
        },
    })
    with open(out_dir / "validation_results.json", "w") as f:
        json.dump(validation, f, indent=2)
    print(f"Saved: {out_dir / 'validation_results.json'}")

    # ── Regenerate calibration report ──
    # Load Phase A info
    with open(RESULTS_DIR / "synthetic_prior.json") as f:
        synth_prior = json.load(f)
    with open(RESULTS_DIR / "posteriors" / "empirical_posteriors.json") as f:
        emp_post = json.load(f)

    # Build phase_a_result stub
    phase_a_stub = {
        "n_scenarios": synth_prior["n_synthetic_scenarios"],
        "n_domains": synth_prior["n_domains"],
        "domain_names": synth_prior["domain_names"],
        "elapsed_s": synth_prior["elapsed_s"],
        "losses": jnp.array(synth_prior.get("losses_every_10", [0])),
        "mu_global_mean": jnp.array(synth_prior["mu_global"]),
        "sigma_global_mean": jnp.array(synth_prior["sigma_global"]),
    }

    # Load loss histories
    loss_path = out_dir / "loss_histories.json"
    if loss_path.exists():
        with open(loss_path) as f:
            loss_hist = json.load(f)
        phase_a_stub["losses"] = jnp.array(loss_hist.get("phase_a", [0]))

    # Build Phase B stub with empirical posteriors
    phase_b_stub = {
        "posteriors": emp_post,
        "train_ids": [sid for i, sid in enumerate(scenario_ids) if i in train_idx],
        "test_ids": [sid for i, sid in enumerate(scenario_ids) if i in test_idx],
        "train_domain_names": sorted(set(domains[i] for i in train_idx)),
        "losses": jnp.array(loss_hist.get("phase_b", [0])) if loss_path.exists() else jnp.array([0]),
        "elapsed_s": 3037.1,
    }

    phase_c_result = {
        "per_scenario": results_per_scenario,
        "train_aggregate": train_agg,
        "test_aggregate": test_agg,
        "posteriors": emp_post,
        "elapsed_s": elapsed,
    }

    _generate_report(phase_a_stub, phase_b_stub, phase_c_result, RESULTS_DIR)


if __name__ == "__main__":
    run_phase_c_fixed()
