"""Hierarchical model v2 with model discrepancy term.

Extends v1 (hierarchical_model.py) with an additive discrepancy δ_s
that captures systematic simulator bias per scenario.

Key difference from v1:
  q_corrected(t) = sigmoid(logit(q_sim(t)) + δ_s)

  where δ_s ~ Normal(δ_d, σ_δ_within)
        δ_d ~ Normal(0, σ_δ_between)

This separates observation noise (σ_outcome) from structural model
discrepancy (δ_s), allowing the model to explain large systematic errors
(WeWork +49pp, Chile -41pp) without inflating observation noise.

Usage:
    from src.inference.hierarchical_model_v2 import (
        hierarchical_model_v2, run_svi_v2, run_phase_bc_v2,
    )
"""

import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.nn import sigmoid

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, init_to_value
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal

import numpy as np

# Import everything from v1 — we reuse data prep, simulation, observation model
from .hierarchical_model import (
    CalibrationData, prepare_calibration_data,
    PARAM_NAMES, COVARIATE_NAMES, N_PARAMS, N_COVARIATES,
    FROZEN_KEYS, _simulate_batch, _SCENARIO_AXES,
    _summarize, extract_posteriors as _extract_posteriors_v1,
)
from .calibration_pipeline import (
    load_empirical_scenario, EMPIRICAL_DIR, RESULTS_DIR,
    _build_domain_index, _stratified_split, _DROP_SCENARIOS,
)
from ..dynamics.opinion_dynamics_jax import ScenarioData, simulate_scenario
from ..dynamics.param_utils import get_default_frozen_params
from ..observation.observation_model import _beta_binomial_logpmf


# ── Constants ────────────────────────────────────────────────────

V2_RESULTS_DIR = RESULTS_DIR / "v2_discrepancy"


# ── Likelihood with discrepancy ──────────────────────────────────

def _likelihood_one_v2(pro_fraction, final_pro_pct, delta_s,
                       obs_pro_pcts, obs_sample_sizes, obs_verified_mask,
                       obs_gt_pro_pct, has_verified,
                       log_phi, log_sigma_outcome):
    """Compute log-likelihood for one scenario with discrepancy correction.

    delta_s is applied in logit space:
      q_corrected = sigmoid(logit(q_sim) + delta_s)
    """
    phi = jnp.exp(log_phi)
    sigma = jnp.exp(log_sigma_outcome)

    eps = 1e-4

    # Apply discrepancy in logit space to final pro_pct
    q_sim_final = jnp.clip(final_pro_pct / 100.0, eps, 1.0 - eps)
    logit_final = jnp.log(q_sim_final / (1.0 - q_sim_final))
    q_corrected_final = sigmoid(logit_final + delta_s) * 100.0

    # Apply discrepancy to per-round pro_fraction
    q_sim = jnp.clip(pro_fraction, eps, 1.0 - eps)
    logit_q = jnp.log(q_sim / (1.0 - q_sim))
    q_corrected = sigmoid(logit_q + delta_s)

    # ── Outcome-only: Normal on corrected final outcome ──
    ll_outcome = (
        -0.5 * jnp.log(2.0 * jnp.pi)
        - jnp.log(sigma)
        - 0.5 * ((obs_gt_pro_pct - q_corrected_final) / sigma) ** 2
    )

    # ── Full: BetaBinomial on corrected verified rounds ──
    q_c = jnp.clip(q_corrected, eps, 1.0 - eps)
    alpha = q_c * phi
    beta = (1.0 - q_c) * phi

    k = jnp.round(obs_pro_pcts / 100.0 * obs_sample_sizes)
    k = jnp.where(jnp.isnan(k), 0.0, k)
    n = obs_sample_sizes

    ll_bb_per_round = _beta_binomial_logpmf(k, n, alpha, beta)
    ll_bb_per_round = jnp.where(jnp.isfinite(ll_bb_per_round), ll_bb_per_round, 0.0)

    mask = obs_verified_mask & (obs_sample_sizes > 0)
    ll_full = jnp.sum(jnp.where(mask, ll_bb_per_round, 0.0))

    return jnp.where(has_verified, ll_full, ll_outcome)


_likelihood_batch_v2 = jax.vmap(
    _likelihood_one_v2,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None),
)


# ── NumPyro Model v2 ─────────────────────────────────────────────

def hierarchical_model_v2(
    data: CalibrationData,
    prior_mu_global: jnp.ndarray = None,
    prior_sigma_global: jnp.ndarray = None,
    batch_size=None,
):
    """Hierarchical model with discrepancy term δ_s.

    Structure:
      Level 1: μ_global, σ_global (global hyperpriors)
      Level 2: μ_d, σ_d (per domain)
      Level 3: θ_s = μ_d + B @ x_s + ε_s (per scenario)

      Discrepancy:
        δ_d ~ Normal(0, σ_δ_between)  (per domain)
        δ_s ~ Normal(δ_d, σ_δ_within) (per scenario)
        q_corrected = sigmoid(logit(q_sim) + δ_s)

    Args:
        data: CalibrationData (pre-padded, stacked).
        prior_mu_global: Informative prior from Phase A (or None for flat).
        prior_sigma_global: Informative prior from Phase A (or None for flat).
        batch_size: Mini-batch size (None = full batch).
    """
    n_scenarios = data.domain_indices.shape[0]
    n_domains = data.n_domains

    # ── Level 1: Global hyperpriors ──
    if prior_mu_global is not None:
        mu_global = numpyro.sample(
            "mu_global",
            dist.Normal(prior_mu_global, prior_sigma_global).to_event(1),
        )
    else:
        mu_global = numpyro.sample(
            "mu_global",
            dist.Normal(jnp.zeros(N_PARAMS), 1.0).to_event(1),
        )
    sigma_global = numpyro.sample(
        "sigma_global",
        dist.HalfNormal(0.5 * jnp.ones(N_PARAMS)).to_event(1),
    )

    # ── Level 2: Domain parameters ──
    with numpyro.plate("domains", n_domains):
        mu_domain = numpyro.sample(
            "mu_domain",
            dist.Normal(mu_global, sigma_global).to_event(1),
        )
        sigma_domain = numpyro.sample(
            "sigma_domain",
            dist.HalfNormal(0.3 * jnp.ones(N_PARAMS)).to_event(1),
        )

    # ── Covariate regression ──
    B = numpyro.sample(
        "B",
        dist.Normal(
            jnp.zeros((N_PARAMS, N_COVARIATES)),
            0.5 * jnp.ones((N_PARAMS, N_COVARIATES)),
        ).to_event(2),
    )

    # ── Observation model parameters ──
    tau_readout = numpyro.sample(
        "tau_readout", dist.LogNormal(-1.5, 0.5),
    )
    log_phi = numpyro.sample("log_phi", dist.Normal(4.0, 1.0))
    # Tight prior on σ_outcome — forces structural bias into δ, not noise
    sigma_outcome = numpyro.sample(
        "sigma_outcome", dist.HalfNormal(3.0),
    )
    log_sigma_outcome = numpyro.deterministic(
        "log_sigma_outcome", jnp.log(sigma_outcome + 1e-8),
    )
    numpyro.deterministic("phi", jnp.exp(log_phi))

    # ── Discrepancy hyperpriors ──
    sigma_delta_between = numpyro.sample(
        "sigma_delta_between", dist.HalfNormal(0.3),
    )
    sigma_delta_within = numpyro.sample(
        "sigma_delta_within", dist.HalfNormal(0.3),
    )

    # Per-domain discrepancy mean
    with numpyro.plate("domains_delta", n_domains):
        delta_domain = numpyro.sample(
            "delta_domain", dist.Normal(0.0, sigma_delta_between),
        )  # [n_domains]

    # ── Level 3: Scenario parameters + discrepancy + likelihood ──
    plate_kwargs = {"name": "scenarios", "size": n_scenarios}
    if batch_size is not None and batch_size < n_scenarios:
        plate_kwargs["subsample_size"] = batch_size

    with numpyro.plate(**plate_kwargs) as idx:
        d = data.domain_indices[idx]
        x = data.covariates[idx]

        # Scenario-level dynamics params
        mu_s = mu_domain[d] + x @ B.T
        sigma_s = sigma_domain[d]
        theta_s = numpyro.sample(
            "theta_s", dist.Normal(mu_s, sigma_s).to_event(1),
        )

        # Scenario-level discrepancy
        delta_s = numpyro.sample(
            "delta_s", dist.Normal(delta_domain[d], sigma_delta_within),
        )  # [batch]

        # ── Simulate batch ──
        batch_sd = ScenarioData(
            initial_positions=data.initial_positions[idx],
            agent_types=data.agent_types[idx],
            agent_rigidities=data.agent_rigidities[idx],
            agent_tolerances=data.agent_tolerances[idx],
            events=data.events[idx],
            llm_shifts=data.llm_shifts[idx],
            interaction_matrix=data.interaction_matrices[idx],
        )
        batch_mask = data.agent_masks[idx]
        batch_nrr = data.n_real_rounds[idx]

        pro_fracs, final_pcts = _simulate_batch(
            theta_s, data.frozen_vec, batch_sd,
            batch_mask, tau_readout, batch_nrr,
        )

        # Corrected final pct (for deterministic tracking)
        eps = 1e-4
        q_sim = jnp.clip(final_pcts / 100.0, eps, 1.0 - eps)
        logit_q = jnp.log(q_sim / (1.0 - q_sim))
        corrected_pcts = sigmoid(logit_q + delta_s) * 100.0

        numpyro.deterministic("sim_final_pcts", final_pcts)
        numpyro.deterministic("corrected_final_pcts", corrected_pcts)
        numpyro.deterministic("delta_s_vals", delta_s)

        # ── Likelihood with discrepancy ──
        ll = _likelihood_batch_v2(
            pro_fracs, final_pcts, delta_s,
            data.obs_pro_pcts[idx],
            data.obs_sample_sizes[idx],
            data.obs_verified_masks[idx],
            data.obs_gt_pro_pcts[idx],
            data.obs_has_verified[idx],
            log_phi, log_sigma_outcome,
        )
        numpyro.factor("likelihood", ll)


# ── SVI Inference v2 ─────────────────────────────────────────────

def run_svi_v2(
    data: CalibrationData,
    prior_mu_global: jnp.ndarray = None,
    prior_sigma_global: jnp.ndarray = None,
    n_steps: int = 2000,
    lr: float = 0.005,
    batch_size: int = None,
    seed: int = 42,
    log_every: int = 100,
):
    """Run SVI for the v2 model with discrepancy."""
    n_scenarios = data.domain_indices.shape[0]
    n_domains = data.n_domains
    if batch_size is not None and batch_size >= n_scenarios:
        batch_size = None

    def model_fn():
        return hierarchical_model_v2(
            data, prior_mu_global, prior_sigma_global,
            batch_size=batch_size,
        )

    # Initial values including discrepancy params
    init_values = {
        "mu_global": jnp.zeros(N_PARAMS),
        "sigma_global": 0.5 * jnp.ones(N_PARAMS),
        "mu_domain": jnp.zeros((n_domains, N_PARAMS)),
        "sigma_domain": 0.3 * jnp.ones((n_domains, N_PARAMS)),
        "B": jnp.zeros((N_PARAMS, N_COVARIATES)),
        "tau_readout": jnp.array(0.02),
        "log_phi": jnp.array(4.0),
        "sigma_outcome": jnp.array(3.0),
        "sigma_delta_between": jnp.array(0.15),
        "sigma_delta_within": jnp.array(0.15),
        "delta_domain": jnp.zeros(n_domains),
        "theta_s": jnp.zeros((n_scenarios, N_PARAMS)),
        "delta_s": jnp.zeros(n_scenarios),
    }

    if prior_mu_global is not None:
        init_values["mu_global"] = prior_mu_global

    guide = AutoLowRankMultivariateNormal(
        model_fn, init_loc_fn=init_to_value(values=init_values),
    )
    optimizer = numpyro.optim.Adam(lr)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(seed)
    svi_state = svi.init(rng_key)

    losses = []
    for step in range(n_steps):
        svi_state, loss = svi.update(svi_state)
        losses.append(float(loss))

        if log_every and (step + 1) % log_every == 0:
            recent = losses[-min(50, len(losses)):]
            avg_loss = sum(recent) / len(recent)
            print(f"  Step {step+1:5d}/{n_steps}  loss={loss:.2f}  avg50={avg_loss:.2f}",
                  flush=True)

    svi_result = svi.get_params(svi_state)
    losses = jnp.array(losses)
    return svi_result, losses, guide


# ── Posterior Extraction v2 ──────────────────────────────────────

def extract_posteriors_v2(
    guide,
    svi_params,
    domain_names: list[str],
    scenario_ids: list[str],
    n_posterior_samples: int = 1000,
    seed: int = 0,
) -> dict:
    """Extract posteriors including discrepancy parameters."""
    predictive = Predictive(guide, params=svi_params,
                            num_samples=n_posterior_samples)
    rng = jax.random.PRNGKey(seed)
    samples = predictive(rng)

    # Start with v1 extraction
    result = {}

    # ── Global ──
    result["global"] = {
        "mu_global": _summarize(samples["mu_global"]),
        "sigma_global": _summarize(samples["sigma_global"]),
    }

    # ── Domains ──
    result["domains"] = {}
    mu_d = samples["mu_domain"]
    sigma_d = samples["sigma_domain"]
    delta_d = samples["delta_domain"]
    for i, name in enumerate(domain_names):
        result["domains"][name] = {
            "mu_d": _summarize(mu_d[:, i, :]),
            "sigma_d": _summarize(sigma_d[:, i, :]),
            "delta_d": _summarize(delta_d[:, i]),
        }

    # ── Scenarios ──
    result["scenarios"] = {}
    theta = samples["theta_s"]
    delta_s_samples = samples["delta_s"]
    for i, sid in enumerate(scenario_ids):
        result["scenarios"][sid] = {
            "theta_s": _summarize(theta[:, i, :]),
            "delta_s": _summarize(delta_s_samples[:, i]),
        }

    # ── Covariates ──
    if "B" in samples:
        from .hierarchical_model import COVARIATE_NAMES as COV_NAMES
        B_samples = samples["B"]
        B_lo = jnp.percentile(B_samples, 2.5, axis=0)
        B_hi = jnp.percentile(B_samples, 97.5, axis=0)
        B_mean = jnp.mean(B_samples, axis=0)
        significant = []
        for p in range(N_PARAMS):
            for c in range(N_COVARIATES):
                if (B_lo[p, c] > 0) or (B_hi[p, c] < 0):
                    significant.append({
                        "param": PARAM_NAMES[p],
                        "covariate": COV_NAMES[c],
                        "effect": float(B_mean[p, c]),
                        "ci95": (float(B_lo[p, c]), float(B_hi[p, c])),
                    })
        result["covariates"] = {
            "B": _summarize(B_samples),
            "significant": significant,
        }

    # ── Discrepancy hyperparams ──
    result["discrepancy"] = {
        "sigma_delta_between": _summarize(samples["sigma_delta_between"]),
        "sigma_delta_within": _summarize(samples["sigma_delta_within"]),
    }

    # ── Observation params ──
    result["observation_params"] = {}
    if "tau_readout" in samples:
        result["observation_params"]["tau_readout"] = _summarize(
            samples["tau_readout"])
    if "log_phi" in samples:
        result["observation_params"]["phi"] = _summarize(
            jnp.exp(samples["log_phi"]))
    if "log_sigma_outcome" in samples:
        result["observation_params"]["sigma_outcome"] = _summarize(
            jnp.exp(samples["log_sigma_outcome"]))

    return result, samples


# ── Phase B+C Runner ─────────────────────────────────────────────

def run_phase_bc_v2(
    n_svi_steps: int = 2000,
    n_pp_samples: int = 200,
    lr: float = 0.005,
    seed: int = 42,
):
    """Run Phase B (empirical fine-tuning) and Phase C (validation) with v2 model.

    Loads synthetic prior from Phase A results, loads empirical scenarios,
    runs SVI with discrepancy model, and validates.
    """
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── Load synthetic prior from Phase A ──
    prior_path = RESULTS_DIR / "synthetic_prior.json"
    if prior_path.exists():
        with open(prior_path) as f:
            prior = json.load(f)
        prior_mu = jnp.array(prior["mu_global"])
        prior_sigma = jnp.array(prior["sigma_global"])
        # Widen sigma slightly for fine-tuning flexibility (same as v1)
        prior_sigma = jnp.maximum(prior_sigma, 0.3)
        print(f"Loaded Phase A prior: μ={list(np.round(prior_mu, 3))}")
    else:
        print("WARNING: No Phase A prior found, using flat prior")
        prior_mu = None
        prior_sigma = None

    # ── Load empirical scenarios (same logic as calibration_pipeline Phase B) ──
    import glob as glob_mod
    json_files = sorted(glob_mod.glob(str(EMPIRICAL_DIR / "*.json")))
    json_files = [f for f in json_files if not f.endswith("manifest.json")
                  and not f.endswith(".meta.json")]

    scenarios_data = []
    observations = []
    covariates_list = []
    domain_list = []
    scenario_ids = []
    failed = 0

    for path in json_files:
        sid = Path(path).stem
        if sid in _DROP_SCENARIOS:
            print(f"  Dropping non-independent: {sid}")
            continue
        try:
            sd, obs, cov, domain = load_empirical_scenario(path, seed=seed)
            scenarios_data.append(sd)
            observations.append(obs)
            covariates_list.append([
                cov.get("initial_polarization", 0.5),
                cov.get("event_volatility", 0.5),
                cov.get("elite_concentration", 0.5),
                cov.get("institutional_trust", 0.5),
                cov.get("undecided_share", 0.1),
            ])
            domain_list.append(domain)
            scenario_ids.append(sid)
        except Exception as e:
            failed += 1
            print(f"  Warning: failed to load {Path(path).name}: {e}")

    n_total = len(scenarios_data)
    print(f"Loaded {n_total} empirical scenarios (failed: {failed})")

    # Stratified train/test split (same as v1)
    train_idx, test_idx = _stratified_split(scenario_ids, domain_list, 0.2, seed)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"Test: {[scenario_ids[i] for i in test_idx]}")

    # ── Prepare calibration data (train set) ──
    train_sds = [scenarios_data[i] for i in train_idx]
    train_obs = [observations[i] for i in train_idx]
    train_dom = [domain_list[i] for i in train_idx]
    train_covs = [covariates_list[i] for i in train_idx]
    train_ids_list = [scenario_ids[i] for i in train_idx]

    train_domain_indices, train_domain_names = _build_domain_index(train_dom)

    cal_data = prepare_calibration_data(
        train_sds, train_obs, train_domain_indices, train_covs,
    )

    # Build full domain index (for test scenarios too)
    all_domain_indices, all_domain_names = _build_domain_index(domain_list)
    domain_to_idx = {d: i for i, d in enumerate(all_domain_names)}

    print(f"Domains: {train_domain_names}")
    print(f"Padded: agents={cal_data.initial_positions.shape[1]}, "
          f"rounds={cal_data.events.shape[1]}")

    # ── Phase B: SVI with discrepancy model ──
    print(f"\n{'='*60}")
    print(f"Phase B (v2): SVI with discrepancy model, {n_svi_steps} steps")
    print(f"{'='*60}")

    t_b = time.time()
    svi_params, losses, guide = run_svi_v2(
        cal_data,
        prior_mu_global=prior_mu,
        prior_sigma_global=prior_sigma,
        n_steps=n_svi_steps,
        lr=lr,
        seed=seed,
        log_every=100,
    )
    t_b_elapsed = time.time() - t_b
    print(f"Phase B elapsed: {t_b_elapsed:.1f}s")
    print(f"Final loss: {float(losses[-1]):.1f}")

    # ── Extract posteriors ──
    posteriors, samples = extract_posteriors_v2(
        guide, svi_params,
        domain_names=train_domain_names,
        scenario_ids=train_ids_list,
        n_posterior_samples=1000,
        seed=seed,
    )

    # ── Phase C: Validation ──
    print(f"\n{'='*60}")
    print("Phase C (v2): Posterior Predictive Validation")
    print(f"{'='*60}")

    results_per_scenario = []
    rng_pp = np.random.RandomState(seed + 100)

    # Get posterior samples for PP
    mu_d_samples = np.array(samples["mu_domain"])      # [S, D, 4]
    sigma_d_samples = np.array(samples["sigma_domain"])
    B_samples_np = np.array(samples["B"])               # [S, 4, 5]
    tau_samples = np.array(samples["tau_readout"])       # [S]
    delta_d_samples = np.array(samples["delta_domain"])  # [S, D]
    sigma_dw_samples = np.array(samples["sigma_delta_within"])  # [S]

    if "log_sigma_outcome" in samples:
        sigma_out_samples = np.exp(np.array(samples["log_sigma_outcome"]))
    else:
        sigma_out_samples = np.abs(rng_pp.normal(0, 3.0, size=1000))
        sigma_out_samples = np.maximum(sigma_out_samples, 1.0)

    n_pp = min(n_pp_samples, mu_d_samples.shape[0])

    # Build train domain_to_idx for covariate normalization
    train_domain_to_idx = {d: i for i, d in enumerate(train_domain_names)}

    for eval_indices, group_name in [
        (train_idx, "train"),
        (test_idx, "test"),
    ]:
        for i in eval_indices:
            sid = scenario_ids[i]
            domain = domain_list[i]
            d_idx = train_domain_to_idx.get(domain)
            if d_idx is None:
                print(f"  {sid}: domain '{domain}' not in train set, skipping")
                continue
            gt = float(observations[i].ground_truth_pro_pct)

            sd = scenarios_data[i]
            x_s = np.array(covariates_list[i])

            # Normalize covariate (approx — use training stats)
            # For simplicity, use raw covariates (normalization done in prepare_calibration_data)
            # We need to re-normalize with training stats
            train_cov_arr = np.array(train_covs)
            cov_mu = train_cov_arr.mean(axis=0)
            cov_std = train_cov_arr.std(axis=0) + 1e-6
            x_norm = (x_s - cov_mu) / cov_std

            pp_preds = []
            for k in range(n_pp):
                # Sample from hierarchy
                mu_dk = mu_d_samples[k, d_idx, :]   # [4]
                sigma_dk = sigma_d_samples[k, d_idx, :]
                B_k = B_samples_np[k]                # [4, 5]

                mu_sk = mu_dk + B_k @ x_norm
                theta_sk = rng_pp.normal(mu_sk, np.maximum(sigma_dk, 0.01))

                # Sample discrepancy
                delta_dk = delta_d_samples[k, d_idx]
                sigma_dw_k = max(float(sigma_dw_samples[k]), 0.01)
                delta_sk = rng_pp.normal(delta_dk, sigma_dw_k)

                # Build params dict
                params = {
                    PARAM_NAMES[j]: float(theta_sk[j]) for j in range(N_PARAMS)
                }
                frozen = get_default_frozen_params()
                for key in FROZEN_KEYS:
                    params[key] = frozen[key]

                # Simulate
                try:
                    result = simulate_scenario(params, sd)
                    q_final = float(result["final_pro_pct"])
                except Exception:
                    continue

                # Apply discrepancy correction
                eps = 1e-4
                q_clip = max(min(q_final / 100.0, 1.0 - eps), eps)
                logit_q = math.log(q_clip / (1.0 - q_clip))
                q_corrected = 1.0 / (1.0 + math.exp(-(logit_q + delta_sk)))
                q_corrected_pct = q_corrected * 100.0

                # Add observation noise
                sigma_k = max(float(sigma_out_samples[k]), 1.0)
                pred = rng_pp.normal(q_corrected_pct, sigma_k)
                pp_preds.append(pred)

            if len(pp_preds) < 10:
                print(f"  {sid}: too few PP samples ({len(pp_preds)})")
                continue

            pp_arr = np.array(pp_preds)
            pp_mean = pp_arr.mean()
            pp_std = pp_arr.std()
            error = pp_mean - gt

            # Coverage
            ci90 = (np.percentile(pp_arr, 5), np.percentile(pp_arr, 95))
            ci50 = (np.percentile(pp_arr, 25), np.percentile(pp_arr, 75))
            in_90 = ci90[0] <= gt <= ci90[1]
            in_50 = ci50[0] <= gt <= ci50[1]

            # CRPS
            crps_term1 = np.mean(np.abs(pp_arr - gt))
            n_s = len(pp_arr)
            if n_s > 1:
                idx1 = rng_pp.choice(n_s, size=n_s)
                idx2 = rng_pp.choice(n_s, size=n_s)
                crps_term2 = np.mean(np.abs(pp_arr[idx1] - pp_arr[idx2]))
            else:
                crps_term2 = 0.0
            crps = crps_term1 - 0.5 * crps_term2

            results_per_scenario.append({
                "id": sid,
                "domain": domain,
                "group": group_name,
                "gt": gt,
                "sim_mean": float(pp_mean),
                "sim_std": float(pp_std),
                "error": float(error),
                "abs_error": abs(float(error)),
                "crps": float(crps),
                "ci90": (float(ci90[0]), float(ci90[1])),
                "in_90": bool(in_90),
                "ci50": (float(ci50[0]), float(ci50[1])),
                "in_50": bool(in_50),
            })

    # ── Generate report ──
    test_ids_set = {scenario_ids[i] for i in test_idx}
    _generate_v2_report(
        posteriors, results_per_scenario,
        train_domain_names, train_ids_list, test_ids_set,
        float(losses[-1]), t_b_elapsed, n_svi_steps,
    )

    # ── Save results ──
    _save_v2_results(posteriors, results_per_scenario, losses)

    total_elapsed = time.time() - t_start
    print(f"\nTotal v2 pipeline: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    return posteriors, results_per_scenario


# ── Report Generation ────────────────────────────────────────────

def _generate_v2_report(
    posteriors, results, domain_names, train_ids, test_ids,
    final_loss, elapsed_b, n_steps,
):
    """Generate markdown calibration report for v2 model."""
    lines = [
        "# Hierarchical Calibration Report — v2 (with Discrepancy)",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Model Description",
        "",
        "Extension of v1 with additive discrepancy term δ_s in logit space:",
        "- `q_corrected = sigmoid(logit(q_sim) + δ_s)`",
        "- δ_s ~ Normal(δ_d, σ_δ_within) — per scenario",
        "- δ_d ~ Normal(0, σ_δ_between) — per domain",
        "- σ_outcome ~ HalfNormal(3.0) — tight, forces bias into δ",
        "",
        "## Phase B: Empirical Fine-tuning",
        "",
        f"- SVI steps: {n_steps}",
        f"- Final loss: {final_loss:.1f}",
        f"- Elapsed: {elapsed_b:.1f}s",
        "",
        "### Calibrated Parameters",
        "",
    ]

    # Global params
    g = posteriors["global"]
    mu = g["mu_global"]["mean"]
    sigma = g["sigma_global"]["mean"]
    lines.append(f"- μ_global: [{', '.join(f'{float(v):.3f}' for v in mu)}]")
    lines.append(f"- σ_global: [{', '.join(f'{float(v):.3f}' for v in sigma)}]")
    lines.append("")

    # Parameter table
    lines.extend([
        "| Parameter | Mean | CI95 Low | CI95 High |",
        "|---|---|---|---|",
    ])
    for k, name in enumerate(PARAM_NAMES):
        s = g["mu_global"]
        lines.append(
            f"| {name} | {float(s['mean'][k]):.3f} | "
            f"{float(s['ci95_lo'][k]):.3f} | {float(s['ci95_hi'][k]):.3f} |"
        )
    lines.append("")

    # Discrepancy params
    disc = posteriors["discrepancy"]
    lines.extend([
        "### Discrepancy Parameters",
        "",
        f"- σ_δ_between: {float(disc['sigma_delta_between']['mean']):.3f} "
        f"(CI: {float(disc['sigma_delta_between']['ci95_lo']):.3f}, "
        f"{float(disc['sigma_delta_between']['ci95_hi']):.3f})",
        f"- σ_δ_within: {float(disc['sigma_delta_within']['mean']):.3f} "
        f"(CI: {float(disc['sigma_delta_within']['ci95_lo']):.3f}, "
        f"{float(disc['sigma_delta_within']['ci95_hi']):.3f})",
        "",
    ])

    # Per-domain delta
    lines.extend([
        "### Per-Domain Discrepancy (δ_d)",
        "",
        "| Domain | δ_d Mean | CI95 Low | CI95 High | Interpretation |",
        "|---|---|---|---|---|",
    ])
    for name in domain_names:
        d = posteriors["domains"][name].get("delta_d", {})
        if d:
            mean = float(d["mean"])
            lo = float(d["ci95_lo"])
            hi = float(d["ci95_hi"])
            # Interpret: positive δ means sim under-predicts approval
            if lo > 0:
                interp = "Sim under-predicts"
            elif hi < 0:
                interp = "Sim over-predicts"
            else:
                interp = "No significant bias"
            lines.append(f"| {name} | {mean:.3f} | {lo:.3f} | {hi:.3f} | {interp} |")
    lines.append("")

    # Validation metrics
    train_results = [r for r in results if r["group"] == "train"]
    test_results = [r for r in results if r["group"] == "test"]

    lines.extend([
        "## Phase C: Validation",
        "",
        "| Metric | Train | Test |",
        "|---|---|---|",
    ])

    for name, group in [("Train", train_results), ("Test", test_results)]:
        pass  # build metrics below

    def _metrics(group):
        if not group:
            return {"n": 0, "mae": 0, "rmse": 0, "cov90": 0, "cov50": 0,
                    "crps": 0, "med_ae": 0}
        n = len(group)
        errors = [abs(r["error"]) for r in group]
        return {
            "n": n,
            "mae": np.mean(errors),
            "rmse": np.sqrt(np.mean([e**2 for e in errors])),
            "cov90": np.mean([r["in_90"] for r in group]),
            "cov50": np.mean([r["in_50"] for r in group]),
            "crps": np.mean([r["crps"] for r in group]),
            "med_ae": np.median(errors),
        }

    m_train = _metrics(train_results)
    m_test = _metrics(test_results)

    for metric, label in [
        ("n", "n"), ("mae", "mae"), ("rmse", "rmse"),
        ("cov90", "coverage_90"), ("cov50", "coverage_50"),
        ("crps", "mean_crps"), ("med_ae", "median_abs_error"),
    ]:
        lines.append(
            f"| {label} | {m_train[metric]:.3f} | {m_test[metric]:.3f} |"
        )
    lines.append("")

    # Per-scenario table
    lines.extend([
        "### Per-Scenario Results",
        "",
        "| Scenario | Domain | Group | GT | Sim | δ_s | Error | CRPS | 90% CI |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    # Sort by absolute error descending
    for r in sorted(results, key=lambda x: x["abs_error"], reverse=True):
        sid_short = r["id"][:40]
        delta_str = ""
        # Find delta_s for this scenario if available
        if r["id"] in posteriors.get("scenarios", {}):
            ds = posteriors["scenarios"][r["id"]].get("delta_s", {})
            if ds:
                delta_str = f"{float(ds['mean']):.3f}"

        lines.append(
            f"| {sid_short} | {r['domain']} | {r['group']} | "
            f"{r['gt']:.1f} | {r['sim_mean']:.1f} | {delta_str} | "
            f"{r['error']:+.1f} | {r['crps']:.2f} | "
            f"[{r['ci90'][0]:.1f}, {r['ci90'][1]:.1f}] |"
        )
    lines.append("")

    # Diagnostic: check if δ is absorbing too much
    lines.extend([
        "## Diagnostic: Discrepancy Health Check",
        "",
    ])
    sigma_db = float(disc['sigma_delta_between']['mean'])
    sigma_dw = float(disc['sigma_delta_within']['mean'])
    total_delta_scale = math.sqrt(sigma_db**2 + sigma_dw**2)

    if total_delta_scale > 1.0:
        lines.extend([
            f"**WARNING**: Total discrepancy scale = {total_delta_scale:.3f} "
            "(> 1.0 in logit space ≈ 25pp in probability).",
            "The discrepancy may be absorbing too much error — consider tightening "
            "the prior (σ_δ ~ HalfNormal(0.15)).",
            "",
        ])
    elif total_delta_scale > 0.5:
        lines.extend([
            f"**NOTE**: Total discrepancy scale = {total_delta_scale:.3f} "
            "(0.5-1.0 in logit space ≈ 12-25pp). Moderate — check per-scenario δ values.",
            "",
        ])
    else:
        lines.extend([
            f"**OK**: Total discrepancy scale = {total_delta_scale:.3f} "
            "(< 0.5 in logit space ≈ <12pp). Discrepancy is modest, simulator remains primary driver.",
            "",
        ])

    report_path = V2_RESULTS_DIR / "calibration_report_v2.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {report_path}")


def _save_v2_results(posteriors, results, losses):
    """Save v2 posteriors and validation results."""
    # Convert jax arrays to lists for JSON
    def _to_serializable(obj):
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    posteriors_path = V2_RESULTS_DIR / "posteriors_v2.json"
    with open(posteriors_path, "w") as f:
        json.dump(_to_serializable(posteriors), f, indent=2)
    print(f"Saved: {posteriors_path}")

    validation_path = V2_RESULTS_DIR / "validation_results_v2.json"
    with open(validation_path, "w") as f:
        json.dump(_to_serializable(results), f, indent=2)
    print(f"Saved: {validation_path}")

    losses_path = V2_RESULTS_DIR / "loss_history_v2.json"
    with open(losses_path, "w") as f:
        json.dump({"phase_b": np.array(losses).tolist()}, f)
    print(f"Saved: {losses_path}")


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hierarchical model v2 with discrepancy")
    parser.add_argument("--svi-steps", type=int, default=2000)
    parser.add_argument("--pp-samples", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_phase_bc_v2(
        n_svi_steps=args.svi_steps,
        n_pp_samples=args.pp_samples,
        lr=args.lr,
        seed=args.seed,
    )
