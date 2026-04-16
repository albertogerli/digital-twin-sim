"""Hierarchical model v3 — v2 + regime switching for crisis dynamics.

Extends v2 (with discrepancy) by adding regime switching parameters that
allow the simulator to model discontinuous trust-collapse dynamics.

Key difference from v2:
    The simulator calls simulate_with_regimes instead of simulate_scenario.
    This adds ~5 crisis/transition parameters (GLOBAL, not per-scenario).

    Expected effect: financial crisis scenarios (WeWork, SVB, FTX) should
    have smaller |δ_s| because the regime switching captures the dynamics
    that the discrepancy term was absorbing.

Usage:
    from src.inference.hierarchical_model_v3 import (
        hierarchical_model_v3, run_svi_v3,
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

from .hierarchical_model import (
    CalibrationData, prepare_calibration_data,
    PARAM_NAMES, COVARIATE_NAMES, N_PARAMS, N_COVARIATES,
    FROZEN_KEYS, _SCENARIO_AXES, _summarize,
)
from .hierarchical_model_v2 import (
    _likelihood_one_v2, _likelihood_batch_v2,
    V2_RESULTS_DIR,
)
from .calibration_pipeline import (
    load_empirical_scenario, EMPIRICAL_DIR, RESULTS_DIR,
    _build_domain_index, _stratified_split, _DROP_SCENARIOS,
)
from ..dynamics.opinion_dynamics_jax import ScenarioData, simulate_scenario
from ..dynamics.regime_switching import RegimeSwitchingSimulator
from ..dynamics.param_utils import get_default_frozen_params
from ..observation.observation_model import _beta_binomial_logpmf


# ── Constants ────────────────────────────────────────────────────

V3_RESULTS_DIR = RESULTS_DIR / "v3_regime_switching"


# ── Single-scenario simulation with regime switching ──────────

def _simulate_one_v3(
    calibrable_vec,      # [4]
    frozen_vec,          # [4]
    scenario_data,       # ScenarioData
    agent_mask,          # [max_agents]
    tau_readout,         # scalar
    n_real_rounds,       # scalar
    crisis_lambda_mult,  # scalar
    crisis_anchor_supp,  # scalar
    crisis_event_amp,    # scalar
    crisis_herd_amp,     # scalar
    shock_trigger,       # scalar
    institutional_trust, # scalar
):
    """Simulate one scenario with regime switching. Returns (pro_fraction, final_pro_pct)."""
    params = {PARAM_NAMES[i]: calibrable_vec[i] for i in range(N_PARAMS)}
    for i, key in enumerate(FROZEN_KEYS):
        params[key] = frozen_vec[i]

    crisis_params = {
        "lambda_multiplier": crisis_lambda_mult,
        "anchor_suppression": crisis_anchor_supp,
        "event_amplification": crisis_event_amp,
        "herd_amplification": crisis_herd_amp,
        "contagion_speed": 0.8,  # fixed
    }
    transition_params = {
        "shock_trigger_threshold": shock_trigger,
        "velocity_trigger_threshold": 0.1,  # fixed
        "trust_sensitivity": -1.0,  # fixed
        "crisis_duration_mean": 2.5,  # fixed
        "recovery_rate": 0.4,  # fixed
    }

    sim = RegimeSwitchingSimulator(crisis_params, transition_params)
    result = sim.simulate(params, scenario_data, float(institutional_trust))

    trajectories = result["trajectories"]  # [max_rounds, max_agents]

    # Recompute pro_fraction with agent mask and τ_readout
    soft_pro = sigmoid((trajectories - 0.05) / tau_readout)
    soft_against = sigmoid((-trajectories - 0.05) / tau_readout)
    soft_decided = soft_pro + soft_against

    mask = agent_mask[None, :]
    pro_count = jnp.sum(soft_pro * mask, axis=1)
    decided_count = jnp.sum(soft_decided * mask, axis=1) + 1e-8
    pro_fraction = pro_count / decided_count

    final_pro_pct = pro_fraction[n_real_rounds - 1] * 100.0
    return pro_fraction, final_pro_pct


# Batched version: vmap over scenario dimension
# Crisis params are shared (None axis), scenario data varies (0 axis)
_simulate_batch_v3 = jax.vmap(
    _simulate_one_v3,
    in_axes=(
        0,     # calibrable_vec
        None,  # frozen_vec
        _SCENARIO_AXES,  # scenario_data
        0,     # agent_mask
        None,  # tau_readout
        0,     # n_real_rounds
        None,  # crisis_lambda_mult — GLOBAL
        None,  # crisis_anchor_supp — GLOBAL
        None,  # crisis_event_amp — GLOBAL
        None,  # crisis_herd_amp — GLOBAL
        None,  # shock_trigger — GLOBAL
        0,     # institutional_trust — per scenario
    ),
)


# ── NumPyro Model v3 ─────────────────────────────────────────────

def hierarchical_model_v3(
    data: CalibrationData,
    institutional_trust_vec: jnp.ndarray = None,
    prior_mu_global: jnp.ndarray = None,
    prior_sigma_global: jnp.ndarray = None,
    batch_size=None,
):
    """v3 = v2 + regime switching.

    New sampled parameters (GLOBAL — not per-domain or per-scenario):
        crisis_lambda_mult ~ LogNormal(log(3), 0.5)
        crisis_anchor_supp ~ Beta(2, 8) — concentrated near 0.1-0.2
        crisis_event_amp ~ LogNormal(log(2.5), 0.5)
        crisis_herd_amp ~ LogNormal(log(2), 0.5)
        shock_trigger ~ Beta(5, 5) — concentrated near 0.5

    With only ~6 financial crisis scenarios, these are GLOBAL parameters.
    Per-domain crisis params would be unidentifiable.
    """
    n_scenarios = data.domain_indices.shape[0]
    n_domains = data.n_domains

    if institutional_trust_vec is None:
        institutional_trust_vec = jnp.full(n_scenarios, 0.5)

    # ── v2 parameters (identical) ──

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

    with numpyro.plate("domains", n_domains):
        mu_domain = numpyro.sample(
            "mu_domain",
            dist.Normal(mu_global, sigma_global).to_event(1),
        )
        sigma_domain = numpyro.sample(
            "sigma_domain",
            dist.HalfNormal(0.3 * jnp.ones(N_PARAMS)).to_event(1),
        )

    B = numpyro.sample(
        "B",
        dist.Normal(
            jnp.zeros((N_PARAMS, N_COVARIATES)),
            0.5 * jnp.ones((N_PARAMS, N_COVARIATES)),
        ).to_event(2),
    )

    tau_readout = numpyro.sample("tau_readout", dist.LogNormal(-1.5, 0.5))
    log_phi = numpyro.sample("log_phi", dist.Normal(4.0, 1.0))
    sigma_outcome = numpyro.sample("sigma_outcome", dist.HalfNormal(3.0))
    log_sigma_outcome = numpyro.deterministic(
        "log_sigma_outcome", jnp.log(sigma_outcome + 1e-8)
    )
    numpyro.deterministic("phi", jnp.exp(log_phi))

    # Discrepancy (same as v2 — should shrink if regime captures crisis)
    sigma_delta_between = numpyro.sample(
        "sigma_delta_between", dist.HalfNormal(0.3)
    )
    sigma_delta_within = numpyro.sample(
        "sigma_delta_within", dist.HalfNormal(0.3)
    )

    with numpyro.plate("domains_delta", n_domains):
        delta_domain = numpyro.sample(
            "delta_domain", dist.Normal(0.0, sigma_delta_between)
        )

    # ── NEW v3: Crisis regime parameters (GLOBAL) ──

    crisis_lambda_mult = numpyro.sample(
        "crisis_lambda_mult", dist.LogNormal(jnp.log(3.0), 0.5)
    )
    crisis_anchor_supp = numpyro.sample(
        "crisis_anchor_supp", dist.Beta(2.0, 8.0)
    )
    crisis_event_amp = numpyro.sample(
        "crisis_event_amp", dist.LogNormal(jnp.log(2.5), 0.5)
    )
    crisis_herd_amp = numpyro.sample(
        "crisis_herd_amp", dist.LogNormal(jnp.log(2.0), 0.5)
    )
    shock_trigger = numpyro.sample(
        "shock_trigger", dist.Beta(5.0, 5.0)
    )

    # ── Scenario-level ──

    plate_kwargs = {"name": "scenarios", "size": n_scenarios}
    if batch_size is not None and batch_size < n_scenarios:
        plate_kwargs["subsample_size"] = batch_size

    with numpyro.plate(**plate_kwargs) as idx:
        d = data.domain_indices[idx]
        x = data.covariates[idx]

        mu_s = mu_domain[d] + x @ B.T
        sigma_s = sigma_domain[d]
        theta_s = numpyro.sample(
            "theta_s", dist.Normal(mu_s, sigma_s).to_event(1)
        )

        delta_s = numpyro.sample(
            "delta_s", dist.Normal(delta_domain[d], sigma_delta_within)
        )

        # Build scenario data for batch simulation
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
        batch_trust = institutional_trust_vec[idx]

        # Simulate with regime switching
        pro_fracs, final_pcts = _simulate_batch_v3(
            theta_s, data.frozen_vec, batch_sd,
            batch_mask, tau_readout, batch_nrr,
            crisis_lambda_mult, crisis_anchor_supp,
            crisis_event_amp, crisis_herd_amp,
            shock_trigger, batch_trust,
        )

        # Track corrected predictions
        eps = 1e-4
        q_sim = jnp.clip(final_pcts / 100.0, eps, 1.0 - eps)
        logit_q = jnp.log(q_sim / (1.0 - q_sim))
        corrected_pcts = sigmoid(logit_q + delta_s) * 100.0

        numpyro.deterministic("sim_final_pcts", final_pcts)
        numpyro.deterministic("corrected_final_pcts", corrected_pcts)
        numpyro.deterministic("delta_s_vals", delta_s)

        # Likelihood (same as v2)
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


# ── SVI v3 ────────────────────────────────────────────────────────

def run_svi_v3(
    data: CalibrationData,
    institutional_trust_vec: jnp.ndarray = None,
    prior_mu_global: jnp.ndarray = None,
    prior_sigma_global: jnp.ndarray = None,
    n_steps: int = 3000,
    lr: float = 0.002,
    batch_size: int = None,
    seed: int = 42,
    log_every: int = 100,
):
    """Run SVI for the v3 model with regime switching."""
    n_scenarios = data.domain_indices.shape[0]
    n_domains = data.n_domains

    if institutional_trust_vec is None:
        institutional_trust_vec = jnp.full(n_scenarios, 0.5)

    if batch_size is not None and batch_size >= n_scenarios:
        batch_size = None

    def model_fn():
        return hierarchical_model_v3(
            data, institutional_trust_vec,
            prior_mu_global, prior_sigma_global,
            batch_size=batch_size,
        )

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
        # v3 crisis params
        "crisis_lambda_mult": jnp.array(3.0),
        "crisis_anchor_supp": jnp.array(0.15),
        "crisis_event_amp": jnp.array(2.5),
        "crisis_herd_amp": jnp.array(2.0),
        "shock_trigger": jnp.array(0.5),
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
            print(
                f"  Step {step+1:5d}/{n_steps}  loss={loss:.2f}  avg50={avg_loss:.2f}",
                flush=True,
            )

    svi_result = svi.get_params(svi_state)
    return svi_result, jnp.array(losses), guide


# ── Posterior extraction v3 ───────────────────────────────────────

def extract_posteriors_v3(
    guide, svi_params,
    domain_names: list[str],
    scenario_ids: list[str],
    n_posterior_samples: int = 1000,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Extract posteriors including crisis regime parameters."""
    predictive = Predictive(guide, params=svi_params,
                            num_samples=n_posterior_samples)
    rng = jax.random.PRNGKey(seed)
    samples = predictive(rng)

    result = {}

    # Global
    result["global"] = {
        "mu_global": _summarize(samples["mu_global"]),
        "sigma_global": _summarize(samples["sigma_global"]),
    }

    # Domains
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

    # Scenarios
    result["scenarios"] = {}
    theta = samples["theta_s"]
    delta_s_samples = samples["delta_s"]
    for i, sid in enumerate(scenario_ids):
        result["scenarios"][sid] = {
            "theta_s": _summarize(theta[:, i, :]),
            "delta_s": _summarize(delta_s_samples[:, i]),
        }

    # Covariates
    if "B" in samples:
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
                        "covariate": COVARIATE_NAMES[c],
                        "effect": float(B_mean[p, c]),
                        "ci95": (float(B_lo[p, c]), float(B_hi[p, c])),
                    })
        result["covariates"] = {"B": _summarize(B_samples), "significant": significant}

    # Discrepancy
    result["discrepancy"] = {
        "sigma_delta_between": _summarize(samples["sigma_delta_between"]),
        "sigma_delta_within": _summarize(samples["sigma_delta_within"]),
    }

    # Observation params
    result["observation_params"] = {}
    if "tau_readout" in samples:
        result["observation_params"]["tau_readout"] = _summarize(samples["tau_readout"])
    if "log_phi" in samples:
        result["observation_params"]["phi"] = _summarize(jnp.exp(samples["log_phi"]))
    if "log_sigma_outcome" in samples:
        result["observation_params"]["sigma_outcome"] = _summarize(
            jnp.exp(samples["log_sigma_outcome"]))

    # NEW v3: Crisis regime params
    result["crisis_regime"] = {
        "crisis_lambda_mult": _summarize(samples["crisis_lambda_mult"]),
        "crisis_anchor_supp": _summarize(samples["crisis_anchor_supp"]),
        "crisis_event_amp": _summarize(samples["crisis_event_amp"]),
        "crisis_herd_amp": _summarize(samples["crisis_herd_amp"]),
        "shock_trigger": _summarize(samples["shock_trigger"]),
    }

    return result, samples
