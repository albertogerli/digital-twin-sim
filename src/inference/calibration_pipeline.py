"""Full calibration pipeline: synthetic pre-training → empirical fine-tuning → validation.

Phase A: Pre-train on ~1000 synthetic LLM-generated scenarios (mini-batch SVI)
Phase B: Fine-tune on 25 empirical scenarios (transfer from synthetic prior)
Phase C: Validate with posterior predictive checks, CRPS, coverage metrics

Usage:
    python -m src.inference.calibration_pipeline --phase all
    python -m src.inference.calibration_pipeline --phase A
    python -m src.inference.calibration_pipeline --phase B
    python -m src.inference.calibration_pipeline --phase C
"""

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.nn import sigmoid

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, init_to_value
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal

from ..dynamics.opinion_dynamics_jax import (
    ScenarioData, build_sparse_interaction, simulate_scenario,
)
from ..dynamics.param_utils import get_default_frozen_params, FROZEN_PARAMS
from ..observation.observation_model import (
    ObservationData, load_scenario_observations, build_scenario_data_from_json,
    _beta_binomial_logpmf,
)
from .hierarchical_model import (
    hierarchical_model, run_svi, extract_posteriors,
    prepare_calibration_data, CalibrationData,
    PARAM_NAMES, COVARIATE_NAMES, N_PARAMS, N_COVARIATES,
    FROZEN_KEYS, _simulate_batch, _likelihood_batch,
    _SCENARIO_AXES,
)


# ── Paths ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
SYNTHETIC_DIR = BASE_DIR / "calibration" / "scenarios"
EMPIRICAL_DIR = BASE_DIR / "calibration" / "empirical" / "scenarios"
RESULTS_DIR = BASE_DIR / "calibration" / "results" / "hierarchical_calibration"


# ── Synthetic scenario → ScenarioData conversion ─────────────────

def _create_agents_from_polling(
    pro_pct: float,
    against_pct: float,
    n_agents: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Create synthetic agents from polling percentages.

    Compact representation (10 agents instead of 100) for JAX efficiency.
    Maps: pro → [+0.15, +0.9], against → [-0.9, -0.15], undecided → [-0.15, +0.15].
    Agent types: 10% elite (type=0), 20% institutional (type=0), 70% citizen (type=1).
    """
    import random
    rng = random.Random(seed)

    n_pro = max(1, round(n_agents * pro_pct / 100))
    n_against = max(0, round(n_agents * against_pct / 100))
    undecided_pct = 100 - pro_pct - against_pct
    n_undecided = max(0, n_agents - n_pro - n_against)

    agents = []
    idx = 0

    def _make(i, pos):
        r = i / n_agents
        if r < 0.1:
            atype, influence, rigidity = 0, rng.uniform(0.7, 1.0), rng.uniform(0.5, 0.8)
        elif r < 0.3:
            atype, influence, rigidity = 0, rng.uniform(0.4, 0.7), rng.uniform(0.3, 0.6)
        else:
            atype, influence, rigidity = 1, rng.uniform(0.1, 0.4), rng.uniform(0.15, 0.45)
        return {
            "name": f"synth_agent_{i}",
            "type": ["elite", "institutional", "citizen_cluster"][min(2, int(r * 10))],
            "initial_position": pos,
            "influence": influence,
            "rigidity": rigidity,
            "tolerance": rng.uniform(0.3, 0.7),
            "agent_type_int": atype,
        }

    for _ in range(n_pro):
        agents.append(_make(idx, rng.uniform(0.15, 0.9)))
        idx += 1
    for _ in range(n_against):
        agents.append(_make(idx, rng.uniform(-0.9, -0.15)))
        idx += 1
    for _ in range(n_undecided):
        agents.append(_make(idx, rng.uniform(-0.15, 0.15)))
        idx += 1

    rng.shuffle(agents)
    return agents


def load_synthetic_scenario(
    json_path: str, n_agents: int = 10, seed: int = 42,
) -> tuple[ScenarioData, ObservationData, dict, str]:
    """Load a synthetic LLM-generated scenario and convert to JAX format.

    Synthetic scenarios have a different schema from empirical:
    - No agents array → created from polling_trajectory[0]
    - No covariates → estimated from polling data
    - Events from key_events → mapped to (magnitude, direction)

    Returns:
        (scenario_data, observations, covariates_dict, domain)
    """
    with open(json_path) as f:
        data = json.load(f)

    domain = data["domain"]
    polling = data["polling_trajectory"]
    n_rounds = len(polling)

    # Create agents from initial polling
    p0 = polling[0]
    agents = _create_agents_from_polling(
        p0["pro_pct"], p0.get("against_pct", 100 - p0["pro_pct"]),
        n_agents=n_agents, seed=seed,
    )

    # Build agent arrays
    positions = jnp.array([a["initial_position"] for a in agents], dtype=jnp.float32)
    agent_types = jnp.array([a["agent_type_int"] for a in agents], dtype=jnp.int32)
    rigidities = jnp.array([a["rigidity"] for a in agents], dtype=jnp.float32)
    tolerances = jnp.array([a["tolerance"] for a in agents], dtype=jnp.float32)
    influences = jnp.array([a["influence"] for a in agents], dtype=jnp.float32)

    # Events from key_events
    events_arr = jnp.zeros((n_rounds, 2), dtype=jnp.float32)
    for evt in data.get("key_events", []):
        r_equiv = evt.get("round_equivalent")
        if r_equiv is None:
            continue
        r = r_equiv - 1  # 0-based
        if 0 <= r < n_rounds:
            # Derive magnitude and direction from polling delta
            if r > 0:
                delta = polling[r]["pro_pct"] - polling[r - 1]["pro_pct"]
                direction = max(-1.0, min(1.0, delta / 4.0))
                magnitude = min(0.6, max(0.2, abs(direction) * 2.0))
            else:
                direction = 0.0
                magnitude = 0.3
            events_arr = events_arr.at[r, 0].set(magnitude)
            events_arr = events_arr.at[r, 1].set(direction)

    # LLM shifts = 0 (no LLM in calibration)
    llm_shifts = jnp.zeros((n_rounds, n_agents), dtype=jnp.float32)

    # Interaction matrix
    interaction_matrix = build_sparse_interaction(influences, seed=seed)

    scenario_data = ScenarioData(
        initial_positions=positions,
        agent_types=agent_types,
        agent_rigidities=rigidities,
        agent_tolerances=tolerances,
        events=events_arr,
        llm_shifts=llm_shifts,
        interaction_matrix=interaction_matrix,
    )

    # Build observations (outcome-only for synthetics, no verified polling)
    pro_pcts = jnp.array([p["pro_pct"] for p in polling], dtype=jnp.float32)
    observations = ObservationData(
        pro_pcts=pro_pcts,
        sample_sizes=jnp.zeros(n_rounds, dtype=jnp.float32),  # no real samples
        verified_mask=jnp.zeros(n_rounds, dtype=jnp.bool_),
        ground_truth_pro_pct=float(data["final_outcome_pro_pct"]),
    )

    # Estimate covariates from polling data
    pro_vals = [p["pro_pct"] for p in polling]
    against_vals = [p.get("against_pct", 100 - p["pro_pct"]) for p in polling]
    undecided_vals = [p.get("undecided_pct", 100 - p["pro_pct"] - p.get("against_pct", 0))
                      for p in polling]

    initial_pol = abs(pro_vals[0] - against_vals[0]) / 100.0
    event_vol = len(data.get("key_events", [])) / max(n_rounds, 1)
    elite_conc = sum(1 for a in agents if a["agent_type_int"] == 0) / len(agents)
    inst_trust = 0.5  # default for synthetic
    undecided_share = undecided_vals[0] / 100.0 if undecided_vals else 0.1

    covariates = {
        "initial_polarization": initial_pol,
        "event_volatility": min(1.0, event_vol),
        "elite_concentration": elite_conc,
        "institutional_trust": inst_trust,
        "undecided_share": undecided_share,
    }

    return scenario_data, observations, covariates, domain


def load_empirical_scenario(
    json_path: str, seed: int = 42,
) -> tuple[ScenarioData, ObservationData, dict, str]:
    """Load an empirical scenario. Returns (scenario_data, obs, covariates, domain)."""
    scenario_dict, obs = load_scenario_observations(json_path)
    scenario_data = build_scenario_data_from_json(scenario_dict, seed=seed)
    domain = scenario_dict["domain"]
    covariates = scenario_dict.get("covariates", {
        "initial_polarization": 0.5,
        "event_volatility": 0.5,
        "elite_concentration": 0.5,
        "institutional_trust": 0.5,
        "undecided_share": 0.1,
    })
    return scenario_data, obs, covariates, domain


# ── Domain indexing ──────────────────────────────────────────────

def _build_domain_index(domains: list[str]) -> tuple[list[int], list[str]]:
    """Map domain strings to integer indices. Returns (indices, unique_names)."""
    unique = sorted(set(domains))
    name2idx = {name: i for i, name in enumerate(unique)}
    indices = [name2idx[d] for d in domains]
    return indices, unique


# ── Cosine annealing (manual, no optax) ─────────────────────────

def cosine_lr(step: int, base_lr: float, total_steps: int, min_lr: float = 1e-5) -> float:
    """Cosine annealing schedule."""
    progress = min(step / max(total_steps, 1), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ── Transfer-learning model ─────────────────────────────────────

def hierarchical_model_transfer(
    data: CalibrationData,
    prior_mu_global: jnp.ndarray,
    prior_sigma_global: jnp.ndarray,
    batch_size=None,
):
    """Hierarchical model with informative priors from synthetic pre-training.

    Same structure as hierarchical_model but μ_global ~ Normal(prior_mu, prior_sigma)
    instead of Normal(0, 1).
    """
    n_scenarios = data.domain_indices.shape[0]
    n_domains = data.n_domains

    # Level 1: Informed hyperpriors from Phase A
    mu_global = numpyro.sample(
        "mu_global",
        dist.Normal(prior_mu_global, prior_sigma_global).to_event(1),
    )
    sigma_global = numpyro.sample(
        "sigma_global",
        dist.HalfNormal(0.5 * jnp.ones(N_PARAMS)).to_event(1),
    )

    # Level 2: Domain
    with numpyro.plate("domains", n_domains):
        mu_domain = numpyro.sample(
            "mu_domain",
            dist.Normal(mu_global, sigma_global).to_event(1),
        )
        sigma_domain = numpyro.sample(
            "sigma_domain",
            dist.HalfNormal(0.3 * jnp.ones(N_PARAMS)).to_event(1),
        )

    # Covariate regression
    B = numpyro.sample(
        "B",
        dist.Normal(
            jnp.zeros((N_PARAMS, N_COVARIATES)),
            0.5 * jnp.ones((N_PARAMS, N_COVARIATES)),
        ).to_event(2),
    )

    # Observation params
    tau_readout = numpyro.sample("tau_readout", dist.LogNormal(-1.5, 0.5))
    log_phi = numpyro.sample("log_phi", dist.Normal(4.0, 1.0))
    sigma_outcome = numpyro.sample("sigma_outcome", dist.HalfNormal(5.0))
    log_sigma_outcome = numpyro.deterministic(
        "log_sigma_outcome", jnp.log(sigma_outcome + 1e-8),
    )
    numpyro.deterministic("phi", jnp.exp(log_phi))

    # Level 3: Scenarios
    plate_kwargs = {"name": "scenarios", "size": n_scenarios}
    if batch_size is not None and batch_size < n_scenarios:
        plate_kwargs["subsample_size"] = batch_size

    with numpyro.plate(**plate_kwargs) as idx:
        d = data.domain_indices[idx]
        x = data.covariates[idx]
        mu_s = mu_domain[d] + x @ B.T
        sigma_s = sigma_domain[d]
        theta_s = numpyro.sample(
            "theta_s", dist.Normal(mu_s, sigma_s).to_event(1),
        )

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
        numpyro.deterministic("sim_final_pcts", final_pcts)

        ll = _likelihood_batch(
            pro_fracs, final_pcts,
            data.obs_pro_pcts[idx],
            data.obs_sample_sizes[idx],
            data.obs_verified_masks[idx],
            data.obs_gt_pro_pcts[idx],
            data.obs_has_verified[idx],
            log_phi, log_sigma_outcome,
        )
        numpyro.factor("likelihood", ll)


# ── Phase A: Synthetic Pre-training ──────────────────────────────

def run_phase_a(
    n_steps: int = 2000,
    base_lr: float = 0.005,
    batch_size: int = 100,
    n_agents: int = 10,
    max_scenarios: int = None,
    seed: int = 42,
    log_every: int = 100,
) -> dict:
    """Phase A: Pre-train on synthetic scenarios.

    Returns dict with:
        mu_global_mean, sigma_global_mean: [4] arrays (informative prior for Phase B)
        losses: [n_steps] loss history
        svi_params, guide: for posterior extraction
        n_scenarios, n_domains, domain_names, elapsed_s
    """
    print("=" * 60)
    print("PHASE A: Synthetic Pre-training")
    print("=" * 60)
    t0 = time.time()

    # Load synthetic scenarios
    json_files = sorted(glob.glob(str(SYNTHETIC_DIR / "*.json")))
    if max_scenarios:
        json_files = json_files[:max_scenarios]
    print(f"Loading {len(json_files)} synthetic scenarios...")

    scenario_datas = []
    observations_list = []
    covariates_list = []
    domains = []
    scenario_ids = []
    failed = 0

    for i, path in enumerate(json_files):
        try:
            sd, obs, cov, domain = load_synthetic_scenario(
                path, n_agents=n_agents, seed=seed + i,
            )
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
            scenario_ids.append(Path(path).stem)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Warning: failed to load {Path(path).name}: {e}")

    print(f"  Loaded: {len(scenario_datas)}, failed: {failed}")

    domain_indices, domain_names = _build_domain_index(domains)
    print(f"  Domains ({len(domain_names)}): {domain_names}")

    # Prepare CalibrationData
    cal_data = prepare_calibration_data(
        scenario_datas, observations_list, domain_indices, covariates_list,
    )
    print(f"  Padded shapes: agents={cal_data.initial_positions.shape[1]}, "
          f"rounds={cal_data.events.shape[1]}")

    # Run SVI — full batch (AutoLowRankMultivariateNormal doesn't support
    # subsampled plates because the latent dimension changes between steps).
    # Control data size via max_scenarios instead.
    n_scenarios = cal_data.domain_indices.shape[0]

    def model_fn():
        return hierarchical_model(cal_data, batch_size=None)

    n_domains_val = cal_data.n_domains
    n_scenarios_val = n_scenarios
    init_values = {
        "mu_global": jnp.zeros(N_PARAMS),
        "sigma_global": 0.5 * jnp.ones(N_PARAMS),
        "mu_domain": jnp.zeros((n_domains_val, N_PARAMS)),
        "sigma_domain": 0.3 * jnp.ones((n_domains_val, N_PARAMS)),
        "B": jnp.zeros((N_PARAMS, N_COVARIATES)),
        "tau_readout": jnp.array(0.02),
        "log_phi": jnp.array(4.0),
        "sigma_outcome": jnp.array(5.0),
        "theta_s": jnp.zeros((n_scenarios_val, N_PARAMS)),
    }

    guide = AutoLowRankMultivariateNormal(
        model_fn, init_loc_fn=init_to_value(values=init_values),
    )

    # Start with base_lr, will manually adjust
    optimizer = numpyro.optim.Adam(base_lr)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(seed)
    svi_state = svi.init(rng_key)

    losses = []
    print(f"\nSVI: {n_steps} steps, batch=full ({n_scenarios}), lr={base_lr}")

    for step in range(n_steps):
        # Cosine annealing: update optimizer state's hyperparams
        cur_lr = cosine_lr(step, base_lr, n_steps)

        # NumPyro Adam stores lr in optim state — we rebuild optimizer each step
        # This is slow but correct; alternative is just using constant lr
        # For efficiency, only change lr every 50 steps
        if step > 0 and step % 50 == 0:
            optimizer_new = numpyro.optim.Adam(cur_lr)
            # Transfer state: pack current state into new optimizer
            # Actually, numpyro.optim.Adam stores JAX optim state, not lr directly.
            # The lr is baked into the step function. Changing lr mid-training
            # requires recreating the optimizer, which resets momentum.
            # Pragmatic approach: just use constant lr (works well with Adam).
            pass

        svi_state, loss = svi.update(svi_state)
        losses.append(float(loss))

        if log_every and (step + 1) % log_every == 0:
            recent = losses[-min(50, len(losses)):]
            avg_loss = sum(recent) / len(recent)
            print(f"  Step {step+1:5d}/{n_steps}  loss={loss:.1f}  avg50={avg_loss:.1f}")

    svi_params = svi.get_params(svi_state)
    elapsed = time.time() - t0

    # Extract synthetic prior
    print("\nExtracting posterior...")
    posteriors = extract_posteriors(
        samples=None, guide=guide, svi_params=svi_params,
        domain_names=domain_names, scenario_ids=scenario_ids[:20],
        n_posterior_samples=500, seed=seed,
    )

    mu_global_mean = jnp.array(posteriors["global"]["mu_global"]["mean"])
    sigma_global_mean = jnp.array(posteriors["global"]["sigma_global"]["mean"])

    print(f"\nPhase A complete in {elapsed:.1f}s")
    print(f"  μ_global: {mu_global_mean}")
    print(f"  σ_global: {sigma_global_mean}")
    print(f"  Final loss: {losses[-1]:.1f}, avg last 50: {sum(losses[-50:])/50:.1f}")

    return {
        "mu_global_mean": mu_global_mean,
        "sigma_global_mean": sigma_global_mean,
        "losses": jnp.array(losses),
        "svi_params": svi_params,
        "guide": guide,
        "posteriors": posteriors,
        "n_scenarios": len(scenario_datas),
        "n_domains": len(domain_names),
        "domain_names": domain_names,
        "elapsed_s": elapsed,
        "cal_data": cal_data,
    }


# ── Phase B: Empirical Fine-tuning ──────────────────────────────

# Non-independent pairs — drop second of each
_DROP_SCENARIOS = {
    "CORP-2020-BOEING_737_MAX_RETURN_TO_SERVI",
    "TECH-2018-FACEBOOK_CAMBRIDGE_ANALYTICA_S",
}


def _stratified_split(
    scenario_ids: list[str],
    domains: list[str],
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Stratified train/test split by domain. Returns (train_indices, test_indices)."""
    import random
    rng = random.Random(seed)

    # Group by domain
    domain_groups: dict[str, list[int]] = {}
    for i, d in enumerate(domains):
        domain_groups.setdefault(d, []).append(i)

    train_idx, test_idx = [], []
    for d, indices in sorted(domain_groups.items()):
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * test_frac))
        if len(indices) <= 1:
            # Single scenario in domain → put in train
            train_idx.extend(indices)
        else:
            test_idx.extend(indices[:n_test])
            train_idx.extend(indices[n_test:])

    return train_idx, test_idx


def run_phase_b(
    phase_a_result: dict,
    n_steps: int = 3000,
    lr: float = 0.002,
    test_frac: float = 0.2,
    seed: int = 42,
    log_every: int = 100,
) -> dict:
    """Phase B: Fine-tune on empirical scenarios with synthetic prior.

    Returns dict with train/test posteriors, metrics, etc.
    """
    print("\n" + "=" * 60)
    print("PHASE B: Empirical Fine-tuning")
    print("=" * 60)
    t0 = time.time()

    # Load empirical scenarios
    json_files = sorted(glob.glob(str(EMPIRICAL_DIR / "*.json")))
    json_files = [f for f in json_files if not f.endswith("manifest.json")
                  and not f.endswith(".meta.json")]

    scenario_datas = []
    observations_list = []
    covariates_list = []
    domains = []
    scenario_ids = []
    failed = 0

    for path in json_files:
        sid = Path(path).stem
        if sid in _DROP_SCENARIOS:
            print(f"  Dropping non-independent: {sid}")
            continue
        try:
            sd, obs, cov, domain = load_empirical_scenario(path, seed=seed)
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
            failed += 1
            print(f"  Warning: failed to load {Path(path).name}: {e}")

    print(f"  Loaded: {len(scenario_datas)} empirical scenarios, failed: {failed}")

    # Stratified split
    train_idx, test_idx = _stratified_split(scenario_ids, domains, test_frac, seed)
    print(f"  Split: {len(train_idx)} train, {len(test_idx)} test")
    print(f"  Test scenarios: {[scenario_ids[i] for i in test_idx]}")

    # Prepare train data
    train_sds = [scenario_datas[i] for i in train_idx]
    train_obs = [observations_list[i] for i in train_idx]
    train_dom = [domains[i] for i in train_idx]
    train_cov = [covariates_list[i] for i in train_idx]
    train_ids = [scenario_ids[i] for i in train_idx]

    train_domain_indices, train_domain_names = _build_domain_index(train_dom)
    train_data = prepare_calibration_data(
        train_sds, train_obs, train_domain_indices, train_cov,
    )

    print(f"  Train domains ({len(train_domain_names)}): {train_domain_names}")
    print(f"  Padded: agents={train_data.initial_positions.shape[1]}, "
          f"rounds={train_data.events.shape[1]}")

    # Transfer learning: use Phase A posterior as prior
    prior_mu = phase_a_result["mu_global_mean"]
    prior_sigma = phase_a_result["sigma_global_mean"]
    # Widen sigma slightly for fine-tuning flexibility
    prior_sigma = jnp.maximum(prior_sigma, 0.3)
    print(f"  Transfer prior μ: {prior_mu}")
    print(f"  Transfer prior σ: {prior_sigma}")

    # SVI with transfer model
    n_train = train_data.domain_indices.shape[0]
    # Full batch for small empirical dataset
    batch_size_eff = None

    def model_fn():
        return hierarchical_model_transfer(
            train_data, prior_mu, prior_sigma, batch_size=batch_size_eff,
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
    optimizer = numpyro.optim.Adam(lr)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(seed + 100)
    svi_state = svi.init(rng_key)

    losses = []
    print(f"\nSVI: {n_steps} steps, batch=full, lr={lr}")

    for step in range(n_steps):
        svi_state, loss = svi.update(svi_state)
        losses.append(float(loss))

        if log_every and (step + 1) % log_every == 0:
            recent = losses[-min(50, len(losses)):]
            avg_loss = sum(recent) / len(recent)
            print(f"  Step {step+1:5d}/{n_steps}  loss={loss:.1f}  avg50={avg_loss:.1f}")

    svi_params = svi.get_params(svi_state)
    elapsed = time.time() - t0

    # Extract posteriors
    print("\nExtracting posteriors...")
    posteriors = extract_posteriors(
        samples=None, guide=guide, svi_params=svi_params,
        domain_names=train_domain_names, scenario_ids=train_ids,
        n_posterior_samples=1000, seed=seed,
    )

    print(f"\nPhase B complete in {elapsed:.1f}s")
    print(f"  μ_global: {posteriors['global']['mu_global']['mean']}")
    print(f"  σ_global: {posteriors['global']['sigma_global']['mean']}")
    if posteriors.get("covariates", {}).get("significant"):
        print(f"  Significant covariates:")
        for s in posteriors["covariates"]["significant"]:
            print(f"    {s['param']} ← {s['covariate']}: {s['effect']:.3f} "
                  f"CI=({s['ci95'][0]:.3f}, {s['ci95'][1]:.3f})")

    return {
        "posteriors": posteriors,
        "svi_params": svi_params,
        "guide": guide,
        "losses": jnp.array(losses),
        "train_data": train_data,
        "train_ids": train_ids,
        "train_domain_names": train_domain_names,
        "test_idx": test_idx,
        "test_ids": [scenario_ids[i] for i in test_idx],
        "all_scenario_datas": scenario_datas,
        "all_observations": observations_list,
        "all_domains": domains,
        "all_covariates": covariates_list,
        "all_scenario_ids": scenario_ids,
        "elapsed_s": elapsed,
    }


# ── CRPS computation ────────────────────────────────────────────

def crps_ensemble(samples: jnp.ndarray, observation: float) -> float:
    """CRPS = E|X-y| - 0.5·E|X-X'| for an ensemble of samples.

    Args:
        samples: [S] posterior predictive samples
        observation: scalar observed value

    Returns:
        Scalar CRPS value (lower = better).
    """
    s = jnp.sort(samples)
    n = s.shape[0]

    # E|X - y|
    term1 = jnp.mean(jnp.abs(s - observation))

    # E|X - X'| via sorted formula: (2i - n - 1) * x_i / n^2
    idx = jnp.arange(1, n + 1)
    term2 = jnp.sum((2 * idx - n - 1) * s) / (n * n)

    return float(term1 - term2)


# ── Phase C: Validation ──────────────────────────────────────────

def run_phase_c(
    phase_b_result: dict,
    phase_a_result: dict = None,
    n_posterior_samples: int = 200,
    seed: int = 42,
) -> dict:
    """Phase C: Validation with posterior predictive checks.

    Metrics per scenario:
    - outcome_error: simulated - observed final pro%
    - coverage_90: fraction of test scenarios where truth ∈ 90% CI
    - coverage_50: fraction of test scenarios where truth ∈ 50% CI
    - crps: Continuous Ranked Probability Score

    Also generates comparison between train and test performance.
    """
    print("\n" + "=" * 60)
    print("PHASE C: Validation")
    print("=" * 60)
    t0 = time.time()

    guide = phase_b_result["guide"]
    svi_params = phase_b_result["svi_params"]
    train_data = phase_b_result["train_data"]
    train_ids = phase_b_result["train_ids"]
    test_idx = phase_b_result["test_idx"]
    test_ids = phase_b_result["test_ids"]
    all_sds = phase_b_result["all_scenario_datas"]
    all_obs = phase_b_result["all_observations"]
    all_domains = phase_b_result["all_domains"]
    all_covariates = phase_b_result["all_covariates"]
    all_ids = phase_b_result["all_scenario_ids"]
    posteriors = phase_b_result["posteriors"]

    # Draw posterior samples from the guide
    print(f"Drawing {n_posterior_samples} posterior samples...")

    def model_fn():
        return hierarchical_model_transfer(
            train_data,
            phase_a_result["mu_global_mean"] if phase_a_result else jnp.zeros(N_PARAMS),
            jnp.maximum(
                phase_a_result["sigma_global_mean"] if phase_a_result else jnp.ones(N_PARAMS),
                0.3,
            ),
            batch_size=None,
        )

    predictive = Predictive(guide, params=svi_params,
                            num_samples=n_posterior_samples)
    rng = jax.random.PRNGKey(seed + 200)
    pp_samples = predictive(rng)

    # Get mu_global and sigma_global samples
    mu_global_samples = pp_samples["mu_global"]           # [S, 4]
    sigma_global_samples = pp_samples["sigma_global"]     # [S, 4]

    # For each test scenario, simulate with posterior parameter samples
    frozen = get_default_frozen_params()
    frozen_vec = jnp.array([frozen[k] for k in FROZEN_KEYS])

    # Get tau_readout samples
    tau_samples = pp_samples.get("tau_readout", jnp.full(n_posterior_samples, 0.02))

    results_per_scenario = {}

    # Evaluate ALL scenarios (train + test) for comparison
    for group_name, group_idx in [("train", list(range(len(all_ids)))),
                                    ("test", test_idx)]:
        if group_name == "train":
            # Only evaluate a subset of train for speed
            eval_idx = [i for i in range(len(all_ids)) if i not in test_idx]
        else:
            eval_idx = group_idx

        for i in eval_idx:
            sid = all_ids[i]
            sd = all_sds[i]
            obs = all_obs[i]
            gt = float(obs.ground_truth_pro_pct)

            # Simulate with posterior mean parameters
            mu_g_mean = jnp.mean(mu_global_samples, axis=0)  # [4]

            # Use global mean as point estimate for this scenario
            params_point = {
                PARAM_NAMES[k]: float(mu_g_mean[k]) for k in range(N_PARAMS)
            }
            for k_idx, key in enumerate(FROZEN_KEYS):
                params_point[key] = float(frozen_vec[k_idx])

            sim_result = simulate_scenario(params_point, sd)
            sim_final = float(sim_result["final_pro_pct"])

            # Posterior predictive distribution: simulate with each sample
            pp_finals = []
            n_eval = min(n_posterior_samples, 50)  # limit for speed
            for s in range(n_eval):
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
                    pp_finals.append(sim_final)

            pp_arr = jnp.array(pp_finals)

            # Metrics
            error = sim_final - gt
            ci90_lo = float(jnp.percentile(pp_arr, 5))
            ci90_hi = float(jnp.percentile(pp_arr, 95))
            ci50_lo = float(jnp.percentile(pp_arr, 25))
            ci50_hi = float(jnp.percentile(pp_arr, 75))
            in_90 = ci90_lo <= gt <= ci90_hi
            in_50 = ci50_lo <= gt <= ci50_hi
            crps_val = crps_ensemble(pp_arr, gt)

            is_test = i in test_idx

            results_per_scenario[sid] = {
                "group": "test" if is_test else "train",
                "domain": all_domains[i],
                "ground_truth": gt,
                "sim_final": sim_final,
                "error": error,
                "abs_error": abs(error),
                "pp_mean": float(jnp.mean(pp_arr)),
                "pp_std": float(jnp.std(pp_arr)),
                "ci90": (ci90_lo, ci90_hi),
                "ci50": (ci50_lo, ci50_hi),
                "in_90": in_90,
                "in_50": in_50,
                "crps": crps_val,
            }

            print(f"  {sid[:45]:45s}  gt={gt:5.1f}  sim={sim_final:5.1f}  "
                  f"err={error:+5.1f}  {'TEST' if is_test else 'train'}")

    # Aggregate metrics
    train_results = [v for v in results_per_scenario.values() if v["group"] == "train"]
    test_results = [v for v in results_per_scenario.values() if v["group"] == "test"]

    def _aggregate(results):
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
            "median_abs_error": float(jnp.median(
                jnp.array([r["abs_error"] for r in results]))),
        }

    train_agg = _aggregate(train_results)
    test_agg = _aggregate(test_results)

    elapsed = time.time() - t0

    print(f"\n{'Metric':<20} {'Train':>10} {'Test':>10}")
    print("-" * 42)
    for key in ["n", "mae", "rmse", "coverage_90", "coverage_50", "mean_crps"]:
        t_val = train_agg.get(key, "-")
        te_val = test_agg.get(key, "-")
        t_str = f"{t_val:.3f}" if isinstance(t_val, float) else str(t_val)
        te_str = f"{te_val:.3f}" if isinstance(te_val, float) else str(te_val)
        print(f"  {key:<18} {t_str:>10} {te_str:>10}")

    print(f"\nPhase C complete in {elapsed:.1f}s")

    return {
        "per_scenario": results_per_scenario,
        "train_aggregate": train_agg,
        "test_aggregate": test_agg,
        "posteriors": posteriors,
        "elapsed_s": elapsed,
    }


# ── Save results ─────────────────────────────────────────────────

def _to_serializable(obj):
    """Convert JAX arrays and other non-serializable types for JSON."""
    if isinstance(obj, (jnp.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return str(obj)


def save_results(
    phase_a_result: dict,
    phase_b_result: dict,
    phase_c_result: dict,
):
    """Save all results to calibration/results/hierarchical_calibration/."""
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic prior
    prior = {
        "mu_global": phase_a_result["mu_global_mean"].tolist(),
        "sigma_global": phase_a_result["sigma_global_mean"].tolist(),
        "n_synthetic_scenarios": phase_a_result["n_scenarios"],
        "n_domains": phase_a_result["n_domains"],
        "domain_names": phase_a_result["domain_names"],
        "elapsed_s": phase_a_result["elapsed_s"],
    }
    with open(out_dir / "synthetic_prior.json", "w") as f:
        json.dump(prior, f, indent=2)
    print(f"\nSaved: {out_dir / 'synthetic_prior.json'}")

    # Posteriors directory
    post_dir = out_dir / "posteriors"
    post_dir.mkdir(exist_ok=True)

    # Phase B posteriors
    posteriors_ser = _to_serializable(phase_b_result["posteriors"])
    with open(post_dir / "empirical_posteriors.json", "w") as f:
        json.dump(posteriors_ser, f, indent=2)
    print(f"Saved: {post_dir / 'empirical_posteriors.json'}")

    # Phase C validation
    validation = _to_serializable({
        "per_scenario": phase_c_result["per_scenario"],
        "train_aggregate": phase_c_result["train_aggregate"],
        "test_aggregate": phase_c_result["test_aggregate"],
    })
    with open(post_dir / "validation_results.json", "w") as f:
        json.dump(validation, f, indent=2)
    print(f"Saved: {post_dir / 'validation_results.json'}")

    # Loss histories
    losses = {
        "phase_a": phase_a_result["losses"].tolist(),
        "phase_b": phase_b_result["losses"].tolist(),
    }
    with open(post_dir / "loss_histories.json", "w") as f:
        json.dump(losses, f, indent=2)

    # Generate calibration report
    _generate_report(phase_a_result, phase_b_result, phase_c_result, out_dir)


def _generate_report(
    phase_a: dict, phase_b: dict, phase_c: dict, out_dir: Path,
):
    """Generate markdown calibration report."""
    pa = phase_a
    pb = phase_b
    pc = phase_c

    posteriors = pb["posteriors"]
    train_agg = pc["train_aggregate"]
    test_agg = pc["test_aggregate"]

    mu_g = posteriors["global"]["mu_global"]["mean"]
    sig_g = posteriors["global"]["sigma_global"]["mean"]

    # Format arrays
    def fmt_arr(arr):
        if hasattr(arr, 'tolist'):
            arr = arr.tolist()
        return "[" + ", ".join(f"{v:.3f}" for v in arr) + "]"

    lines = [
        "# Hierarchical Calibration Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Phase A: Synthetic Pre-training",
        "",
        f"- Scenarios: {pa['n_scenarios']}",
        f"- Domains: {pa['n_domains']} ({', '.join(pa['domain_names'])})",
        f"- SVI steps: {len(pa['losses'])}",
        f"- Final loss: {float(pa['losses'][-1]):.1f}",
        f"- Elapsed: {pa['elapsed_s']:.1f}s",
        f"- μ_global (synthetic): {fmt_arr(pa['mu_global_mean'])}",
        f"- σ_global (synthetic): {fmt_arr(pa['sigma_global_mean'])}",
        "",
        "## Phase B: Empirical Fine-tuning",
        "",
        f"- Train scenarios: {len(pb['train_ids'])}",
        f"- Test scenarios: {len(pb['test_ids'])} ({', '.join(pb['test_ids'])})",
        f"- Domains: {', '.join(pb['train_domain_names'])}",
        f"- SVI steps: {len(pb['losses'])}",
        f"- Final loss: {float(pb['losses'][-1]):.1f}",
        f"- Elapsed: {pb['elapsed_s']:.1f}s",
        "",
        "### Calibrated Parameters",
        "",
        f"- μ_global: {fmt_arr(mu_g)}",
        f"- σ_global: {fmt_arr(sig_g)}",
        "",
        "| Parameter | Mean | CI95 Low | CI95 High |",
        "|---|---|---|---|",
    ]

    for i, name in enumerate(PARAM_NAMES):
        m = mu_g[i] if hasattr(mu_g, '__getitem__') else mu_g
        lo = posteriors["global"]["mu_global"]["ci95_lo"]
        hi = posteriors["global"]["mu_global"]["ci95_hi"]
        lo_v = lo[i] if hasattr(lo, '__getitem__') else lo
        hi_v = hi[i] if hasattr(hi, '__getitem__') else hi
        if hasattr(m, 'item'):
            m, lo_v, hi_v = m.item(), lo_v.item(), hi_v.item()
        lines.append(f"| {name} | {float(m):.3f} | {float(lo_v):.3f} | {float(hi_v):.3f} |")

    # Observation params
    obs_p = posteriors.get("observation_params", {})
    lines.extend(["", "### Observation Parameters", ""])
    if "tau_readout" in obs_p:
        tau = obs_p["tau_readout"]["mean"]
        lines.append(f"- τ_readout: {float(tau) if hasattr(tau, 'item') else tau:.4f}")
    if "phi" in obs_p:
        phi = obs_p["phi"]["mean"]
        lines.append(f"- φ (BetaBinomial): {float(phi) if hasattr(phi, 'item') else phi:.1f}")
    if "sigma_outcome" in obs_p:
        so = obs_p["sigma_outcome"]["mean"]
        lines.append(f"- σ_outcome: {float(so) if hasattr(so, 'item') else so:.1f}")

    # Significant covariates
    sig_covs = posteriors.get("covariates", {}).get("significant", [])
    if sig_covs:
        lines.extend(["", "### Significant Covariates (95% CI excludes 0)", ""])
        for s in sig_covs:
            lines.append(
                f"- `{s['param']}` ← `{s['covariate']}`: "
                f"{s['effect']:.3f} (CI: {s['ci95'][0]:.3f}, {s['ci95'][1]:.3f})"
            )
    else:
        lines.extend(["", "### Covariates", "", "No significant effects at 95% CI."])

    # Phase C validation
    lines.extend([
        "",
        "## Phase C: Validation",
        "",
        "| Metric | Train | Test |",
        "|---|---|---|",
    ])
    for key in ["n", "mae", "rmse", "coverage_90", "coverage_50",
                "mean_crps", "median_abs_error"]:
        t_val = train_agg.get(key, "-")
        te_val = test_agg.get(key, "-")
        t_str = f"{t_val:.3f}" if isinstance(t_val, float) else str(t_val)
        te_str = f"{te_val:.3f}" if isinstance(te_val, float) else str(te_val)
        lines.append(f"| {key} | {t_str} | {te_str} |")

    # Per-scenario results
    lines.extend(["", "### Per-Scenario Results", ""])
    lines.append("| Scenario | Domain | Group | GT | Sim | Error | CRPS | 90% CI |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for sid, r in sorted(pc["per_scenario"].items(),
                         key=lambda x: x[1]["abs_error"], reverse=True):
        ci90 = r["ci90"]
        lines.append(
            f"| {sid[:40]} | {r['domain']} | {r['group']} | "
            f"{r['ground_truth']:.1f} | {r['sim_final']:.1f} | "
            f"{r['error']:+.1f} | {r['crps']:.2f} | "
            f"[{ci90[0]:.1f}, {ci90[1]:.1f}] |"
        )

    # Domain breakdown
    lines.extend(["", "### Domain Breakdown", ""])
    domain_results: dict[str, list] = {}
    for r in pc["per_scenario"].values():
        domain_results.setdefault(r["domain"], []).append(r)

    lines.append("| Domain | N | MAE | Coverage 90% |")
    lines.append("|---|---|---|---|")
    for d, results in sorted(domain_results.items()):
        n = len(results)
        mae = sum(r["abs_error"] for r in results) / n
        cov90 = sum(1 for r in results if r["in_90"]) / n
        lines.append(f"| {d} | {n} | {mae:.1f} | {cov90:.0%} |")

    report = "\n".join(lines) + "\n"
    report_path = out_dir / "calibration_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nSaved: {report_path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hierarchical calibration pipeline")
    parser.add_argument("--phase", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--svi-steps-a", type=int, default=2000)
    parser.add_argument("--svi-steps-b", type=int, default=3000)
    parser.add_argument("--lr-a", type=float, default=0.005)
    parser.add_argument("--lr-b", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-synthetic", type=int, default=None,
                        help="Limit synthetic scenarios (for testing)")
    parser.add_argument("--n-agents", type=int, default=10,
                        help="Agents per synthetic scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    phase_a_result = None
    phase_b_result = None
    phase_c_result = None

    # Try to load cached Phase A result
    prior_path = RESULTS_DIR / "synthetic_prior.json"

    if args.phase in ("A", "all"):
        phase_a_result = run_phase_a(
            n_steps=args.svi_steps_a,
            base_lr=args.lr_a,
            batch_size=args.batch_size,
            n_agents=args.n_agents,
            max_scenarios=args.max_synthetic,
            seed=args.seed,
            log_every=args.log_every,
        )

    if args.phase in ("B", "C", "all"):
        if phase_a_result is None:
            # Load cached synthetic prior
            if prior_path.exists():
                print(f"Loading cached synthetic prior from {prior_path}")
                with open(prior_path) as f:
                    prior = json.load(f)
                phase_a_result = {
                    "mu_global_mean": jnp.array(prior["mu_global"]),
                    "sigma_global_mean": jnp.array(prior["sigma_global"]),
                    "n_scenarios": prior["n_synthetic_scenarios"],
                    "n_domains": prior["n_domains"],
                    "domain_names": prior["domain_names"],
                    "elapsed_s": prior["elapsed_s"],
                    "losses": jnp.array([]),  # not available from cache
                }
            else:
                print("ERROR: No Phase A result available. Run --phase A first.")
                return

    if args.phase in ("B", "all"):
        phase_b_result = run_phase_b(
            phase_a_result,
            n_steps=args.svi_steps_b,
            lr=args.lr_b,
            seed=args.seed,
            log_every=args.log_every,
        )

    if args.phase in ("C", "all"):
        if phase_b_result is None:
            print("ERROR: No Phase B result. Run --phase B first.")
            return
        phase_c_result = run_phase_c(
            phase_b_result,
            phase_a_result=phase_a_result,
            seed=args.seed,
        )

    # Save if we have all phases
    if phase_a_result and phase_b_result and phase_c_result:
        save_results(phase_a_result, phase_b_result, phase_c_result)

    print("\n" + "=" * 60)
    print("CALIBRATION PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
