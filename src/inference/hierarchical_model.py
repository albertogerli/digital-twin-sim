"""Hierarchical Bayesian calibration model for DynamicsV2.

Three-level hierarchy:
  Level 1 — Global hyperpriors:  μ_global_k, σ_global_k
  Level 2 — Domain parameters:   μ_d_k, σ_d_k  (per domain)
  Level 3 — Scenario parameters: θ_s_k = μ_d_k + B_k @ x_s + ε_s_k
             with covariate regression B (4×5 matrix)

Observation model:
  - Verified rounds → BetaBinomial(n, q·φ, (1-q)·φ)
  - Outcome-only    → Normal(q_final·100, σ_outcome)

Inference:
  - SVI with AutoLowRankMultivariateNormal (default)
  - NUTS with init_to_value (optional, slow for large datasets)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.nn import sigmoid

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, NUTS, MCMC, Predictive, init_to_value, init_to_median
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal

from ..dynamics.opinion_dynamics_jax import (
    simulate_scenario, ScenarioData, build_sparse_interaction,
)
from ..dynamics.param_utils import get_default_frozen_params, FROZEN_PARAMS
from ..observation.observation_model import _beta_binomial_logpmf


# ── Constants ────────────────────────────────────────────────────

PARAM_NAMES = ["alpha_herd", "alpha_anchor", "alpha_social", "alpha_event"]
FROZEN_KEYS = [
    "log_lambda_elite", "log_lambda_citizen",
    "logit_herd_threshold", "logit_anchor_drift",
]
COVARIATE_NAMES = [
    "initial_polarization", "event_volatility",
    "elite_concentration", "institutional_trust", "undecided_share",
]
N_PARAMS = len(PARAM_NAMES)
N_COVARIATES = len(COVARIATE_NAMES)


# ── Data Structures ──────────────────────────────────────────────

class CalibrationData(NamedTuple):
    """Pre-padded and stacked data for the hierarchical model.

    All scenario fields are padded to (max_agents, max_rounds).
    N = number of scenarios, A = max_agents, R = max_rounds.
    """
    # Scenario arrays (padded)
    initial_positions: jnp.ndarray      # [N, A]
    agent_types: jnp.ndarray            # [N, A]
    agent_rigidities: jnp.ndarray       # [N, A]
    agent_tolerances: jnp.ndarray       # [N, A]
    events: jnp.ndarray                 # [N, R, 2]
    llm_shifts: jnp.ndarray            # [N, R, A]
    interaction_matrices: jnp.ndarray   # [N, A, A]
    agent_masks: jnp.ndarray            # [N, A] — True for real agents

    # Observation arrays (padded)
    obs_pro_pcts: jnp.ndarray           # [N, R]
    obs_sample_sizes: jnp.ndarray       # [N, R]
    obs_verified_masks: jnp.ndarray     # [N, R]
    obs_gt_pro_pcts: jnp.ndarray        # [N]
    obs_has_verified: jnp.ndarray       # [N] — True if any verified round

    # Metadata
    n_real_rounds: jnp.ndarray          # [N]
    domain_indices: jnp.ndarray         # [N] — int index into domain list
    covariates: jnp.ndarray             # [N, 5] — normalized
    n_domains: int                      # number of unique domains (Python int, trace-safe)

    # Frozen params (shared)
    frozen_vec: jnp.ndarray             # [4]


# ── Data Preparation ─────────────────────────────────────────────

def prepare_calibration_data(
    scenario_datas: list[ScenarioData],
    observations: list,  # list of ObservationData
    domain_indices: list[int],
    covariates_raw: list[list[float]],
) -> CalibrationData:
    """Pad, normalize, and stack all data for the hierarchical model.

    Args:
        scenario_datas: List of ScenarioData (variable n_agents, n_rounds).
        observations: List of ObservationData.
        domain_indices: Per-scenario domain index (int).
        covariates_raw: Per-scenario covariate vectors [5].

    Returns:
        CalibrationData ready for hierarchical_model.
    """
    n = len(scenario_datas)
    max_a = max(sd.initial_positions.shape[0] for sd in scenario_datas)
    max_r = max(sd.events.shape[0] for sd in scenario_datas)

    # Pad scenario data
    init_pos = []
    a_types = []
    a_rigid = []
    a_toler = []
    evts = []
    llm_sh = []
    int_mat = []
    a_masks = []
    n_real_r = []

    for sd in scenario_datas:
        na = sd.initial_positions.shape[0]
        nr = sd.events.shape[0]
        n_real_r.append(nr)

        # Pad agents: position=0, rigidity=1 (frozen), tolerance=0, type=1
        pad_a = max_a - na
        init_pos.append(jnp.pad(sd.initial_positions, (0, pad_a)))
        a_types.append(jnp.pad(sd.agent_types, (0, pad_a), constant_values=1))
        a_rigid.append(jnp.pad(sd.agent_rigidities, (0, pad_a), constant_values=1.0))
        a_toler.append(jnp.pad(sd.agent_tolerances, (0, pad_a)))
        a_masks.append(
            jnp.concatenate([jnp.ones(na, dtype=jnp.bool_),
                             jnp.zeros(pad_a, dtype=jnp.bool_)])
        )

        # Pad rounds: events=(0,0), llm_shifts=0
        pad_r = max_r - nr
        evts.append(jnp.pad(sd.events, ((0, pad_r), (0, 0))))
        llm_padded = jnp.pad(sd.llm_shifts, ((0, pad_r), (0, pad_a)))
        llm_sh.append(llm_padded)

        # Pad interaction matrix
        im = jnp.pad(sd.interaction_matrix, ((0, pad_a), (0, pad_a)))
        int_mat.append(im)

    # Pad observations
    obs_pp = []
    obs_ss = []
    obs_vm = []
    obs_gt = []
    obs_hv = []

    for obs in observations:
        nr = obs.pro_pcts.shape[0]
        pad_r = max_r - nr
        pp = jnp.pad(obs.pro_pcts, (0, pad_r), constant_values=jnp.nan)
        ss = jnp.pad(obs.sample_sizes, (0, pad_r))
        vm = jnp.pad(obs.verified_mask, (0, pad_r))
        obs_pp.append(pp)
        obs_ss.append(ss)
        obs_vm.append(vm)
        obs_gt.append(obs.ground_truth_pro_pct)
        obs_hv.append(bool(jnp.any(obs.verified_mask)))

    # Normalize covariates
    cov = jnp.array(covariates_raw, dtype=jnp.float32)  # [N, 5]
    cov_mu = jnp.mean(cov, axis=0)
    cov_std = jnp.std(cov, axis=0) + 1e-6
    cov_norm = (cov - cov_mu) / cov_std

    # Frozen params
    frozen = get_default_frozen_params()
    frozen_vec = jnp.array([frozen[k] for k in FROZEN_KEYS])

    n_domains = int(max(domain_indices)) + 1

    return CalibrationData(
        initial_positions=jnp.stack(init_pos),
        agent_types=jnp.stack(a_types),
        agent_rigidities=jnp.stack(a_rigid),
        agent_tolerances=jnp.stack(a_toler),
        events=jnp.stack(evts),
        llm_shifts=jnp.stack(llm_sh),
        interaction_matrices=jnp.stack(int_mat),
        agent_masks=jnp.stack(a_masks),
        obs_pro_pcts=jnp.stack(obs_pp),
        obs_sample_sizes=jnp.stack(obs_ss),
        obs_verified_masks=jnp.stack(obs_vm),
        obs_gt_pro_pcts=jnp.array(obs_gt),
        obs_has_verified=jnp.array(obs_hv),
        n_real_rounds=jnp.array(n_real_r, dtype=jnp.int32),
        domain_indices=jnp.array(domain_indices, dtype=jnp.int32),
        covariates=cov_norm,
        n_domains=n_domains,
        frozen_vec=frozen_vec,
    )


# ── Masked Simulation ────────────────────────────────────────────

def _simulate_one(calibrable_vec, frozen_vec, scenario_data,
                  agent_mask, tau_readout, n_real_rounds):
    """Simulate one padded scenario. Returns pro_fraction and final_pro_pct."""
    params = {
        PARAM_NAMES[i]: calibrable_vec[i] for i in range(N_PARAMS)
    }
    for i, key in enumerate(FROZEN_KEYS):
        params[key] = frozen_vec[i]

    result = simulate_scenario(params, scenario_data)
    trajectories = result["trajectories"]  # [max_rounds, max_agents]

    # Recompute pro_fraction with agent mask and τ_readout
    soft_pro = sigmoid((trajectories - 0.05) / tau_readout)
    soft_against = sigmoid((-trajectories - 0.05) / tau_readout)
    soft_decided = soft_pro + soft_against

    mask = agent_mask[None, :]  # [1, max_agents]
    pro_count = jnp.sum(soft_pro * mask, axis=1)
    decided_count = jnp.sum(soft_decided * mask, axis=1) + 1e-8
    pro_fraction = pro_count / decided_count  # [max_rounds]

    # Final pro_pct from the last real round
    final_pro_pct = pro_fraction[n_real_rounds - 1] * 100.0

    return pro_fraction, final_pro_pct


# vmap over batch: calibrable varies, frozen shared, tau_readout shared
_SCENARIO_AXES = ScenarioData(0, 0, 0, 0, 0, 0, 0)

_simulate_batch = jax.vmap(
    _simulate_one,
    in_axes=(0, None, _SCENARIO_AXES, 0, None, 0),
)


# ── Likelihood Computation ───────────────────────────────────────

def _likelihood_one(pro_fraction, final_pro_pct,
                    obs_pro_pcts, obs_sample_sizes, obs_verified_mask,
                    obs_gt_pro_pct, has_verified,
                    log_phi, log_sigma_outcome):
    """Compute log-likelihood for one scenario."""
    phi = jnp.exp(log_phi)
    sigma = jnp.exp(log_sigma_outcome)

    # ── Outcome-only: Normal on final outcome ──
    ll_outcome = (
        -0.5 * jnp.log(2.0 * jnp.pi)
        - jnp.log(sigma)
        - 0.5 * ((obs_gt_pro_pct - final_pro_pct) / sigma) ** 2
    )

    # ── Full: BetaBinomial on verified rounds ──
    q = pro_fraction  # [max_rounds], 0-1
    eps = 1e-4
    q_c = jnp.clip(q, eps, 1.0 - eps)
    alpha = q_c * phi
    beta = (1.0 - q_c) * phi

    k = jnp.round(obs_pro_pcts / 100.0 * obs_sample_sizes)
    # Replace NaN in obs_pro_pcts with 0 for safe computation
    k = jnp.where(jnp.isnan(k), 0.0, k)
    n = obs_sample_sizes

    ll_bb_per_round = _beta_binomial_logpmf(k, n, alpha, beta)
    # Replace NaN/Inf from rounds with n=0 (lgamma(0) issues)
    ll_bb_per_round = jnp.where(jnp.isfinite(ll_bb_per_round), ll_bb_per_round, 0.0)

    mask = obs_verified_mask & (obs_sample_sizes > 0)
    ll_full = jnp.sum(jnp.where(mask, ll_bb_per_round, 0.0))

    # Select: verified → full BB, otherwise → outcome-only
    return jnp.where(has_verified, ll_full, ll_outcome)


_likelihood_batch = jax.vmap(
    _likelihood_one,
    in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
)


# ── NumPyro Model ────────────────────────────────────────────────

def hierarchical_model(data: CalibrationData, batch_size=None):
    """Three-level hierarchical model.

    Args:
        data: CalibrationData (pre-padded, stacked).
        batch_size: Mini-batch size for SVI subsampling. None = full batch.
    """
    n_scenarios = data.domain_indices.shape[0]
    n_domains = data.n_domains

    # ── Level 1: Global hyperpriors ──
    # .to_event(1) makes the [4] vector a single event (not 4 independent batch dims)
    mu_global = numpyro.sample(
        "mu_global",
        dist.Normal(jnp.zeros(N_PARAMS), 1.0).to_event(1),
    )  # [4]
    sigma_global = numpyro.sample(
        "sigma_global",
        dist.HalfNormal(0.5 * jnp.ones(N_PARAMS)).to_event(1),
    )  # [4]

    # ── Level 2: Domain parameters ──
    with numpyro.plate("domains", n_domains):
        mu_domain = numpyro.sample(
            "mu_domain",
            dist.Normal(mu_global, sigma_global).to_event(1),
        )  # [n_domains, 4]
        sigma_domain = numpyro.sample(
            "sigma_domain",
            dist.HalfNormal(0.3 * jnp.ones(N_PARAMS)).to_event(1),
        )  # [n_domains, 4]

    # ── Covariate regression matrix ──
    B = numpyro.sample(
        "B",
        dist.Normal(
            jnp.zeros((N_PARAMS, N_COVARIATES)),
            0.5 * jnp.ones((N_PARAMS, N_COVARIATES)),
        ).to_event(2),
    )  # [4, 5]

    # ── Observation model parameters ──
    tau_readout = numpyro.sample(
        "tau_readout", dist.LogNormal(-1.5, 0.5),
    )  # scalar, ~0.22
    log_phi = numpyro.sample("log_phi", dist.Normal(4.0, 1.0))  # ~55
    sigma_outcome = numpyro.sample(
        "sigma_outcome", dist.HalfNormal(5.0),
    )
    log_sigma_outcome = numpyro.deterministic(
        "log_sigma_outcome", jnp.log(sigma_outcome + 1e-8),
    )
    numpyro.deterministic("phi", jnp.exp(log_phi))

    # ── Level 3: Scenario parameters + likelihood ──
    plate_kwargs = {"name": "scenarios", "size": n_scenarios}
    if batch_size is not None and batch_size < n_scenarios:
        plate_kwargs["subsample_size"] = batch_size

    with numpyro.plate(**plate_kwargs) as idx:
        # Domain and covariate lookups
        d = data.domain_indices[idx]       # [batch]
        x = data.covariates[idx]           # [batch, 5]

        # Scenario-level prior: θ_s ~ Normal(μ_d + B @ x_s, σ_d)
        mu_s = mu_domain[d] + x @ B.T     # [batch, 4]
        sigma_s = sigma_domain[d]          # [batch, 4]

        theta_s = numpyro.sample(
            "theta_s", dist.Normal(mu_s, sigma_s).to_event(1),
        )  # [batch, 4]

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
        # pro_fracs: [batch, max_rounds], final_pcts: [batch]

        numpyro.deterministic("sim_final_pcts", final_pcts)

        # ── Likelihood ──
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


# ── SVI Inference ────────────────────────────────────────────────

def run_svi(
    data: CalibrationData,
    n_steps: int = 2000,
    lr: float = 0.005,
    batch_size: int = 50,
    seed: int = 42,
    log_every: int = 100,
):
    """Run SVI with AutoLowRankMultivariateNormal guide.

    Args:
        data: CalibrationData.
        n_steps: Number of SVI optimization steps.
        lr: Learning rate for Adam.
        batch_size: Mini-batch size (None = full batch).
        seed: Random seed.
        log_every: Print loss every N steps.

    Returns:
        (svi_result, losses, guide)
    """
    n_scenarios = data.domain_indices.shape[0]
    if batch_size is not None and batch_size >= n_scenarios:
        batch_size = None  # full batch

    def model_fn():
        return hierarchical_model(data, batch_size=batch_size)

    # Provide explicit initial values to bypass numpyro's while_loop search
    # which can't trace through the vmap+scan simulation
    n_domains_val = data.n_domains
    n_scenarios_val = data.domain_indices.shape[0]
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
            print(f"  Step {step+1:5d}/{n_steps}  loss={loss:.2f}  avg50={avg_loss:.2f}")

    svi_result = svi.get_params(svi_state)
    losses = jnp.array(losses)

    return svi_result, losses, guide


# ── NUTS Inference ───────────────────────────────────────────────

def run_nuts(
    data: CalibrationData,
    n_warmup: int = 200,
    n_samples: int = 500,
    seed: int = 42,
    svi_params=None,
    guide=None,
):
    """Run NUTS sampler (slow for large datasets — use SVI by default).

    Args:
        data: CalibrationData.
        n_warmup: Warmup steps.
        n_samples: Number of posterior samples.
        seed: Random seed.
        svi_params: Optional SVI result for init_to_value initialization.
        guide: Optional guide (needed if svi_params provided).

    Returns:
        mcmc_samples (dict of arrays).
    """
    def model_fn():
        return hierarchical_model(data, batch_size=None)

    # Initialize from SVI if available
    init_strategy = None
    if svi_params is not None and guide is not None:
        rng_key = jax.random.PRNGKey(seed + 1)
        predictive = Predictive(guide, params=svi_params, num_samples=1)
        init_samples = predictive(rng_key)
        init_values = {k: v[0] for k, v in init_samples.items()
                       if not k.endswith("_base")}
        init_strategy = init_to_value(values=init_values)

    kernel = NUTS(
        model_fn,
        target_accept_prob=0.8,
        init_strategy=init_strategy,
    )
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples)
    mcmc.run(jax.random.PRNGKey(seed))
    mcmc.print_summary()

    return mcmc.get_samples()


# ── Posterior Extraction ─────────────────────────────────────────

def _summarize(arr, axis=0):
    """Compute mean and 95% CI."""
    mean = jnp.mean(arr, axis=axis)
    lo = jnp.percentile(arr, 2.5, axis=axis)
    hi = jnp.percentile(arr, 97.5, axis=axis)
    return {"mean": mean, "ci95_lo": lo, "ci95_hi": hi}


def extract_posteriors(
    samples: dict,
    guide=None,
    svi_params=None,
    domain_names: list[str] = None,
    scenario_ids: list[str] = None,
    n_posterior_samples: int = 1000,
    seed: int = 0,
) -> dict:
    """Extract structured posteriors from SVI or MCMC results.

    For SVI: pass guide + svi_params to draw posterior samples.
    For MCMC: pass samples dict directly.

    Returns structured dict with global, domain, scenario, covariate,
    and observation parameter posteriors.
    """
    # Draw samples from SVI guide if needed
    if guide is not None and svi_params is not None:
        predictive = Predictive(guide, params=svi_params,
                                num_samples=n_posterior_samples)
        rng = jax.random.PRNGKey(seed)
        samples = predictive(rng)

    result = {}

    # ── Global ──
    result["global"] = {
        "mu_global": _summarize(samples["mu_global"]),
        "sigma_global": _summarize(samples["sigma_global"]),
    }

    # ── Domains ──
    if domain_names is not None:
        result["domains"] = {}
        mu_d = samples["mu_domain"]      # [S, n_domains, 4]
        sigma_d = samples["sigma_domain"]
        for i, name in enumerate(domain_names):
            result["domains"][name] = {
                "mu_d": _summarize(mu_d[:, i, :]),
                "sigma_d": _summarize(sigma_d[:, i, :]),
            }

    # ── Scenarios ──
    if scenario_ids is not None and "theta_s" in samples:
        result["scenarios"] = {}
        theta = samples["theta_s"]  # [S, n_scenarios, 4]
        for i, sid in enumerate(scenario_ids):
            result["scenarios"][sid] = {
                "theta_s": _summarize(theta[:, i, :]),
            }

    # ── Covariates ──
    if "B" in samples:
        B_samples = samples["B"]  # [S, 4, 5]
        result["covariates"] = {
            "B": _summarize(B_samples),
            "significant": [],
        }
        # Check which B elements are significantly ≠ 0
        # (95% CI excludes zero)
        B_lo = jnp.percentile(B_samples, 2.5, axis=0)
        B_hi = jnp.percentile(B_samples, 97.5, axis=0)
        B_mean = jnp.mean(B_samples, axis=0)
        for p in range(N_PARAMS):
            for c in range(N_COVARIATES):
                if (B_lo[p, c] > 0) or (B_hi[p, c] < 0):
                    result["covariates"]["significant"].append({
                        "param": PARAM_NAMES[p],
                        "covariate": COVARIATE_NAMES[c],
                        "effect": float(B_mean[p, c]),
                        "ci95": (float(B_lo[p, c]), float(B_hi[p, c])),
                    })

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

    return result
