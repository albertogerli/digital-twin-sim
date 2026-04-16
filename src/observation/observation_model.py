"""Observation model — connects JAX DynamicsV2 simulator to empirical data.

Two likelihood variants:
  (A) Full BetaBinomial — per-round likelihood for scenarios with verified polling
  (B) Outcome-Only Normal — final outcome only, for all scenarios

Both are pure JAX, jit-compilable, and differentiable (for use with jax.grad
and NumPyro's NUTS/SVI).

The readout uses the existing soft pro_fraction from simulate_scenario
(τ=0.02 sigmoid), so no additional τ_readout parameter is needed.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.nn import sigmoid
import json
import os

from ..dynamics.opinion_dynamics_jax import simulate_scenario, ScenarioData, build_sparse_interaction


# ── Observation Data ─────────────────────────────────────────────

class ObservationData(NamedTuple):
    """Empirical observations for a single scenario.

    Fields:
        pro_pcts: [n_rounds] observed pro% per round (0-100 scale).
                  Rounds without data should be NaN.
        sample_sizes: [n_rounds] sample size per round.
                      Rounds without polling data should be 0.
        verified_mask: [n_rounds] boolean — True for rounds with verified polling.
        ground_truth_pro_pct: scalar — final outcome pro% (0-100).
    """
    pro_pcts: jnp.ndarray        # [n_rounds]
    sample_sizes: jnp.ndarray    # [n_rounds] (int, 0 = no data)
    verified_mask: jnp.ndarray   # [n_rounds] (bool)
    ground_truth_pro_pct: float  # scalar


# ── (A) Full BetaBinomial Likelihood ─────────────────────────────

def _beta_binomial_logpmf(k, n, alpha, beta):
    """Log-PMF of BetaBinomial(n, α, β) at k.

    log P(k | n, α, β) = log C(n,k) + log B(k+α, n-k+β) - log B(α, β)

    where B is the Beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b).

    Pure JAX, differentiable w.r.t. α, β.
    """
    log_comb = (
        jax.lax.lgamma(n + 1.0)
        - jax.lax.lgamma(k + 1.0)
        - jax.lax.lgamma(n - k + 1.0)
    )
    log_beta_num = (
        jax.lax.lgamma(k + alpha)
        + jax.lax.lgamma(n - k + beta)
        - jax.lax.lgamma(n + alpha + beta)
    )
    log_beta_den = (
        jax.lax.lgamma(alpha)
        + jax.lax.lgamma(beta)
        - jax.lax.lgamma(alpha + beta)
    )
    return log_comb + log_beta_num - log_beta_den


def log_likelihood_full(
    params: dict,
    scenario_data: ScenarioData,
    observations: ObservationData,
    log_phi: jnp.ndarray,
) -> jnp.ndarray:
    """Full BetaBinomial log-likelihood over verified rounds.

    For each verified round t:
        q(t) = simulated pro_fraction(t)  (from simulator, 0-1 scale)
        φ = exp(log_phi)                  (concentration, >0)
        α(t) = q(t) * φ
        β(t) = (1 - q(t)) * φ
        y(t) ~ BetaBinomial(n=sample_size(t), α(t), β(t))
        where y(t) = round(pro_pct(t)/100 * sample_size(t))

    Args:
        params: Full parameter dict (calibrable + frozen).
        scenario_data: ScenarioData for the JAX simulator.
        observations: ObservationData with polling trajectory.
        log_phi: Scalar — log concentration parameter.

    Returns:
        Scalar log-likelihood (sum over verified rounds).
    """
    result = simulate_scenario(params, scenario_data)
    q = result["pro_fraction"]  # [n_rounds], 0-1 scale

    phi = jnp.exp(log_phi)

    # Observed counts: k(t) = round(pro_pct/100 * sample_size)
    k = jnp.round(observations.pro_pcts / 100.0 * observations.sample_sizes)
    n = observations.sample_sizes

    # BetaBinomial parameters
    # Clamp q to (eps, 1-eps) to avoid α=0 or β=0
    eps = 1e-4
    q_clamped = jnp.clip(q, eps, 1.0 - eps)
    alpha = q_clamped * phi
    beta = (1.0 - q_clamped) * phi

    # Per-round log-likelihood
    ll_per_round = _beta_binomial_logpmf(k, n, alpha, beta)

    # Mask: only sum over verified rounds with sample_size > 0
    mask = observations.verified_mask & (observations.sample_sizes > 0)
    ll = jnp.sum(jnp.where(mask, ll_per_round, 0.0))

    return ll


# ── (B) Outcome-Only Normal Likelihood ───────────────────────────

def log_likelihood_outcome(
    params: dict,
    scenario_data: ScenarioData,
    ground_truth_pro_pct: jnp.ndarray,
    log_sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Normal log-likelihood on final outcome only.

    y_final ~ Normal(q_final * 100, σ)

    where q_final = simulated pro_fraction at last round (0-1),
    y_final = ground truth pro% (0-100).

    Args:
        params: Full parameter dict.
        scenario_data: ScenarioData for the JAX simulator.
        ground_truth_pro_pct: Scalar — observed final pro% (0-100).
        log_sigma: Scalar — log of observation noise σ.

    Returns:
        Scalar log-likelihood.
    """
    result = simulate_scenario(params, scenario_data)
    q_final = result["final_pro_pct"]  # 0-100 scale

    sigma = jnp.exp(log_sigma)
    residual = ground_truth_pro_pct - q_final

    # Normal log-likelihood (up to constant)
    ll = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) - 0.5 * (residual / sigma) ** 2
    return ll


# ── Diagnostics ──────────────────────────────────────────────────

def compute_diagnostics(
    params: dict,
    scenario_data: ScenarioData,
    observations: ObservationData,
) -> dict:
    """Compute ex-post diagnostic metrics.

    Returns:
        {
            'outcome_error': float — (simulated - observed) final pro%,
            'outcome_abs_error': float — |outcome_error|,
            'trajectory_mae': float — MAE over verified rounds (pro%),
            'trajectory_rmse': float — RMSE over verified rounds (pro%),
            'simulated_trajectory': array[n_rounds] — simulated pro% per round,
            'simulated_final': float — simulated final pro%,
        }
    """
    result = simulate_scenario(params, scenario_data)
    sim_pct = result["pro_fraction"] * 100.0  # [n_rounds], 0-100

    # Final outcome error
    outcome_error = result["final_pro_pct"] - observations.ground_truth_pro_pct

    # Trajectory errors (only verified rounds)
    mask = observations.verified_mask & (observations.sample_sizes > 0)
    residuals = sim_pct - observations.pro_pcts
    n_verified = jnp.sum(mask.astype(jnp.float32))

    # MAE and RMSE over verified rounds
    abs_residuals = jnp.abs(residuals)
    mae = jnp.sum(jnp.where(mask, abs_residuals, 0.0)) / jnp.maximum(n_verified, 1.0)
    rmse = jnp.sqrt(
        jnp.sum(jnp.where(mask, residuals ** 2, 0.0)) / jnp.maximum(n_verified, 1.0)
    )

    return {
        "outcome_error": outcome_error,
        "outcome_abs_error": jnp.abs(outcome_error),
        "trajectory_mae": mae,
        "trajectory_rmse": rmse,
        "simulated_trajectory": sim_pct,
        "simulated_final": result["final_pro_pct"],
    }


# ── Scenario Loader ──────────────────────────────────────────────

def load_scenario_observations(
    scenario_json_path: str,
) -> tuple[dict, ObservationData]:
    """Load empirical scenario JSON and build ObservationData.

    Reads the scenario JSON and its companion .meta.json (if present)
    to determine which polling rounds are verified.

    A round is considered "verified" if:
      1. Its meta provenance is "verified_url" or "verified_csv", AND
      2. Its sample_size is not null/None.

    If no .meta.json exists, all rounds with non-null sample_size are
    treated as verified.

    Args:
        scenario_json_path: Path to the scenario .json file.

    Returns:
        (scenario_dict, observations)
        - scenario_dict: raw JSON dict (for building ScenarioData externally)
        - observations: ObservationData ready for likelihood functions
    """
    with open(scenario_json_path, "r") as f:
        scenario = json.load(f)

    n_rounds = scenario["n_rounds"]
    polling = scenario.get("polling_trajectory", [])

    # Build polling arrays indexed by round (1-based in JSON → 0-based here)
    pro_pcts = jnp.full(n_rounds, jnp.nan)
    sample_sizes = jnp.zeros(n_rounds, dtype=jnp.float32)

    for entry in polling:
        r = entry["round"] - 1  # 0-based
        if 0 <= r < n_rounds:
            pro_pcts = pro_pcts.at[r].set(float(entry["pro_pct"]))
            ss = entry.get("sample_size")
            if ss is not None:
                sample_sizes = sample_sizes.at[r].set(float(ss))

    # Load meta.json for provenance info
    meta_path = scenario_json_path.replace(".json", ".meta.json")
    verified_provenance = set()
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        prov_list = meta.get("field_provenance", {}).get("polling_trajectory", [])
        for i, prov in enumerate(prov_list):
            if prov in ("verified_url", "verified_csv"):
                verified_provenance.add(i)

    # Build verified mask
    verified_mask = jnp.zeros(n_rounds, dtype=jnp.bool_)
    for r in range(n_rounds):
        has_sample = sample_sizes[r] > 0
        if os.path.exists(meta_path):
            is_verified = r in verified_provenance
        else:
            # No meta → trust all rounds with sample_size
            is_verified = True
        if has_sample and is_verified:
            verified_mask = verified_mask.at[r].set(True)

    # Ground truth
    gt = scenario.get("ground_truth_outcome", {})
    ground_truth_pro_pct = float(gt.get("pro_pct", 50.0))

    obs = ObservationData(
        pro_pcts=pro_pcts,
        sample_sizes=sample_sizes,
        verified_mask=verified_mask,
        ground_truth_pro_pct=ground_truth_pro_pct,
    )

    return scenario, obs


# ── Scenario Builder Helper ──────────────────────────────────────

def build_scenario_data_from_json(
    scenario: dict,
    seed: int = 42,
) -> ScenarioData:
    """Build a ScenarioData from an empirical scenario JSON dict.

    Maps the heterogeneous agent list (elite, institutional, citizen_cluster)
    to the flat arrays expected by the JAX simulator.

    Agent properties:
        - type: 0=elite/institutional, 1=citizen_cluster
        - rigidity: elite=0.7, institutional=0.8, citizen=0.3
        - tolerance: elite=0.3, institutional=0.4, citizen=0.6
        - influence: from JSON

    Events are mapped to (magnitude, direction) per round.
    LLM shifts are set to zero (no LLM forcing in calibration mode).

    Args:
        scenario: Parsed scenario JSON dict.
        seed: Random seed for interaction matrix.

    Returns:
        ScenarioData ready for simulate_scenario.
    """
    agents = scenario["agents"]
    n_agents = len(agents)
    n_rounds = scenario["n_rounds"]

    # Agent arrays
    positions = []
    agent_types = []
    rigidities = []
    tolerances = []
    influences = []

    type_map = {"elite": 0, "institutional": 0, "citizen_cluster": 1}
    rigidity_map = {"elite": 0.7, "institutional": 0.8, "citizen_cluster": 0.3}
    tolerance_map = {"elite": 0.3, "institutional": 0.4, "citizen_cluster": 0.6}

    for a in agents:
        atype = a["type"]
        positions.append(a["initial_position"])
        agent_types.append(type_map.get(atype, 1))
        rigidities.append(rigidity_map.get(atype, 0.5))
        tolerances.append(tolerance_map.get(atype, 0.5))
        influences.append(a.get("influence", 0.5))

    # Implicit "Public Opinion" agent from polling trajectory.
    # Anchors simulation to empirical baseline sentiment. Low rigidity
    # makes it responsive to events; moderate influence creates social
    # pressure. Position from first polling data point.
    polling = scenario.get("polling_trajectory", [])
    if polling:
        first_pro = polling[0].get("pro_pct", 50.0)
        pub_position = (first_pro / 100.0) * 2.0 - 1.0  # map 0-100% → [-1, +1]
        positions.append(pub_position)
        agent_types.append(1)     # citizen
        rigidities.append(0.1)    # very reactive to events
        tolerances.append(0.9)    # broad tolerance
        influences.append(0.7)    # moderate-high social influence
        n_agents += 1

    initial_positions = jnp.array(positions, dtype=jnp.float32)
    agent_types_arr = jnp.array(agent_types, dtype=jnp.int32)
    agent_rigidities = jnp.array(rigidities, dtype=jnp.float32)
    agent_tolerances = jnp.array(tolerances, dtype=jnp.float32)
    influences_arr = jnp.array(influences, dtype=jnp.float32)

    # Events: map to per-round (magnitude, direction)
    events_arr = jnp.zeros((n_rounds, 2), dtype=jnp.float32)
    for evt in scenario.get("events", []):
        r = evt["round"] - 1  # 0-based
        if 0 <= r < n_rounds:
            mag = evt.get("shock_magnitude", 0.0)
            dir_ = evt.get("shock_direction", 0.0)
            # If multiple events in same round, sum magnitudes
            events_arr = events_arr.at[r, 0].add(float(mag))
            events_arr = events_arr.at[r, 1].set(float(dir_))

    # LLM shifts: zero in calibration mode
    llm_shifts = jnp.zeros((n_rounds, n_agents), dtype=jnp.float32)

    # Interaction matrix
    interaction_matrix = build_sparse_interaction(influences_arr, seed=seed)

    return ScenarioData(
        initial_positions=initial_positions,
        agent_types=agent_types_arr,
        agent_rigidities=agent_rigidities,
        agent_tolerances=agent_tolerances,
        events=events_arr,
        llm_shifts=llm_shifts,
        interaction_matrix=interaction_matrix,
    )
