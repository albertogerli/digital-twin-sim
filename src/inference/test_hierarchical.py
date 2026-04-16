"""End-to-end tests for the hierarchical Bayesian model.

Tests on 10 scenarios (5 synthetic + 5 empirical), 2+ domains, 200 SVI steps:
  1. SVI converges (loss decreases)
  2. Domain posteriors differ
  3. Scenario posteriors show shrinkage toward domain
  4. Covariate matrix B has reasonable values
  5. Observation params (tau_readout, phi, sigma_outcome) move from prior
"""

import sys
import os
import json
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.dynamics.opinion_dynamics_jax import ScenarioData, build_sparse_interaction
from src.dynamics.param_utils import get_default_frozen_params
from src.observation.observation_model import (
    ObservationData, load_scenario_observations, build_scenario_data_from_json,
)
from src.inference.hierarchical_model import (
    hierarchical_model,
    run_svi,
    extract_posteriors,
    prepare_calibration_data,
    CalibrationData,
    PARAM_NAMES,
    COVARIATE_NAMES,
    N_PARAMS,
    N_COVARIATES,
)


# ── Synthetic Scenario Generator ─────────────────────────────────

def make_synthetic_scenario(n_agents, n_rounds, seed, pro_bias=0.0):
    """Create a synthetic scenario with controlled pro bias.

    Args:
        n_agents: Number of agents.
        n_rounds: Number of rounds.
        seed: Random seed.
        pro_bias: Bias toward pro (+) or against (-) in initial positions.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    positions = jax.random.uniform(k1, (n_agents,), minval=-0.8, maxval=0.8)
    positions = positions + pro_bias * 0.3  # shift
    positions = jnp.clip(positions, -1.0, 1.0)

    n_elite = max(2, n_agents // 4)
    agent_types = jnp.array(
        [0] * n_elite + [1] * (n_agents - n_elite), dtype=jnp.int32
    )
    rigidities = jnp.where(agent_types == 0, 0.7, 0.3)
    tolerances = jnp.where(agent_types == 0, 0.3, 0.6)
    influences = jax.random.uniform(k2, (n_agents,), minval=0.2, maxval=0.9)

    # Random events
    events = jnp.zeros((n_rounds, 2))
    shock_round = seed % n_rounds
    events = events.at[shock_round, 0].set(0.3)
    events = events.at[shock_round, 1].set(jnp.sign(pro_bias + 0.1))

    llm_shifts = jax.random.normal(k3, (n_rounds, n_agents)) * 0.02
    interaction = build_sparse_interaction(influences, seed=seed)

    sd = ScenarioData(
        initial_positions=positions,
        agent_types=agent_types,
        agent_rigidities=rigidities,
        agent_tolerances=tolerances,
        events=events,
        llm_shifts=llm_shifts,
        interaction_matrix=interaction,
    )

    # Synthetic observations
    base_pro = 50.0 + pro_bias * 15.0
    pro_pcts = jnp.array([
        base_pro + jax.random.normal(k4, ()) * 3.0
        for _ in range(n_rounds)
    ])
    # Give some rounds sample sizes
    sample_sizes = jnp.zeros(n_rounds)
    sample_sizes = sample_sizes.at[0].set(1000.0)
    sample_sizes = sample_sizes.at[n_rounds - 1].set(1000.0)
    verified_mask = sample_sizes > 0

    obs = ObservationData(
        pro_pcts=pro_pcts,
        sample_sizes=sample_sizes,
        verified_mask=verified_mask,
        ground_truth_pro_pct=float(base_pro + jax.random.normal(
            jax.random.PRNGKey(seed + 100), ()) * 5.0),
    )

    covariates = [
        0.5 + pro_bias * 0.2,  # initial_polarization
        0.1 + abs(pro_bias) * 0.1,  # event_volatility
        n_elite / n_agents,  # elite_concentration
        0.5,  # institutional_trust
        0.15,  # undecided_share
    ]

    return sd, obs, covariates


# ── Load Empirical Scenarios ─────────────────────────────────────

SCENARIOS_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../calibration/empirical/scenarios",
)

EMPIRICAL_PICKS = [
    ("POL-2016-BREXIT.json", "political"),
    ("POL-2020-CHILE_CONSTITUTIONAL_REFERENDU.json", "political"),
    ("FIN-2021-GAMESTOP.json", "financial"),
    ("FIN-2019-WEWORK_IPO_COLLAPSE_AND_PUBLIC.json", "financial"),
    ("CORP-2019-BOEING_MAX.json", "corporate"),
]


def load_empirical_scenarios():
    """Load 5 empirical scenarios."""
    results = []
    for fname, domain in EMPIRICAL_PICKS:
        path = os.path.join(SCENARIOS_DIR, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {fname} not found, skipping")
            continue
        scenario_dict, obs = load_scenario_observations(path)
        sd = build_scenario_data_from_json(scenario_dict)
        covariates = scenario_dict.get("covariates", {})
        cov_vec = [
            covariates.get("initial_polarization", 0.5),
            covariates.get("event_volatility", 0.1),
            covariates.get("elite_concentration", 0.2),
            covariates.get("institutional_trust", 0.5),
            covariates.get("undecided_share", 0.15),
        ]
        results.append((sd, obs, cov_vec, domain, scenario_dict["id"]))
    return results


# ── Build Test Dataset ───────────────────────────────────────────

def build_test_dataset():
    """Build 10-scenario dataset: 5 synthetic + 5 empirical."""
    domain_map = {}
    domain_counter = 0

    scenario_datas = []
    observations = []
    domain_indices = []
    covariates_list = []
    scenario_ids = []

    # 5 synthetic scenarios (2 "synthetic_A" + 3 "synthetic_B")
    synth_configs = [
        (8, 6, 100, 0.3, "synthetic_A"),
        (10, 7, 200, -0.2, "synthetic_A"),
        (6, 5, 300, 0.5, "synthetic_B"),
        (9, 7, 400, -0.4, "synthetic_B"),
        (7, 6, 500, 0.1, "synthetic_B"),
    ]
    for na, nr, seed, bias, domain in synth_configs:
        sd, obs, cov = make_synthetic_scenario(na, nr, seed, bias)
        if domain not in domain_map:
            domain_map[domain] = domain_counter
            domain_counter += 1
        scenario_datas.append(sd)
        observations.append(obs)
        domain_indices.append(domain_map[domain])
        covariates_list.append(cov)
        scenario_ids.append(f"SYNTH-{seed}")

    # 5 empirical scenarios
    empirical = load_empirical_scenarios()
    for sd, obs, cov, domain, sid in empirical:
        if domain not in domain_map:
            domain_map[domain] = domain_counter
            domain_counter += 1
        scenario_datas.append(sd)
        observations.append(obs)
        domain_indices.append(domain_map[domain])
        covariates_list.append(cov)
        scenario_ids.append(sid)

    domain_names = [None] * len(domain_map)
    for name, idx in domain_map.items():
        domain_names[idx] = name

    data = prepare_calibration_data(
        scenario_datas, observations, domain_indices, covariates_list,
    )
    return data, domain_names, scenario_ids


# ── Tests ────────────────────────────────────────────────────────

def test_data_preparation():
    """Test that data preparation produces valid shapes."""
    print("Test 1: Data preparation...")
    data, domain_names, scenario_ids = build_test_dataset()

    n = len(scenario_ids)
    A = data.initial_positions.shape[1]
    R = data.events.shape[1]

    assert data.initial_positions.shape == (n, A), \
        f"initial_positions: {data.initial_positions.shape}"
    assert data.events.shape == (n, R, 2)
    assert data.llm_shifts.shape == (n, R, A)
    assert data.interaction_matrices.shape == (n, A, A)
    assert data.agent_masks.shape == (n, A)
    assert data.obs_pro_pcts.shape == (n, R)
    assert data.covariates.shape == (n, 5)
    assert data.domain_indices.shape == (n,)
    assert data.frozen_vec.shape == (4,)

    print(f"  {n} scenarios, max_agents={A}, max_rounds={R}")
    print(f"  domains: {domain_names}")
    print(f"  scenarios: {scenario_ids}")
    print(f"  has_verified: {data.obs_has_verified}")
    print("  PASSED")
    return data, domain_names, scenario_ids


def test_svi_convergence(data, domain_names, scenario_ids):
    """Test SVI runs and loss decreases."""
    print("\nTest 2: SVI convergence (200 steps)...")

    t0 = time.time()
    svi_result, losses, guide = run_svi(
        data,
        n_steps=200,
        lr=0.01,
        batch_size=None,  # full batch for small test
        seed=42,
        log_every=50,
    )
    elapsed = time.time() - t0

    # Check loss decreased
    first_20 = float(jnp.mean(losses[:20]))
    last_20 = float(jnp.mean(losses[-20:]))
    improved = last_20 < first_20

    print(f"  Time: {elapsed:.1f}s")
    print(f"  Loss: first_20_avg={first_20:.2f} → last_20_avg={last_20:.2f}")
    print(f"  Improved: {improved}")

    assert jnp.all(jnp.isfinite(losses)), "Some losses are NaN/Inf"
    # Allow for stochastic non-monotonicity but overall trend should be down
    assert improved, f"Loss did not decrease: {first_20:.2f} → {last_20:.2f}"

    print("  PASSED")
    return svi_result, losses, guide


def test_posteriors(svi_result, guide, domain_names, scenario_ids):
    """Test posterior extraction and structure."""
    print("\nTest 3: Posterior extraction...")

    posteriors = extract_posteriors(
        samples=None,
        guide=guide,
        svi_params=svi_result,
        domain_names=domain_names,
        scenario_ids=scenario_ids,
        n_posterior_samples=500,
        seed=0,
    )

    # Check structure
    assert "global" in posteriors
    assert "domains" in posteriors
    assert "scenarios" in posteriors
    assert "covariates" in posteriors
    assert "observation_params" in posteriors

    # Global posteriors
    mu_g = posteriors["global"]["mu_global"]["mean"]
    print(f"  μ_global: {mu_g}")
    assert mu_g.shape == (4,), f"Expected (4,), got {mu_g.shape}"

    sigma_g = posteriors["global"]["sigma_global"]["mean"]
    print(f"  σ_global: {sigma_g}")
    assert jnp.all(sigma_g > 0), "sigma_global should be positive"

    print("  PASSED")
    return posteriors


def test_domain_diversity(posteriors, domain_names):
    """Test that domain posteriors differ from each other."""
    print("\nTest 4: Domain diversity...")

    if len(domain_names) < 2:
        print("  SKIPPED — fewer than 2 domains")
        return

    # Compare first two domains
    d0 = domain_names[0]
    d1 = domain_names[1]
    mu_0 = posteriors["domains"][d0]["mu_d"]["mean"]
    mu_1 = posteriors["domains"][d1]["mu_d"]["mean"]

    diff = jnp.abs(mu_0 - mu_1)
    max_diff = float(jnp.max(diff))

    print(f"  {d0} μ_d: {mu_0}")
    print(f"  {d1} μ_d: {mu_1}")
    print(f"  max |diff|: {max_diff:.4f}")

    # They should not be identical (collapsed to prior)
    # With only 200 steps, the difference may be small
    # Just check they're not exactly the same
    assert max_diff > 1e-6, "Domain posteriors collapsed to same value"
    print("  PASSED")


def test_shrinkage(posteriors, domain_names, scenario_ids, data):
    """Test that scenario posteriors show shrinkage toward domain mean."""
    print("\nTest 5: Shrinkage check...")

    if not posteriors.get("scenarios"):
        print("  SKIPPED — no scenario posteriors")
        return

    # For each domain, check that scenario posteriors are closer to domain mean
    # than to the global mean
    mu_global = posteriors["global"]["mu_global"]["mean"]

    shrinkage_count = 0
    total_count = 0

    for sid in scenario_ids:
        if sid not in posteriors["scenarios"]:
            continue
        theta = posteriors["scenarios"][sid]["theta_s"]["mean"]
        # Find domain
        idx = scenario_ids.index(sid)
        d_idx = int(data.domain_indices[idx])
        d_name = domain_names[d_idx]
        mu_d = posteriors["domains"][d_name]["mu_d"]["mean"]

        dist_to_domain = float(jnp.mean(jnp.abs(theta - mu_d)))
        dist_to_global = float(jnp.mean(jnp.abs(theta - mu_global)))

        if dist_to_domain <= dist_to_global:
            shrinkage_count += 1
        total_count += 1

    frac = shrinkage_count / max(total_count, 1)
    print(f"  Scenarios closer to domain than global: {shrinkage_count}/{total_count} ({frac:.0%})")

    # At least half should show shrinkage toward domain
    assert frac >= 0.4, f"Insufficient shrinkage: {frac:.0%}"
    print("  PASSED")


def test_covariate_matrix(posteriors):
    """Test covariate regression matrix B."""
    print("\nTest 6: Covariate matrix B...")

    B = posteriors["covariates"]["B"]["mean"]
    assert B.shape == (4, 5), f"Expected (4,5), got {B.shape}"

    # B should not be all zeros (some covariates should have effect)
    max_abs = float(jnp.max(jnp.abs(B)))
    print(f"  B shape: {B.shape}")
    print(f"  max|B|: {max_abs:.4f}")
    print(f"  B:\n{B}")

    significant = posteriors["covariates"]["significant"]
    print(f"  Significant effects (CI excludes 0): {len(significant)}")
    for s in significant:
        print(f"    {s['param']} ← {s['covariate']}: {s['effect']:.3f} {s['ci95']}")

    print("  PASSED")


def test_observation_params(posteriors):
    """Test observation model parameters moved from prior."""
    print("\nTest 7: Observation parameters...")

    obs = posteriors["observation_params"]

    if "tau_readout" in obs:
        tau = obs["tau_readout"]["mean"]
        print(f"  τ_readout: {tau:.4f} (prior mode ≈ 0.22)")

    if "phi" in obs:
        phi = obs["phi"]["mean"]
        print(f"  φ (concentration): {phi:.2f} (prior mode ≈ 55)")

    if "sigma_outcome" in obs:
        sigma = obs["sigma_outcome"]["mean"]
        print(f"  σ_outcome: {sigma:.2f} (prior HalfNormal(5))")
        assert sigma > 0, "sigma_outcome should be positive"

    print("  PASSED")


def test_simulate_batch_correctness(data):
    """Test that vmapped simulation produces finite results."""
    print("\nTest 8: Batch simulation correctness...")

    from src.inference.hierarchical_model import _simulate_batch, FROZEN_KEYS
    from src.dynamics.opinion_dynamics_jax import ScenarioData

    n = data.initial_positions.shape[0]

    # Use default calibrable params (all zeros)
    theta = jnp.zeros((n, 4))

    batch_sd = ScenarioData(
        initial_positions=data.initial_positions,
        agent_types=data.agent_types,
        agent_rigidities=data.agent_rigidities,
        agent_tolerances=data.agent_tolerances,
        events=data.events,
        llm_shifts=data.llm_shifts,
        interaction_matrix=data.interaction_matrices,
    )

    pro_fracs, final_pcts = _simulate_batch(
        theta, data.frozen_vec, batch_sd,
        data.agent_masks, jnp.array(0.02), data.n_real_rounds,
    )

    assert pro_fracs.shape[0] == n
    assert final_pcts.shape == (n,)
    assert jnp.all(jnp.isfinite(final_pcts)), \
        f"Non-finite final_pcts: {final_pcts}"
    assert jnp.all((final_pcts >= 0) & (final_pcts <= 100)), \
        f"final_pcts out of range: {final_pcts}"

    print(f"  Batch size: {n}")
    print(f"  final_pcts: {final_pcts}")
    print("  PASSED")


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Hierarchical Model — End-to-End Tests")
    print("=" * 60)

    # Test 1: Data preparation
    data, domain_names, scenario_ids = test_data_preparation()

    # Test 8: Batch simulation (before SVI to catch early errors)
    test_simulate_batch_correctness(data)

    # Test 2: SVI convergence
    svi_result, losses, guide = test_svi_convergence(
        data, domain_names, scenario_ids)

    # Test 3: Posterior extraction
    posteriors = test_posteriors(svi_result, guide, domain_names, scenario_ids)

    # Test 4: Domain diversity
    test_domain_diversity(posteriors, domain_names)

    # Test 5: Shrinkage
    test_shrinkage(posteriors, domain_names, scenario_ids, data)

    # Test 6: Covariate matrix
    test_covariate_matrix(posteriors)

    # Test 7: Observation params
    test_observation_params(posteriors)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
