"""End-to-end tests for the observation model.

Tests:
  1. Synthetic data: BetaBinomial likelihood is finite and sensible
  2. Outcome-only likelihood: finite and peaks near true value
  3. Gradient check: jax.grad through both likelihoods
  4. Variant coherence: full and outcome-only agree in direction
  5. Diagnostics: outputs are reasonable
  6. Scenario loader: loads Brexit JSON correctly
"""

import sys
import os
import json

import jax
import jax.numpy as jnp

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.dynamics.opinion_dynamics_jax import ScenarioData, build_sparse_interaction, simulate_scenario
from src.dynamics.param_utils import get_default_params
from src.observation.observation_model import (
    log_likelihood_full,
    log_likelihood_outcome,
    compute_diagnostics,
    load_scenario_observations,
    build_scenario_data_from_json,
    ObservationData,
    _beta_binomial_logpmf,
)


def _make_synthetic_scenario(n_agents=12, n_rounds=6, seed=42):
    """Build a synthetic ScenarioData for testing."""
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    positions = jax.random.uniform(k1, (n_agents,), minval=-0.8, maxval=0.8)
    # 4 elite (type=0), 8 citizen (type=1)
    agent_types = jnp.array([0]*4 + [1]*8, dtype=jnp.int32)[:n_agents]
    rigidities = jnp.where(agent_types == 0, 0.7, 0.3)
    tolerances = jnp.where(agent_types == 0, 0.3, 0.6)
    influences = jax.random.uniform(k2, (n_agents,), minval=0.2, maxval=0.9)

    events = jnp.zeros((n_rounds, 2))
    events = events.at[2, 0].set(0.3)  # shock in round 3
    events = events.at[2, 1].set(0.5)

    llm_shifts = jax.random.normal(k3, (n_rounds, n_agents)) * 0.05

    interaction = build_sparse_interaction(influences, seed=seed)

    return ScenarioData(
        initial_positions=positions,
        agent_types=agent_types,
        agent_rigidities=rigidities,
        agent_tolerances=tolerances,
        events=events,
        llm_shifts=llm_shifts,
        interaction_matrix=interaction,
    )


def _make_synthetic_observations(n_rounds=6, sim_pro_fractions=None):
    """Build synthetic ObservationData.

    If sim_pro_fractions provided, generate observations near those values.
    Otherwise use fixed values.
    """
    if sim_pro_fractions is not None:
        pro_pcts = sim_pro_fractions * 100.0  # convert to %
    else:
        pro_pcts = jnp.array([42.0, 43.0, 44.0, 43.0, 45.0, 44.0])[:n_rounds]

    sample_sizes = jnp.array([1000.0, 1000.0, 0.0, 1200.0, 0.0, 1000.0])[:n_rounds]
    verified_mask = jnp.array([True, True, False, True, False, True])[:n_rounds]

    return ObservationData(
        pro_pcts=pro_pcts,
        sample_sizes=sample_sizes,
        verified_mask=verified_mask,
        ground_truth_pro_pct=51.89,
    )


def test_beta_binomial_logpmf():
    """Test BetaBinomial log-PMF is finite and sensible."""
    print("Test 1: BetaBinomial log-PMF...")

    # k=50 successes out of n=100, with α=5, β=5 (uniform-ish)
    ll = _beta_binomial_logpmf(
        k=jnp.array(50.0),
        n=jnp.array(100.0),
        alpha=jnp.array(5.0),
        beta=jnp.array(5.0),
    )
    assert jnp.isfinite(ll), f"BetaBinomial logpmf not finite: {ll}"
    assert ll < 0, f"Log-probability should be negative: {ll}"

    # Higher concentration → tighter around mode
    ll_tight = _beta_binomial_logpmf(
        k=jnp.array(50.0), n=jnp.array(100.0),
        alpha=jnp.array(50.0), beta=jnp.array(50.0),
    )
    ll_wide = _beta_binomial_logpmf(
        k=jnp.array(50.0), n=jnp.array(100.0),
        alpha=jnp.array(1.0), beta=jnp.array(1.0),
    )
    # At mode, tight distribution should give higher probability
    assert ll_tight > ll_wide, f"Expected tight > wide: {ll_tight} vs {ll_wide}"

    print(f"  logpmf(50|100, 5, 5) = {ll:.4f}")
    print(f"  logpmf(50|100, 50, 50) = {ll_tight:.4f}  (tight)")
    print(f"  logpmf(50|100, 1, 1) = {ll_wide:.4f}  (wide)")
    print("  PASSED")


def test_full_likelihood():
    """Test full BetaBinomial likelihood runs and returns finite scalar."""
    print("\nTest 2: Full BetaBinomial likelihood...")

    params = get_default_params()
    scenario = _make_synthetic_scenario()

    # First simulate to get pro fractions for realistic observations
    result = simulate_scenario(params, scenario)
    obs = _make_synthetic_observations(
        n_rounds=6,
        sim_pro_fractions=result["pro_fraction"],
    )

    log_phi = jnp.log(jnp.array(50.0))
    ll = log_likelihood_full(params, scenario, obs, log_phi)

    assert jnp.isfinite(ll), f"Full likelihood not finite: {ll}"
    assert ll.shape == (), f"Expected scalar, got shape {ll.shape}"

    print(f"  log_likelihood_full = {ll:.4f}")
    print(f"  phi = {jnp.exp(log_phi):.1f}")
    print("  PASSED")


def test_outcome_likelihood():
    """Test outcome-only Normal likelihood."""
    print("\nTest 3: Outcome-only likelihood...")

    params = get_default_params()
    scenario = _make_synthetic_scenario()

    result = simulate_scenario(params, scenario)
    sim_final = result["final_pro_pct"]

    # Test at true value → should be higher than at far value
    log_sigma = jnp.log(jnp.array(5.0))

    ll_near = log_likelihood_outcome(params, scenario, sim_final, log_sigma)
    ll_far = log_likelihood_outcome(params, scenario, sim_final + 30.0, log_sigma)

    assert jnp.isfinite(ll_near), f"Near ll not finite: {ll_near}"
    assert jnp.isfinite(ll_far), f"Far ll not finite: {ll_far}"
    assert ll_near > ll_far, f"Expected near > far: {ll_near} vs {ll_far}"

    print(f"  sim_final_pro_pct = {sim_final:.2f}")
    print(f"  ll(at true) = {ll_near:.4f}")
    print(f"  ll(+30pp away) = {ll_far:.4f}")
    print("  PASSED")


def test_gradients():
    """Test jax.grad flows through both likelihoods."""
    print("\nTest 4: Gradient check...")

    params = get_default_params()
    scenario = _make_synthetic_scenario()
    result = simulate_scenario(params, scenario)
    obs = _make_synthetic_observations(
        n_rounds=6,
        sim_pro_fractions=result["pro_fraction"],
    )

    # Grad of full likelihood w.r.t. alpha_herd
    def loss_full(alpha_herd):
        p = {**params, "alpha_herd": alpha_herd}
        return log_likelihood_full(p, scenario, obs, jnp.log(jnp.array(50.0)))

    grad_full = jax.grad(loss_full)(params["alpha_herd"])
    assert jnp.isfinite(grad_full), f"Full grad not finite: {grad_full}"
    print(f"  d(ll_full)/d(alpha_herd) = {grad_full:.6f}")

    # Grad of outcome likelihood w.r.t. alpha_herd
    def loss_outcome(alpha_herd):
        p = {**params, "alpha_herd": alpha_herd}
        return log_likelihood_outcome(p, scenario, jnp.array(51.89), jnp.log(jnp.array(5.0)))

    grad_outcome = jax.grad(loss_outcome)(params["alpha_herd"])
    assert jnp.isfinite(grad_outcome), f"Outcome grad not finite: {grad_outcome}"
    print(f"  d(ll_outcome)/d(alpha_herd) = {grad_outcome:.6f}")

    # Grad w.r.t. log_phi
    def loss_phi(log_phi):
        return log_likelihood_full(params, scenario, obs, log_phi)

    grad_phi = jax.grad(loss_phi)(jnp.log(jnp.array(50.0)))
    assert jnp.isfinite(grad_phi), f"Phi grad not finite: {grad_phi}"
    print(f"  d(ll_full)/d(log_phi) = {grad_phi:.6f}")

    print("  PASSED")


def test_diagnostics():
    """Test diagnostic computation."""
    print("\nTest 5: Diagnostics...")

    params = get_default_params()
    scenario = _make_synthetic_scenario()
    obs = _make_synthetic_observations(n_rounds=6)

    diag = compute_diagnostics(params, scenario, obs)

    assert jnp.isfinite(diag["outcome_error"]), "outcome_error not finite"
    assert jnp.isfinite(diag["trajectory_mae"]), "trajectory_mae not finite"
    assert jnp.isfinite(diag["trajectory_rmse"]), "trajectory_rmse not finite"
    assert diag["trajectory_mae"] >= 0, "MAE should be non-negative"
    assert diag["trajectory_rmse"] >= 0, "RMSE should be non-negative"
    assert diag["trajectory_rmse"] >= diag["trajectory_mae"], "RMSE >= MAE"
    assert diag["simulated_trajectory"].shape == (6,), \
        f"Expected (6,), got {diag['simulated_trajectory'].shape}"

    print(f"  outcome_error = {diag['outcome_error']:.2f}pp")
    print(f"  trajectory_mae = {diag['trajectory_mae']:.2f}pp")
    print(f"  trajectory_rmse = {diag['trajectory_rmse']:.2f}pp")
    print(f"  simulated_final = {diag['simulated_final']:.2f}%")
    print("  PASSED")


def test_scenario_loader():
    """Test loading the Brexit scenario."""
    print("\nTest 6: Scenario loader (Brexit)...")

    base = os.path.join(os.path.dirname(__file__), "../../calibration/empirical/scenarios")
    brexit_path = os.path.join(base, "POL-2016-BREXIT.json")

    if not os.path.exists(brexit_path):
        print("  SKIPPED — Brexit JSON not found")
        return

    scenario_dict, obs = load_scenario_observations(brexit_path)

    assert scenario_dict["id"] == "POL-2016-BREXIT"
    assert obs.pro_pcts.shape == (6,), f"Expected (6,), got {obs.pro_pcts.shape}"
    assert obs.sample_sizes.shape == (6,)
    assert obs.verified_mask.shape == (6,)
    assert obs.ground_truth_pro_pct == 51.89

    # Brexit has no verified sample_sizes (all null) → no verified rounds
    n_verified = int(jnp.sum(obs.verified_mask.astype(jnp.int32)))
    print(f"  scenario: {scenario_dict['id']}")
    print(f"  n_rounds: {scenario_dict['n_rounds']}")
    print(f"  polling pro_pcts: {obs.pro_pcts}")
    print(f"  sample_sizes: {obs.sample_sizes}")
    print(f"  verified_rounds: {n_verified}")
    print(f"  ground_truth: {obs.ground_truth_pro_pct}%")

    # Build ScenarioData and run simulation
    sd = build_scenario_data_from_json(scenario_dict)
    params = get_default_params()
    result = simulate_scenario(params, sd)

    print(f"  simulated_final: {result['final_pro_pct']:.2f}%")
    print("  PASSED")


def test_build_scenario_data():
    """Test building ScenarioData from empirical JSON."""
    print("\nTest 7: build_scenario_data_from_json...")

    base = os.path.join(os.path.dirname(__file__), "../../calibration/empirical/scenarios")
    brexit_path = os.path.join(base, "POL-2016-BREXIT.json")

    if not os.path.exists(brexit_path):
        print("  SKIPPED — Brexit JSON not found")
        return

    with open(brexit_path) as f:
        scenario = json.load(f)

    sd = build_scenario_data_from_json(scenario)

    assert sd.initial_positions.shape == (12,), f"Expected 12 agents, got {sd.initial_positions.shape}"
    assert sd.agent_types.shape == (12,)
    assert sd.events.shape == (6, 2), f"Expected (6,2), got {sd.events.shape}"
    assert sd.llm_shifts.shape == (6, 12)
    assert sd.interaction_matrix.shape == (12, 12)

    # Check events are correctly mapped
    # Round 2 (idx=1) should have Boris Johnson shock (mag=0.35, dir=0.5)
    assert sd.events[1, 0] > 0.3, f"Expected shock at round 2: {sd.events[1]}"

    # Verify simulation runs
    params = get_default_params()
    result = simulate_scenario(params, sd)
    assert jnp.isfinite(result["final_pro_pct"]), "Simulation produced NaN"

    print(f"  agents: {sd.initial_positions.shape[0]}")
    print(f"  events mapped: {jnp.sum(sd.events[:, 0] > 0)} rounds with shocks")
    print(f"  simulation final_pro_pct: {result['final_pro_pct']:.2f}%")
    print("  PASSED")


def test_variant_coherence():
    """Test that both likelihood variants agree on direction."""
    print("\nTest 8: Variant coherence...")

    params = get_default_params()
    scenario = _make_synthetic_scenario()
    result = simulate_scenario(params, scenario)

    sim_final = result["final_pro_pct"]
    obs_near = _make_synthetic_observations(
        n_rounds=6,
        sim_pro_fractions=result["pro_fraction"],
    )
    obs_near = obs_near._replace(ground_truth_pro_pct=float(sim_final))

    obs_far = obs_near._replace(
        pro_pcts=obs_near.pro_pcts + 20.0,
        ground_truth_pro_pct=float(sim_final + 20.0),
    )

    log_phi = jnp.log(jnp.array(50.0))
    log_sigma = jnp.log(jnp.array(5.0))

    ll_full_near = log_likelihood_full(params, scenario, obs_near, log_phi)
    ll_full_far = log_likelihood_full(params, scenario, obs_far, log_phi)

    ll_out_near = log_likelihood_outcome(params, scenario, obs_near.ground_truth_pro_pct, log_sigma)
    ll_out_far = log_likelihood_outcome(params, scenario, obs_far.ground_truth_pro_pct, log_sigma)

    # Both should prefer the near observation
    assert ll_full_near > ll_full_far, \
        f"Full: near ({ll_full_near:.2f}) should > far ({ll_full_far:.2f})"
    assert ll_out_near > ll_out_far, \
        f"Outcome: near ({ll_out_near:.2f}) should > far ({ll_out_far:.2f})"

    print(f"  Full:    near={ll_full_near:.2f}  far={ll_full_far:.2f}  Δ={ll_full_near-ll_full_far:.2f}")
    print(f"  Outcome: near={ll_out_near:.2f}  far={ll_out_far:.2f}  Δ={ll_out_near-ll_out_far:.2f}")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Observation Model — End-to-End Tests")
    print("=" * 60)

    test_beta_binomial_logpmf()
    test_full_likelihood()
    test_outcome_likelihood()
    test_gradients()
    test_diagnostics()
    test_scenario_loader()
    test_build_scenario_data()
    test_variant_coherence()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
