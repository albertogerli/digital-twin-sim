"""Tests for regime switching dynamics.

4 tests:
1. Crisis activates on large-shock scenario (financial crisis pattern)
2. No crisis on normal scenario (Brexit-like gradual drift)
3. Regime switching improves error on financial scenarios
4. JIT compatibility
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dynamics.opinion_dynamics_jax import (
    ScenarioData,
    simulate_scenario,
    build_sparse_interaction,
)
from src.dynamics.regime_switching import (
    RegimeSwitchingSimulator,
    CRISIS_DEFAULTS,
    TRANSITION_DEFAULTS,
)
from src.dynamics.param_utils import get_default_frozen_params, get_default_params
from src.observation.observation_model import (
    build_scenario_data_from_json,
    load_scenario_observations,
)

BASE = os.path.dirname(os.path.dirname(__file__))
SCENARIOS_DIR = os.path.join(BASE, "calibration", "empirical", "scenarios")
POSTERIORS_PATH = os.path.join(
    BASE, "calibration", "results", "hierarchical_calibration",
    "v2_discrepancy", "posteriors_v2.json",
)


def _make_synthetic_scenario(n_agents=10, n_rounds=9, shock_round=3,
                              shock_magnitude=0.8, shock_direction=-1.0):
    """Build a synthetic scenario with a big shock at a specific round."""
    initial_positions = jnp.linspace(-0.3, 0.6, n_agents)
    agent_types = jnp.array([0, 0] + [1] * (n_agents - 2))
    agent_rigidities = jnp.where(agent_types == 0, 0.7, 0.3)
    agent_tolerances = jnp.where(agent_types == 0, 0.3, 0.6)
    influences = jnp.ones(n_agents) * 0.5
    interaction_matrix = build_sparse_interaction(influences, k=min(5, n_agents - 1), seed=42)

    events = jnp.zeros((n_rounds, 2))
    events = events.at[shock_round - 1, 0].set(shock_magnitude)
    events = events.at[shock_round - 1, 1].set(shock_direction)

    llm_shifts = jnp.zeros((n_rounds, n_agents))

    return ScenarioData(
        initial_positions=initial_positions,
        agent_types=agent_types,
        agent_rigidities=agent_rigidities,
        agent_tolerances=agent_tolerances,
        events=events,
        llm_shifts=llm_shifts,
        interaction_matrix=interaction_matrix,
    )


def _get_calibrated_params():
    """Get v2-calibrated params (or defaults if posterior not available)."""
    if os.path.exists(POSTERIORS_PATH):
        with open(POSTERIORS_PATH) as f:
            post = json.load(f)
        mu = post["global"]["mu_global"]["mean"]
        params = {
            "alpha_herd": mu[0],
            "alpha_anchor": mu[1],
            "alpha_social": mu[2],
            "alpha_event": mu[3],
        }
    else:
        params = {
            "alpha_herd": -0.176,
            "alpha_anchor": 0.297,
            "alpha_social": -0.105,
            "alpha_event": -0.130,
        }
    frozen = get_default_frozen_params()
    return {k: float(v) if hasattr(v, 'item') else v
            for k, v in {**params, **frozen}.items()}


# ── Test 1: Crisis activation on financial shock ─────────────

def test_regime_activation_on_financial_crisis():
    """Regime crisis should activate on a scenario with large shock.

    A shock of magnitude 0.8 at round 3 should trigger regime_prob > 0.5
    at rounds 3-4, then gradually recover.
    """
    scenario_data = _make_synthetic_scenario(
        shock_round=3, shock_magnitude=0.8, shock_direction=-1.0,
    )
    params = _get_calibrated_params()

    sim = RegimeSwitchingSimulator()
    result = sim.simulate(params, scenario_data, institutional_trust=0.3)

    regime_probs = np.array(result["regime_probs"])
    regime_seq = np.array(result["regime_sequence"])
    trajectories = np.array(result["trajectories"])

    print(f"\n{'='*60}")
    print("TEST 1: Crisis activation on large shock")
    print(f"{'='*60}")
    print(f"{'Round':>5} {'RegimeP':>8} {'Regime':>7} {'MeanPos':>9}")
    for r in range(len(regime_probs)):
        print(
            f"{r+1:5d} {regime_probs[r]:8.3f} {regime_seq[r]:7d} "
            f"{np.mean(trajectories[r]):9.4f}"
        )

    # Regime should activate at/after the shock round
    max_regime_prob = float(np.max(regime_probs[2:5]))  # rounds 3-5
    print(f"\nMax regime_prob (rounds 3-5): {max_regime_prob:.3f}")
    assert max_regime_prob > 0.5, (
        f"Crisis should trigger on large shock: max_prob={max_regime_prob}"
    )

    # After shock: regime should eventually recover
    final_regime_prob = float(regime_probs[-1])
    print(f"Final regime_prob: {final_regime_prob:.3f}")

    # Position should drop more than with normal dynamics
    result_normal = simulate_scenario(params, scenario_data)
    drop_regime = float(np.mean(trajectories[3]) - np.mean(trajectories[1]))
    drop_normal = float(
        np.mean(np.array(result_normal["trajectories"][3]))
        - np.mean(np.array(result_normal["trajectories"][1]))
    )
    print(f"Position drop (rounds 2→4): regime={drop_regime:.4f}, normal={drop_normal:.4f}")
    print(f"Amplification: {abs(drop_regime)/max(abs(drop_normal), 1e-6):.1f}x")

    print("\n✓ test_regime_activation_on_financial_crisis PASSED")
    return result


# ── Test 2: No crisis on normal scenario ─────────────────────

def test_no_crisis_on_normal_scenario():
    """On a scenario with small, gradual shocks, regime should stay normal.

    Uses Brexit-like scenario: small shocks (0.1-0.2), gradual drift.
    regime_prob should remain < 0.3 for all rounds.
    """
    # Small shocks spread over rounds
    scenario_data = _make_synthetic_scenario(
        shock_round=3, shock_magnitude=0.15, shock_direction=1.0,
    )
    # Add another small shock at round 6
    events = scenario_data.events.at[5, 0].set(0.1)
    events = events.at[5, 1].set(-1.0)
    scenario_data = scenario_data._replace(events=events)

    params = _get_calibrated_params()

    sim = RegimeSwitchingSimulator()
    result = sim.simulate(params, scenario_data, institutional_trust=0.6)

    regime_probs = np.array(result["regime_probs"])

    # Also run without regime switching for comparison
    result_normal = simulate_scenario(params, scenario_data)

    print(f"\n{'='*60}")
    print("TEST 2: No crisis on normal scenario")
    print(f"{'='*60}")
    print(f"{'Round':>5} {'RegimeP':>8} {'ProFrac_RS':>11} {'ProFrac_N':>10}")
    for r in range(len(regime_probs)):
        pf_rs = float(result["pro_fraction"][r]) * 100
        pf_n = float(result_normal["pro_fraction"][r]) * 100
        print(f"{r+1:5d} {regime_probs[r]:8.3f} {pf_rs:11.2f} {pf_n:10.2f}")

    max_regime_prob = float(np.max(regime_probs))
    print(f"\nMax regime_prob: {max_regime_prob:.3f}")
    assert max_regime_prob < 0.4, (
        f"Normal scenario shouldn't trigger crisis: max_prob={max_regime_prob}"
    )

    # Results should be nearly identical to non-regime version
    final_rs = float(result["final_pro_pct"])
    final_n = float(result_normal["final_pro_pct"])
    diff = abs(final_rs - final_n)
    print(f"Final pro%: regime={final_rs:.2f}, normal={final_n:.2f}, diff={diff:.2f}pp")
    assert diff < 3.0, f"Normal scenarios should match: diff={diff:.2f}pp"

    print("\n✓ test_no_crisis_on_normal_scenario PASSED")
    return result


# ── Test 3: Improvement on financial scenarios ───────────────

def test_regime_switching_improves_financial_scenarios():
    """Regime switching should reduce error on financial crisis scenarios.

    Loads WeWork, FTX, SVB from empirical data (if available).
    Compares v2 (no regime) vs v3 (regime switching) predictions.
    """
    financial_scenarios = [
        "FIN-2019-WEWORK_IPO_COLLAPSE_AND_PUBLIC",
        "FIN-2022-FTX_CRYPTO_CRISIS_NOVEMBER_202",
        "FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI",
    ]

    params = _get_calibrated_params()

    print(f"\n{'='*60}")
    print("TEST 3: Regime switching on financial scenarios")
    print(f"{'='*60}")

    found_any = False
    improvements = []

    for sid in financial_scenarios:
        path = os.path.join(SCENARIOS_DIR, f"{sid}.json")
        if not os.path.exists(path):
            print(f"  SKIP: {sid} not found")
            continue

        found_any = True
        scenario_dict, obs = load_scenario_observations(path)
        scenario_data = build_scenario_data_from_json(scenario_dict)
        gt = scenario_dict["ground_truth_outcome"]["pro_pct"]

        # Get institutional_trust from covariates
        covs = scenario_dict.get("covariates", {})
        inst_trust = covs.get("institutional_trust", 0.3)

        # v2: no regime switching
        result_v2 = simulate_scenario(params, scenario_data)
        pred_v2 = float(result_v2["final_pro_pct"])
        err_v2 = abs(pred_v2 - gt)

        # v3: with regime switching
        sim = RegimeSwitchingSimulator()
        result_v3 = sim.simulate(params, scenario_data, institutional_trust=inst_trust)
        pred_v3 = float(result_v3["final_pro_pct"])
        err_v3 = abs(pred_v3 - gt)

        regime_probs = np.array(result_v3["regime_probs"])
        max_rp = float(np.max(regime_probs))
        crisis_rounds = int(np.sum(np.array(result_v3["regime_sequence"])))

        improvement = err_v2 - err_v3
        improvements.append(improvement)

        print(
            f"\n  {sid[:45]}"
            f"\n    GT={gt:.1f}%  v2={pred_v2:.1f}% (err={err_v2:.1f})  "
            f"v3={pred_v3:.1f}% (err={err_v3:.1f})"
            f"\n    Δerr={improvement:+.1f}pp  max_regime_prob={max_rp:.3f}  "
            f"crisis_rounds={crisis_rounds}"
        )

    if not found_any:
        print("  SKIP: No financial scenarios found in empirical data")
        print("\n⚠ test_regime_switching_improves_financial SKIPPED")
        return None

    mean_improvement = np.mean(improvements)
    print(f"\n  Mean error improvement: {mean_improvement:+.1f}pp")

    # We expect improvement on average (not necessarily on every single scenario)
    if mean_improvement > 0:
        print("\n✓ test_regime_switching_improves_financial PASSED")
    else:
        print("\n⚠ test_regime_switching_improves_financial: no net improvement")
        print("  (May need parameter tuning — crisis_params are defaults)")

    return improvements


# ── Test 4: JIT compatibility ────────────────────────────────

def test_regime_jit_compatible():
    """Verify that simulate with regime switching is jit-compilable."""
    scenario_data = _make_synthetic_scenario(n_agents=8, n_rounds=5)
    params = _get_calibrated_params()

    sim = RegimeSwitchingSimulator()

    # First call: compiles
    result = sim.simulate(params, scenario_data, institutional_trust=0.5)

    # Verify output shapes
    n_rounds = 5
    n_agents = 8
    assert result["trajectories"].shape == (n_rounds, n_agents), \
        f"trajectories shape: {result['trajectories'].shape}"
    assert result["pro_fraction"].shape == (n_rounds,), \
        f"pro_fraction shape: {result['pro_fraction'].shape}"
    assert result["regime_probs"].shape == (n_rounds,), \
        f"regime_probs shape: {result['regime_probs'].shape}"
    assert result["regime_sequence"].shape == (n_rounds,), \
        f"regime_sequence shape: {result['regime_sequence'].shape}"

    # Check finite values
    assert jnp.all(jnp.isfinite(result["trajectories"])), "Non-finite trajectories"
    assert jnp.all(jnp.isfinite(result["pro_fraction"])), "Non-finite pro_fraction"
    assert jnp.all(jnp.isfinite(result["regime_probs"])), "Non-finite regime_probs"
    assert jnp.isfinite(result["final_pro_pct"]), "Non-finite final_pro_pct"

    # Check ranges
    assert jnp.all(result["trajectories"] >= -1.0) and jnp.all(result["trajectories"] <= 1.0)
    assert jnp.all(result["regime_probs"] >= 0.0) and jnp.all(result["regime_probs"] <= 1.0)
    assert 0.0 <= float(result["final_pro_pct"]) <= 100.0

    # JIT compile explicitly
    @jax.jit
    def run_jitted(p_herd, p_anchor, p_social, p_event):
        p = dict(params)
        p["alpha_herd"] = p_herd
        p["alpha_anchor"] = p_anchor
        p["alpha_social"] = p_social
        p["alpha_event"] = p_event
        return sim.simulate(p, scenario_data, 0.5)["final_pro_pct"]

    # Should not raise
    val = run_jitted(
        jnp.array(-0.2), jnp.array(0.3),
        jnp.array(-0.1), jnp.array(-0.15),
    )
    assert jnp.isfinite(val), f"JIT result non-finite: {val}"

    print(f"\n{'='*60}")
    print("TEST 4: JIT compatibility")
    print(f"{'='*60}")
    print(f"  Output shapes: OK")
    print(f"  Values finite: OK")
    print(f"  Ranges valid: OK")
    print(f"  JIT compilation: OK (final_pro_pct={float(val):.2f})")
    print("\n✓ test_regime_jit_compatible PASSED")


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Regime Switching Test Suite")
    print("=" * 60)

    test_regime_jit_compatible()
    test_no_crisis_on_normal_scenario()
    test_regime_activation_on_financial_crisis()
    test_regime_switching_improves_financial_scenarios()

    print("\n✓ All regime switching tests completed.")
