# Observation Model — Validation Report

## Overview

The observation model connects the JAX DynamicsV2 simulator to empirical data
via two likelihood variants:

| Variant | Distribution | Use Case | Input |
|---|---|---|---|
| **Full** | BetaBinomial(n, q·φ, (1-q)·φ) | ~8 scenarios with verified polling | per-round pro_pct + sample_size |
| **Outcome-Only** | Normal(q_final·100, σ) | All 24 scenarios | final ground_truth_pro_pct only |

## Design Decisions

1. **τ_readout**: Reuses existing τ=0.02 from `simulate_scenario`'s soft pro_fraction.
   No additional parameter needed.

2. **BetaBinomial**: Pure JAX implementation using `lgamma` (not NumPyro's distribution).
   This avoids NumPyro dependency in the core observation model while remaining
   differentiable. NumPyro will wrap these likelihoods in its own model functions.

3. **Verified mask**: Rounds are only included in the full likelihood if both:
   - `sample_size > 0` (polling data exists)
   - Meta provenance is `"verified_url"` or `"verified_csv"`

4. **Scenario builder**: Maps empirical JSON agents to JAX arrays with fixed
   rigidity/tolerance by type (elite=0.7/0.3, institutional=0.8/0.4, citizen=0.3/0.6).

## Test Results

| Test | Status | Key Metric |
|---|---|---|
| BetaBinomial log-PMF | PASSED | Correct monotonicity (tight > wide at mode) |
| Full likelihood | PASSED | ll = -20.32 (finite, negative) |
| Outcome-only likelihood | PASSED | ll_near > ll_far by 8.0 nats |
| Gradient check | PASSED | All 3 grads finite and non-zero |
| Diagnostics | PASSED | RMSE ≥ MAE ≥ 0 |
| Scenario loader (Brexit) | PASSED | 6 rounds, 0 verified (all sample_size=null) |
| build_scenario_data | PASSED | 12 agents, 3 shock rounds, sim=55.66% |
| Variant coherence | PASSED | Both prefer near observation (Δ_full=23.7, Δ_out=8.0) |

## Brexit Scenario (Smoke Test)

Using default parameters (all α=0, frozen at defaults):
- Simulated final pro_pct: **55.66%**
- Ground truth: **51.89%**
- Outcome error: **+3.77pp** (before any calibration!)

The default parameters already produce a reasonable ballpark for Brexit,
suggesting the model structure and agent properties are sensible.

## Files

| File | Description |
|---|---|
| `src/observation/__init__.py` | Package exports |
| `src/observation/observation_model.py` | Core module (likelihoods, diagnostics, loader) |
| `src/observation/test_observation_model.py` | 8 end-to-end tests |

## API Summary

```python
# Full BetaBinomial likelihood (verified rounds only)
ll = log_likelihood_full(params, scenario_data, observations, log_phi)

# Outcome-only Normal likelihood
ll = log_likelihood_outcome(params, scenario_data, ground_truth_pro_pct, log_sigma)

# Diagnostics
diag = compute_diagnostics(params, scenario_data, observations)
# → outcome_error, trajectory_mae, trajectory_rmse, simulated_trajectory

# Scenario loading
scenario_dict, obs = load_scenario_observations("path/to/scenario.json")
scenario_data = build_scenario_data_from_json(scenario_dict)
```

## Next Step

Proceed to Phase 1 calibration: wrap these likelihoods in NumPyro models
and run NUTS/SVI on the empirical dataset.
