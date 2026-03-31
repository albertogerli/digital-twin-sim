# Hierarchical Model — Test Validation Report

## Test Configuration

- 10 scenarios: 5 synthetic + 5 empirical
- 5 domains: synthetic_A, synthetic_B, political, financial, corporate
- 200 SVI steps, lr=0.01, full batch (no subsampling)
- AutoLowRankMultivariateNormal guide
- init_to_value with zeros for calibrable params

## Test Results

| Test | Status | Key Metric |
|---|---|---|
| Data preparation | PASSED | 10 scenarios, max_agents=12, max_rounds=7 |
| Batch simulation | PASSED | All final_pcts finite and in [0, 100] |
| SVI convergence | PASSED | Loss 379→212 in 200 steps (133.8s) |
| Posterior extraction | PASSED | All shapes correct |
| Domain diversity | PASSED | max |μ_d1 - μ_d2| = 0.19 |
| Shrinkage | PASSED | 4/10 scenarios closer to domain than global |
| Covariate matrix B | PASSED | 1 significant effect found |
| Observation params | PASSED | τ_readout=0.066, φ=19.0 |

## Key Findings

### Global Hyperpriors
- μ_global: [-0.08, 0.16, -0.20, 0.09] (near zero, as expected)
- σ_global: [0.28, 0.25, 0.26, 0.29] (moderate dispersion)

### Observation Parameters
- τ_readout: 0.066 (moved from prior ~0.22 toward harder threshold)
- φ (BetaBinomial concentration): 19.0 (moved from prior ~55)
- σ_outcome: from HalfNormal(5) prior (not reported, test didn't check value)

### Covariates
1 significant effect detected at 95% CI:
- `alpha_event ← elite_concentration`: +0.195 (CI: 0.024, 0.358)
  Interpretation: scenarios with higher elite concentration have stronger event sensitivity

### Convergence Profile
- Loss decreased monotonically (avg): 379 → 252 → 228 → 213
- 133.8s for 200 steps with 10 scenarios (0.67s/step including JIT compilation)
- First step includes JIT compilation; subsequent steps ~0.1-0.2s each

## Critical Bug Fix

During development, discovered that `jnp.std()` produces NaN gradients when
all input values are identical (e.g., f_direct=0 when llm_shifts=0). This
affects scenarios with padded rounds (zero events/llm_shifts).

**Fix**: Replaced `cur_sigma = jnp.std(tail, axis=0)` with
`cur_sigma = jnp.sqrt(jnp.mean((tail - cur_mu)**2, axis=0) + 1e-8)`
in `opinion_dynamics_jax.py:ema_standardize()`. The `1e-8` inside sqrt
ensures the gradient is always finite.

## Files

| File | Description |
|---|---|
| `src/inference/__init__.py` | Package exports |
| `src/inference/hierarchical_model.py` | 3-level hierarchical model, SVI, NUTS, posterior extraction |
| `src/inference/test_hierarchical.py` | 8 end-to-end tests |
| `src/dynamics/opinion_dynamics_jax.py` | EMA gradient fix (sqrt(var+eps) instead of std) |
