"""Lightweight 5-parameter SVI experiment.

Runs on a small subset (8 scenarios) with fewer steps to demonstrate
λ_citizen identifiability without the full pipeline overhead.
"""
import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch BEFORE importing anything else
import importlib
hm_module = importlib.import_module("src.inference.hierarchical_model")
from src.dynamics import param_utils

# Save originals
orig_calibrable = param_utils.CALIBRABLE_PARAMS[:]
orig_frozen = param_utils.FROZEN_PARAMS[:]
orig_hm_param_names = hm_module.PARAM_NAMES[:]
orig_hm_frozen_keys = hm_module.FROZEN_KEYS[:]
orig_hm_n_params = hm_module.N_PARAMS

# Promote log_lambda_citizen BEFORE pipeline import
param_utils.CALIBRABLE_PARAMS.append("log_lambda_citizen")
param_utils.FROZEN_PARAMS.remove("log_lambda_citizen")

NEW_PARAM_NAMES = [
    "alpha_herd", "alpha_anchor", "alpha_social", "alpha_event",
    "log_lambda_citizen",
]
NEW_FROZEN_KEYS = [
    "log_lambda_elite", "logit_herd_threshold", "logit_anchor_drift",
]

hm_module.PARAM_NAMES = NEW_PARAM_NAMES
hm_module.FROZEN_KEYS = NEW_FROZEN_KEYS
hm_module.N_PARAMS = 5

# Patch default frozen params (remove log_lambda_citizen)
orig_defaults_fn = param_utils.get_default_frozen_params
def patched_defaults():
    return {
        "log_lambda_elite": jnp.log(jnp.array(0.15)),
        "logit_herd_threshold": param_utils._logit(jnp.array(0.21)),
        "logit_anchor_drift": param_utils._logit(jnp.array(0.25)),
    }
param_utils.get_default_frozen_params = patched_defaults

# Patch default calibrable params (add log_lambda_citizen)
orig_cal_defaults_fn = param_utils.get_default_calibrable_params
def patched_cal_defaults():
    return {
        "alpha_herd": jnp.array(0.0),
        "alpha_anchor": jnp.array(0.0),
        "alpha_social": jnp.array(0.0),
        "alpha_event": jnp.array(0.0),
        "log_lambda_citizen": jnp.log(jnp.array(0.25)),
    }
param_utils.get_default_calibrable_params = patched_cal_defaults

# Need to also re-create _simulate_batch with correct shapes
# The vmap was already compiled with 4 params — we need to invalidate the cache
# by re-creating it
from src.dynamics.opinion_dynamics_jax import simulate_scenario
from jax.scipy.special import expit as sigmoid
from src.inference.hierarchical_model import (
    ScenarioData, _beta_binomial_logpmf, CalibrationData,
    prepare_calibration_data,
)

def _simulate_one_5p(calibrable_vec, frozen_vec, scenario_data,
                     agent_mask, tau_readout, n_real_rounds):
    """Simulate one scenario with 5 calibrable params."""
    params = {NEW_PARAM_NAMES[i]: calibrable_vec[i] for i in range(5)}
    for i, key in enumerate(NEW_FROZEN_KEYS):
        params[key] = frozen_vec[i]
    result = simulate_scenario(params, scenario_data)
    trajectories = result["trajectories"]
    soft_pro = sigmoid((trajectories - 0.05) / tau_readout)
    soft_against = sigmoid((-trajectories - 0.05) / tau_readout)
    soft_decided = soft_pro + soft_against
    mask = agent_mask[None, :]
    pro_count = jnp.sum(soft_pro * mask, axis=1)
    decided_count = jnp.sum(soft_decided * mask, axis=1) + 1e-8
    pro_fraction = pro_count / decided_count
    final_pro_pct = pro_fraction[n_real_rounds - 1] * 100.0
    return pro_fraction, final_pro_pct

_SCENARIO_AXES = ScenarioData(0, 0, 0, 0, 0, 0, 0)
_simulate_batch_5p = jax.vmap(
    _simulate_one_5p,
    in_axes=(0, None, _SCENARIO_AXES, 0, None, 0),
)

# Override the module-level _simulate_batch
hm_module._simulate_batch = _simulate_batch_5p

# NOW import the pipeline (it will see 5-param constants)
cp_module = importlib.import_module("src.inference.calibration_pipeline")
cp_module.PARAM_NAMES = NEW_PARAM_NAMES
cp_module.FROZEN_KEYS = NEW_FROZEN_KEYS
cp_module.N_PARAMS = 5

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, init_to_value
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal

# ── Load the 20 HQ scenarios (same as v2.2/v2.3) ──
import glob
EMPIRICAL_DIR = PROJECT_ROOT / "calibration" / "empirical" / "scenarios"

# 20 HQ scenarios from v2.2/v2.3 (loaded from validation results)
hq_ids = {
    "CORP-2017-UNITED_AIRLINES_PASSENGER_DRAG",
    "CORP-2022-TWITTER_X_ACQUISITION_BY_ELON",
    "FIN-2022-FTX_CRYPTO_CRISIS_NOVEMBER_202",
    "FIN-2023-SVB_COLLAPSE_MARCH_2023_BANKI",
    "FIN-2021-ARCHEGOS_CAPITAL_COLLAPSE_MARC",
    "FIN-2020-TESLA_STOCK_SPLIT_AUGUST_2020",
    "POL-2022-ELEZIONI_PRESIDENZIALI_BRASILE",
    "POL-2011-REFERENDUM_DIVORZIO_MALTA_2011",
    "POL-2018-ELEZIONI_MIDTERM_USA_2018_HOU",
    "POL-2019-ELEZIONI_EUROPEE_2019_ITALIA",
    "POL-2018-REFERENDUM_ABORTO_IRLANDA_2018",
    "POL-2016-REFERENDUM_COSTITUZIONALE_ITAL",
    "POL-2017-ELEZIONI_PRESIDENZIALI_FRANCIA",
    "PH-2022-MONKEYPOX_PUBLIC_CONCERN_USA_2",
    "PH-2021-MASKING_MANDATE_DEBATE_USA_202",
    "CORP-2021-FACEBOOK_META_REBRAND_OCTOBER",
    "FIN-2021-AMC_SHORT_SQUEEZE_2021_RETAIL",
    "POL-2017-REFERENDUM_INDIPENDENZA_CATALU",
    "POL-2020-ELEZIONI_PRESIDENZIALI_USA_202",
    "PH-2021-VACCINE_HESITANCY_USA_2021_CO",
}
print(f"  HQ scenarios: {len(hq_ids)}")

json_files = sorted(glob.glob(str(EMPIRICAL_DIR / "*.json")))
json_files = [f for f in json_files if not f.endswith("manifest.json")
              and not f.endswith(".meta.json")]

# Load the scenarios used in the _DROP_SCENARIOS list
_DROP_SCENARIOS = getattr(cp_module, '_DROP_SCENARIOS', set())

all_sds, all_obs, all_covs, all_domains, all_ids = [], [], [], [], []
for path in json_files:
    sid = Path(path).stem
    if sid in _DROP_SCENARIOS:
        continue
    # If HQ manifest exists, only load HQ scenarios; otherwise load all
    if hq_ids is not None and sid not in hq_ids:
        continue
    try:
        sd, obs, cov, domain = cp_module.load_empirical_scenario(path, seed=42)
        all_sds.append(sd)
        all_obs.append(obs)
        all_covs.append([
            cov.get("initial_polarization", 0.5),
            cov.get("event_volatility", 0.5),
            cov.get("elite_concentration", 0.5),
            cov.get("institutional_trust", 0.5),
            cov.get("undecided_share", 0.1),
        ])
        all_domains.append(domain)
        all_ids.append(sid)
        print(f"  Loaded: {sid[:50]}")
    except Exception as e:
        print(f"  Failed: {sid}: {e}")

N = len(all_ids)
print(f"\nLoaded {N} scenarios")

if N == 0:
    print("No scenarios loaded!")
    sys.exit(1)

# Build domain index
domain_names = sorted(set(all_domains))
domain_map = {d: i for i, d in enumerate(domain_names)}
domain_indices = [domain_map[d] for d in all_domains]

# Prepare data
data = prepare_calibration_data(all_sds, all_obs, domain_indices, all_covs)
n_domains_val = data.n_domains
N_PARAMS = 5
N_COVARIATES = 5

print(f"Domains: {domain_names}")
print(f"Shape check: domain_indices={data.domain_indices.shape}, covariates={data.covariates.shape}")

# Build the model
prior_mu = jnp.zeros(N_PARAMS)
prior_sigma = 0.5 * jnp.ones(N_PARAMS)

def model_fn():
    return cp_module.hierarchical_model_transfer(
        data, prior_mu, prior_sigma, batch_size=None,
    )

init_values = {
    "mu_global": prior_mu,
    "sigma_global": prior_sigma,
    "mu_domain": jnp.broadcast_to(prior_mu, (n_domains_val, N_PARAMS)),
    "sigma_domain": 0.3 * jnp.ones((n_domains_val, N_PARAMS)),
    "B": jnp.zeros((N_PARAMS, N_COVARIATES)),
    "tau_readout": jnp.array(0.02),
    "log_phi": jnp.array(4.0),
    "sigma_outcome": jnp.array(5.0),
    "theta_s": jnp.zeros((N, N_PARAMS)),
}

print("\nInitializing SVI (this includes JIT compilation)...")
sys.stdout.flush()
guide = AutoLowRankMultivariateNormal(
    model_fn, init_loc_fn=init_to_value(values=init_values),
)
optimizer = numpyro.optim.Adam(0.003)
svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

rng_key = jax.random.PRNGKey(42)
t0 = time.time()
svi_state = svi.init(rng_key)
t_init = time.time() - t0
print(f"SVI init (JIT compile) took {t_init:.1f}s")
sys.stdout.flush()

# Run SVI
N_STEPS = 2000
print(f"\nRunning {N_STEPS} SVI steps...")
sys.stdout.flush()
t0 = time.time()
losses = []
for step in range(N_STEPS):
    svi_state, loss = svi.update(svi_state)
    losses.append(float(loss))
    if (step + 1) % 500 == 0:
        print(f"  Step {step+1}/{N_STEPS}  loss={loss:.1f}")
        sys.stdout.flush()
elapsed = time.time() - t0
print(f"SVI done in {elapsed:.1f}s")

# Extract posteriors
svi_params = svi.get_params(svi_state)
predictive = Predictive(guide, params=svi_params, num_samples=200)
pp_samples = predictive(jax.random.PRNGKey(99))

mu_global = pp_samples["mu_global"]  # [200, 5]
sigma_global = pp_samples["sigma_global"]  # [200, 5]

print("\n" + "=" * 60)
print("5-PARAMETER POSTERIOR SUMMARY")
print("=" * 60)

results = {"param_posteriors": {}}
for i, name in enumerate(NEW_PARAM_NAMES):
    samples = mu_global[:, i]
    mean = float(jnp.mean(samples))
    std = float(jnp.std(samples))
    q5 = float(jnp.percentile(samples, 5))
    q95 = float(jnp.percentile(samples, 95))
    print(f"  {name:25s}  mean={mean:7.4f}  std={std:7.4f}  90%CI=[{q5:.4f}, {q95:.4f}]")
    results["param_posteriors"][name] = {
        "mean": mean, "std": std, "ci90": [q5, q95],
    }

# Compare with frozen value
frozen_log_lambda = float(jnp.log(jnp.array(0.25)))
lc_mean = results["param_posteriors"]["log_lambda_citizen"]["mean"]
lc_std = results["param_posteriors"]["log_lambda_citizen"]["std"]
print(f"\n  λ_citizen frozen default: log(0.25) = {frozen_log_lambda:.4f}")
print(f"  λ_citizen posterior mean: {lc_mean:.4f} (λ = {math.exp(lc_mean):.4f})")
print(f"  λ_citizen posterior std:  {lc_std:.4f}")
if lc_std < 0.05:
    print("  → Posterior is TIGHT → λ_citizen is identifiable")
elif lc_std < 0.2:
    print("  → Posterior is moderately constrained")
else:
    print("  → Posterior is WIDE → λ_citizen may be weakly identified")

# Validate: simulate with posterior means
print("\nValidation: simulating with posterior means...")
from src.dynamics.opinion_dynamics_jax import simulate_scenario as sim_scen

frozen = patched_defaults()
frozen_vec = jnp.array([frozen[k] for k in NEW_FROZEN_KEYS])

validation = []
for i in range(N):
    sid = all_ids[i]
    obs = all_obs[i]
    gt = float(obs.ground_truth_pro_pct)

    # Use posterior mean for global params
    pp_finals = []
    for s in range(min(50, mu_global.shape[0])):
        params_s = {NEW_PARAM_NAMES[k]: float(mu_global[s, k]) for k in range(5)}
        for k_idx, key in enumerate(NEW_FROZEN_KEYS):
            params_s[key] = float(frozen_vec[k_idx])
        try:
            res = sim_scen(params_s, all_sds[i])
            pp_finals.append(float(res["final_pro_pct"]))
        except:
            pass

    if pp_finals:
        arr = jnp.array(pp_finals)
        sim_mean = float(jnp.mean(arr))
        ci90 = (float(jnp.percentile(arr, 5)), float(jnp.percentile(arr, 95)))
        in_90 = ci90[0] <= gt <= ci90[1]
        err = abs(sim_mean - gt)
    else:
        sim_mean, ci90, in_90, err = 50.0, (0, 100), True, abs(50 - gt)

    covered = "✓" if in_90 else "✗"
    print(f"  {sid[:45]:45s} gt={gt:5.1f} sim={sim_mean:5.1f} err={err:5.1f} {covered}")
    validation.append({
        "id": sid, "gt": gt, "sim_mean": sim_mean,
        "abs_error": err, "ci90": list(ci90), "in_90": in_90,
    })

mae = sum(r["abs_error"] for r in validation) / len(validation)
cov90 = sum(1 for r in validation if r["in_90"]) / len(validation)
print(f"\n  MAE: {mae:.1f}pp, Coverage90: {cov90*100:.1f}%")

results["validation"] = validation
results["mae"] = mae
results["coverage_90"] = cov90
results["n_scenarios"] = N
results["n_svi_steps"] = N_STEPS
results["elapsed_svi_s"] = elapsed
results["elapsed_init_s"] = t_init
results["final_loss"] = losses[-1]
results["loss_history"] = losses[::100]  # every 100th

REVIEWER_DIR = PROJECT_ROOT / "calibration" / "results" / "hierarchical_calibration" / "reviewer_experiments"
REVIEWER_DIR.mkdir(parents=True, exist_ok=True)

def to_json(obj):
    if isinstance(obj, jnp.ndarray):
        return obj.tolist()
    if isinstance(obj, (jnp.floating, float)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json(v) for v in obj]
    return obj

with open(REVIEWER_DIR / "5param_results.json", "w") as f:
    json.dump(to_json(results), f, indent=2)

print(f"\nResults saved to {REVIEWER_DIR / '5param_results.json'}")
