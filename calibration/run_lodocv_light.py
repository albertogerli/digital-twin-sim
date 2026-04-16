"""Lightweight LODO-CV on the 20 HQ scenarios (4 domains).

Leave-one-domain-out: train on 3 domains, predict the held-out.
Uses the same 5-param patching infrastructure.
"""
import json
import math
import sys
import time
from pathlib import Path
import glob

import jax
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib
hm_module = importlib.import_module("src.inference.hierarchical_model")
from src.dynamics import param_utils
from src.dynamics.opinion_dynamics_jax import simulate_scenario
from jax.scipy.special import expit as sigmoid

# Keep 4-param defaults (no patching needed for LODO)
from src.inference.hierarchical_model import (
    PARAM_NAMES, N_PARAMS, N_COVARIATES, FROZEN_KEYS,
    prepare_calibration_data, CalibrationData,
)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, init_to_value
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal

cp_module = importlib.import_module("src.inference.calibration_pipeline")

# 20 HQ scenarios
HQ_IDS = {
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

EMPIRICAL_DIR = PROJECT_ROOT / "calibration" / "empirical" / "scenarios"
_DROP_SCENARIOS = getattr(cp_module, '_DROP_SCENARIOS', set())

# Load all HQ scenarios
json_files = sorted(glob.glob(str(EMPIRICAL_DIR / "*.json")))
json_files = [f for f in json_files if not f.endswith("manifest.json")
              and not f.endswith(".meta.json")]

all_sds, all_obs, all_covs, all_domains, all_ids = [], [], [], [], []
for path in json_files:
    sid = Path(path).stem
    if sid in _DROP_SCENARIOS or sid not in HQ_IDS:
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
    except Exception as e:
        print(f"  Failed: {sid}: {e}")

N = len(all_ids)
unique_domains = sorted(set(all_domains))
print(f"Loaded {N} scenarios across {len(unique_domains)} domains: {unique_domains}")
sys.stdout.flush()

frozen = param_utils.get_default_frozen_params()
frozen_vec = jnp.array([frozen[k] for k in FROZEN_KEYS])

lodocv_results = {}

for held_out_domain in unique_domains:
    print(f"\n{'='*60}")
    print(f"Holding out: {held_out_domain}")
    print(f"{'='*60}")
    sys.stdout.flush()

    train_idx = [i for i, d in enumerate(all_domains) if d != held_out_domain]
    test_idx = [i for i, d in enumerate(all_domains) if d == held_out_domain]
    n_test = len(test_idx)
    n_train = len(train_idx)

    if n_test == 0:
        print(f"  No scenarios — skipping")
        continue

    print(f"  Train: {n_train}, Test: {n_test}")

    # Prepare train data
    train_sds = [all_sds[i] for i in train_idx]
    train_obs = [all_obs[i] for i in train_idx]
    train_dom = [all_domains[i] for i in train_idx]
    train_cov = [all_covs[i] for i in train_idx]

    train_domain_names = sorted(set(train_dom))
    train_domain_map = {d: i for i, d in enumerate(train_domain_names)}
    train_domain_indices = [train_domain_map[d] for d in train_dom]

    train_data = prepare_calibration_data(
        train_sds, train_obs, train_domain_indices, train_cov,
    )

    n_domains_val = train_data.n_domains
    prior_mu = jnp.zeros(N_PARAMS)
    prior_sigma = 0.5 * jnp.ones(N_PARAMS)

    def model_fn():
        return cp_module.hierarchical_model_transfer(
            train_data, prior_mu, prior_sigma, batch_size=None,
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
        "theta_s": jnp.zeros((n_train, N_PARAMS)),
    }

    guide = AutoLowRankMultivariateNormal(
        model_fn, init_loc_fn=init_to_value(values=init_values),
    )
    optimizer = numpyro.optim.Adam(0.003)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(42 + hash(held_out_domain) % 1000)

    print("  Initializing SVI (JIT)...")
    sys.stdout.flush()
    t0 = time.time()
    svi_state = svi.init(rng_key)
    t_init = time.time() - t0
    print(f"  JIT: {t_init:.1f}s")
    sys.stdout.flush()

    N_STEPS = 1500
    print(f"  Running {N_STEPS} SVI steps...")
    sys.stdout.flush()
    t0 = time.time()
    for step in range(N_STEPS):
        svi_state, loss = svi.update(svi_state)
        if (step + 1) % 500 == 0:
            print(f"    Step {step+1}/{N_STEPS}  loss={loss:.1f}")
            sys.stdout.flush()
    elapsed = time.time() - t0
    print(f"  SVI done in {elapsed:.1f}s")
    sys.stdout.flush()

    # Predict on held-out domain
    svi_params = svi.get_params(svi_state)
    predictive = Predictive(guide, params=svi_params, num_samples=100)
    pp_samples = predictive(jax.random.PRNGKey(42))
    mu_global_samples = pp_samples["mu_global"]  # [S, 4]

    domain_results = []
    for i in test_idx:
        sid = all_ids[i]
        obs = all_obs[i]
        gt = float(obs.ground_truth_pro_pct)

        pp_finals = []
        for s in range(min(50, mu_global_samples.shape[0])):
            params_s = {
                PARAM_NAMES[k]: float(mu_global_samples[s, k])
                for k in range(N_PARAMS)
            }
            for k_idx, key in enumerate(FROZEN_KEYS):
                params_s[key] = float(frozen_vec[k_idx])
            try:
                res_s = simulate_scenario(params_s, all_sds[i])
                pp_finals.append(float(res_s["final_pro_pct"]))
            except:
                pass

        if pp_finals:
            pp_arr = jnp.array(pp_finals)
            sim_mean = float(jnp.mean(pp_arr))
            ci90 = (float(jnp.percentile(pp_arr, 5)), float(jnp.percentile(pp_arr, 95)))
            in_90 = ci90[0] <= gt <= ci90[1]
            err = abs(sim_mean - gt)
        else:
            sim_mean, ci90, in_90, err = 50.0, (0, 100), True, abs(50 - gt)

        covered = "\u2713" if in_90 else "\u2717"
        print(f"    {sid[:45]:45s} gt={gt:5.1f} sim={sim_mean:5.1f} err={err:5.1f} {covered}")
        domain_results.append({
            "id": sid, "gt": gt, "sim_mean": sim_mean,
            "abs_error": err, "ci90": list(ci90), "in_90": in_90,
        })

    mae = sum(r["abs_error"] for r in domain_results) / len(domain_results)
    cov90 = sum(1 for r in domain_results if r["in_90"]) / len(domain_results)
    lodocv_results[held_out_domain] = {
        "n_scenarios": n_test, "mae": mae, "coverage_90": cov90,
        "elapsed_s": elapsed, "per_scenario": domain_results,
    }
    print(f"  Domain MAE: {mae:.1f}pp, Cov90: {cov90*100:.1f}%")
    sys.stdout.flush()

# Summary
print(f"\n{'='*60}")
print("LODO-CV Summary (20 HQ Scenarios)")
print(f"{'='*60}")
total_n = sum(v["n_scenarios"] for v in lodocv_results.values())
total_covered = sum(
    sum(1 for r in v["per_scenario"] if r["in_90"])
    for v in lodocv_results.values()
)
all_errors = [
    r["abs_error"] for v in lodocv_results.values() for r in v["per_scenario"]
]
overall_mae = sum(all_errors) / len(all_errors) if all_errors else 0
overall_cov90 = total_covered / total_n if total_n else 0

for domain, v in sorted(lodocv_results.items()):
    print(f"  {domain:20s}  N={v['n_scenarios']:2d}  MAE={v['mae']:5.1f}pp  Cov90={v['coverage_90']*100:5.1f}%")
print(f"  {'OVERALL':20s}  N={total_n:2d}  MAE={overall_mae:5.1f}pp  Cov90={overall_cov90*100:5.1f}%")

summary = {
    "overall_mae": overall_mae,
    "overall_coverage_90": overall_cov90,
    "n_total": total_n,
    "per_domain": lodocv_results,
}

REVIEWER_DIR = PROJECT_ROOT / "calibration" / "results" / "hierarchical_calibration" / "reviewer_experiments"
REVIEWER_DIR.mkdir(parents=True, exist_ok=True)

def to_json(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, (float, int)):
        return obj
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json(v) for v in obj]
    return obj

with open(REVIEWER_DIR / "lodocv_results.json", "w") as f:
    json.dump(to_json(summary), f, indent=2)

print(f"\nResults saved to {REVIEWER_DIR / 'lodocv_results.json'}")
