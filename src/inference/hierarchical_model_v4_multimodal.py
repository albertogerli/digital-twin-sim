"""Hierarchical model v4 — multi-modal: polling + financial market observations.

Extends v2 (discrepancy model) with a financial observation likelihood.
When scenarios have market_observations (actual stock/sector returns),
the model jointly fits:
  1. Polling trajectory → BetaBinomial / Normal likelihood (from v2)
  2. Market returns → Normal likelihood via opinion→return linkage function

Key additions vs v2:
  - w_opinion, w_event, w_polar: learnable linkage weights (shared across scenarios)
  - log_sigma_market: market observation noise
  - lambda_fin: modality weight (Beta(2,5) prior, ~0.28 mean)

The financial likelihood only applies to scenarios with market data
(typically FIN and CORP domains). Non-financial scenarios are unaffected.

Usage:
    python -m src.inference.hierarchical_model_v4_multimodal
    python -m src.inference.hierarchical_model_v4_multimodal --svi-steps 3000
"""

import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.nn import sigmoid

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive, init_to_value
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal

import numpy as np

# Import from v2 — we extend it
from .hierarchical_model_v2 import (
    _likelihood_batch_v2, extract_posteriors_v2,
)
from .hierarchical_model import (
    CalibrationData, prepare_calibration_data,
    PARAM_NAMES, COVARIATE_NAMES, N_PARAMS, N_COVARIATES,
    FROZEN_KEYS, _simulate_batch, _SCENARIO_AXES,
    _summarize,
)
from .calibration_pipeline import (
    load_empirical_scenario, EMPIRICAL_DIR, RESULTS_DIR,
    _build_domain_index, _stratified_split, _DROP_SCENARIOS,
)
from ..dynamics.opinion_dynamics_jax import ScenarioData, simulate_scenario
from ..dynamics.param_utils import get_default_frozen_params
from ..observation.financial_obs_model import (
    FinancialObsData, load_financial_observations,
    compute_expected_returns, log_likelihood_financial_one,
)


# ── Constants ────────────────────────────────────────────────────────

V4_RESULTS_DIR = RESULTS_DIR / "v4_multimodal"


# ── Financial likelihood wrapper ─────────────────────────────────────

def _financial_ll_one(
    pro_fraction, trajectories, events, agent_mask,
    fin_returns, fin_mask, fin_sector_beta, fin_has_data,
    w_opinion, w_event, w_polar, log_sigma_market,
):
    """Financial log-likelihood for one scenario, masked by fin_has_data."""
    ll = log_likelihood_financial_one(
        pro_fraction, trajectories, events, agent_mask,
        fin_returns, fin_mask, fin_sector_beta,
        w_opinion, w_event, w_polar, log_sigma_market,
    )
    return jnp.where(fin_has_data, ll, 0.0)


# ── Data preparation ─────────────────────────────────────────────────

def prepare_data_v4(
    scenario_datas, observations, domain_indices, covariates_raw,
    financial_obs,
):
    """Prepare CalibrationData + financial arrays."""
    cal_data = prepare_calibration_data(
        scenario_datas, observations, domain_indices, covariates_raw,
    )

    max_r = cal_data.events.shape[1]
    fin_returns = []
    fin_masks = []
    fin_betas = []
    fin_has = []

    for fobs in financial_obs:
        if fobs is not None:
            nr = fobs.returns_pct.shape[0]
            pad_r = max_r - nr
            fin_returns.append(jnp.pad(fobs.returns_pct, (0, pad_r)))
            fin_masks.append(jnp.pad(fobs.return_mask, (0, pad_r)))
            fin_betas.append(fobs.sector_beta)
            fin_has.append(True)
        else:
            fin_returns.append(jnp.zeros(max_r))
            fin_masks.append(jnp.zeros(max_r, dtype=jnp.bool_))
            fin_betas.append(1.0)
            fin_has.append(False)

    fin_arrays = {
        "fin_returns": jnp.stack(fin_returns),
        "fin_masks": jnp.stack(fin_masks),
        "fin_sector_betas": jnp.array(fin_betas),
        "fin_has_data": jnp.array(fin_has),
    }
    return cal_data, fin_arrays


# ── NumPyro Model v4 ─────────────────────────────────────────────────

def hierarchical_model_v4(
    data: CalibrationData,
    fin_arrays: dict,
    prior_mu_global=None,
    prior_sigma_global=None,
    batch_size=None,
):
    """v2 hierarchy + discrepancy + financial market likelihood."""
    n_scenarios = data.domain_indices.shape[0]
    n_domains = data.n_domains

    # ── Level 1: Global ──
    if prior_mu_global is not None:
        mu_global = numpyro.sample(
            "mu_global",
            dist.Normal(prior_mu_global, prior_sigma_global).to_event(1),
        )
    else:
        mu_global = numpyro.sample(
            "mu_global",
            dist.Normal(jnp.zeros(N_PARAMS), 1.0).to_event(1),
        )
    sigma_global = numpyro.sample(
        "sigma_global",
        dist.HalfNormal(0.5 * jnp.ones(N_PARAMS)).to_event(1),
    )

    # ── Level 2: Domain ──
    with numpyro.plate("domains", n_domains):
        mu_domain = numpyro.sample(
            "mu_domain",
            dist.Normal(mu_global, sigma_global).to_event(1),
        )
        sigma_domain = numpyro.sample(
            "sigma_domain",
            dist.HalfNormal(0.3 * jnp.ones(N_PARAMS)).to_event(1),
        )

    # ── Covariates ──
    B = numpyro.sample(
        "B",
        dist.Normal(
            jnp.zeros((N_PARAMS, N_COVARIATES)),
            0.5 * jnp.ones((N_PARAMS, N_COVARIATES)),
        ).to_event(2),
    )

    # ── Observation params ──
    tau_readout = numpyro.sample("tau_readout", dist.LogNormal(-1.5, 0.5))
    log_phi = numpyro.sample("log_phi", dist.Normal(4.0, 1.0))
    sigma_outcome = numpyro.sample("sigma_outcome", dist.HalfNormal(3.0))
    log_sigma_outcome = numpyro.deterministic(
        "log_sigma_outcome", jnp.log(sigma_outcome + 1e-8),
    )

    # ── Discrepancy ──
    sigma_delta_between = numpyro.sample("sigma_delta_between", dist.HalfNormal(0.3))
    sigma_delta_within = numpyro.sample("sigma_delta_within", dist.HalfNormal(0.3))
    with numpyro.plate("domains_delta", n_domains):
        delta_domain = numpyro.sample(
            "delta_domain", dist.Normal(0.0, sigma_delta_between),
        )

    # ── NEW: Financial parameters ──
    w_opinion = numpyro.sample("w_opinion", dist.Normal(1.0, 1.0))
    w_event = numpyro.sample("w_event", dist.Normal(-1.0, 1.0))
    w_polar = numpyro.sample("w_polar", dist.Normal(-0.5, 1.0))
    log_sigma_market = numpyro.sample("log_sigma_market", dist.Normal(1.0, 0.5))
    lambda_fin = numpyro.sample("lambda_fin", dist.Beta(2.0, 5.0))

    # ── Level 3: Scenarios ──
    plate_kwargs = {"name": "scenarios", "size": n_scenarios}
    if batch_size is not None and batch_size < n_scenarios:
        plate_kwargs["subsample_size"] = batch_size

    with numpyro.plate(**plate_kwargs) as idx:
        d = data.domain_indices[idx]
        x = data.covariates[idx]

        mu_s = mu_domain[d] + x @ B.T
        theta_s = numpyro.sample(
            "theta_s", dist.Normal(mu_s, sigma_domain[d]).to_event(1),
        )
        delta_s = numpyro.sample(
            "delta_s", dist.Normal(delta_domain[d], sigma_delta_within),
        )

        # ── Simulate ──
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

        # Get trajectories for financial likelihood
        def _get_trajectories(calibrable_vec, scenario_data):
            params = {PARAM_NAMES[i]: calibrable_vec[i] for i in range(N_PARAMS)}
            for i, key in enumerate(FROZEN_KEYS):
                params[key] = data.frozen_vec[i]
            return simulate_scenario(params, scenario_data)["trajectories"]

        batch_traj = jax.vmap(
            _get_trajectories, in_axes=(0, _SCENARIO_AXES),
        )(theta_s, batch_sd)

        # Deterministics
        eps = 1e-4
        q_sim = jnp.clip(final_pcts / 100.0, eps, 1.0 - eps)
        logit_q = jnp.log(q_sim / (1.0 - q_sim))
        corrected_pcts = sigmoid(logit_q + delta_s) * 100.0
        numpyro.deterministic("corrected_final_pcts", corrected_pcts)

        # ── Polling likelihood ──
        ll_polling = _likelihood_batch_v2(
            pro_fracs, final_pcts, delta_s,
            data.obs_pro_pcts[idx], data.obs_sample_sizes[idx],
            data.obs_verified_masks[idx], data.obs_gt_pro_pcts[idx],
            data.obs_has_verified[idx], log_phi, log_sigma_outcome,
        )
        numpyro.factor("likelihood_polling", ll_polling)

        # ── Financial likelihood ──
        fin_ll = jax.vmap(
            _financial_ll_one,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0,
                     None, None, None, None),
        )(
            pro_fracs, batch_traj, batch_sd.events, batch_mask,
            fin_arrays["fin_returns"][idx], fin_arrays["fin_masks"][idx],
            fin_arrays["fin_sector_betas"][idx], fin_arrays["fin_has_data"][idx],
            w_opinion, w_event, w_polar, log_sigma_market,
        )
        numpyro.factor("likelihood_financial", lambda_fin * fin_ll)


# ── SVI ──────────────────────────────────────────────────────────────

def run_svi_v4(
    data, fin_arrays,
    prior_mu_global=None, prior_sigma_global=None,
    n_steps=2500, lr=0.003, seed=42, log_every=100,
):
    """Run SVI for v4 model."""
    n_scenarios = data.domain_indices.shape[0]
    n_domains = data.n_domains

    def model_fn():
        return hierarchical_model_v4(
            data, fin_arrays, prior_mu_global, prior_sigma_global,
        )

    init_values = {
        "mu_global": jnp.zeros(N_PARAMS),
        "sigma_global": 0.5 * jnp.ones(N_PARAMS),
        "mu_domain": jnp.zeros((n_domains, N_PARAMS)),
        "sigma_domain": 0.3 * jnp.ones((n_domains, N_PARAMS)),
        "B": jnp.zeros((N_PARAMS, N_COVARIATES)),
        "tau_readout": jnp.array(0.02),
        "log_phi": jnp.array(4.0),
        "sigma_outcome": jnp.array(3.0),
        "sigma_delta_between": jnp.array(0.15),
        "sigma_delta_within": jnp.array(0.15),
        "delta_domain": jnp.zeros(n_domains),
        "theta_s": jnp.zeros((n_scenarios, N_PARAMS)),
        "delta_s": jnp.zeros(n_scenarios),
        "w_opinion": jnp.array(1.0),
        "w_event": jnp.array(-1.0),
        "w_polar": jnp.array(-0.5),
        "log_sigma_market": jnp.array(1.0),
        "lambda_fin": jnp.array(0.25),
    }

    if prior_mu_global is not None:
        init_values["mu_global"] = prior_mu_global

    guide = AutoLowRankMultivariateNormal(
        model_fn, init_loc_fn=init_to_value(values=init_values),
    )
    svi = SVI(model_fn, guide, numpyro.optim.Adam(lr), loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(seed)
    svi_state = svi.init(rng_key)

    losses = []
    for step in range(n_steps):
        svi_state, loss = svi.update(svi_state)
        losses.append(float(loss))
        if log_every and (step + 1) % log_every == 0:
            recent = losses[-min(50, len(losses)):]
            print(f"  Step {step+1:5d}/{n_steps}  loss={loss:.2f}  "
                  f"avg50={sum(recent)/len(recent):.2f}", flush=True)

    return svi.get_params(svi_state), jnp.array(losses), guide


# ── Phase B+C Runner ─────────────────────────────────────────────────

def run_phase_bc_v4(
    n_svi_steps=2500, n_pp_samples=200, lr=0.003, seed=42,
):
    """Full pipeline: load data, fit v4 model, validate."""
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    V4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── Load prior ──
    prior_path = RESULTS_DIR / "synthetic_prior.json"
    if prior_path.exists():
        with open(prior_path) as f:
            prior = json.load(f)
        prior_mu = jnp.array(prior["mu_global"])
        prior_sigma = jnp.maximum(jnp.array(prior["sigma_global"]), 0.3)
        print(f"Prior: μ={list(np.round(prior_mu, 3))}")
    else:
        prior_mu = prior_sigma = None

    # ── Load scenarios ──
    import glob as glob_mod
    json_files = sorted(glob_mod.glob(str(EMPIRICAL_DIR / "*.json")))
    json_files = [f for f in json_files
                  if not f.endswith("manifest.json") and not f.endswith(".meta.json")]

    scenario_datas, observations, covariates_list = [], [], []
    domain_list, scenario_ids, financial_obs = [], [], []
    n_financial = 0

    for path in json_files:
        sid = Path(path).stem
        if sid in _DROP_SCENARIOS:
            continue
        try:
            sd, obs, cov, domain = load_empirical_scenario(path, seed=seed)
            scenario_datas.append(sd)
            observations.append(obs)
            covariates_list.append([
                cov.get("initial_polarization", 0.5),
                cov.get("event_volatility", 0.5),
                cov.get("elite_concentration", 0.5),
                cov.get("institutional_trust", 0.5),
                cov.get("undecided_share", 0.1),
            ])
            domain_list.append(domain)
            scenario_ids.append(sid)

            has_fin, fin_obs = load_financial_observations(path, sd.events.shape[0])
            financial_obs.append(fin_obs)
            if has_fin:
                n_financial += 1
                print(f"  {sid}: financial ✓ (β={fin_obs.sector_beta:.2f})")
        except Exception as e:
            print(f"  WARN: {Path(path).name}: {e}")

    print(f"\n{len(scenario_datas)} scenarios loaded ({n_financial} with financial data)")

    # ── Split ──
    train_idx, test_idx = _stratified_split(scenario_ids, domain_list, 0.2, seed)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    train_sds = [scenario_datas[i] for i in train_idx]
    train_obs = [observations[i] for i in train_idx]
    train_dom = [domain_list[i] for i in train_idx]
    train_covs = [covariates_list[i] for i in train_idx]
    train_fin = [financial_obs[i] for i in train_idx]
    train_ids = [scenario_ids[i] for i in train_idx]

    train_di, train_dn = _build_domain_index(train_dom)
    cal_data, fin_arrs = prepare_data_v4(
        train_sds, train_obs, train_di, train_covs, train_fin,
    )
    print(f"Domains: {train_dn}")

    # ── SVI ──
    print(f"\n{'='*60}\nPhase B (v4 multimodal): {n_svi_steps} steps\n{'='*60}")
    t_b = time.time()
    svi_params, losses, guide = run_svi_v4(
        cal_data, fin_arrs, prior_mu, prior_sigma,
        n_steps=n_svi_steps, lr=lr, seed=seed,
    )
    elapsed_b = time.time() - t_b
    print(f"Elapsed: {elapsed_b:.1f}s, Final loss: {float(losses[-1]):.1f}")

    # ── Posteriors ──
    result, samples = extract_posteriors_v2(
        guide, svi_params, train_dn, train_ids, 1000, seed,
    )

    # Add financial params
    result["financial_linkage"] = {}
    for key in ["w_opinion", "w_event", "w_polar", "lambda_fin"]:
        if key in samples:
            result["financial_linkage"][key] = _summarize(samples[key])
    if "log_sigma_market" in samples:
        result["financial_linkage"]["sigma_market"] = _summarize(
            jnp.exp(samples["log_sigma_market"]))

    fl = result["financial_linkage"]
    print(f"\nFinancial linkage:")
    for k in ["w_opinion", "w_event", "w_polar", "lambda_fin", "sigma_market"]:
        if k in fl:
            print(f"  {k}: {float(fl[k]['mean']):.3f} "
                  f"[{float(fl[k]['ci95_lo']):.3f}, {float(fl[k]['ci95_hi']):.3f}]")

    # ── Validation ──
    print(f"\n{'='*60}\nPhase C: Validation\n{'='*60}")

    mu_d = np.array(samples["mu_domain"])
    sigma_d = np.array(samples["sigma_domain"])
    B_np = np.array(samples["B"])
    delta_d_s = np.array(samples["delta_domain"])
    sigma_dw = np.array(samples["sigma_delta_within"])
    sigma_out = (np.exp(np.array(samples["log_sigma_outcome"]))
                 if "log_sigma_outcome" in samples
                 else np.abs(np.random.normal(0, 3, 1000)))

    rng = np.random.RandomState(seed + 100)
    n_pp = min(n_pp_samples, mu_d.shape[0])
    d2i = {d: i for i, d in enumerate(train_dn)}

    results_list = []
    for eval_idx, group in [(train_idx, "train"), (test_idx, "test")]:
        for i in eval_idx:
            sid = scenario_ids[i]
            d_idx = d2i.get(domain_list[i])
            if d_idx is None:
                continue
            gt = float(observations[i].ground_truth_pro_pct)
            sd = scenario_datas[i]

            cov_arr = np.array(train_covs)
            x_norm = (np.array(covariates_list[i]) - cov_arr.mean(0)) / (cov_arr.std(0) + 1e-6)

            preds = []
            for k in range(n_pp):
                mu_sk = mu_d[k, d_idx] + B_np[k] @ x_norm
                theta = rng.normal(mu_sk, np.maximum(sigma_d[k, d_idx], 0.01))
                delta = rng.normal(delta_d_s[k, d_idx], max(float(sigma_dw[k]), 0.01))

                params = {PARAM_NAMES[j]: float(theta[j]) for j in range(N_PARAMS)}
                for ii, key in enumerate(FROZEN_KEYS):
                    params[key] = get_default_frozen_params()[key]

                try:
                    res = simulate_scenario(params, sd)
                    qf = float(res["final_pro_pct"])
                except Exception:
                    continue

                eps = 1e-4
                qc = max(min(qf / 100, 1 - eps), eps)
                lq = math.log(qc / (1 - qc))
                corrected = 1 / (1 + math.exp(-(lq + delta))) * 100

                sig = max(float(sigma_out[k]), 1.0)
                preds.append(rng.normal(corrected, sig))

            if len(preds) < 10:
                continue

            pp = np.array(preds)
            ci90 = (np.percentile(pp, 5), np.percentile(pp, 95))
            ci50 = (np.percentile(pp, 25), np.percentile(pp, 75))

            crps1 = np.mean(np.abs(pp - gt))
            ix1, ix2 = rng.choice(len(pp), len(pp)), rng.choice(len(pp), len(pp))
            crps = crps1 - 0.5 * np.mean(np.abs(pp[ix1] - pp[ix2]))

            results_list.append({
                "id": sid, "domain": domain_list[i], "group": group,
                "gt": gt, "sim_mean": float(pp.mean()), "sim_std": float(pp.std()),
                "error": float(pp.mean() - gt), "abs_error": abs(float(pp.mean() - gt)),
                "crps": float(crps),
                "ci90": (float(ci90[0]), float(ci90[1])),
                "in_90": bool(ci90[0] <= gt <= ci90[1]),
                "ci50": (float(ci50[0]), float(ci50[1])),
                "in_50": bool(ci50[0] <= gt <= ci50[1]),
                "has_financial_data": financial_obs[i] is not None,
            })

    # ── Report ──
    _save_and_report(result, results_list, losses, train_dn, n_financial,
                     float(losses[-1]), elapsed_b, n_svi_steps)

    total = time.time() - t_start
    print(f"\nTotal: {total:.1f}s ({total/60:.1f}min)")

    # ── Summary ──
    test_r = [r for r in results_list if r["group"] == "test"]
    fin_r = [r for r in results_list if r.get("has_financial_data")]
    for label, grp in [("All", results_list), ("Test", test_r), ("Financial", fin_r)]:
        if grp:
            ae = [r["abs_error"] for r in grp]
            print(f"  {label:12s}: MAE={np.mean(ae):.1f}pp  "
                  f"RMSE={np.sqrt(np.mean([e**2 for e in ae])):.1f}pp  "
                  f"Cov90={np.mean([r['in_90'] for r in grp])*100:.0f}%  "
                  f"(n={len(grp)})")

    return result, results_list


def _save_and_report(posteriors, results, losses, domain_names, n_fin,
                     final_loss, elapsed, n_steps):
    """Save results and generate report."""
    def _ser(obj):
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_ser(v) for v in obj]
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    for name, data in [
        ("posteriors_v4.json", posteriors),
        ("validation_results_v4.json", results),
        ("loss_history_v4.json", {"phase_b": np.array(losses).tolist()}),
    ]:
        p = V4_RESULTS_DIR / name
        with open(p, "w") as f:
            json.dump(_ser(data), f, indent=2)
        print(f"Saved: {p}")

    # Markdown report
    train_r = [r for r in results if r["group"] == "train"]
    test_r = [r for r in results if r["group"] == "test"]
    fin_r = [r for r in results if r.get("has_financial_data")]

    def _m(grp):
        if not grp:
            return {"n": 0, "mae": 0, "rmse": 0, "cov90": 0}
        ae = [r["abs_error"] for r in grp]
        return {"n": len(grp), "mae": np.mean(ae),
                "rmse": np.sqrt(np.mean([e**2 for e in ae])),
                "cov90": np.mean([r["in_90"] for r in grp])}

    mt, mte, mf = _m(train_r), _m(test_r), _m(fin_r)

    lines = [
        "# v4 Multi-Modal Calibration (Polling + Financial Markets)",
        f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"\nScenarios with financial data: {n_fin}",
        f"SVI steps: {n_steps}, Loss: {final_loss:.1f}, Time: {elapsed:.0f}s",
        "",
        "## Headline Metrics",
        "",
        "| Metric | v2 (paper) | v4 (multimodal) Train | v4 Test | v4 Financial |",
        "|---|---|---|---|---|",
        f"| MAE | 14.3 / 19.2pp | {mt['mae']:.1f}pp | {mte['mae']:.1f}pp | {mf['mae']:.1f}pp |",
        f"| RMSE | 18.8 / 26.6pp | {mt['rmse']:.1f}pp | {mte['rmse']:.1f}pp | {mf['rmse']:.1f}pp |",
        f"| Cov90 | 79.4 / 75.0% | {mt['cov90']*100:.0f}% | {mte['cov90']*100:.0f}% | {mf['cov90']*100:.0f}% |",
        "",
    ]

    # Financial linkage
    fl = posteriors.get("financial_linkage", {})
    if fl:
        lines.append("## Financial Linkage Parameters\n")
        lines.append("| Param | Mean | CI95 |")
        lines.append("|---|---|---|")
        for k in ["w_opinion", "w_event", "w_polar", "lambda_fin", "sigma_market"]:
            if k in fl:
                lines.append(f"| {k} | {float(fl[k]['mean']):.3f} | "
                             f"[{float(fl[k]['ci95_lo']):.3f}, {float(fl[k]['ci95_hi']):.3f}] |")
        lines.append("")

    # Per-scenario
    lines.append("## Per-Scenario\n")
    lines.append("| Scenario | Group | GT | Sim | Error | Fin? | 90%CI |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: x["abs_error"], reverse=True):
        lines.append(
            f"| {r['id'][:42]} | {r['group']} | {r['gt']:.1f} | "
            f"{r['sim_mean']:.1f} | {r['error']:+.1f} | "
            f"{'Y' if r.get('has_financial_data') else ''} | "
            f"{'YES' if r['in_90'] else 'no'} |"
        )

    rpt = V4_RESULTS_DIR / "calibration_report_v4.md"
    with open(rpt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {rpt}")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--svi-steps", type=int, default=2500)
    p.add_argument("--pp-samples", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run_phase_bc_v4(a.svi_steps, a.pp_samples, a.lr, a.seed)
