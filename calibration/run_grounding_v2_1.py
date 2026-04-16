#!/usr/bin/env python3
"""Grounding v2.1: Selective grounding + recalibration.

Strategy: ground only scenarios with high |delta_s| (by domain), keep the rest.
Then recalibrate Phase B on the mixed dataset.

Usage:
    .venv_cal/bin/python calibration/run_grounding_v2_1.py
"""

import copy
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("grounding_v2.1")

# ── Paths ───────────────────────────────────────────────────────────────

SCENARIOS_DIR = ROOT / "calibration" / "empirical" / "scenarios"
POSTERIORS_V2_PATH = ROOT / "calibration" / "results" / "hierarchical_calibration" / "v2_discrepancy" / "posteriors_v2.json"
SYNTHETIC_PRIOR_PATH = ROOT / "calibration" / "results" / "hierarchical_calibration" / "synthetic_prior.json"
GROUNDING_OUTPUT_DIR = ROOT / "calibration" / "results" / "grounding_v2.1"
SCENARIOS_V21_DIR = ROOT / "calibration" / "empirical" / "scenarios_v2.1"
RECAL_OUTPUT_DIR = ROOT / "calibration" / "results" / "hierarchical_calibration" / "v2.1_grounded"

# Domains to ground (high |delta_s|)
GROUND_DOMAINS = {"financial", "corporate", "energy", "public_health"}
# Domains to keep unchanged (low |delta_s|)
KEEP_DOMAINS = {"political", "technology", "commercial", "environmental", "labor", "social"}

# Drop non-independent scenarios (same as calibration_pipeline.py)
DROP_SCENARIOS = {
    "CORP-2020-BOEING_737_MAX_RETURN_TO_SERVI",
    "TECH-2018-FACEBOOK_CAMBRIDGE_ANALYTICA_S",
}


# ══════════════════════════════════════════════════════════════════════════
# STEP 0: Identify scenarios to ground
# ══════════════════════════════════════════════════════════════════════════

def step0_identify_scenarios():
    """Extract |delta_s| per scenario, classify GROUND vs KEEP."""
    print("=" * 70)
    print("STEP 0: Identify scenarios to ground")
    print("=" * 70)

    with open(POSTERIORS_V2_PATH) as f:
        posteriors = json.load(f)

    # Get all scenario JSON files
    json_files = sorted(SCENARIOS_DIR.glob("*.json"))
    json_files = [f for f in json_files
                  if not f.name.endswith("manifest.json")
                  and not f.name.endswith(".meta.json")]

    scenarios = []
    for path in json_files:
        sid = path.stem
        if sid in DROP_SCENARIOS:
            continue
        with open(path) as f:
            data = json.load(f)
        domain = data.get("domain", "unknown")
        gt_pct = data.get("ground_truth_outcome", {}).get("pro_pct", 50.0)

        # Get delta_s from posteriors
        s_data = posteriors.get("scenarios", {}).get(sid, {})
        delta_s = s_data.get("delta_s", {}).get("mean", 0.0)

        # Determine action
        action = "GROUND" if domain in GROUND_DOMAINS else "KEEP"

        scenarios.append({
            "sid": sid,
            "domain": domain,
            "gt_pct": gt_pct,
            "delta_s": delta_s,
            "abs_delta_s": abs(delta_s),
            "action": action,
            "path": str(path),
            "n_elite": len([a for a in data.get("agents", []) if a.get("type") == "elite"]),
            "n_rounds": data.get("n_rounds", 6),
            "topic": data.get("topic", sid),
            "country": data.get("country", ""),
            "timeframe": data.get("timeframe", ""),
        })

    # Sort by |delta_s| descending
    scenarios.sort(key=lambda x: x["abs_delta_s"], reverse=True)

    # Print table
    print(f"\n{'#':>3} {'Scenario ID':<45} {'Domain':<15} {'|δ_s|':>7} {'GT%':>6} {'Action':<8}")
    print("-" * 90)
    n_ground = 0
    n_keep = 0
    for i, s in enumerate(scenarios):
        mark = "→" if s["action"] == "GROUND" else " "
        print(f"{i+1:>3} {s['sid']:<45} {s['domain']:<15} {s['abs_delta_s']:>7.3f} "
              f"{s['gt_pct']:>6.1f} {mark}{s['action']:<8}")
        if s["action"] == "GROUND":
            n_ground += 1
        else:
            n_keep += 1

    print(f"\nTotal: {len(scenarios)} scenarios")
    print(f"  GROUND: {n_ground} (domains: {sorted(GROUND_DOMAINS)})")
    print(f"  KEEP:   {n_keep} (domains: {sorted(KEEP_DOMAINS)})")

    return scenarios


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Setup LLM and Search (reuse from PoC)
# ══════════════════════════════════════════════════════════════════════════

def step1_setup():
    """Setup LLM and search functions."""
    print("\n" + "=" * 70)
    print("STEP 1: Setup LLM and Search")
    print("=" * 70)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = "gemini-3.1-flash-lite-preview"

    # LLM function
    def llm_fn(prompt: str) -> str:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"temperature": 0.2, "max_output_tokens": 4000},
                )
                return response.text
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"LLM retry {attempt+1} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise
        return ""

    # Search function using Google Search grounding
    def search_fn(query: str) -> list[dict]:
        prompt = (
            f"Search the web for: {query}\n\n"
            "Return a detailed factual summary with specific dates, names, numbers, "
            "and events. Include as many concrete facts as possible."
        )
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=4000,
                ),
            )
            results = []
            for candidate in response.candidates or []:
                gm = getattr(candidate, "grounding_metadata", None)
                if not gm:
                    continue
                for chunk in getattr(gm, "grounding_chunks", []) or []:
                    web = getattr(chunk, "web", None)
                    if web:
                        url = getattr(web, "uri", "") or ""
                        title = getattr(web, "title", "") or ""
                        results.append({"url": url, "text": title})
                for support in getattr(gm, "grounding_supports", []) or []:
                    seg = getattr(support, "segment", None)
                    if seg:
                        text = getattr(seg, "text", "") or ""
                        if text and len(text) > 30:
                            urls = []
                            for idx_obj in getattr(support, "grounding_chunk_indices", []) or []:
                                idx = int(idx_obj) if not isinstance(idx_obj, int) else idx_obj
                                if idx < len(results):
                                    urls.append(results[idx]["url"])
                            url = urls[0] if urls else ""
                            results.append({"url": url, "text": text})
            if response.text:
                results.append({"url": "gemini-grounded-search", "text": response.text})
            time.sleep(0.3)
            return results
        except Exception as e:
            logger.warning(f"Grounded search failed: {e}")
            return []

    print(f"✓ LLM: Google Gemini ({model})")
    print(f"✓ Search: Gemini Google Search Grounding")
    return llm_fn, search_fn


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Ground selected scenarios
# ══════════════════════════════════════════════════════════════════════════

def step2_ground_scenarios(scenarios, llm_fn, search_fn):
    """Ground all GROUND scenarios, save results."""
    print("\n" + "=" * 70)
    print("STEP 2: Ground selected scenarios")
    print("=" * 70)

    from src.grounding.pipeline import ground_scenario

    to_ground = [s for s in scenarios if s["action"] == "GROUND"]
    print(f"\nGrounding {len(to_ground)} scenarios...\n")

    results = {}
    failed = []

    for i, s in enumerate(to_ground):
        sid = s["sid"]
        print(f"\n── [{i+1}/{len(to_ground)}] {sid} ──")

        out_dir = GROUNDING_OUTPUT_DIR / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            fact_sheet, grounded_agents, grounded_events = ground_scenario(
                topic=s["topic"],
                domain=s["domain"],
                country=s["country"],
                timeframe=s["timeframe"],
                scenario_id=sid,
                n_elite=max(s["n_elite"], 3),  # at least 3 elite
                n_rounds=s["n_rounds"],
                llm_fn=llm_fn,
                search_fn=search_fn,
            )

            # Save intermediates
            with open(out_dir / "fact_sheet.json", "w") as f:
                json.dump(fact_sheet.to_dict(), f, indent=2, default=str)
            with open(out_dir / "agents.json", "w") as f:
                json.dump(grounded_agents, f, indent=2)
            with open(out_dir / "events.json", "w") as f:
                json.dump(grounded_events, f, indent=2)
            with open(out_dir / "llm_context.txt", "w") as f:
                f.write(fact_sheet.to_llm_context())

            quality = fact_sheet.quality_score
            n_agents = len(grounded_agents)
            n_events = len(grounded_events)

            print(f"  ✓ quality={quality}, agents={n_agents}, events={n_events}")
            for a in grounded_agents[:5]:
                print(f"    {a['name'][:35]:35s} pos={a['position']:+.2f} inf={a['influence']:.2f}")

            if quality < 30 and n_agents == 0:
                print(f"  ⚠ Quality too low, keeping original")
                failed.append(sid)
                results[sid] = {"action": "KEEP_FALLBACK", "reason": f"quality={quality}"}
            else:
                results[sid] = {
                    "action": "GROUNDED",
                    "quality": quality,
                    "n_agents": n_agents,
                    "n_events": n_events,
                    "agents": grounded_agents,
                    "events": grounded_events,
                }

        except Exception as e:
            print(f"  ⚠ FAILED: {e}")
            failed.append(sid)
            results[sid] = {"action": "KEEP_FALLBACK", "reason": str(e)}

    n_grounded = sum(1 for r in results.values() if r["action"] == "GROUNDED")
    n_fallback = sum(1 for r in results.values() if r["action"] == "KEEP_FALLBACK")
    print(f"\n\nGrounding complete: {n_grounded} grounded, {n_fallback} fallback to original")
    if failed:
        print(f"  Failed: {failed}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Prepare mixed dataset
# ══════════════════════════════════════════════════════════════════════════

def step3_prepare_dataset(scenarios, grounding_results):
    """Create scenarios_v2.1/ with grounded + original scenarios."""
    print("\n" + "=" * 70)
    print("STEP 3: Prepare mixed dataset")
    print("=" * 70)

    SCENARIOS_V21_DIR.mkdir(parents=True, exist_ok=True)

    grounded_list = []
    original_list = []

    for s in scenarios:
        sid = s["sid"]
        src_path = Path(s["path"])
        dst_path = SCENARIOS_V21_DIR / f"{sid}.json"

        gr = grounding_results.get(sid, {})

        if gr.get("action") == "GROUNDED":
            # Create modified scenario JSON
            with open(src_path) as f:
                scenario = json.load(f)

            # Replace elite agents with grounded ones
            non_elite = [a for a in scenario["agents"] if a.get("type") != "elite"]
            grounded_elite = []
            for ga in gr["agents"]:
                grounded_elite.append({
                    "type": "elite",
                    "name": ga.get("name", "unknown"),
                    "initial_position": ga.get("position", 0.0),
                    "influence": ga.get("influence", 0.5),
                    "rigidity": ga.get("rigidity", 0.7),
                    "_grounded": True,
                    "_stake": ga.get("_stake", "unknown"),
                    "_role": ga.get("role", ""),
                })
            scenario["agents"] = grounded_elite + non_elite

            # Replace events with grounded ones (if any)
            if gr["events"]:
                scenario["events"] = []
                for evt in gr["events"]:
                    scenario["events"].append({
                        "round": evt["round"],
                        "description": evt.get("description", ""),
                        "shock_magnitude": evt.get("shock_magnitude", 0.5),
                        "shock_direction": evt.get("shock_direction", -1),
                        "_verified": True,
                    })

            scenario["_grounding_v2.1"] = {
                "grounded": True,
                "quality": gr["quality"],
                "n_grounded_agents": gr["n_agents"],
                "n_grounded_events": gr["n_events"],
            }

            with open(dst_path, "w") as f:
                json.dump(scenario, f, indent=2)
            grounded_list.append(sid)

        else:
            # Copy original
            shutil.copy2(src_path, dst_path)
            # Also copy meta if exists
            meta_path = src_path.with_suffix(".meta.json")
            if meta_path.exists():
                shutil.copy2(meta_path, SCENARIOS_V21_DIR / f"{sid}.meta.json")
            original_list.append(sid)

    # Create manifest
    manifest = {
        "version": "v2.1_grounded",
        "n_scenarios": len(scenarios),
        "n_grounded": len(grounded_list),
        "n_original": len(original_list),
        "grounded_scenarios": sorted(grounded_list),
        "original_scenarios": sorted(original_list),
        "grounding_method": "AgentGrounder + ScenarioResearcher (Gemini Google Search)",
        "grounding_date": time.strftime("%Y-%m-%d"),
    }
    with open(SCENARIOS_V21_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Dataset created: {SCENARIOS_V21_DIR}")
    print(f"  Total: {len(scenarios)} scenarios")
    print(f"  Grounded: {len(grounded_list)}")
    print(f"  Original: {len(original_list)}")

    # Verify count
    json_count = len(list(SCENARIOS_V21_DIR.glob("*.json")))
    meta_count = len(list(SCENARIOS_V21_DIR.glob("*.meta.json")))
    manifest_count = 1
    scenario_count = json_count - meta_count - manifest_count
    print(f"  JSON files: {scenario_count} scenarios + {meta_count} meta + 1 manifest")

    return grounded_list, original_list


# ══════════════════════════════════════════════════════════════════════════
# STEP 4+5: Recalibrate Phase B
# ══════════════════════════════════════════════════════════════════════════

def step4_5_recalibrate():
    """Recalibrate Phase B using the v2.1 dataset."""
    print("\n" + "=" * 70)
    print("STEP 4+5: Build ScenarioData + Recalibrate Phase B")
    print("=" * 70)

    # Monkey-patch the pipeline to use our directory
    import src.inference.calibration_pipeline as pipeline
    original_dir = pipeline.EMPIRICAL_DIR
    pipeline.EMPIRICAL_DIR = SCENARIOS_V21_DIR

    # Load Phase A synthetic prior
    with open(SYNTHETIC_PRIOR_PATH) as f:
        prior_data = json.load(f)

    import jax.numpy as jnp
    phase_a_result = {
        "mu_global_mean": jnp.array(prior_data["mu_global"]),
        "sigma_global_mean": jnp.array(prior_data["sigma_global"]),
    }

    print(f"  Transfer prior μ: {prior_data['mu_global']}")
    print(f"  Transfer prior σ: {prior_data['sigma_global']}")

    # Run Phase B
    phase_b_result = pipeline.run_phase_b(
        phase_a_result,
        n_steps=3000,
        lr=0.002,
        seed=42,
        log_every=200,
    )

    # Run Phase C
    phase_c_result = pipeline.run_phase_c(
        phase_b_result,
        phase_a_result=phase_a_result,
        seed=42,
    )

    # Save posteriors
    RECAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    posteriors = phase_b_result["posteriors"]
    posteriors_ser = pipeline._to_serializable(posteriors)
    with open(RECAL_OUTPUT_DIR / "posteriors_v2.1.json", "w") as f:
        json.dump(posteriors_ser, f, indent=2)

    # Restore original dir
    pipeline.EMPIRICAL_DIR = original_dir

    return phase_b_result, phase_c_result, posteriors


# ══════════════════════════════════════════════════════════════════════════
# STEP 6+7: Compare v2 vs v2.1
# ══════════════════════════════════════════════════════════════════════════

def step6_7_compare(scenarios, posteriors_v21, phase_b_result, phase_c_result):
    """Compare v2 vs v2.1 metrics."""
    print("\n" + "=" * 70)
    print("STEP 6+7: Compare v2 vs v2.1")
    print("=" * 70)

    with open(POSTERIORS_V2_PATH) as f:
        posteriors_v2 = json.load(f)

    # Extract Phase C aggregates
    train_agg = phase_c_result.get("train_aggregate", {}) if phase_c_result else {}
    test_agg = phase_c_result.get("test_aggregate", {}) if phase_c_result else {}
    per_scenario = phase_c_result.get("per_scenario", {}) if phase_c_result else {}

    # ── Table A: Headline Metrics ──
    print("\n── Table A: Headline Metrics ──")
    print(f"{'Metric':<35} {'v2':>12} {'v2.1':>12} {'Δ':>10}")
    print("-" * 72)

    v2_metrics_table = [
        ("MAE test",             19.2,  test_agg.get("mae")),
        ("MAE train",            14.3,  train_agg.get("mae")),
        ("RMSE test",            26.6,  test_agg.get("rmse")),
        ("RMSE train",           None,  train_agg.get("rmse")),
        ("Coverage 90% train",   79.4,  train_agg.get("coverage_90", 0) * 100 if train_agg.get("coverage_90") is not None else None),
        ("Coverage 90% test",    None,  test_agg.get("coverage_90", 0) * 100 if test_agg.get("coverage_90") is not None else None),
        ("CRPS test",            None,  test_agg.get("mean_crps")),
    ]

    for name, v2_val, v21_val in v2_metrics_table:
        v2_str = f"{v2_val:>10.1f}pp" if v2_val is not None else f"{'—':>12}"
        if v21_val is not None:
            v21_str = f"{v21_val:>10.1f}pp"
            if v2_val is not None:
                delta = v21_val - v2_val
                d_str = f"{delta:>+8.1f}pp"
            else:
                d_str = f"{'—':>10}"
        else:
            v21_str = f"{'?':>12}"
            d_str = f"{'?':>10}"
        print(f"{name:<35} {v2_str} {v21_str} {d_str}")

    # ── Table B: Discrepancy / Global Params ──
    print("\n── Table B: Global Parameter Comparison ──")
    v2_global = posteriors_v2.get("global", {})
    v21_global = posteriors_v21.get("global", {}) if posteriors_v21 else {}

    param_names = ["alpha_media", "alpha_herd", "alpha_event", "decay_rate"]
    print(f"{'Parameter':<20} {'v2 μ':>10} {'v2.1 μ':>10} {'Δ':>10}")
    print("-" * 55)
    for i, pn in enumerate(param_names):
        v2_mu = v2_global.get("mu_global", {}).get("mean", [0]*4)
        v21_mu = v21_global.get("mu_global", {}).get("mean", [0]*4)
        v2_val = v2_mu[i] if isinstance(v2_mu, list) and len(v2_mu) > i else 0
        v21_val = v21_mu[i] if isinstance(v21_mu, list) and len(v21_mu) > i else 0
        print(f"  {pn:<18} {v2_val:>+10.4f} {v21_val:>+10.4f} {v21_val-v2_val:>+10.4f}")

    # v2 discrepancy (v2.1 doesn't have discrepancy model)
    v2_disc = posteriors_v2.get("discrepancy", {})
    if v2_disc:
        sigma_b = v2_disc.get("sigma_delta_between", {}).get("mean", 0)
        sigma_w = v2_disc.get("sigma_delta_within", {}).get("mean", 0)
        print(f"\n  v2 σ_b,between: {sigma_b:.3f}")
        print(f"  v2 σ_b,within:  {sigma_w:.3f}")
        print(f"  (v2.1 uses transfer model — no discrepancy term)")

    # ── Per-domain error comparison ──
    # Use Phase C sim errors as proxy for bias (v2 uses delta_s, v2.1 uses sim error)
    v2_scenarios = posteriors_v2.get("scenarios", {})
    domain_map = {s["sid"]: s["domain"] for s in scenarios}
    gt_map = {s["sid"]: s["gt_pct"] for s in scenarios}

    print("\n── Per-domain mean |error| (sim vs ground truth) ──")
    print(f"  {'Domain':<15} {'Action':>6}  {'v2 |δ_s|':>9}  {'v2.1 |err|':>11}  {'Δ':>8}")
    print("  " + "-" * 55)

    for domain in sorted(set(domain_map.values())):
        sids_in_domain = [sid for sid, d in domain_map.items() if d == domain]

        v2_deltas = [abs(v2_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0))
                     for sid in sids_in_domain if sid in v2_scenarios]
        v21_errors = [abs(per_scenario.get(sid, {}).get("error", 0))
                      for sid in sids_in_domain if sid in per_scenario]

        v2_mean = sum(v2_deltas) / len(v2_deltas) if v2_deltas else 0
        v21_mean = sum(v21_errors) / len(v21_errors) if v21_errors else 0

        action = "GROUND" if domain in GROUND_DOMAINS else "KEEP"
        # Note: v2 uses delta_s (latent space), v2.1 uses sim error (pp).
        # Not directly comparable but directionally informative.
        print(f"  {domain:<15} {action:>6}  {v2_mean:>9.3f}  {v21_mean:>9.1f}pp  {'':>8}")

    # ── Table C: Per-scenario on grounded scenarios ──
    print("\n── Table C: Per-Scenario (Grounded) ──")
    print(f"{'Scenario':<45} {'Domain':<12} {'GT%':>5} {'v2 δ_s':>8} {'v2.1 err':>9} {'v2.1 sim%':>9}")
    print("-" * 95)

    grounded_sids = [s["sid"] for s in scenarios if s["action"] == "GROUND"]
    for sid in sorted(grounded_sids):
        s = next((x for x in scenarios if x["sid"] == sid), None)
        if not s:
            continue
        v2_ds = v2_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
        v21_r = per_scenario.get(sid, {})
        v21_err = v21_r.get("error", 0)
        v21_sim = v21_r.get("sim_final", 0)
        print(f"{sid:<45} {s['domain']:<12} {s['gt_pct']:>5.1f} {v2_ds:>+8.3f} {v21_err:>+9.1f} {v21_sim:>9.1f}")

    # ── Table D: Sanity check on kept scenarios ──
    print("\n── Table D: Sanity Check (Kept Scenarios, sample) ──")
    print(f"{'Scenario':<45} {'Domain':<12} {'GT%':>5} {'v2 δ_s':>8} {'v2.1 err':>9}")
    print("-" * 85)

    kept_sids = [s["sid"] for s in scenarios if s["action"] == "KEEP"]
    for sid in sorted(kept_sids)[:10]:
        s = next((x for x in scenarios if x["sid"] == sid), None)
        domain = s["domain"] if s else "?"
        gt = s["gt_pct"] if s else 0
        v2_ds = v2_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
        v21_r = per_scenario.get(sid, {})
        v21_err = v21_r.get("error", 0)
        print(f"{sid:<45} {domain:<12} {gt:>5.1f} {v2_ds:>+8.3f} {v21_err:>+9.1f}")

    return {
        "train_agg": train_agg,
        "test_agg": test_agg,
        "per_scenario": per_scenario,
    }


# ══════════════════════════════════════════════════════════════════════════
# STEP 8: Report
# ══════════════════════════════════════════════════════════════════════════

def step8_report(scenarios, grounding_results, posteriors_v21, v2_metrics):
    """Generate markdown report."""
    print("\n" + "=" * 70)
    print("STEP 8: Generate Report")
    print("=" * 70)

    with open(POSTERIORS_V2_PATH) as f:
        posteriors_v2 = json.load(f)

    lines = [
        "# Grounding v2.1 Recalibration Report",
        "",
        "## 1. Strategy",
        "",
        "Selective grounding: ground only high-|δ_s| domains (financial, corporate, energy, public_health),",
        "keep low-|δ_s| domains unchanged (political, technology, etc.).",
        "Then recalibrate Phase B on the mixed dataset.",
        "",
        "## 2. Scenarios",
        "",
    ]

    grounded = [s for s in scenarios if s["action"] == "GROUND"]
    kept = [s for s in scenarios if s["action"] == "KEEP"]
    lines.append(f"- **Grounded**: {len(grounded)} scenarios")
    lines.append(f"- **Kept**: {len(kept)} scenarios")
    lines.append("")

    n_success = sum(1 for sid in [s["sid"] for s in grounded]
                    if grounding_results.get(sid, {}).get("action") == "GROUNDED")
    n_fallback = len(grounded) - n_success
    lines.append(f"- Grounding success: {n_success}/{len(grounded)}")
    if n_fallback:
        lines.append(f"- Fallback to original: {n_fallback}")
    lines.append("")

    # Phase C metrics comparison
    train_agg = v2_metrics.get("train_agg", {}) if isinstance(v2_metrics, dict) else {}
    test_agg = v2_metrics.get("test_agg", {}) if isinstance(v2_metrics, dict) else {}
    per_scenario = v2_metrics.get("per_scenario", {}) if isinstance(v2_metrics, dict) else {}

    lines.extend([
        "## 3. Calibration Metrics Comparison",
        "",
        "| Metric | v2 | v2.1 | Δ |",
        "|---|---|---|---|",
    ])

    v2_mae_test = 19.2
    v21_mae_test = test_agg.get("mae", None)
    if v21_mae_test is not None:
        lines.append(f"| MAE test | {v2_mae_test:.1f}pp | {v21_mae_test:.1f}pp | {v21_mae_test-v2_mae_test:+.1f}pp |")
    else:
        lines.append(f"| MAE test | {v2_mae_test:.1f}pp | ? | ? |")

    v21_mae_train = train_agg.get("mae", None)
    if v21_mae_train is not None:
        lines.append(f"| MAE train | 14.3pp | {v21_mae_train:.1f}pp | {v21_mae_train-14.3:+.1f}pp |")

    v21_rmse_test = test_agg.get("rmse", None)
    if v21_rmse_test is not None:
        lines.append(f"| RMSE test | 26.6pp | {v21_rmse_test:.1f}pp | {v21_rmse_test-26.6:+.1f}pp |")

    v21_cov90_train = train_agg.get("coverage_90", None)
    if v21_cov90_train is not None:
        lines.append(f"| Coverage 90% train | 79.4% | {v21_cov90_train*100:.1f}% | {v21_cov90_train*100-79.4:+.1f}pp |")

    # Per-scenario highlights
    lines.extend([
        "",
        "## 4. Notable Scenarios",
        "",
        "| Scenario | GT% | v2 δ_s | v2.1 sim% | v2.1 err |",
        "|---|---|---|---|---|",
    ])

    v2_scenarios = posteriors_v2.get("scenarios", {})
    grounded_sids = [s["sid"] for s in scenarios if s["action"] == "GROUND"]
    for sid in sorted(grounded_sids):
        s = next((x for x in scenarios if x["sid"] == sid), None)
        if not s:
            continue
        v2_ds = v2_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
        v21_r = per_scenario.get(sid, {})
        v21_err = v21_r.get("error", 0)
        v21_sim = v21_r.get("sim_final", 0)
        lines.append(f"| {sid[:40]} | {s['gt_pct']:.1f} | {v2_ds:+.3f} | {v21_sim:.1f} | {v21_err:+.1f}pp |")

    lines.extend([
        "",
        "## 5. Verdict",
        "",
    ])

    if v21_mae_test is not None:
        if v21_mae_test < v2_mae_test * 0.8:
            verdict = "MAJOR IMPROVEMENT"
        elif v21_mae_test < v2_mae_test:
            verdict = "MEANINGFUL IMPROVEMENT"
        elif v21_mae_test < v2_mae_test * 1.1:
            verdict = "COMPARABLE — grounding didn't hurt, but didn't help test MAE"
        else:
            verdict = "REGRESSION — test MAE increased"
        lines.append(f"**{verdict}**: MAE test v2={v2_mae_test:.1f}pp → v2.1={v21_mae_test:.1f}pp ({v21_mae_test-v2_mae_test:+.1f}pp)")
    else:
        lines.append("**INCOMPLETE — Phase C metrics not available**")

    lines.extend([
        "",
        "---",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*",
    ])

    report_path = GROUNDING_OUTPUT_DIR / "recalibration_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    # Also save to recal output
    RECAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RECAL_OUTPUT_DIR / "calibration_report_v2.1.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n✓ Report saved: {report_path}")
    print(f"✓ Report saved: {RECAL_OUTPUT_DIR / 'calibration_report_v2.1.md'}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("╔" + "═" * 68 + "╗")
    print("║    GROUNDING v2.1: Selective Grounding + Recalibration           ║")
    print("╚" + "═" * 68 + "╝")

    # Step 0
    scenarios = step0_identify_scenarios()

    # Step 1
    llm_fn, search_fn = step1_setup()

    # Step 2
    grounding_results = step2_ground_scenarios(scenarios, llm_fn, search_fn)

    # Step 3
    grounded_list, original_list = step3_prepare_dataset(scenarios, grounding_results)

    # Save grounding checkpoint
    checkpoint = {
        "scenarios": [{k: v for k, v in s.items() if k != "path"} for s in scenarios],
        "grounding_results": {
            sid: {k: v for k, v in r.items() if k not in ("agents", "events")}
            for sid, r in grounding_results.items()
        },
        "grounded_list": grounded_list,
        "original_list": original_list,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    GROUNDING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(GROUNDING_OUTPUT_DIR / "checkpoint.json", "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)

    t_grounding = time.time()
    print(f"\n⏱ Grounding completed in {(t_grounding - t_start)/60:.1f} min")

    # Steps 4+5
    try:
        phase_b_result, phase_c_result, posteriors_v21 = step4_5_recalibrate()
    except Exception as e:
        print(f"\n⚠ RECALIBRATION FAILED: {e}")
        print("Saving partial report...")
        posteriors_v21 = {}
        phase_b_result = None
        phase_c_result = None

    t_recal = time.time()
    print(f"\n⏱ Recalibration completed in {(t_recal - t_grounding)/60:.1f} min")

    # Steps 6+7
    v2_metrics = step6_7_compare(scenarios, posteriors_v21, phase_b_result, phase_c_result)

    # Step 8
    step8_report(scenarios, grounding_results, posteriors_v21, v2_metrics)

    t_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"TOTAL TIME: {t_total/60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
