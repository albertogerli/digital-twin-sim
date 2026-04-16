#!/usr/bin/env python3
"""Re-run only steps 6-8 (comparison + report) using saved Phase C results."""

import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calibration.run_grounding_v2_1 import (
    step0_identify_scenarios, step6_7_compare, step8_report,
    GROUNDING_OUTPUT_DIR, RECAL_OUTPUT_DIR,
)

# Step 0
scenarios = step0_identify_scenarios()

# Load checkpoint
with open(GROUNDING_OUTPUT_DIR / "checkpoint.json") as f:
    ckpt = json.load(f)
grounding_results = ckpt.get("grounding_results", {})

# Load posteriors
with open(RECAL_OUTPUT_DIR / "posteriors_v2.1.json") as f:
    posteriors_v21 = json.load(f)

# Reconstruct phase_c_result from the console output data
# We need per_scenario and aggregates. Let's recompute from posteriors + sim.
# Actually, let's just re-run Phase C from the saved Phase B result.
print("\nRe-running Phase C from saved posteriors...")

import src.inference.calibration_pipeline as pipeline
SCENARIOS_V21_DIR = ROOT / "calibration" / "empirical" / "scenarios_v2.1"
SYNTHETIC_PRIOR_PATH = ROOT / "calibration" / "results" / "hierarchical_calibration" / "synthetic_prior.json"

original_dir = pipeline.EMPIRICAL_DIR
pipeline.EMPIRICAL_DIR = SCENARIOS_V21_DIR

with open(SYNTHETIC_PRIOR_PATH) as f:
    prior_data = json.load(f)

import jax.numpy as jnp
phase_a_result = {
    "mu_global_mean": jnp.array(prior_data["mu_global"]),
    "sigma_global_mean": jnp.array(prior_data["sigma_global"]),
}

# Re-run Phase B (needed for Phase C guide/params)
phase_b_result = pipeline.run_phase_b(
    phase_a_result, n_steps=3000, lr=0.002, seed=42, log_every=200,
)

# Save proper posteriors
posteriors_ser = pipeline._to_serializable(phase_b_result["posteriors"])
with open(RECAL_OUTPUT_DIR / "posteriors_v2.1.json", "w") as f:
    json.dump(posteriors_ser, f, indent=2)

# Run Phase C
phase_c_result = pipeline.run_phase_c(
    phase_b_result, phase_a_result=phase_a_result, seed=42,
)

pipeline.EMPIRICAL_DIR = original_dir

posteriors_v21 = posteriors_ser

# Steps 6+7
v2_metrics = step6_7_compare(scenarios, posteriors_v21, phase_b_result, phase_c_result)

# Step 8
step8_report(scenarios, grounding_results, posteriors_v21, v2_metrics)

print("\nDone!")
