#!/usr/bin/env python3
"""Re-run steps 4-8 of v2.1 pipeline (recalibration + comparison + report)."""

import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calibration.run_grounding_v2_1 import (
    step0_identify_scenarios, step4_5_recalibrate, step6_7_compare, step8_report,
    GROUNDING_OUTPUT_DIR,
)

t_start = time.time()

# Step 0
scenarios = step0_identify_scenarios()

# Load checkpoint
with open(GROUNDING_OUTPUT_DIR / "checkpoint.json") as f:
    ckpt = json.load(f)
grounding_results = ckpt.get("grounding_results", {})

# Steps 4+5
try:
    phase_b_result, phase_c_result, posteriors_v21 = step4_5_recalibrate()
except Exception as e:
    import traceback
    print(f"\nRECALIBRATION FAILED: {e}")
    traceback.print_exc()
    posteriors_v21 = {}
    phase_b_result = None
    phase_c_result = None

t_recal = time.time()
print(f"\n⏱ Recalibration: {(t_recal - t_start)/60:.1f} min")

# Steps 6+7
v2_metrics = step6_7_compare(scenarios, posteriors_v21, phase_b_result, phase_c_result)

# Step 8
step8_report(scenarios, grounding_results, posteriors_v21, v2_metrics)

print(f"\nTOTAL: {(time.time() - t_start)/60:.1f} min")
