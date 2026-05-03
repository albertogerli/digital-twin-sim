"""Sprint 15 — re-run NumPyro SVI calibration on the empirical scenarios after
all Sprint 1-13 simulator changes (stakeholder graph, country aliases, realism
gate fixes, agent prompt improvements, engine logic fixes).

The goal is a fresh MAE / coverage / CRPS baseline directly comparable to the
previous v2_discrepancy run so we can confirm the simulator changes didn't
regress the calibrated posteriors.

Output goes to a NEW directory so the canonical v2_discrepancy results stay
intact for diffing.

Usage:
    python -m calibration.sprint15_recalibrate           # full 3000-step run
    python -m calibration.sprint15_recalibrate --quick   # 500-step smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch V2_RESULTS_DIR before importing run_phase_bc_v2 so the run writes
# into the sprint15 sub-directory.
SPRINT15_DIR = PROJECT_ROOT / "calibration" / "results" / "hierarchical_calibration" / "sprint15"
SPRINT15_DIR.mkdir(parents=True, exist_ok=True)

from src.inference import hierarchical_model_v2 as hm_v2  # noqa: E402

hm_v2.V2_RESULTS_DIR = SPRINT15_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svi-steps", type=int, default=3000,
                        help="SVI steps (3000 matches the previous baseline; 500 for smoke)")
    parser.add_argument("--pp-samples", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 500 SVI steps, 50 PP samples")
    args = parser.parse_args()

    if args.quick:
        args.svi_steps = 500
        args.pp_samples = 50

    print(f"┌─ Sprint 15 re-calibration ────────────────────────────────────")
    print(f"│  SVI steps : {args.svi_steps}")
    print(f"│  PP samples: {args.pp_samples}")
    print(f"│  LR        : {args.lr}")
    print(f"│  Seed      : {args.seed}")
    print(f"│  Out dir   : {SPRINT15_DIR.relative_to(PROJECT_ROOT)}")
    print(f"└────────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    t0 = time.time()
    posteriors, results = hm_v2.run_phase_bc_v2(
        n_svi_steps=args.svi_steps,
        n_pp_samples=args.pp_samples,
        lr=args.lr,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    print(f"\nRe-calibration complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results: {SPRINT15_DIR}")


if __name__ == "__main__":
    main()
