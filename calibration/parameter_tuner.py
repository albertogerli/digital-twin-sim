"""Grid search over OpinionDynamics parameters for calibration."""

import itertools
import json
import logging
from pathlib import Path
from typing import Optional

from .historical_scenario import GroundTruth
from .trajectory_comparator import compute_calibration_score

logger = logging.getLogger(__name__)

# Parameter grid for grid search
DEFAULT_GRID = {
    "anchor_weight": [0.05, 0.10, 0.15, 0.20],
    "social_weight": [0.05, 0.10, 0.15, 0.20, 0.25],
    "event_weight": [0.02, 0.05, 0.08, 0.12],
    "herd_weight": [0.02, 0.05, 0.08],
    "herd_threshold": [0.15, 0.20, 0.30],
    "direct_shift_weight": [0.2, 0.4, 0.6],
    "anchor_drift_rate": [0.10, 0.20, 0.35],
}


def generate_parameter_combinations(grid: dict = None) -> list[dict]:
    """Generate all parameter combinations from the grid."""
    grid = grid or DEFAULT_GRID
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


class CalibrationResult:
    """Result from a single calibration run."""

    def __init__(self, params: dict, metrics: dict, simulated_pro_pct: float):
        self.params = params
        self.metrics = metrics
        self.simulated_pro_pct = simulated_pro_pct

    @property
    def composite_score(self) -> float:
        return self.metrics.get("composite_score", float("inf"))


def find_best_params(results: list[CalibrationResult]) -> Optional[CalibrationResult]:
    """Find the parameter combination with the lowest composite score."""
    if not results:
        return None
    return min(results, key=lambda r: r.composite_score)


def save_calibrated_params(
    params: dict,
    metrics: dict,
    output_path: str,
    domain: str = "political",
):
    """Save calibrated parameters to JSON file."""
    output = {
        "domain": domain,
        "calibrated_params": params,
        "calibration_metrics": metrics,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved calibrated params to {output_path}")


def load_calibrated_params(path: str) -> Optional[dict]:
    """Load calibrated parameters from JSON file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("calibrated_params", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return None
