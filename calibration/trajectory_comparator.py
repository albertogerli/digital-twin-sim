"""Metrics for comparing simulated trajectories to ground truth."""

import math
from typing import Optional

from .historical_scenario import GroundTruth


def outcome_accuracy(simulated_pro_pct: float, ground_truth: GroundTruth) -> float:
    """Absolute error between simulated and actual final outcome (lower is better)."""
    return abs(simulated_pro_pct - ground_truth.final_outcome_pro_pct)


def position_mae(
    simulated_positions_per_round: list[float],
    ground_truth: GroundTruth,
) -> Optional[float]:
    """Mean Absolute Error of simulated avg position vs real polling per round.

    Converts polling % to [-1, +1] scale: position = (pro - against) / 100.
    """
    if not ground_truth.polling_trajectory:
        return None

    errors = []
    for dp in ground_truth.polling_trajectory:
        round_idx = dp.round_equivalent - 1
        if round_idx < len(simulated_positions_per_round):
            real_position = (dp.pro_pct - dp.against_pct) / 100.0
            sim_position = simulated_positions_per_round[round_idx]
            errors.append(abs(sim_position - real_position))

    return sum(errors) / len(errors) if errors else None


def trajectory_dtw(
    simulated: list[float],
    real: list[float],
) -> float:
    """Dynamic Time Warping distance between two trajectories.
    Lower = more similar shape.
    """
    n, m = len(simulated), len(real)
    if n == 0 or m == 0:
        return float("inf")

    dtw = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dtw[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(simulated[i - 1] - real[j - 1])
            dtw[i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])

    return dtw[n][m]


def compute_calibration_score(
    simulated_pro_pct: float,
    simulated_positions_per_round: list[float],
    ground_truth: GroundTruth,
) -> dict:
    """Compute all calibration metrics and a composite score."""
    oa = outcome_accuracy(simulated_pro_pct, ground_truth)
    mae = position_mae(simulated_positions_per_round, ground_truth)

    # Convert polling to position trajectory for DTW
    real_trajectory = [
        (dp.pro_pct - dp.against_pct) / 100.0
        for dp in ground_truth.polling_trajectory
    ]
    dtw = trajectory_dtw(simulated_positions_per_round, real_trajectory) if real_trajectory else None

    # Composite: outcome accuracy is most important (weight 0.5)
    # Lower composite = better calibration
    composite = oa * 0.5
    if mae is not None:
        composite += mae * 100 * 0.3  # Scale MAE to percentage
    if dtw is not None:
        composite += dtw * 10 * 0.2   # Scale DTW

    return {
        "outcome_error_pct": round(oa, 2),
        "position_mae": round(mae, 4) if mae is not None else None,
        "trajectory_dtw": round(dtw, 4) if dtw is not None else None,
        "composite_score": round(composite, 2),
    }
