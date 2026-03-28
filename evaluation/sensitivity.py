"""Sensitivity analysis — run same scenario N times and measure variance."""

import logging
import statistics
from typing import Optional

logger = logging.getLogger(__name__)


def compute_run_variance(
    run_results: list[dict],
) -> dict:
    """Compute variance metrics across multiple runs of the same scenario.

    Each run_result should contain:
    - final_positions: dict[str, float] (agent_id -> final position)
    - final_polarization: float
    - realism_score: float (optional)

    Returns variance metrics. Target: variance < 15%.
    """
    if len(run_results) < 2:
        return {"error": "Need at least 2 runs for variance analysis"}

    # Polarization variance
    polarizations = [r.get("final_polarization", 0) for r in run_results]
    pol_mean = statistics.mean(polarizations)
    pol_stdev = statistics.stdev(polarizations) if len(polarizations) > 1 else 0
    pol_cv = (pol_stdev / pol_mean * 100) if pol_mean > 0 else 0

    # Per-agent position variance
    all_agent_ids = set()
    for r in run_results:
        all_agent_ids.update(r.get("final_positions", {}).keys())

    agent_variances = {}
    for agent_id in all_agent_ids:
        positions = [
            r.get("final_positions", {}).get(agent_id)
            for r in run_results
            if agent_id in r.get("final_positions", {})
        ]
        positions = [p for p in positions if p is not None]
        if len(positions) >= 2:
            agent_variances[agent_id] = {
                "mean": round(statistics.mean(positions), 3),
                "stdev": round(statistics.stdev(positions), 3),
            }

    avg_agent_stdev = (
        statistics.mean(v["stdev"] for v in agent_variances.values())
        if agent_variances else 0
    )

    # Realism score variance
    realism_scores = [r.get("realism_score") for r in run_results if r.get("realism_score") is not None]
    realism_stdev = statistics.stdev(realism_scores) if len(realism_scores) > 1 else 0

    return {
        "num_runs": len(run_results),
        "polarization_mean": round(pol_mean, 2),
        "polarization_stdev": round(pol_stdev, 2),
        "polarization_cv_pct": round(pol_cv, 1),
        "avg_agent_position_stdev": round(avg_agent_stdev, 3),
        "realism_score_stdev": round(realism_stdev, 1) if realism_scores else None,
        "is_stable": pol_cv < 15 and avg_agent_stdev < 0.15,
    }
