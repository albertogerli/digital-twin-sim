"""Post-simulation realism assessment — checks outputs for plausibility."""

import logging
from typing import Optional

from core.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

EVENT_COHERENCE_PROMPT = """Rate the overall coherence and realism of this simulation's event sequence.

SCENARIO: {scenario_name}
EVENTS:
{events_text}

Score from 0 to 100:
- 0-30: Events are disconnected, implausible, or contradictory
- 30-60: Some events are plausible but causal chains are weak
- 60-80: Events form a coherent, plausible narrative
- 80-100: Events are highly realistic, well-connected, and grounded

Respond with JSON: {{"score": <0-100>, "issues": ["list of specific issues if any"]}}"""


def check_distribution_plausibility(agent_positions: list[float]) -> tuple[float, list[str]]:
    """Check if final agent positions have a plausible distribution.

    Flags: too many extremes, all clustered, unrealistic uniformity.
    Returns (score 0-100, list of issues).
    """
    if not agent_positions:
        return 50.0, ["No agent positions to evaluate"]

    n = len(agent_positions)
    issues = []
    score = 100.0

    # Check for extreme clustering
    extremes = sum(1 for p in agent_positions if abs(p) > 0.8)
    extreme_ratio = extremes / n
    if extreme_ratio > 0.6:
        score -= 30
        issues.append(f"{extreme_ratio:.0%} of agents at extreme positions (>0.8)")

    # Check for lack of diversity
    avg = sum(agent_positions) / n
    variance = sum((p - avg) ** 2 for p in agent_positions) / n
    if variance < 0.05:
        score -= 25
        issues.append(f"Very low position variance ({variance:.3f}) — positions too similar")
    elif variance > 0.8:
        score -= 15
        issues.append(f"Very high variance ({variance:.3f}) — positions too scattered")

    # Check for realistic center mass
    center_agents = sum(1 for p in agent_positions if abs(p) < 0.3)
    center_ratio = center_agents / n
    if center_ratio < 0.1:
        score -= 15
        issues.append(f"Only {center_ratio:.0%} in the center — missing moderates")

    return max(0, score), issues


def check_drift_realism(
    agents_start_positions: dict[str, float],
    agents_end_positions: dict[str, float],
    max_total_drift: float = 0.5,
) -> tuple[float, list[str]]:
    """Check if total position drift is realistic.

    Flags agents who drifted more than max_total_drift.
    """
    issues = []
    excessive_drifters = 0

    for agent_id, start_pos in agents_start_positions.items():
        end_pos = agents_end_positions.get(agent_id)
        if end_pos is None:
            continue
        total_drift = abs(end_pos - start_pos)
        if total_drift > max_total_drift:
            excessive_drifters += 1
            issues.append(f"{agent_id}: drifted {total_drift:.2f} (>{max_total_drift})")

    n = len(agents_start_positions)
    if n == 0:
        return 50.0, ["No agents to evaluate"]

    ratio = excessive_drifters / n
    score = max(0, 100 - ratio * 200)  # 50% excessive drifters = score 0
    return score, issues


def check_alliance_consistency(
    alliances: list[tuple[str, str]],
    agent_positions: dict[str, float],
    max_distance: float = 0.8,
) -> tuple[float, list[str]]:
    """Check if alliances are consistent with position proximity."""
    if not alliances:
        return 80.0, ["No alliances to evaluate"]

    issues = []
    inconsistent = 0

    for a, b in alliances:
        pos_a = agent_positions.get(a)
        pos_b = agent_positions.get(b)
        if pos_a is None or pos_b is None:
            continue
        distance = abs(pos_a - pos_b)
        if distance > max_distance:
            inconsistent += 1
            issues.append(f"Alliance {a}↔{b}: position distance {distance:.2f}")

    n = len(alliances)
    ratio = inconsistent / n if n > 0 else 0
    score = max(0, 100 - ratio * 150)
    return score, issues


async def check_event_coherence(
    llm: BaseLLMClient,
    scenario_name: str,
    events: list[dict],
) -> tuple[float, list[str]]:
    """LLM-as-judge: evaluate coherence of the full event sequence."""
    events_text = "\n".join(
        f"Round {e.get('round', '?')} ({e.get('timeline_label', '?')}): {e.get('event', '')[:200]}"
        for e in events
    )

    prompt = EVENT_COHERENCE_PROMPT.format(
        scenario_name=scenario_name,
        events_text=events_text,
    )

    try:
        result = await llm.generate_json(
            prompt=prompt,
            temperature=0.2,
            max_output_tokens=300,
            component="realism_evaluation",
        )
        score = float(result.get("score", 50))
        issues = result.get("issues", [])
        return max(0, min(100, score)), issues
    except Exception as e:
        logger.warning(f"Event coherence check failed: {e}")
        return 50.0, [f"LLM evaluation failed: {e}"]


async def compute_realism_score(
    llm: Optional[BaseLLMClient],
    scenario_name: str,
    agents_start_positions: dict[str, float],
    agents_end_positions: dict[str, float],
    all_final_positions: list[float],
    alliances: list[tuple[str, str]],
    events: list[dict],
) -> dict:
    """Compute composite realism score (0-100) for a completed simulation.

    Checks:
    1. Distribution plausibility (25% weight)
    2. Drift realism (25% weight)
    3. Alliance consistency (20% weight)
    4. Event coherence via LLM (30% weight)
    """
    dist_score, dist_issues = check_distribution_plausibility(all_final_positions)
    drift_score, drift_issues = check_drift_realism(agents_start_positions, agents_end_positions)
    alliance_score, alliance_issues = check_alliance_consistency(
        alliances, agents_end_positions
    )

    event_score = 50.0
    event_issues = []
    if llm and events:
        event_score, event_issues = await check_event_coherence(llm, scenario_name, events)

    composite = (
        dist_score * 0.25
        + drift_score * 0.25
        + alliance_score * 0.20
        + event_score * 0.30
    )

    return {
        "realism_score": round(composite, 1),
        "distribution_plausibility": round(dist_score, 1),
        "drift_realism": round(drift_score, 1),
        "alliance_consistency": round(alliance_score, 1),
        "event_coherence": round(event_score, 1),
        "issues": {
            "distribution": dist_issues,
            "drift": drift_issues,
            "alliance": alliance_issues,
            "events": event_issues,
        },
    }
