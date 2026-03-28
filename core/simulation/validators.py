"""Output validation and constraint enforcement for simulation realism."""

import logging
from typing import Optional

from ..llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

# Position delta caps per agent tier
ELITE_DELTA_CAP = 0.15
CLUSTER_DELTA_CAP = 0.10
INSTITUTIONAL_DELTA_CAP = 0.12

# Shock magnitude bounds
SHOCK_MIN = 0.1
SHOCK_MAX = 0.6

PLAUSIBILITY_PROMPT = """Rate the plausibility of this event in the given scenario context.

SCENARIO: {scenario_context}

EVENT: {event_text}

Score from 1 (completely implausible/invented) to 10 (highly plausible/grounded in reality).
Consider: Does it reference real institutions? Are the causal chains logical? Is the scale realistic?

Respond with JSON only: {{"score": <1-10>, "reason": "one sentence"}}"""


def clamp_position_delta(
    old_position: float,
    new_position: float,
    delta_cap: float,
) -> float:
    """Clamp position change to max delta, preserving direction."""
    delta = new_position - old_position
    if abs(delta) > delta_cap:
        clamped_delta = delta_cap if delta > 0 else -delta_cap
        new_position = old_position + clamped_delta
    return max(-1.0, min(1.0, new_position))


def clamp_shock_magnitude(magnitude: float) -> float:
    """Clamp shock magnitude to realistic bounds."""
    return max(SHOCK_MIN, min(SHOCK_MAX, magnitude))


def validate_agent_references(
    references: list[str],
    valid_agent_ids: set[str],
) -> list[str]:
    """Filter alliance/target lists to only existing agent IDs."""
    return [ref for ref in references if ref in valid_agent_ids]


def normalize_sentiment_distribution(dist: dict) -> dict:
    """Force sentiment distribution values to sum to 100."""
    if not dist:
        return dist
    total = sum(dist.values())
    if total == 0:
        n = len(dist)
        return {k: round(100 / n) for k in dist}
    factor = 100.0 / total
    normalized = {k: round(v * factor) for k, v in dist.items()}
    # Fix rounding error
    diff = 100 - sum(normalized.values())
    if diff != 0 and normalized:
        first_key = next(iter(normalized))
        normalized[first_key] += diff
    return normalized


async def check_event_plausibility(
    llm: BaseLLMClient,
    event_text: str,
    scenario_context: str,
    min_score: int = 4,
) -> tuple[bool, int, str]:
    """Lightweight LLM-as-judge plausibility check for generated events.

    Returns (is_plausible, score, reason).
    Cost: ~$0.001 per check.
    """
    prompt = PLAUSIBILITY_PROMPT.format(
        scenario_context=scenario_context[:500],
        event_text=event_text[:500],
    )
    try:
        result = await llm.generate_json(
            prompt=prompt,
            temperature=0.1,
            max_output_tokens=200,
            component="plausibility_check",
        )
        score = int(result.get("score", 5))
        reason = result.get("reason", "")
        is_plausible = score >= min_score
        if not is_plausible:
            logger.warning(
                f"Event failed plausibility check (score={score}): {reason}"
            )
        return is_plausible, score, reason
    except Exception as e:
        logger.warning(f"Plausibility check failed: {e}")
        return True, 5, "Check failed, assuming plausible"
