"""Automatic seed data generation from entity research.

Phase 5: Converts entity research results into a SeedDataBundle so that
every scenario gets grounding data, not just those with manual seed files.
"""

import logging
from typing import Optional

from core.llm.base_client import BaseLLMClient
from seed_data.schema import SeedDataBundle, VerifiedStakeholder, VerifiedDemographic

logger = logging.getLogger(__name__)


def build_seed_from_entity_research(
    entity_context: str,
    analysis: dict,
) -> Optional[SeedDataBundle]:
    """Convert entity research + generated analysis into a SeedDataBundle.

    This provides grounding for scenarios that don't have manual seed data.
    Uses the generated elite agents and clusters as verified-ish data points,
    enriched by entity research context.

    Args:
        entity_context: Formatted entity research string from entity_researcher
        analysis: The merged analysis dict from agent_generator

    Returns:
        SeedDataBundle or None if insufficient data
    """
    if not analysis:
        return None

    elite = analysis.get("suggested_elite_agents", [])
    clusters = analysis.get("suggested_citizen_clusters", [])

    if not elite and not clusters:
        return None

    # Build stakeholders from elite agents
    stakeholders = []
    for a in elite:
        # Determine confidence based on source
        confidence = "medium"  # Default for LLM-generated
        if entity_context and a.get("name", "").lower() in entity_context.lower():
            confidence = "high"  # Confirmed by entity research

        stakeholders.append(VerifiedStakeholder(
            name=a.get("name", ""),
            role=a.get("role", ""),
            known_position=float(a.get("position", 0)),
            position_source="entity_research" if confidence == "high" else "llm_estimate",
            bio_verified=a.get("bio", ""),
            key_quotes=[],  # Could be enriched from entity research
            archetype=a.get("archetype", ""),
            communication_style=a.get("communication_style", ""),
            influence=float(a.get("influence", 0.5)),
            rigidity=float(a.get("rigidity", 0.5)),
            confidence=confidence,
        ))

    # Build demographics from citizen clusters
    demographics = []
    for c in clusters:
        demographics.append(VerifiedDemographic(
            name=c.get("name", ""),
            description=c.get("description", ""),
            population_share=_estimate_share(c.get("size", 1000), clusters),
            known_position=float(c.get("position", 0)),
            position_source="llm_estimate",
            key_concerns=[],
            info_channel=c.get("info_channel", ""),
            demographic_attributes=c.get("demographic_attributes", {}),
            confidence="medium",
        ))

    bundle = SeedDataBundle(
        context_text=analysis.get("scenario_context", ""),
        stakeholders=stakeholders,
        demographics=demographics,
        historical_text="",
        known_events=[],
    )

    logger.info(
        f"Auto seed data: {len(stakeholders)} stakeholders, "
        f"{len(demographics)} demographics"
    )
    return bundle


def _estimate_share(size: int, clusters: list[dict]) -> float:
    """Estimate population share of a cluster relative to total."""
    total = sum(c.get("size", 1000) for c in clusters)
    if total == 0:
        return 0.1
    return round(size / total, 3)
