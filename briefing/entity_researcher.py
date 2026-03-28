"""Entity deep-dive — extracts named entities from web research and does targeted lookups.

Phase 2: After initial web research, we:
1. Extract named entities from the synthesis (1 LLM call)
2. Do targeted DuckDuckGo searches per entity (free, max 5)
3. Synthesize into structured stakeholder profiles (1 LLM call)
"""

import asyncio
import logging
from typing import Optional

from core.llm.base_client import BaseLLMClient
from .web_research import _execute_searches

logger = logging.getLogger(__name__)


ENTITY_EXTRACTION_PROMPT = """Extract the most important named entities from this research context.
Focus on people, organizations, and institutions that are key stakeholders in the scenario.

SCENARIO BRIEF: {brief}

RESEARCH CONTEXT:
{web_context}

Respond with JSON:
{{
  "entities": [
    {{
      "name": "Full Name or Organization Name",
      "type": "person|organization|institution",
      "role_hint": "brief description of their role/relevance",
      "search_query": "best search query to learn more about their position on this topic"
    }}
  ]
}}

Return at most 8 entities, prioritized by relevance to the scenario. Include ONLY real, identifiable entities."""


ENTITY_SYNTHESIS_PROMPT = """You are a research analyst. Based on targeted research about specific stakeholders,
produce structured profiles for use in a simulation.

SCENARIO BRIEF: {brief}

ENTITY RESEARCH RESULTS:
{entity_results}

For each entity found, produce a structured stakeholder profile.
Respond with JSON:
{{
  "stakeholders": [
    {{
      "name": "Full Name",
      "role": "Their role/title",
      "type": "person|organization",
      "estimated_position": 0.0,
      "position_evidence": "Why this position — cite specific quotes, votes, statements",
      "key_quote": "A real or closely paraphrased quote if available",
      "influence_estimate": 0.5,
      "communication_style": "How they communicate publicly"
    }}
  ]
}}

RULES:
- estimated_position is a float from -1.0 (strongly against) to +1.0 (strongly in favor)
- Only include entities you have actual evidence for — do NOT fabricate profiles
- Cite specific evidence for position estimates"""


async def research_entities(
    brief: str,
    web_context: str,
    llm: BaseLLMClient,
    max_entities: int = 5,
    progress_callback=None,
) -> str:
    """Extract entities from web context and do targeted deep-dive research.

    Returns a formatted string of structured stakeholder profiles
    suitable for injection into agent generation prompts.
    """
    if not web_context:
        return ""

    try:
        # Step 1: Extract entities
        if progress_callback:
            await progress_callback("round_phase", {
                "phase": "entity_research", "message": "Extracting key entities..."
            })

        extraction = await llm.generate_json(
            prompt=ENTITY_EXTRACTION_PROMPT.format(
                brief=brief,
                web_context=web_context[:6000],
            ),
            temperature=0.3,
            max_output_tokens=1000,
            component="entity_extraction",
        )

        entities = extraction.get("entities", [])[:max_entities]
        if not entities:
            logger.info("Entity extraction: no entities found")
            return ""

        logger.info(f"Entity extraction: {len(entities)} entities found")

        # Step 2: Targeted searches per entity
        if progress_callback:
            await progress_callback("round_phase", {
                "phase": "entity_research",
                "message": f"Deep-dive research on {len(entities)} entities..."
            })

        search_queries = [e.get("search_query", e["name"]) for e in entities]
        search_results = await _execute_searches(search_queries)

        if not search_results.strip():
            logger.warning("Entity deep-dive: no search results")
            # Still useful to return extraction info
            return _format_extraction_only(entities)

        # Step 3: Synthesize into structured profiles
        if progress_callback:
            await progress_callback("round_phase", {
                "phase": "entity_research", "message": "Synthesizing entity profiles..."
            })

        synthesis = await llm.generate_json(
            prompt=ENTITY_SYNTHESIS_PROMPT.format(
                brief=brief,
                entity_results=search_results[:8000],
            ),
            temperature=0.3,
            max_output_tokens=2000,
            component="entity_synthesis",
        )

        stakeholders = synthesis.get("stakeholders", [])
        if not stakeholders:
            return _format_extraction_only(entities)

        # Format as context string for agent generation
        return _format_stakeholder_profiles(stakeholders)

    except Exception as e:
        logger.warning(f"Entity research failed (non-fatal): {e}")
        return ""


def _format_extraction_only(entities: list[dict]) -> str:
    """Format basic entity info when deep-dive search fails."""
    lines = ["IDENTIFIED KEY ENTITIES (from web research):"]
    for e in entities:
        lines.append(f"- {e['name']} ({e.get('type', '?')}): {e.get('role_hint', '')}")
    return "\n".join(lines)


def _format_stakeholder_profiles(stakeholders: list[dict]) -> str:
    """Format synthesized stakeholder profiles for injection into agent generation."""
    lines = ["RESEARCHED STAKEHOLDER PROFILES (use these to ground agent generation):"]
    for s in stakeholders:
        pos = s.get("estimated_position", 0)
        pos_label = "PRO" if pos > 0.2 else ("AGAINST" if pos < -0.2 else "NEUTRAL")
        lines.append(
            f"\n- {s['name']} ({s.get('role', '')})"
            f"\n  Position: {pos:+.2f} ({pos_label}) — {s.get('position_evidence', 'no evidence')}"
        )
        if s.get("key_quote"):
            lines.append(f'  Quote: "{s["key_quote"]}"')
        if s.get("communication_style"):
            lines.append(f"  Style: {s['communication_style']}")
    return "\n".join(lines)
