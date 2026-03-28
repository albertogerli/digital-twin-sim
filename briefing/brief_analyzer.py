"""Analyzes a free-text scenario brief and produces structured scenario config."""

import logging
from core.llm.base_client import BaseLLMClient
from core.llm.json_parser import parse_json_response

logger = logging.getLogger(__name__)

BRIEF_ANALYSIS_PROMPT = """You are an expert simulation designer. A user wants to run a digital twin simulation.
Analyze their scenario brief and produce a structured configuration.

USER'S SCENARIO BRIEF:
{brief}

{web_context}

AVAILABLE DOMAINS: {domains}

Analyze the brief and respond with JSON:
{{
  "domain": "the most appropriate domain from the list above",
  "scenario_name": "short descriptive name for the scenario",
  "description": "2-3 sentence description of what will be simulated",
  "language": "detect the language of the user's brief and use its ISO 639-1 code (e.g. 'it' for Italian, 'en' for English, 'es' for Spanish)",
  "num_rounds": "integer, number of simulation rounds (typically 6-12)",
  "timeline_unit": "day|week|month|quarter",
  "timeline_labels": ["label for each round, e.g. 'Month 1', 'Week 2'"],
  "position_axis": {{
    "negative_label": "what -1 means in this context",
    "positive_label": "what +1 means in this context",
    "neutral_label": "what 0 means"
  }},
  "initial_event": "detailed description of the triggering event (4-5 sentences). Use real-world context if available.",
  "scenario_context": "broader context for the simulation, incorporating real-world data and stakeholder positions",
  "metrics_to_track": ["list of relevant metrics to measure"],
  "suggested_channels": [
    {{"id": "channel_id", "description": "what this channel represents", "max_length": 280, "channel_type": "short_form|long_form|official"}}
  ],
  "suggested_elite_agents": [
    {{
      "id": "unique_id",
      "name": "Full Name or Title (use REAL names of relevant public figures when appropriate)",
      "role": "their role/title",
      "archetype": "category (politician, business_leader, journalist, influencer, etc.)",
      "position": "float -1 to +1, initial stance based on their known real-world positions",
      "influence": "float 0-1",
      "rigidity": "float 0-1",
      "bio": "2-3 sentence bio based on real-world information",
      "communication_style": "how they communicate",
      "key_traits": ["trait1", "trait2"]
    }}
  ],
  "suggested_institutional_agents": [
    {{
      "id": "unique_id",
      "name": "Organization Name (use REAL organization names)",
      "role": "their institutional role",
      "category": "category",
      "position": "float -1 to +1",
      "influence": "float 0-1",
      "rigidity": "float 0-1",
      "key_trait": "defining characteristic"
    }}
  ],
  "suggested_citizen_clusters": [
    {{
      "id": "cluster_id",
      "name": "Cluster Name",
      "description": "demographic description",
      "size": "population count",
      "position": "float -1 to +1, initial stance",
      "engagement_base": "float 0-1",
      "info_channel": "primary information channel id",
      "demographic_attributes": {{"age_range": "", "education": "", "location": "", "income": ""}}
    }}
  ]
}}

Generate 8-12 elite agents, 6-10 institutional agents, and 5-8 citizen clusters.

CRITICAL RULES FOR AGENT GENERATION:
- You MUST use REAL names of public figures, politicians, journalists, executives, and organizations.
- NEVER invent generic names like "Marco Rossi" or "Elena Bianchi" — every elite and institutional agent must be a real, identifiable person or organization found in the web context or publicly known.
- If the web context provides real stakeholder names, USE THEM ALL as agents.
- For citizen clusters, use realistic demographic descriptions (not invented names).
- Ensure positions span the full -1 to +1 range with natural distribution.
- Keep bios and descriptions concise (1 sentence each) to stay within output limits.
- If you don't know real names for a niche topic, generate fewer elite agents rather than inventing fake ones."""


async def analyze_brief(
    brief: str,
    llm: BaseLLMClient,
    available_domains: list[str],
    web_context: str = "",
    seed_data_bundle=None,
    progress_callback=None,
) -> dict:
    """Analyze a scenario brief and return structured config.

    If seed_data_bundle is provided, verified stakeholders and demographics
    are injected into the prompt so the LLM grounds its output in real data.
    """
    # Format web context section
    context_section = ""
    if web_context:
        context_section = f"""REAL-WORLD CONTEXT (from web research):
{web_context}

Use this real-world context to make agents, events, and positions more realistic and grounded in reality.
"""

    # Inject seed data if available
    seed_section = ""
    if seed_data_bundle:
        stakeholder_text = seed_data_bundle.format_stakeholders_for_prompt()
        demographic_text = seed_data_bundle.format_demographics_for_prompt()
        if stakeholder_text:
            seed_section += f"\n\n{stakeholder_text}\n"
        if demographic_text:
            seed_section += f"\n\n{demographic_text}\n"
        if seed_data_bundle.context_text:
            seed_section += f"\n\nGROUNDING CONTEXT:\n{seed_data_bundle.context_text[:2000]}\n"
        seed_section += "\nCRITICAL: Use the verified stakeholders above as the PRIMARY source for elite agents. Use their exact names, positions, and quotes. The LLM should fill gaps (bios, communication details) but NOT invent alternative stakeholders when verified ones exist.\n"

    prompt = BRIEF_ANALYSIS_PROMPT.format(
        brief=brief,
        domains=", ".join(available_domains),
        web_context=context_section + seed_section,
    )

    result = await llm.generate_json(
        prompt=prompt,
        temperature=0.5,
        max_output_tokens=16000,
        component="brief_analysis",
    )

    logger.info(f"Brief analyzed: domain={result.get('domain')}, "
                f"name={result.get('scenario_name')}")
    return result
