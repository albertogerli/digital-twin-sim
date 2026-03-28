"""Multi-step agent generation — replaces monolithic single-call approach.

Splits agent generation into 4 focused LLM calls:
  3a. Scaffold — domain, axis, channels, initial event
  3b. Elite Agents — 8-14 agents with archetype guidance
  3c. Institutional Agents — 6-10 orgs, cross-referencing elite agents
  3d. Citizen Clusters — 5-8 demographic clusters
"""

import logging
from core.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


# ── Step 3a: Scaffold ──────────────────────────────────────────────────────

SCAFFOLD_PROMPT = """You are an expert simulation designer. Analyze the user's scenario brief and
produce the structural scaffold for a digital-twin simulation.

USER'S SCENARIO BRIEF:
{brief}

{web_context}

AVAILABLE DOMAINS: {domains}

{archetype_guidance}

Respond with JSON:
{{
  "domain": "the most appropriate domain from the list above",
  "scenario_name": "short descriptive name",
  "description": "2-3 sentence description of the simulation",
  "language": "ISO 639-1 code detected from the brief (e.g. 'it', 'en', 'es')",
  "num_rounds": 9,
  "timeline_unit": "day|week|month|quarter",
  "timeline_labels": ["label for each round"],
  "position_axis": {{
    "negative_label": "what -1 means",
    "positive_label": "what +1 means",
    "neutral_label": "what 0 means"
  }},
  "initial_event": "detailed triggering event (4-5 sentences, use real-world context)",
  "scenario_context": "broader context incorporating real-world data",
  "metrics_to_track": ["metric1", "metric2"],
  "suggested_channels": [
    {{"id": "channel_id", "description": "description", "max_length": 280, "channel_type": "short_form|long_form|official"}}
  ]
}}

Keep bios/descriptions concise. Use real-world context when available."""


# ── Step 3b: Elite Agents ──────────────────────────────────────────────────

ELITE_AGENTS_PROMPT = """You are an expert simulation designer. Generate elite agents (tier 1, individual actors)
for this simulation scenario.

SCENARIO SCAFFOLD:
- Name: {scenario_name}
- Domain: {domain}
- Axis: [{neg_label}] <--> [{pos_label}]
- Context: {scenario_context}
- Initial event: {initial_event}

{web_context}

{entity_context}

{seed_context}

ARCHETYPE GUIDANCE:
Required archetypes (MUST include at least one of each): {required_archetypes}
Optional archetypes (include if relevant): {optional_archetypes}
Distribution hint: {distribution_hint}

CRITICAL RULES:
- Use REAL names of public figures, politicians, journalists, executives. NEVER invent generic names.
- If the web context or entity research provides real stakeholder names, USE THEM ALL.
- If verified stakeholders (seed data) are provided, use their exact names and positions.
- Ensure positions span the full -1 to +1 range with natural distribution.
- Generate {elite_min}-{elite_max} elite agents.
- Keep bios to 1-2 sentences to stay within output limits.

Respond with JSON:
{{
  "elite_agents": [
    {{
      "id": "unique_snake_case_id",
      "name": "Full Real Name",
      "role": "their role/title",
      "archetype": "from the archetype list above",
      "position": 0.0,
      "influence": 0.5,
      "rigidity": 0.5,
      "bio": "1-2 sentence bio based on real-world information",
      "communication_style": "how they communicate",
      "key_traits": ["trait1", "trait2"]
    }}
  ]
}}"""


# ── Step 3c: Institutional Agents ──────────────────────────────────────────

INSTITUTIONAL_AGENTS_PROMPT = """You are an expert simulation designer. Generate institutional agents (tier 2, organizations)
for this simulation scenario.

SCENARIO:
- Name: {scenario_name}
- Domain: {domain}
- Axis: [{neg_label}] <--> [{pos_label}]
- Context: {scenario_context}

ELITE AGENTS ALREADY GENERATED (cross-reference these — institutions should interact with them):
{elite_summary}

{web_context}

{entity_context}

CRITICAL RULES:
- Use REAL organization names (ministries, unions, companies, NGOs, media outlets).
- Each institution should have clear relationships to at least 1-2 elite agents.
- Generate {inst_min}-{inst_max} institutional agents.
- Ensure position diversity across the axis.

Respond with JSON:
{{
  "institutional_agents": [
    {{
      "id": "unique_snake_case_id",
      "name": "Real Organization Name",
      "role": "institutional role",
      "category": "category (government, media, business, union, ngo, academic, etc.)",
      "position": 0.0,
      "influence": 0.3,
      "rigidity": 0.5,
      "key_trait": "defining characteristic"
    }}
  ]
}}"""


# ── Step 3d: Citizen Clusters ──────────────────────────────────────────────

CITIZEN_CLUSTERS_PROMPT = """You are an expert simulation designer. Generate citizen clusters (tier 3, demographic segments)
for this simulation scenario.

SCENARIO:
- Name: {scenario_name}
- Domain: {domain}
- Axis: [{neg_label}] <--> [{pos_label}]
- Context: {scenario_context}
- Channels: {channels}

{web_context}

{seed_demographics}

CRITICAL RULES:
- Use realistic demographic descriptions, NOT invented character names.
- Cluster sizes should sum to a realistic population (e.g. 5000-50000 total depending on scenario).
- Ensure position diversity — at least 1 pro, 1 anti, 1 neutral cluster.
- Generate {cluster_min}-{cluster_max} clusters.
- If verified demographics are provided, use them as basis.

Respond with JSON:
{{
  "citizen_clusters": [
    {{
      "id": "cluster_id",
      "name": "Cluster Name",
      "description": "demographic description",
      "size": 5000,
      "position": 0.0,
      "engagement_base": 0.5,
      "info_channel": "primary channel id from the scenario",
      "demographic_attributes": {{"age_range": "", "education": "", "location": "", "income": ""}}
    }}
  ]
}}"""


async def generate_agents_multistep(
    brief: str,
    llm: BaseLLMClient,
    available_domains: list[str],
    web_context: str = "",
    seed_data_bundle=None,
    entity_context: str = "",
    domain_plugin=None,
    progress_callback=None,
) -> dict:
    """Generate scenario config via 4 focused LLM calls instead of 1 monolithic call.

    Returns a dict compatible with the old analyze_brief output format.
    """
    # Get archetype guidance from domain plugin (if known already)
    guidance = domain_plugin.get_agent_generation_guidance() if domain_plugin else {}
    guidance_text = ""
    if guidance.get("required_archetypes"):
        guidance_text = (
            f"DOMAIN ARCHETYPE GUIDANCE:\n"
            f"Required: {', '.join(guidance['required_archetypes'])}\n"
            f"Optional: {', '.join(guidance.get('optional_archetypes', []))}\n"
            f"Hint: {guidance.get('position_distribution_hint', '')}"
        )

    # Format context sections
    web_section = ""
    if web_context:
        web_section = f"REAL-WORLD CONTEXT (from web research):\n{web_context[:6000]}"

    seed_section = ""
    seed_demo_section = ""
    if seed_data_bundle:
        stakeholder_text = seed_data_bundle.format_stakeholders_for_prompt()
        if stakeholder_text:
            seed_section = f"\n{stakeholder_text}\nCRITICAL: Use verified stakeholders as PRIMARY source for elite agents.\n"
        demo_text = seed_data_bundle.format_demographics_for_prompt()
        if demo_text:
            seed_demo_section = f"\n{demo_text}\nUse verified demographics as basis for citizen clusters.\n"

    entity_section = ""
    if entity_context:
        entity_section = f"ENTITY RESEARCH (deep-dive on key stakeholders):\n{entity_context[:4000]}"

    # ── Step 3a: Scaffold ──────────────────────────────────────────────────
    if progress_callback:
        await progress_callback("round_phase", {
            "phase": "agent_generation", "message": "Building scenario scaffold..."
        })

    scaffold = await llm.generate_json(
        prompt=SCAFFOLD_PROMPT.format(
            brief=brief,
            domains=", ".join(available_domains),
            web_context=web_section,
            archetype_guidance=guidance_text,
        ),
        temperature=0.5,
        max_output_tokens=2500,
        component="agent_gen_scaffold",
    )

    logger.info(f"Scaffold: domain={scaffold.get('domain')}, name={scaffold.get('scenario_name')}")

    # Resolve domain plugin if not provided yet
    if not domain_plugin:
        from domains.domain_registry import DomainRegistry
        domain_plugin = DomainRegistry.get(scaffold.get("domain", "political"))
        if domain_plugin:
            guidance = domain_plugin.get_agent_generation_guidance()

    axis = scaffold.get("position_axis", {})
    neg_label = axis.get("negative_label", "Against")
    pos_label = axis.get("positive_label", "In favor")

    elite_range = guidance.get("elite_count_range", (8, 14))
    inst_range = guidance.get("institutional_count_range", (6, 10))
    cluster_range = guidance.get("cluster_count_range", (5, 8))

    # ── Step 3b: Elite Agents ──────────────────────────────────────────────
    if progress_callback:
        await progress_callback("round_phase", {
            "phase": "agent_generation", "message": "Generating elite agents..."
        })

    elite_result = await llm.generate_json(
        prompt=ELITE_AGENTS_PROMPT.format(
            scenario_name=scaffold.get("scenario_name", ""),
            domain=scaffold.get("domain", ""),
            neg_label=neg_label,
            pos_label=pos_label,
            scenario_context=scaffold.get("scenario_context", "")[:2000],
            initial_event=scaffold.get("initial_event", "")[:500],
            web_context=web_section[:4000],
            entity_context=entity_section,
            seed_context=seed_section,
            required_archetypes=", ".join(guidance.get("required_archetypes", [])),
            optional_archetypes=", ".join(guidance.get("optional_archetypes", [])),
            distribution_hint=guidance.get("position_distribution_hint", ""),
            elite_min=elite_range[0],
            elite_max=elite_range[1],
        ),
        temperature=0.5,
        max_output_tokens=6000,
        component="agent_gen_elite",
    )

    elite_agents = elite_result.get("elite_agents", [])
    logger.info(f"Elite agents generated: {len(elite_agents)}")

    # Build summary for institutional cross-referencing
    elite_summary = "\n".join(
        f"- {a['name']} ({a.get('role', '')}) — position: {a.get('position', 0):+.2f}, archetype: {a.get('archetype', '')}"
        for a in elite_agents
    )

    # ── Step 3c: Institutional Agents ──────────────────────────────────────
    if progress_callback:
        await progress_callback("round_phase", {
            "phase": "agent_generation", "message": "Generating institutional agents..."
        })

    inst_result = await llm.generate_json(
        prompt=INSTITUTIONAL_AGENTS_PROMPT.format(
            scenario_name=scaffold.get("scenario_name", ""),
            domain=scaffold.get("domain", ""),
            neg_label=neg_label,
            pos_label=pos_label,
            scenario_context=scaffold.get("scenario_context", "")[:2000],
            elite_summary=elite_summary,
            web_context=web_section[:3000],
            entity_context=entity_section[:2000],
            inst_min=inst_range[0],
            inst_max=inst_range[1],
        ),
        temperature=0.5,
        max_output_tokens=3000,
        component="agent_gen_institutional",
    )

    inst_agents = inst_result.get("institutional_agents", [])
    logger.info(f"Institutional agents generated: {len(inst_agents)}")

    # ── Step 3d: Citizen Clusters ──────────────────────────────────────────
    if progress_callback:
        await progress_callback("round_phase", {
            "phase": "agent_generation", "message": "Generating citizen clusters..."
        })

    channels_list = ", ".join(
        ch.get("id", "") for ch in scaffold.get("suggested_channels", [])
    )

    cluster_result = await llm.generate_json(
        prompt=CITIZEN_CLUSTERS_PROMPT.format(
            scenario_name=scaffold.get("scenario_name", ""),
            domain=scaffold.get("domain", ""),
            neg_label=neg_label,
            pos_label=pos_label,
            scenario_context=scaffold.get("scenario_context", "")[:2000],
            channels=channels_list,
            web_context=web_section[:2000],
            seed_demographics=seed_demo_section,
            cluster_min=cluster_range[0],
            cluster_max=cluster_range[1],
        ),
        temperature=0.5,
        max_output_tokens=2500,
        component="agent_gen_clusters",
    )

    clusters = cluster_result.get("citizen_clusters", [])
    logger.info(f"Citizen clusters generated: {len(clusters)}")

    # ── Merge into unified output ──────────────────────────────────────────
    result = {**scaffold}
    result["suggested_elite_agents"] = elite_agents
    result["suggested_institutional_agents"] = inst_agents
    result["suggested_citizen_clusters"] = clusters

    return result
