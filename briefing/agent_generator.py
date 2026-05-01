"""Multi-step agent generation — replaces monolithic single-call approach.

Splits agent generation into 4 focused LLM calls:
  3a. Scaffold — domain, axis, channels, initial event
  3b. Elite Agents — 8-14 agents with archetype guidance
  3c. Institutional Agents — 6-10 orgs, cross-referencing elite agents
  3d. Citizen Clusters — 5-8 demographic clusters
"""

import logging
from typing import Optional

from core.llm.base_client import BaseLLMClient
from .brief_scope import BriefScope

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

{scope_section}

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
- If SCOPE CONSTRAINTS are provided above, they are NON-NEGOTIABLE. Any agent whose public role
  falls outside the scope sector / geography / tier must be excluded — no exceptions for fame.

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

{scope_section}

ELITE AGENTS ALREADY GENERATED (cross-reference these — institutions should interact with them):
{elite_summary}

{web_context}

{entity_context}

CRITICAL RULES:
- Use REAL organization names (ministries, unions, companies, NGOs, media outlets).
- Each institution should have clear relationships to at least 1-2 elite agents.
- Generate {inst_min}-{inst_max} institutional agents.
- Ensure position diversity across the axis.
- If SCOPE CONSTRAINTS are provided above, restrict organizations to the scoped sector / geography.
  Global bodies (IMF, UN, World Bank) only if the scope tier is global.

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
    scope: Optional[BriefScope] = None,
) -> dict:
    """Generate scenario config via 4 focused LLM calls instead of 1 monolithic call.

    Returns a dict compatible with the old analyze_brief output format.

    When `scope` is provided (Layer 0), its constraints are injected into the
    elite + institutional prompts so generation stays on-scope by construction
    (not just post-audited by the realism gate).
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

    # Layer-0 scope block (empty if no scope was provided — backward compat)
    scope_section = scope.prompt_block() if scope is not None else ""

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
        max_output_tokens=4000,
        component="agent_gen_scaffold",
    )

    # Guard: unwrap list if LLM returned array
    if isinstance(scaffold, list):
        scaffold = scaffold[0] if scaffold and isinstance(scaffold[0], dict) else {}

    logger.info(f"Scaffold: domain={scaffold.get('domain')}, name={scaffold.get('scenario_name')}")

    # Resolve domain plugin if not provided yet.
    # DomainRegistry.get() raises ValueError on unknown domain; we treat that
    # as "no plugin — use LLM defaults" so the pipeline stays robust when a
    # domain isn't registered (e.g. in tests or novel briefs).
    if not domain_plugin:
        try:
            from domains.domain_registry import DomainRegistry
            domain_plugin = DomainRegistry.get(scaffold.get("domain", "political"))
        except Exception as exc:
            logger.debug(f"domain plugin not resolvable: {exc}")
            domain_plugin = None
        if domain_plugin:
            guidance = domain_plugin.get_agent_generation_guidance()

    axis = scaffold.get("position_axis", {})
    neg_label = axis.get("negative_label", "Against")
    pos_label = axis.get("positive_label", "In favor")

    elite_range = guidance.get("elite_count_range", (8, 14))
    inst_range = guidance.get("institutional_count_range", (6, 10))
    cluster_range = guidance.get("cluster_count_range", (5, 8))

    # ── Step 3b: Elite Agents ──────────────────────────────────────────────
    # Try Semantic Retriever first (zero LLM cost, wave-based activation)
    elite_agents = []
    graph_used = False
    activation_plan = None
    try:
        from stakeholder_graph.db import StakeholderDB
        from core.orchestrator.retriever import SemanticRetriever
        db = StakeholderDB()
        retriever = SemanticRetriever(db)

        # Extract topic tags from scaffold for richer retrieval
        llm_topics = None
        scaffold_context = scaffold.get("scenario_context", "")
        if scaffold_context:
            from stakeholder_graph.integration import infer_topic_tags
            llm_topics = infer_topic_tags(scaffold_context, scaffold.get("domain", ""))

        # Derive primary country from scope so the retriever doesn't pull
        # all global stakeholders. Default "IT" if no scope (back-compat).
        scope_country = "IT"
        scope_is_global = False
        if scope is not None:
            scope_is_global = scope.scope_tier == "global"
            geo_codes = [g.upper() for g in (scope.geography or [])]
            if geo_codes and not scope_is_global:
                # Pick the first non-supranational country code
                for g in geo_codes:
                    if g not in ("EU", "GLOBAL", "EUROZONE", "WORLD"):
                        scope_country = g
                        break

        activation_plan = retriever.retrieve(
            brief=brief,
            country=scope_country,
            max_total=elite_range[1] + 15,  # extra for waves 2-3
            llm_topics=llm_topics,
        )

        # ── HARD GUARD: heads-of-state / global tech billionaires ─────────
        # Hard-block list applied BEFORE the scope filter, because we keep
        # finding Biden/Trump/Musk/DeSantis in IT-scoped sims. Either the
        # LLM scope detector returned scope_tier="global" (it shouldn't for
        # a bank pricing brief) or scope was None entirely. Defence-in-depth:
        # certain very-recognisable global figures are dropped UNLESS the
        # brief verbatim names them.
        _HARD_BLOCK_NAMES = {
            "Donald Trump", "Joe Biden", "Kamala Harris", "JD Vance",
            "Elon Musk", "Ron DeSantis", "Alexandria Ocasio-Cortez",
            "Bernie Sanders", "Vladimir Putin", "Xi Jinping",
            "Volodymyr Zelensky", "Mark Zuckerberg", "Jeff Bezos",
            "Bill Gates", "Tim Cook", "Sundar Pichai", "Satya Nadella",
        }
        brief_lower = (brief or "").lower()
        _hard_block_active = {
            n for n in _HARD_BLOCK_NAMES if n.lower() not in brief_lower
        }

        # Convert all wave agents to agent spec format
        all_wave_agents = []
        for wave in activation_plan.all_waves:
            for score in wave:
                stakeholder = db.get(score.stakeholder_id)
                if stakeholder:
                    # Hard-block filter (defence-in-depth)
                    if stakeholder.name in _hard_block_active:
                        logger.info(
                            f"hard_block: dropping {stakeholder.name} "
                            f"(global figure not named in brief)"
                        )
                        continue
                    # Geography filter for narrow-scope briefs: drop foreign
                    # stakeholders unless they have international reach AND
                    # the scope tier is broader than national.
                    if scope is not None and not scope_is_global:
                        s_country = (getattr(stakeholder, "country", "") or "").upper()
                        scope_geo = [g.upper() for g in (scope.geography or [])]
                        if s_country and scope_geo and s_country not in scope_geo:
                            # Allow EU figures for IT briefs and vice versa
                            ok_supranational = (
                                ("EU" in scope_geo and s_country in {"DE","FR","IT","ES","NL","BE","AT","PT","IE","FI","GR","DK","SE","PL"})
                                or ("EU" in {s_country} and "IT" in scope_geo)
                            )
                            if not ok_supranational:
                                logger.debug(
                                    f"retriever filter: dropping {stakeholder.name} "
                                    f"(country={s_country}, scope={scope_geo})"
                                )
                                continue
                    spec = stakeholder.to_agent_spec()
                    spec["_activation_tier"] = score.activation_tier
                    spec["_relevance_score"] = score.total
                    spec["_activation_reason"] = score.activation_reason
                    all_wave_agents.append(spec)

        if len(all_wave_agents) >= elite_range[0]:
            elite_agents = all_wave_agents
            graph_used = True
            n_w1 = len(activation_plan.wave_1)
            n_w2 = len(activation_plan.wave_2)
            n_w3 = len(activation_plan.wave_3)
            logger.info(
                f"Semantic retriever: {len(elite_agents)} agents "
                f"(wave1={n_w1}, wave2={n_w2}, wave3={n_w3})"
            )
            if progress_callback:
                await progress_callback("round_phase", {
                    "phase": "agent_generation",
                    "message": f"Orchestrator: {n_w1} immediate + {n_w2} secondary + {n_w3} tertiary agents",
                })
        else:
            logger.info(f"Semantic retriever: only {len(all_wave_agents)} agents, falling back to LLM")
            activation_plan = None
    except ImportError:
        logger.debug("orchestrator not available — using LLM generation")
    except Exception as e:
        logger.warning(f"Semantic retriever failed: {e}")
        activation_plan = None

    # Fall back to LLM generation if graph didn't provide enough
    if not graph_used:
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
                scope_section=scope_section,
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

        if isinstance(elite_result, list):
            elite_result = elite_result[0] if elite_result and isinstance(elite_result[0], dict) else {}
        elite_agents = elite_result.get("elite_agents", [])
        logger.info(f"Elite agents from LLM: {len(elite_agents)}")

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
            scope_section=scope_section,
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

    if isinstance(inst_result, list):
        inst_result = inst_result[0] if inst_result and isinstance(inst_result[0], dict) else {}
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

    if isinstance(cluster_result, list):
        cluster_result = cluster_result[0] if cluster_result and isinstance(cluster_result[0], dict) else {}
    clusters = cluster_result.get("citizen_clusters", [])
    logger.info(f"Citizen clusters generated: {len(clusters)}")

    # ── Final safety net: drop action-token impostors (e.g. "remove_agent")
    # that may have slipped past upstream LLM calls. ────────────────────────
    from .realism_gate import filter_invalid_agents
    elite_agents, dropped_e = filter_invalid_agents(elite_agents)
    inst_agents, dropped_i = filter_invalid_agents(inst_agents)
    if dropped_e or dropped_i:
        logger.warning(
            f"agent_generator final filter: dropped {len(dropped_e)} elite + "
            f"{len(dropped_i)} institutional agents with invalid names"
        )

    # ── Merge into unified output ──────────────────────────────────────────
    result = {**scaffold}
    result["suggested_elite_agents"] = elite_agents
    result["suggested_institutional_agents"] = inst_agents
    result["suggested_citizen_clusters"] = clusters

    # Pass activation plan through for engine integration
    if activation_plan:
        result["_activation_plan"] = activation_plan

    return result
