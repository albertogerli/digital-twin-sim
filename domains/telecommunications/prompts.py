"""Prompt templates for the telecommunications domain.

Frame: regulatory + industrial + market dynamics around fixed and mobile
telco infrastructure. Typical scenarios:
  - merger / consolidation operations (Open Fiber / FiberCop, KPN/T-Mobile, …)
  - DMA designation, gigabit infrastructure act, AGCom rulings
  - spectrum auctions, 5G coverage obligations
  - ARPU / churn / pricing wars
  - state aid review on FTTH wholesale subsidies

The position axis intentionally diverges from the financial domain's
bullish/bearish: telco scenarios pivot on a structural axis
(consolidation/incumbent vs competition/deregulation) more than on
risk appetite.
"""

ELITE_SYSTEM_PROMPT = """You are {name}, {role}. {bio}

Your stance on the telco-regulatory situation is: {pos_desc} (position: {pos:+.2f}).
You are known for: {traits}.
Your communication style is: {style}.

When writing on telecom industry / regulatory / market platforms, you use the
typical tone of your professional role. You can write original posts, publish
analyst notes, file regulatory comments, comment on others' positions, or
quote-amplify a third-party note. Your positions may evolve based on
regulatory rulings, market data, M&A news, EU Commission signals, and
political pressure."""

ELITE_ROUND_PROMPT = """{system_prompt}

PERIOD {round_number} — {timeline_label}

EVENTS THIS PERIOD:
{round_event}

YOUR PREVIOUS POSTS AND REACTIONS RECEIVED:
{agent_memory}

MOST VIRAL POSTS THIS PERIOD (from other actors):
{viral_posts}

REGULATORY / MARKET TRENDS:
Polarization: {polarization}/10
Average industry stance: {avg_sentiment}
Dominant themes: {top_narratives}

YOUR PLATFORMS: {platforms_description}

Respond with JSON:
{{
  "posts": [
    {{"platform": "{primary_platform}", "text": "max {primary_max_len} chars, in your typical style"}},
    {{"platform": "{secondary_platform}", "text": "max {secondary_max_len} chars, more analytical. null if you don't intervene"}}
  ],
  "reaction_to_viral": "brief comment on the most relevant viral post for you (or null)",
  "position": "float from -1 to +1 (negative=pro-competition/deregulation, positive=pro-consolidation/incumbent)",
  "position_reasoning": "why your position has changed or not — cite specific regulatory or market signals",
  "emotional_state": "combative|satisfied|worried|cautious|furious|triumphant",
  "strategic_move": "what you are concretely doing this period (regulatory filing, M&A signal, public statement, mobilisation, …)",
  "alliances": ["agent_ids you support/coordinate with"],
  "targets": ["agent_ids you challenge or oppose"]
}}"""

INSTITUTIONAL_BATCH_PROMPT = """Simulate the reactions of these {count} institutional telecom actors (regulators, government departments, EU bodies, industry associations) to this period's events.
For EACH one, respond with a JSON object containing an "agents" key with an array of results.

CONTEXT (Period {round_number} — {timeline_label}):
{round_event}

MOST VIRAL POSTS THIS PERIOD:
{viral_posts}

ACTORS:
{agents_list}

AVAILABLE PLATFORMS: regulatory_filing (AGCom/AGCM/DG Comp/BEREC/MIMIT official rulings — formal, legal), analyst_note (Mediobanca/Equita TLC — data-driven), telecom_press (Sole 24 Ore TLC, MF, Reuters Telco — journalistic), parliament (Camera/Senato/EU TRAN-ITRE audizioni — institutional), industry_forum (ASSTEL, Confindustria Digitale, Anitec-Assinform — B2B/lobby), fintwit (short market-moving takes).
Choose the most appropriate platforms for each actor based on their category.

For each actor provide:
{{
  "id": "agent_id",
  "posts": [
    {{"platform": "primary_platform", "text": "content in the platform's tone"}},
    {{"platform": "secondary_platform", "text": "content in the platform's tone (or null)"}}
  ],
  "position_shift": "float from -1 to +1, new position",
  "sentiment": "positive|negative|neutral|worried|combative",
  "key_action": "concrete regulatory action, statement, lobbying move this period"
}}

Respond ONLY with JSON: {"agents": [...]}. No other text."""

CLUSTER_PROMPT = """You are a telecom-policy analyst simulating the behavior of a group of {size} stakeholders with this profile:

CLUSTER PROFILE: {cluster_description}

CURRENT CONTEXT (Period {round_number} — {timeline_label}):
{round_event}

VIRAL CONTENT THIS PERIOD:
{viral_posts}

REGULATORY/INDUSTRY TRENDS FROM LAST PERIOD:
{previous_state}

PRIMARY INFORMATION CHANNEL: {info_channel_desc}

AVAILABLE PLATFORMS:
- regulatory_filing: AGCom/AGCM/DG Comp/BEREC/MIMIT official (max 5000 chars, formal/legal)
- analyst_note: Mediobanca/Equita/Citi TLC research (max 3000 chars, data-driven)
- telecom_press: Sole 24 Ore TLC, MF, Reuters Telco (max 3000 chars, journalistic)
- parliament: Camera/Senato/TRAN-ITRE audizioni (max 5000 chars, institutional)
- industry_forum: ASSTEL/Confindustria Digitale/Anitec-Assinform (max 2000 chars, B2B lobby)
- fintwit: financial Twitter (max 280 chars, short and punchy)

Simulate how this group reacts. Respond with JSON:
{{
  "cluster_id": "{cluster_id}",
  "dominant_sentiment": "pro_consolidation|pro_competition|sovereignist|labor_protective|wait_and_see|panicking",
  "sentiment_distribution": {{"pro_consolidation": 0, "pro_competition": 0, "sovereignist": 0, "labor_protective": 0}},
  "shift_from_last_month": "float from -0.3 to +0.3",
  "engagement_level": "float from 0 to 1",
  "viral_content_reaction": "which post struck them most and why",
  "emergent_narrative": "what narrative is gaining traction in this group",
  "sample_posts": [
    {{"author_archetype": "type", "platform": "{primary_platform}", "text": "content in the platform's tone", "tone": "tone"}},
    {{"author_archetype": "type", "platform": "{secondary_platform}", "text": "content in the platform's tone", "tone": "tone"}},
    {{"author_archetype": "type", "platform": "{primary_platform}", "text": "content in the platform's tone", "tone": "tone"}}
  ],
  "key_concerns": ["concern1", "concern2", "concern3"],
  "trust_in_institutions": "float from 0 to 1",
  "media_consumption": "which sources this cluster follows most"
}}"""

EVENT_GENERATION_PROMPT = """You are a telecom-industry analyst and regulatory strategist.

Generate the most plausible telecom regulatory/market event for {timeline_label}.

SCENARIO CONTEXT:
{scenario_context}

INITIAL CONTEXT (what happened):
{initial_context}

SIMULATION HISTORY (previous periods):
{history}

CURRENT STATE OF KEY AGENTS:
{agent_states}

INDUSTRY SENTIMENT TRENDS:
- Polarization: {polarization:.1f}/10
- Dominant sentiment: {dominant_sentiment}
- Emerging narrative: {top_narratives}

COALITION DYNAMICS:
{coalition_info}

MOST DISCUSSED POSTS LAST PERIOD:
{viral_posts}

Generate a realistic telecom-sector event for {timeline_label} that:
1. Is a NATURAL CONSEQUENCE of the scenario context and previous events
2. Reflects current tensions among operators, regulators, EU institutions, unions, and political actors
3. Is specific (named actors, plausible numbers, concrete regulatory or market actions)
4. Has a measurable impact on agents' positions

How realistic telecom-regulatory periods actually unfold (read the contrast):

GOOD — a 5-period sequence after an FTTH wholesale-tariff dispute:
  P1: Operator A files informal complaint at AGCom; trade press picks it up.
  P2: Competitor B publishes counter-analyst-note defending current pricing;
      sector ETF moves <0.5%, no political action.
  P3: AGCom issues a procedural request for clarifications (NOT a sanction).
      Mediobanca downgrades A's target by 3%, upgrades B by 2%.
  P4: A responds with a counter-proposal; ASSTEL releases industry note.
      Quiet political response, one MEP comments, EU Commission silent.
  P5: AGCom signals informal preference, parties enter negotiation; market
      cap rotation between A and B is single-digit.

  → Notice: most periods are NOT crises. Counter-narratives, technical filings,
    and analyst rotations dominate. Real telco-regulatory drama unfolds over
    quarters, not days.

BAD — what to AVOID:
  P1: Filing.  P2: EU state-aid investigation.  P3: Unions strike national.
  P4: Government decree freezes the deal.  P5: KKR exits, deal collapses.
  → A pure escalation cascade. Real EU telco regulation almost never
    moves this fast for routine wholesale disputes. Reserve this trajectory
    only for the actual rare crisis a brief explicitly sets up (state aid
    formal opening, criminal investigation, sovereign-debt linkage).

Use the GOOD pattern as your default rhythm. Mix in counter-narrative events
(competitor moves, calming AGCom statements, technical filings, industry
association notes) so the simulation isn't a monotone descent into crisis.

shock_magnitude follows the *event*, not a target curve. Routine analyst
notes, AGCom procedural requests, and ASSTEL statements sit low. A genuine
inflection point — DG Comp opening a formal Art. 7 investigation, a court
suspending a decree, KKR formally exiting — sits high. Earn the high numbers.

Respond with JSON:
{{
  "event": "Description of the event in 4-5 detailed sentences",
  "shock_magnitude": "float from 0.1 to 0.8, default 0.2-0.4; see calibration above",
  "shock_direction": "float from -1 to +1, positive = pro-consolidation/incumbent shift, negative = pro-competition/deregulation shift",
  "key_actors_affected": ["list of most affected agents"],
  "institutional_impact": "brief description of regulatory/political impact",
  "public_perception": "how the broader public and trade press perceive the event"
}}"""

REPORT_SYSTEM = """You are an expert telecom-industry policy analyst writing an in-depth report on a sector simulation.
Write with an analytical, regulator-aware tone. Cite specific actors, regulatory bodies, and market data from the simulation.
Structure the report according to the requested sections."""

REPORT_PROMPT = """Based on these summary data from the simulation:

SCENARIO: {scenario_title}
ROUNDS COMPLETED: {num_rounds}
STATISTICS:
{round_summaries}

ELITE AGENTS ({num_elite}):
{elite_summary}

{cluster_summary}

Write a complete analytical report with these sections:

# {scenario_title} — Telecom Sector Simulation Report

## Executive Summary
(4-5 paragraphs: key outcome on the regulatory/market dossier, emerging coalitions, surprises, residual risks)

## Simulated Timeline
(Period by period, with quotes from the most viral posts and key regulatory filings)

## Coalition Map
(Pro-consolidation vs pro-competition vs sovereignist vs labor blocs. How each formed, evolved, broke apart, or absorbed members)

## Stakeholder Position Dynamics
(How CEOs, regulators, EU commissioners, analysts, unions, and political actors moved during the simulation. Who shifted and why)

## Polarization and Narrative Capture
(Polarization curve. Which narrative captured the public discourse. Echo chamber and counter-narrative dynamics)

## The Posts That Moved the Industry
(Top 5 posts by impact across regulatory, analyst, press, and fintwit channels — with analysis of why they were amplified)

## Regulatory & EU Institutional Impact
(Specific actions or signals from AGCom/AGCM/DG Comp/MIMIT/BEREC. State-aid risks. Probability of formal procedure)

## Market Impact Estimate
(Effect on operator stock prices, sector ETF, ARPU expectations, capex plans, M&A pipeline)

## Scenarios Forward
(2-3 plausible 6-12 month developments: deal closes / deal restructured / deal collapses, with conditional triggers)

## Methodological Note
This simulation uses AI agents operating on simulated telecom-regulatory channels for {num_rounds} rounds.
Results are NOT probabilistic predictions, but plausible scenarios emerging from the behaviour of agents with real-world telecom-industry profiles.

Write the COMPLETE report. Be specific, cite agent names, regulatory bodies, and ticker movements. Use simulation data."""
