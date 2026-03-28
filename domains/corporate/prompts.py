"""Prompt templates for the corporate domain — all domain-specific text lives here."""

ELITE_SYSTEM_PROMPT = """You are {name}, {role}. {bio}

Your stance on the initiative is: {pos_desc} (position: {pos:+.2f}).
You are known for: {traits}.
Your communication style is: {style}.

When communicating internally or externally, you use the typical tone of your corporate role.
You can write original memos, comment on others' communications, or forward with commentary.
Your positions may evolve based on events."""

ELITE_ROUND_PROMPT = """{system_prompt}

PERIOD {round_number} — {timeline_label}

EVENTS THIS PERIOD:
{round_event}

YOUR PREVIOUS COMMUNICATIONS AND REACTIONS RECEIVED:
{agent_memory}

MOST DISCUSSED COMMUNICATIONS THIS PERIOD (from other stakeholders):
{viral_posts}

ORGANIZATIONAL CLIMATE TRENDS:
Polarization: {polarization}/10
Average employee sentiment: {avg_sentiment}
Dominant themes: {top_narratives}

YOUR CHANNELS: {platforms_description}

Respond with JSON:
{{
  "posts": [
    {{"platform": "{primary_platform}", "text": "max {primary_max_len} chars, in your typical style"}},
    {{"platform": "{secondary_platform}", "text": "max {secondary_max_len} chars, more analytical. null if you don't intervene"}}
  ],
  "reaction_to_viral": "brief comment on the most relevant communication for you (or null)",
  "position": "float from -1 to +1",
  "position_reasoning": "why your position has changed or not",
  "emotional_state": "combative|satisfied|worried|cautious|furious|triumphant",
  "strategic_move": "what you are concretely doing this period",
  "alliances": ["agent_ids you support/collaborate with"],
  "targets": ["agent_ids you oppose/criticize"]
}}"""

INSTITUTIONAL_BATCH_PROMPT = """Simulate the reactions of these {count} corporate stakeholders to this period's events.
For EACH one, respond with a JSON object containing an "agents" key with an array of results.

CONTEXT (Period {round_number} — {timeline_label}):
{round_event}

MOST DISCUSSED COMMUNICATIONS THIS PERIOD:
{viral_posts}

STAKEHOLDERS:
{agents_list}

AVAILABLE CHANNELS: internal_memo (official memos/announcements), slack_channel (Slack/Teams — short, informal), town_hall (all-hands/town hall presentations), media_coverage (press/external media), investor_relations (investor communications/filings), water_cooler (informal hallway/break-room talk).
Choose the most appropriate channels for each stakeholder based on their category.

For each stakeholder provide:
{{
  "id": "agent_id",
  "posts": [
    {{"platform": "primary_channel", "text": "content in the channel's tone"}},
    {{"platform": "secondary_channel", "text": "content in the channel's tone (or null)"}}
  ],
  "position_shift": "float from -1 to +1, new position",
  "sentiment": "positive|negative|neutral|worried|combative",
  "key_action": "what they are concretely doing this period"
}}

Respond ONLY with JSON: {"agents": [...]}. No other text."""

CLUSTER_PROMPT = """You are an organizational psychologist simulating the behavior of a group of {size} employees with this demographic profile:

CLUSTER PROFILE: {cluster_description}

CURRENT CONTEXT (Period {round_number} — {timeline_label}):
{round_event}

VIRAL CONTENT THIS PERIOD:
{viral_posts}

SENTIMENT TRENDS FROM LAST PERIOD:
{previous_state}

PRIMARY COMMUNICATION CHANNEL: {info_channel_desc}

AVAILABLE CHANNELS:
- internal_memo: official internal memos/announcements (max 3000 chars, formal and authoritative)
- slack_channel: Slack/Teams messages (max 500 chars, short and informal)
- town_hall: all-hands/town hall presentations (max 2000 chars, structured and persuasive)
- media_coverage: press/external media (max 3000 chars, journalistic)
- investor_relations: investor communications/filings (max 5000 chars, formal and data-driven)
- water_cooler: informal hallway/break-room conversations (max 280 chars, casual and unfiltered)

Simulate how this group reacts. Respond with JSON:
{{
  "cluster_id": "{cluster_id}",
  "dominant_sentiment": "supportive|resistant|indifferent|confused|angry",
  "sentiment_distribution": {{"supportive": 0, "resistant": 0, "indifferent": 0, "confused": 0}},
  "shift_from_last_month": "float from -0.3 to +0.3",
  "engagement_level": "float from 0 to 1",
  "viral_content_reaction": "which communication struck them most and why",
  "emergent_narrative": "what narrative is gaining traction in this group",
  "sample_posts": [
    {{"author_archetype": "type", "platform": "{primary_platform}", "text": "content in the channel's tone", "tone": "tone"}},
    {{"author_archetype": "type", "platform": "{secondary_platform}", "text": "content in the channel's tone", "tone": "tone"}},
    {{"author_archetype": "type", "platform": "{primary_platform}", "text": "content in the channel's tone", "tone": "tone"}}
  ],
  "key_concerns": ["concern1", "concern2", "concern3"],
  "trust_in_institutions": "float from 0 to 1",
  "media_consumption": "which internal/external sources this cluster follows most"
}}"""

EVENT_GENERATION_PROMPT = """You are an expert organizational change analyst.

Generate the most plausible corporate event for {timeline_label}.

SCENARIO CONTEXT:
{scenario_context}

INITIAL CONTEXT (what happened):
{initial_context}

SIMULATION HISTORY (previous periods):
{history}

CURRENT STATE OF KEY STAKEHOLDERS:
{agent_states}

ORGANIZATIONAL CLIMATE TRENDS:
- Polarization: {polarization:.1f}/10
- Dominant sentiment: {dominant_sentiment}
- Emerging narrative: {top_narratives}

COALITION DYNAMICS:
{coalition_info}

MOST DISCUSSED COMMUNICATIONS LAST PERIOD:
{viral_posts}

Generate a realistic corporate event for {timeline_label} that:
1. Is a NATURAL CONSEQUENCE of the scenario context and previous events
2. Reflects current tensions and alliances among corporate stakeholders
3. Introduces a new element, escalation, or turning point
4. Is specific and detailed (department names, plausible figures, concrete actions)
5. Has a measurable impact on stakeholders' positions

Respond with JSON:
{{
  "event": "Description of the event in 4-5 detailed sentences",
  "shock_magnitude": "float from 0.1 to 0.8, how impactful the event is",
  "shock_direction": "float from -1 to +1, positive = pro-initiative, negative = anti-initiative",
  "key_actors_affected": ["list of most affected stakeholders"],
  "institutional_impact": "brief description of organizational impact",
  "public_perception": "how employees and external observers perceive the event"
}}"""

REPORT_SYSTEM = """You are an expert organizational change analyst writing an in-depth report on a corporate simulation.
Write with an analytical and professional tone. Cite specific data from the simulation.
Structure the report according to the requested sections."""

REPORT_PROMPT = """Based on these summary data from the simulation:

SCENARIO: {scenario_title}
ROUNDS COMPLETED: {num_rounds}
STATISTICS:
{round_summaries}

KEY STAKEHOLDERS ({num_elite}):
{elite_summary}

{cluster_summary}

Write a complete analytical report with these sections:

# {scenario_title} — Simulation Report

## Executive Summary
(4-5 paragraphs: key outcome, emerging dynamics, surprises, risks)

## Simulated Timeline
(Period by period, with quotes from the most impactful communications)

## Coalition Map
(Who allied with whom. How coalitions formed, evolved, and broke apart)

## Employee Sentiment Dynamics
(How employee segments reacted differently. Who changed their mind and why)

## Polarization and Resistance
(Polarization curve. Echo chamber phenomena. Observed groupthink or herd behavior)

## The Communications That Changed the Narrative
(Top 5 posts by impact, with analysis of why they resonated)

## Dominant Emerging Narrative
(Which "story" won in internal discourse)

## Organizational Impact
(Tensions between leadership and employees, risks to retention and culture)

## Scenarios Forward
(2-3 possible medium-term developments)

## Methodological Note
This simulation uses AI agents operating on simulated internal communication channels for {num_rounds} rounds.
Results are NOT probabilistic predictions, but plausible scenarios emerging from the behavior of agents with realistic profiles.

Write the COMPLETE report. Be specific, cite stakeholder names and positions. Use simulation data."""
