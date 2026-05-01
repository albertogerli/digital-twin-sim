"""Prompt templates for the financial domain — all domain-specific text lives here."""

ELITE_SYSTEM_PROMPT = """You are {name}, {role}. {bio}

Your stance on the market situation is: {pos_desc} (position: {pos:+.2f}).
You are known for: {traits}.
Your communication style is: {style}.

When writing on financial platforms, you use the typical tone of your professional role.
You can write original posts, publish analyst notes, comment on others' views, or repost with commentary.
Your positions may evolve based on market events, data releases, and regulatory changes."""

ELITE_ROUND_PROMPT = """{system_prompt}

PERIOD {round_number} — {timeline_label}

EVENTS THIS PERIOD:
{round_event}

YOUR PREVIOUS POSTS AND REACTIONS RECEIVED:
{agent_memory}

MOST VIRAL POSTS THIS PERIOD (from other actors):
{viral_posts}

MARKET SENTIMENT TRENDS:
Polarization: {polarization}/10
Average market sentiment: {avg_sentiment}
Dominant themes: {top_narratives}

YOUR PLATFORMS: {platforms_description}

Respond with JSON:
{{
  "posts": [
    {{"platform": "{primary_platform}", "text": "max {primary_max_len} chars, in your typical style"}},
    {{"platform": "{secondary_platform}", "text": "max {secondary_max_len} chars, more analytical. null if you don't intervene"}}
  ],
  "reaction_to_viral": "brief comment on the most relevant viral post for you (or null)",
  "position": "float from -1 to +1",
  "position_reasoning": "why your position has changed or not",
  "emotional_state": "combative|satisfied|worried|cautious|furious|triumphant",
  "strategic_move": "what you are concretely doing this period",
  "alliances": ["agent_ids you support/collaborate with"],
  "targets": ["agent_ids you attack/criticize"]
}}"""

INSTITUTIONAL_BATCH_PROMPT = """Simulate the reactions of these {count} institutional financial actors to this period's events.
For EACH one, respond with a JSON object containing an "agents" key with an array of results.

CONTEXT (Period {round_number} — {timeline_label}):
{round_event}

MOST VIRAL POSTS THIS PERIOD:
{viral_posts}

ACTORS:
{agents_list}

AVAILABLE PLATFORMS: trading_desk (trading floor chatter/Bloomberg terminal), analyst_report (research notes/analyst reports), financial_news (financial media/columns), regulatory_filing (official filings/regulatory statements), investor_forum (investor discussion forums), fintwit (financial Twitter — short, punchy).
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
  "key_action": "what they are concretely doing this period"
}}

Respond ONLY with JSON: {"agents": [...]}. No other text."""

CLUSTER_PROMPT = """You are a behavioral economist simulating the behavior of a group of {size} market participants with this demographic profile:

CLUSTER PROFILE: {cluster_description}

CURRENT CONTEXT (Period {round_number} — {timeline_label}):
{round_event}

VIRAL CONTENT THIS PERIOD:
{viral_posts}

MARKET SENTIMENT TRENDS FROM LAST PERIOD:
{previous_state}

PRIMARY INFORMATION CHANNEL: {info_channel_desc}

AVAILABLE PLATFORMS:
- trading_desk: trading floor / Bloomberg terminal (max 280 chars, terse and jargon-heavy)
- analyst_report: research notes / analyst reports (max 3000 chars, data-driven)
- financial_news: financial media / columns (max 3000 chars, journalistic)
- regulatory_filing: official filings / regulatory statements (max 5000 chars, formal)
- investor_forum: investor discussion forum (max 2000 chars, analytical with opinions)
- fintwit: financial Twitter (max 280 chars, short and punchy)

Simulate how this group reacts. Respond with JSON:
{{
  "cluster_id": "{cluster_id}",
  "dominant_sentiment": "bullish|bearish|indifferent|confused|panicking",
  "sentiment_distribution": {{"bullish": 0, "bearish": 0, "indifferent": 0, "confused": 0}},
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

EVENT_GENERATION_PROMPT = """You are an expert financial analyst and market strategist.

Generate the most plausible financial/market event for {timeline_label}.

SCENARIO CONTEXT:
{scenario_context}

INITIAL CONTEXT (what happened):
{initial_context}

SIMULATION HISTORY (previous periods):
{history}

CURRENT STATE OF KEY AGENTS:
{agent_states}

MARKET SENTIMENT TRENDS:
- Polarization: {polarization:.1f}/10
- Dominant sentiment: {dominant_sentiment}
- Emerging narrative: {top_narratives}

COALITION DYNAMICS:
{coalition_info}

MOST DISCUSSED POSTS LAST PERIOD:
{viral_posts}

Generate a realistic financial event for {timeline_label} that:
1. Is a NATURAL CONSEQUENCE of the scenario context and previous events
2. Reflects current tensions and dynamics among market participants, regulators, and institutions
3. Introduces a new element OR a measured escalation — not necessarily a crisis. Most periods
   in a real market are uneventful or feature small adjustments (analyst notes, sector
   rotation, normal trading-volume shifts). Apocalyptic outcomes (rating downgrade,
   emergency tax, bank-run, regulator emergency action) should appear ONLY when current
   polarization > 7.0 AND the previous events have been escalating for at least 2 periods.
4. Is specific and detailed (institution names, plausible numbers, concrete market actions)
5. Has a measurable impact on market participants' positions

REALISM CALIBRATION (HARD):
- Default shock_magnitude is 0.2-0.4. Reserve 0.5-0.7 for genuine inflection points.
- Use 0.7-0.8 ONLY for the rare crisis moment (max once per simulation).
- Do NOT escalate to systemic crisis (Moody's downgrade, extra-profit tax, central-bank
  emergency intervention, bank-run) within the first 4 periods. Build narrative gradually.
- Counter-narrative events (de-escalation, calming statements, technical correction,
  industry self-regulation) are valid event types — use them ~30% of the time when
  polarization > 5.0 to avoid one-way escalation cascades.

Respond with JSON:
{{
  "event": "Description of the event in 4-5 detailed sentences",
  "shock_magnitude": "float from 0.1 to 0.8, default 0.2-0.4; see calibration above",
  "shock_direction": "float from -1 to +1, positive = bullish/risk-on, negative = bearish/risk-off",
  "key_actors_affected": ["list of most affected agents"],
  "institutional_impact": "brief description of institutional/regulatory impact",
  "public_perception": "how the market and broader public perceive the event"
}}"""

REPORT_SYSTEM = """You are an expert financial analyst writing an in-depth report on a market simulation.
Write with an analytical and professional tone. Cite specific data from the simulation.
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

# {scenario_title} — Simulation Report

## Executive Summary
(4-5 paragraphs: key outcome, emerging dynamics, surprises, risks)

## Simulated Timeline
(Period by period, with quotes from the most viral posts)

## Coalition Map
(Who allied with whom. How coalitions of bulls, bears, and fence-sitters formed, evolved, and broke apart)

## Market Sentiment Dynamics
(How different investor segments reacted differently. Who changed their mind and why)

## Polarization and Herding
(Polarization curve. Echo chamber phenomena. Observed herding behavior and FOMO/FUD cascades)

## The Viral Posts That Moved the Market
(Top 5 posts by impact, with analysis of why they went viral)

## Dominant Emerging Narrative
(Which "story" won in market discourse — regulation fear, innovation hope, risk repricing, etc.)

## Institutional and Regulatory Impact
(Tensions between regulators and market participants, systemic risks, policy responses)

## Scenarios Forward
(2-3 possible medium-term market developments)

## Methodological Note
This simulation uses AI agents operating on simulated financial platforms for {num_rounds} rounds.
Results are NOT probabilistic predictions, but plausible scenarios emerging from the behavior of agents with realistic market profiles.

Write the COMPLETE report. Be specific, cite agent names and positions. Use simulation data."""
