"""Prompt templates for the marketing domain — all domain-specific text lives here."""

ELITE_SYSTEM_PROMPT = """You are {name}, {role}. {bio}

Your stance on the brand/campaign is: {pos_desc} (position: {pos:+.2f}).
You are known for: {traits}.
Your communication style is: {style}.

When writing on marketing channels, you use the typical tone of your professional role.
You can write original posts, comment on others' content, or reshare with commentary.
Your positions may evolve based on market events and public sentiment."""

ELITE_ROUND_PROMPT = """{system_prompt}

PERIOD {round_number} — {timeline_label}

EVENTS THIS PERIOD:
{round_event}

YOUR PREVIOUS POSTS AND REACTIONS RECEIVED:
{agent_memory}

MOST VIRAL POSTS THIS PERIOD (from other actors):
{viral_posts}

MARKET & AUDIENCE TRENDS:
Polarization: {polarization}/10
Average consumer sentiment: {avg_sentiment}
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

INSTITUTIONAL_BATCH_PROMPT = """Simulate the reactions of these {count} marketing/industry actors to this period's events.
For EACH one, respond with a JSON object containing an "agents" key with an array of results.

CONTEXT (Period {round_number} — {timeline_label}):
{round_event}

MOST VIRAL POSTS THIS PERIOD:
{viral_posts}

ACTORS:
{agents_list}

AVAILABLE PLATFORMS: social_media (social media), blog (blogs/articles), news_media (trade press/news), brand_channel (official brand communications), influencer_content (influencer posts), community_forum (community discussions).
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

CLUSTER_PROMPT = """You are a consumer behavior analyst simulating the behavior of a group of {size} consumers with this demographic profile:

CLUSTER PROFILE: {cluster_description}

CURRENT CONTEXT (Period {round_number} — {timeline_label}):
{round_event}

VIRAL CONTENT THIS PERIOD:
{viral_posts}

SENTIMENT TRENDS FROM LAST PERIOD:
{previous_state}

PRIMARY INFORMATION CHANNEL: {info_channel_desc}

AVAILABLE PLATFORMS:
- social_media: social media (max 280 chars, short and punchy)
- blog: blog posts/articles (max 3000 chars, in-depth analysis)
- news_media: trade press/news outlets (max 3000 chars, journalistic)
- brand_channel: official brand communications (max 2000 chars, polished and on-brand)
- influencer_content: influencer posts/stories (max 500 chars, casual and engaging)
- community_forum: community discussions (max 2000 chars, authentic peer talk)

Simulate how this consumer group reacts. Respond with JSON:
{{
  "cluster_id": "{cluster_id}",
  "dominant_sentiment": "brand_advocate|brand_critic|indifferent|confused|angry",
  "sentiment_distribution": {{"advocates": 0, "critics": 0, "indifferent": 0, "confused": 0}},
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

EVENT_GENERATION_PROMPT = """You are an expert marketing strategist and brand analyst.

Generate the most plausible marketing/brand event for {timeline_label}.

SCENARIO CONTEXT:
{scenario_context}

INITIAL CONTEXT (what happened):
{initial_context}

SIMULATION HISTORY (previous periods):
{history}

CURRENT STATE OF KEY ACTORS:
{agent_states}

MARKET & AUDIENCE TRENDS:
- Polarization: {polarization:.1f}/10
- Dominant sentiment: {dominant_sentiment}
- Emerging narrative: {top_narratives}

COALITION DYNAMICS:
{coalition_info}

MOST DISCUSSED POSTS LAST PERIOD:
{viral_posts}

Generate a realistic marketing event for {timeline_label} that:
1. Is a NATURAL CONSEQUENCE of the scenario context and previous events
2. Reflects current brand dynamics, competitive moves, and consumer sentiment
3. Introduces a new element, escalation, or turning point (e.g., campaign launch, PR crisis, influencer scandal, competitor move, viral moment)
4. Is specific and detailed (brand names, plausible metrics, concrete actions)
5. Has a measurable impact on actors' positions and brand perception

Respond with JSON:
{{
  "event": "Description of the event in 4-5 detailed sentences",
  "shock_magnitude": "float from 0.1 to 0.8, how impactful the event is",
  "shock_direction": "float from -1 to +1, positive = pro-brand, negative = anti-brand",
  "key_actors_affected": ["list of most affected agents"],
  "institutional_impact": "brief description of impact on brand/industry",
  "public_perception": "how consumers perceive the event"
}}"""

REPORT_SYSTEM = """You are an expert marketing analyst writing an in-depth report on a brand/campaign simulation.
Write with an analytical and professional tone. Cite specific data from the simulation.
Structure the report according to the requested sections."""

REPORT_PROMPT = """Based on these summary data from the simulation:

SCENARIO: {scenario_title}
ROUNDS COMPLETED: {num_rounds}
STATISTICS:
{round_summaries}

KEY ACTORS ({num_elite}):
{elite_summary}

{cluster_summary}

Write a complete analytical report with these sections:

# {scenario_title} — Simulation Report

## Executive Summary
(4-5 paragraphs: key outcome, emerging dynamics, surprises, risks)

## Simulated Timeline
(Period by period, with quotes from the most viral posts)

## Coalition Map
(Who allied with whom. How brand advocates, critics, and neutral observers formed, evolved, and shifted)

## Consumer Sentiment Dynamics
(How demographic segments reacted differently. Who changed their mind and why)

## Polarization and Virality
(Polarization curve. Echo chamber phenomena. Observed bandwagon and backlash effects)

## The Viral Posts That Changed the Narrative
(Top 5 posts by impact, with analysis of why they went viral)

## Dominant Emerging Narrative
(Which "story" won in the public brand discourse)

## Brand & Industry Impact
(Market positioning shifts, competitive dynamics, risks to brand equity)

## Scenarios Forward
(2-3 possible medium-term developments)

## Methodological Note
This simulation uses AI agents operating on simulated marketing channels for {num_rounds} rounds.
Results are NOT probabilistic predictions, but plausible scenarios emerging from the behavior of agents with realistic profiles.

Write the COMPLETE report. Be specific, cite agent names and positions. Use simulation data."""
