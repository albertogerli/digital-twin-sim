"""Prompt templates for LLM article analysis.

All prompts isolated here for easy iteration and testing.
"""

SYSTEM_PROMPT = """You are an expert political analyst specializing in European politics, economics, and public affairs. You analyze news articles to extract signals about public figures' positions, influence, and relationships.

You are precise, evidence-based, and conservative in your assessments. You only report what the article actually says or strongly implies — never speculate beyond the text."""

ARTICLE_ANALYSIS_PROMPT = """Analyze this article for signals about the stakeholder below.

STAKEHOLDER PROFILE:
- Name: {name}
- Role: {role}
- Party/Org: {party_or_org}
- Current positions: {positions_json}

ARTICLE:
- Title: {title}
- Source: {source_name}
- Published: {published}
- Text:
{body}

---

Extract structured signals. Return JSON with this exact schema:

{{
  "relevant": true/false,
  "position_signals": [
    {{
      "topic_tag": "string (use existing tags when possible: {available_topics})",
      "direction": float (-1.0 to 1.0, where -1=strongly against/left, +1=strongly for/right),
      "strength": "strong|moderate|weak",
      "evidence": "exact quote or close paraphrase from article",
      "is_new_topic": false
    }}
  ],
  "quotes": ["exact stakeholder quotes from the article, if any"],
  "influence_signal": null | {{
    "direction": "up|down|stable",
    "magnitude": float (0.0-1.0),
    "reason": "brief explanation"
  }},
  "relationship_signals": [
    {{
      "target_name": "name of other stakeholder mentioned",
      "relation_type": "ally|rival|coalition|opposition|neutral",
      "evidence": "brief evidence"
    }}
  ]
}}

RULES:
1. Set "relevant" to false if the article doesn't contain meaningful signals about this stakeholder's positions or influence.
2. Only include position_signals backed by DIRECT EVIDENCE in the article text.
3. "evidence" MUST be a real quote or close paraphrase from the article — never fabricate.
4. Use the stakeholder's EXISTING topic_tags when possible. Only use is_new_topic=true for genuinely new policy areas.
5. "direction" reflects what the article SUGGESTS the stakeholder's position is, not the article's editorial stance.
6. For influence_signal, only report if the article shows a clear change (new role, scandal, major victory, political crisis).
7. Prefer fewer, high-confidence signals over many weak ones.

Available topic tags for reference: {available_topics}"""


BATCH_RELEVANCE_PROMPT = """Given these article titles and a stakeholder name, rate which articles are most likely to contain meaningful position signals.

STAKEHOLDER: {name} ({role}, {party_or_org})

ARTICLES:
{article_list}

Return JSON array of article indices (0-based) that are LIKELY RELEVANT, sorted by relevance:
{{"relevant_indices": [0, 3, 7]}}

Only include articles where the stakeholder likely takes a position, makes a statement, or has their influence affected. Exclude articles that merely mention them in passing."""


# Standard topic tags used across the graph
STANDARD_TOPICS = [
    "general_left_right", "eu_integration", "immigration", "fiscal_policy",
    "judiciary_reform", "premierato", "autonomia_differenziata", "labor_reform",
    "environment", "diritti_civili", "education_reform", "media_freedom",
    "defense_spending", "industrial_policy", "reddito_cittadinanza", "sanita",
    "energy_policy", "trade_policy", "welfare", "foreign_policy",
    "digital_regulation", "housing", "pensions",
]
