"""Layer 0 — analyze a brief's scope before generating agents.

Produces a structured `BriefScope` with sector, geography, scope tier, and
archetypes of plausible commenters. NO person names in output — that's the
realism guardrail: scope first, names later (in agent_generator, constrained
by this scope).
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Literal

from core.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

ScopeTier = Literal["global", "national", "regional", "niche"]


@dataclass
class BriefScope:
    """Structured scope analysis of a scenario brief.

    This is intentionally person-name-free. Its job is to define the *scope*
    within which later generation steps must pick realistic commenters.
    """

    sector: str                         # e.g. "fashion_retail", "fintech", "pharma_regulation"
    sub_sector: str = ""                # e.g. "luxury_streetwear", "sme_lending"
    geography: list[str] = field(default_factory=list)   # ISO country codes, e.g. ["IT", "EU"]
    scope_tier: ScopeTier = "national"
    named_entities: list[str] = field(default_factory=list)  # brands/people explicitly in brief
    stakeholder_archetypes: list[str] = field(default_factory=list)  # who would plausibly comment
    excluded_archetypes: list[str] = field(default_factory=list)  # who would NOT plausibly comment
    language_register: str = "formal"   # formal|informal|technical
    rationale: str = ""                 # one-line explanation for logs

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"[{self.scope_tier}] {self.sector}"
            f"{f'/{self.sub_sector}' if self.sub_sector else ''} "
            f"in {','.join(self.geography) or '?'} "
            f"(archetypes: {','.join(self.stakeholder_archetypes[:4])})"
        )

    def prompt_block(self) -> str:
        """Render a prompt-ready scope constraint block for downstream generators.

        Returns a string meant to be dropped into ELITE_AGENTS_PROMPT /
        INSTITUTIONAL_AGENTS_PROMPT / ENTITY_EXTRACTION_PROMPT etc.
        Empty string if scope is degenerate.

        When the scope is narrow (sub_sector set, or tier in {niche, regional}),
        the block adds a stronger "roster composition" hint pushing the LLM
        toward sub-sector specialists and community voices instead of the
        default generalist-analyst bias.
        """
        if not self.sector or self.sector == "unknown":
            return ""
        geo = ", ".join(self.geography) if self.geography else "UNKNOWN"
        allowed = ", ".join(self.stakeholder_archetypes) or "(none specified — err on the side of in-sector roles)"
        excluded = ", ".join(self.excluded_archetypes) or "(none specified)"
        named = ", ".join(self.named_entities[:8]) if self.named_entities else "(none — pick realistic public figures in this sector/geography)"

        narrow = bool(self.sub_sector) or self.scope_tier in ("niche", "regional")
        if narrow:
            composition = (
                "ROSTER COMPOSITION (this brief is narrow — sub-sector or niche/regional tier):\n"
                "  * PREFER specialist voices whose public track record is tied SPECIFICALLY to\n"
                "    this sub-sector — independent critics, community curators, long-running\n"
                "    publication editors, sub-sector historians, newsletter authors, domain\n"
                "    practitioners with a public footprint — OVER generalist industry-wide\n"
                "    analysts, big-firm consulting partners, or broad category commentators.\n"
                "  * The audience of a narrow sub-sector follows specific voices that\n"
                "    industry-generalists do not reach. Include those voices.\n"
                "  * Cap generalist analysts at roughly 1-2 per roster; the rest should be\n"
                "    actors with documented depth in this exact sub-sector.\n"
                "  * If the sub-sector has a well-known critic, curator, archivist, or community\n"
                "    figure with a real public footprint (books, long-running publication,\n"
                "    dedicated following), INCLUDE THEM even if their 'title' is unconventional.\n"
            )
        else:
            composition = (
                "ROSTER COMPOSITION:\n"
                "  * Include at least 2-3 specialist voices — sector-specific critics,\n"
                "    community curators, long-running publication editors, independent\n"
                "    commentators — alongside the industry-generalist analysts. The audience\n"
                "    of a sector reads both, and specialists often shape opinion ahead of\n"
                "    generalists.\n"
            )

        return (
            "SCOPE CONSTRAINTS (HARD — violating these invalidates the output):\n"
            f"- Sector: {self.sector}{('/' + self.sub_sector) if self.sub_sector else ''}\n"
            f"- Geography: {geo}\n"
            f"- Scope tier: {self.scope_tier}\n"
            f"- Allowed archetypes (whitelist): {allowed}\n"
            f"- Forbidden archetypes (denylist): {excluded}\n"
            f"- Entities mentioned in brief: {named}\n"
            "RULES:\n"
            "  * Only pick people/orgs whose public role is IN the sector AND IN the geography.\n"
            "  * Do NOT pick global celebrities, heads of state, central bankers, or tech billionaires\n"
            "    unless the brief genuinely operates at that tier (scope_tier == 'global' and the role fits).\n"
            "  * If unsure whether a figure is in-scope, skip them and pick a tighter, sector-specific actor.\n"
            "  * Prefer concrete, verifiable individuals over generic composite roles.\n"
            f"{composition}"
        )


SCOPE_PROMPT = """You are a scoping analyst for a digital-twin simulation system. Your ONLY job is
to classify the *scope* of a scenario brief. You do NOT name any specific people.
You do NOT generate agents. You define the boundaries within which realistic
commenters will later be selected.

USER'S SCENARIO BRIEF:
{brief}

{web_context}

Produce a JSON analysis with these fields — no additional fields, no person names:

{{
  "sector": "snake_case sector tag, e.g. fashion_retail, fintech_lending, pharma_regulation, automotive_ev, food_beverage, consumer_electronics, italian_politics, eurozone_monetary. Be specific enough to exclude unrelated domains.",
  "sub_sector": "optional narrower tag, e.g. luxury_streetwear, sme_credit, pediatric_vaccines. Empty string if none.",
  "geography": ["list of ISO country codes or supranational blocs that the scenario actually operates in, e.g. [\\"IT\\"], [\\"IT\\",\\"EU\\"], [\\"GLOBAL\\"]"],
  "scope_tier": "one of: global | national | regional | niche. Use global ONLY for scenarios whose primary playing field is international (G7 policy, global platform shifts, multinational M&A). Use niche for narrow verticals (single-city events, B2B-only topics).",
  "named_entities": ["brands, companies, products, or public figures mentioned EXPLICITLY in the brief — extract verbatim. If none, empty list."],
  "stakeholder_archetypes": [
    "archetypes of actors who would PLAUSIBLY comment on this brief. Use archetype labels, NOT names. Examples: industry_ceo, sector_analyst, trade_press_journalist, industry_association, regulator, competitor_executive, consumer_advocate, niche_influencer, academic_expert, labor_union, local_politician. Be tight — only include archetypes that would realistically weigh in."
  ],
  "excluded_archetypes": [
    "archetypes that must NOT be included because they are out of scope. Examples: head_of_state, global_tech_billionaire, central_bank_governor, world_religious_leader — include any archetype whose scope exceeds the scenario's. Default to including these for non-global / non-policy briefs."
  ],
  "language_register": "formal | informal | technical",
  "rationale": "one short sentence summarizing why this scope was chosen"
}}

CRITICAL SCOPING RULES:
- If the brief is about a consumer product launch, a corporate decision, a vertical-industry event:
  it is NOT global. Heads of state, global tech billionaires, central bank governors are OUT.
- A national/niche brief in Italy has Italian stakeholders as primary commenters. World-tier
  figures enter ONLY if the brief explicitly invokes them.
- Err on the side of narrower scope. A too-narrow scope means realistic agents; a too-broad
  scope invites hallucinated celebrity commentary.
- Geography must reflect the scenario's real operating area, not the nationality of the company
  (e.g., an Italian company launching a product globally → geography includes GLOBAL if the
  launch is genuinely multi-market; otherwise IT only).
"""


async def analyze_scope(
    brief: str,
    llm: BaseLLMClient,
    web_context: str = "",
) -> BriefScope:
    """Layer 0: classify scope of a brief before agent generation."""

    ctx_section = ""
    if web_context:
        trimmed = web_context[:4000]
        ctx_section = (
            f"REAL-WORLD CONTEXT (for scope inference only — do not name any person):\n"
            f"{trimmed}\n"
        )

    prompt = SCOPE_PROMPT.format(brief=brief.strip(), web_context=ctx_section)

    result = await llm.generate_json(
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=1500,
        component="scope_analysis",
    )

    scope = BriefScope(
        sector=str(result.get("sector", "") or "").strip() or "unknown",
        sub_sector=str(result.get("sub_sector", "") or "").strip(),
        geography=[str(g).strip().upper() for g in (result.get("geography") or []) if str(g).strip()] or ["UNKNOWN"],
        scope_tier=_coerce_tier(result.get("scope_tier", "national")),
        named_entities=[str(e).strip() for e in (result.get("named_entities") or []) if str(e).strip()],
        stakeholder_archetypes=[str(a).strip() for a in (result.get("stakeholder_archetypes") or []) if str(a).strip()],
        excluded_archetypes=[str(a).strip() for a in (result.get("excluded_archetypes") or []) if str(a).strip()],
        language_register=str(result.get("language_register", "formal") or "formal").strip().lower(),
        rationale=str(result.get("rationale", "") or "").strip(),
    )

    logger.info(f"Scope analyzed: {scope.summary()}")
    return scope


def _coerce_tier(value) -> ScopeTier:
    v = str(value or "").strip().lower()
    if v in ("global", "national", "regional", "niche"):
        return v  # type: ignore
    return "national"
