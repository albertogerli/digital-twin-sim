"""Layer 1 — deterministic relevance score for stakeholder pre-filtering.

Why this exists
---------------
The SemanticRetriever pulls 30-50 candidates from the global stakeholder DB.
Many are noise for a given brief (e.g. US politicians on a Banca Sella
pricing brief). Without filtering, the realism gate (Layer 2) has to
reject 60% of them — wastes LLM tokens and sometimes leaks through (we
saw Biden, Trump, Musk, Vance in a Sella sim).

This module provides a fast, deterministic score in [0, 1] that combines:

  - country_match:        is the stakeholder's country in scope.geography?
  - sector_match:         do their declared topic_tags overlap the brief's sector?
  - brief_mention_bonus:  is their name / party / org named verbatim in the brief?
  - archetype_whitelist:  is their archetype on the scope's allow list?
  - archetype_denylist:   is their archetype on the scope's deny list?
  - category_alignment:   small domain-specific bonus

Threshold is conservative (default 0.40) — we want few false negatives
(in-scope stakeholders dropped) at the cost of more false positives
(out-of-scope stakeholders that pass to Layer 2 for LLM audit).

Pure Python, no LLM call, no network. Sub-millisecond per stakeholder.
Safe to run on every retriever pass.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .brief_scope import BriefScope

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.40

# Component weights — combine into a [0, 1] score after clamping.
# These are ad-hoc heuristics, NOT learned. Tune on a labelled test set
# (5-10 brief × ~30 candidates with human relevance verdict) when ready.
WEIGHTS = {
    "country_match":      0.40,   # strongest signal: geography is the bedrock
    "sector_match":       0.25,   # second strongest: domain alignment
    "brief_mention":      0.40,   # huge bonus: explicitly named in the brief
    "archetype_allow":    0.15,   # mild bonus
    "archetype_deny":     0.50,   # strong PENALTY (subtracted)
    "category_align":     0.10,
    "global_figure_pen":  0.30,   # PENALTY for known global figures unless
                                   # geography is also global / they're named
}

# Categories that signal "head of state / global tech billionaire" — these
# get an extra penalty unless the brief geography is global or they're named.
_GLOBAL_INFLUENCE_CATEGORIES = {"politician_head_of_state", "global_tech_billionaire"}

# Archetype <-> category equivalence (some stakeholder JSONs use one or the other).
_HEAD_OF_STATE_HINTS = {
    "head_of_state", "president", "prime_minister", "vice_president",
    "secretary_of_state", "chancellor",
}
_GLOBAL_TECH_HINTS = {
    "global_tech_billionaire", "tech_ceo", "platform_owner",
}


# ── Sector / topic ↔ stakeholder topic_tag overlap ─────────────────────────

# Map from BriefScope.sector → set of likely stakeholder topic_tags.
# Add entries as new domains/sectors emerge.
_SECTOR_TO_TOPIC_TAGS: dict[str, set[str]] = {
    # banking / consumer finance
    "fintech_lending":         {"consumer_credit_rates", "banking_sector_regulation",
                                "fintech_disruption", "consumer_protection",
                                "monetary_policy"},
    "consumer_banking":        {"consumer_credit_rates", "banking_sector_regulation",
                                "consumer_protection", "monetary_policy"},
    "italian_banking":         {"consumer_credit_rates", "banking_sector_regulation",
                                "consumer_protection", "extra_profit_tax",
                                "monetary_policy", "btp_spread"},
    "consumer_credit":         {"consumer_credit_rates", "consumer_protection",
                                "fintech_disruption"},
    "mortgage":                {"consumer_credit_rates", "monetary_policy",
                                "banking_sector_regulation"},
    # insurance / asset mgmt
    "insurance_eu":            {"insurance_regulation", "solvency", "pension_reform"},
    "asset_management":        {"market_regulation", "fund_regulation", "monetary_policy"},
    # politics
    "italian_politics":        {"general_left_right", "premierato", "judiciary_reform",
                                "eu_integration"},
    "eu_monetary":             {"monetary_policy", "btp_spread", "eu_integration"},
    "eurozone_monetary":       {"monetary_policy", "btp_spread", "eu_integration"},
    # generic fallback handled by partial-match scan
}


def _hash_brief(brief: str) -> str:
    """Stable short hash of a brief — useful for caching."""
    return hashlib.sha1((brief or "").encode("utf-8")).hexdigest()[:12]


# ── Per-stakeholder scoring ────────────────────────────────────────────────

def _country_score(stakeholder_country: str, geography: list[str]) -> float:
    """Return [0, 1] match between stakeholder country and brief geography."""
    if not geography:
        return 0.5  # neutral when scope geography unknown
    sc = (stakeholder_country or "").upper()
    geo = [g.upper() for g in geography]
    if not sc:
        return 0.3  # no country info → conservative low
    if sc in geo:
        return 1.0
    # Supranational tolerance
    if "EU" in geo and sc in {"DE", "FR", "IT", "ES", "NL", "BE", "AT",
                               "PT", "IE", "FI", "GR", "DK", "SE", "PL"}:
        return 0.7
    if "EU" in {sc}:  # stakeholder country literally "EU"
        return 0.5 if any(g in {"DE","FR","IT","ES","NL"} for g in geo) else 0.3
    if "GLOBAL" in geo:
        return 0.6
    return 0.0


def _sector_overlap_score(stakeholder, sector: str, sub_sector: str = "") -> float:
    """Score how well stakeholder's declared topic_tags match the brief sector."""
    if not sector:
        return 0.3
    tags_set = set()
    for pos in (getattr(stakeholder, "positions", []) or []):
        tag = getattr(pos, "topic_tag", None) or (
            pos.get("topic_tag", "") if isinstance(pos, dict) else ""
        )
        if tag:
            tags_set.add(tag.lower())
    if not tags_set:
        return 0.4  # neutral when stakeholder hasn't declared topic positions
    expected = _SECTOR_TO_TOPIC_TAGS.get(sector.lower(), set())
    expected |= _SECTOR_TO_TOPIC_TAGS.get(sub_sector.lower(), set())
    # Also accept any tag that contains a substring of sector
    sector_lower = sector.lower().replace("_", " ")
    fuzzy = {t for t in tags_set if any(w in t for w in sector_lower.split())}
    overlap = (tags_set & expected) | fuzzy
    if not overlap:
        return 0.0
    # Saturating: 1 match = 0.5, 3+ matches = 1.0
    return min(1.0, 0.5 + 0.2 * (len(overlap) - 1))


def _brief_mention_bonus(stakeholder, brief_lower: str) -> float:
    """1.0 if stakeholder name or party/org appears verbatim in the brief."""
    if not brief_lower:
        return 0.0
    name = (getattr(stakeholder, "name", "") or "").lower()
    party = (getattr(stakeholder, "party_or_org", "") or "").lower()
    if name and len(name) > 3 and name in brief_lower:
        return 1.0
    if party and len(party) > 3 and party in brief_lower:
        return 0.7
    # Last name only (e.g. "Meloni" mentioned without first name)
    if name and " " in name:
        last = name.rsplit(" ", 1)[1]
        if len(last) > 4 and re.search(rf"\b{re.escape(last)}\b", brief_lower):
            return 0.6
    return 0.0


def _archetype_score(stakeholder, scope) -> tuple[float, float]:
    """Return (allow_bonus, deny_penalty), each in [0, 1]."""
    arch = (getattr(stakeholder, "archetype", "") or "").lower()
    if not arch or scope is None:
        return 0.0, 0.0
    allowed = {a.lower() for a in (getattr(scope, "stakeholder_archetypes", []) or [])}
    denied = {a.lower() for a in (getattr(scope, "excluded_archetypes", []) or [])}
    bonus = 1.0 if arch in allowed else 0.0
    penalty = 1.0 if arch in denied else 0.0
    return bonus, penalty


def _global_figure_penalty(stakeholder, geography: list[str], brief_lower: str) -> float:
    """Penalise FOREIGN heads-of-state and global tech billionaires UNLESS the
    brief is global-tier or names them verbatim. In-country heads-of-state
    (e.g. Italian PM on an Italian brief) are NOT penalised — they're often
    the most natural commenter.

    Note: matches archetype only (not role substring), because Italian roles
    like "Presidente della Repubblica" / "Presidente del Consiglio" / "Presidente
    ABI" all contain "presidente" and would false-trigger.
    """
    if not geography:
        return 0.0
    geo_upper = [g.upper() for g in geography]
    if "GLOBAL" in geo_upper:
        return 0.0  # global brief: no penalty
    sc = (getattr(stakeholder, "country", "") or "").upper()
    # In-scope country: Italian PM on Italian brief, German chancellor on
    # German brief, etc. — they're the natural commenters, no penalty.
    if sc and sc in geo_upper:
        return 0.0
    # Stakeholder operates in one of the EU partner countries on an EU/IT brief
    if sc and "EU" in geo_upper and sc in {
        "DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR", "DK", "SE", "PL"
    }:
        return 0.0
    name = (getattr(stakeholder, "name", "") or "").lower()
    if name and name in brief_lower:
        return 0.0  # explicitly named: no penalty
    # Foreign + not named: check if they're a global figure type
    arch = (getattr(stakeholder, "archetype", "") or "").lower()
    role = (getattr(stakeholder, "role", "") or "").lower()
    is_head_of_state = any(h in arch for h in _HEAD_OF_STATE_HINTS)
    is_global_tech = any(h in arch for h in _GLOBAL_TECH_HINTS)
    # Role-based fallback for English-language roles (US/UK presidents),
    # safe because in-country case already returned 0 above.
    if not is_head_of_state and any(h in role for h in {"president", "prime minister", "chancellor"}):
        is_head_of_state = True
    if is_head_of_state or is_global_tech:
        return 1.0
    return 0.0


# ── Public API ─────────────────────────────────────────────────────────────

@dataclass
class RelevanceVerdict:
    stakeholder_id: str
    name: str
    score: float
    components: dict
    kept: bool
    reason: str = ""


def score_stakeholder_relevance(
    stakeholder, brief: str, scope=None,
) -> RelevanceVerdict:
    """Compute the relevance score for one stakeholder against one brief.

    Returns a RelevanceVerdict with the decomposed score, useful for audit.
    """
    brief_lower = (brief or "").lower()
    geography = list(getattr(scope, "geography", []) or []) if scope else []
    sector = (getattr(scope, "sector", "") or "") if scope else ""
    sub_sector = (getattr(scope, "sub_sector", "") or "") if scope else ""

    c_score = _country_score(getattr(stakeholder, "country", ""), geography)
    s_score = _sector_overlap_score(stakeholder, sector, sub_sector)
    m_bonus = _brief_mention_bonus(stakeholder, brief_lower)
    a_allow, a_deny = _archetype_score(stakeholder, scope)
    g_pen = _global_figure_penalty(stakeholder, geography, brief_lower)

    # Combine — additive, then clamp to [0, 1]
    raw = (
        WEIGHTS["country_match"]    * c_score
        + WEIGHTS["sector_match"]   * s_score
        + WEIGHTS["brief_mention"]  * m_bonus
        + WEIGHTS["archetype_allow"]* a_allow
        - WEIGHTS["archetype_deny"] * a_deny
        - WEIGHTS["global_figure_pen"] * g_pen
    )
    # Cap mention_bonus override: if explicitly named, never drop
    if m_bonus >= 0.9:
        raw = max(raw, 0.7)

    score = max(0.0, min(1.0, raw))

    components = {
        "country": round(c_score, 3),
        "sector": round(s_score, 3),
        "brief_mention": round(m_bonus, 3),
        "archetype_allow": a_allow,
        "archetype_deny": a_deny,
        "global_figure_pen": g_pen,
        "raw": round(raw, 3),
    }

    return RelevanceVerdict(
        stakeholder_id=getattr(stakeholder, "id", ""),
        name=getattr(stakeholder, "name", ""),
        score=round(score, 3),
        components=components,
        kept=False,  # filled by caller
        reason="",
    )


def filter_stakeholders_by_relevance(
    stakeholders: list,
    brief: str,
    scope=None,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list, list[RelevanceVerdict]]:
    """Score every stakeholder, drop those below threshold.

    Returns (kept_stakeholders, full_verdict_list). Full verdict list is
    useful for audit/debugging — it includes both kept and dropped, with
    score components. Logs dropped at INFO level.
    """
    kept = []
    verdicts = []
    dropped_count = 0
    for s in stakeholders:
        v = score_stakeholder_relevance(s, brief, scope)
        if v.score >= threshold:
            v.kept = True
            v.reason = "above threshold"
            kept.append(s)
        else:
            v.kept = False
            v.reason = f"score {v.score} < threshold {threshold}"
            dropped_count += 1
            logger.info(
                f"relevance_filter: dropped {v.name} "
                f"(score {v.score}, components={v.components})"
            )
        verdicts.append(v)
    if dropped_count > 0:
        logger.info(
            f"relevance_filter: kept {len(kept)}/{len(stakeholders)} "
            f"(dropped {dropped_count} below {threshold})"
        )
    return kept, verdicts
