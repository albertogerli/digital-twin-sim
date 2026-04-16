"""Integration layer between StakeholderDB and the simulation pipeline.

Replaces on-the-fly LLM elite agent generation with stakeholder graph lookups.
Falls back to LLM generation when the graph has insufficient coverage.
"""

import logging
from typing import Optional

from stakeholder_graph.db import StakeholderDB
from stakeholder_graph.schema import Stakeholder

logger = logging.getLogger(__name__)

# Singleton DB instance
_db: Optional[StakeholderDB] = None


def get_db() -> StakeholderDB:
    """Get or create the singleton StakeholderDB."""
    global _db
    if _db is None:
        _db = StakeholderDB()
    return _db


def infer_topic_tags(brief: str, domain: str) -> list[str]:
    """Infer topic tags from the scenario brief and domain.

    Simple keyword matching — no LLM call needed.
    """
    brief_lower = brief.lower()
    tags = []

    # Italian political topics
    keyword_map = {
        "judiciary_reform": ["separazione carriere", "riforma giustizia", "judiciary", "magistrat"],
        "premierato": ["premierato", "premier", "elezione diretta"],
        "autonomia_differenziata": ["autonomia differenziata", "autonomia", "regionalismo"],
        "immigration": ["immigra", "migranti", "sbarchi", "frontiere", "borders"],
        "eu_integration": ["europa", "ue", "eu ", "european", "bruxelles", "brussels"],
        "fiscal_policy": ["tasse", "fisco", "bilancio", "manovra", "tax", "budget", "deficit"],
        "labor_reform": ["lavoro", "occupazione", "salario", "labor", "employment", "wage", "licenziament", "sciopero", "sindacato", "operai"],
        "environment": ["clima", "ambiente", "green", "climate", "energy", "energia"],
        "diritti_civili": ["diritti civili", "lgbtq", "aborto", "eutanasia", "civil rights"],
        "education_reform": ["scuola", "istruzione", "università", "education"],
        "media_freedom": ["stampa", "informazione", "rai", "media", "press freedom"],
        "defense_spending": ["difesa", "nato", "militare", "defense", "armi"],
        "industrial_policy": ["industria", "automotive", "made in italy", "manufacturing"],
        "reddito_cittadinanza": ["reddito", "sussidio", "welfare"],
    }

    for tag, keywords in keyword_map.items():
        if any(kw in brief_lower for kw in keywords):
            tags.append(tag)

    # Always include general positioning
    if not tags:
        tags.append("general_left_right")

    return tags


def infer_country(brief: str) -> str:
    """Infer country from brief text."""
    brief_lower = brief.lower()
    if any(w in brief_lower for w in ["italia", "italian", "roma", "parlamento", "senato", "camera"]):
        return "IT"
    if any(w in brief_lower for w in ["france", "français", "paris", "assemblée"]):
        return "FR"
    if any(w in brief_lower for w in ["germany", "deutsch", "berlin", "bundestag"]):
        return "DE"
    if any(w in brief_lower for w in ["spain", "español", "madrid", "congreso"]):
        return "ES"
    if any(w in brief_lower for w in ["uk", "britain", "london", "parliament", "westminster"]):
        return "GB"
    return ""


def stakeholders_for_scenario(
    brief: str,
    domain: str = "",
    country: str = "",
    n_elite: int = 12,
    n_institutional: int = 8,
    min_coverage: int = 5,
) -> Optional[dict]:
    """Query the stakeholder graph for a scenario.

    Args:
        brief: The user's scenario brief text.
        domain: Scenario domain (political, financial, etc.).
        country: ISO country code. If empty, inferred from brief.
        n_elite: Target number of elite agents.
        n_institutional: Target number of institutional agents.
        min_coverage: Minimum stakeholders needed to skip LLM generation.

    Returns:
        Dict with 'elite_agents' and 'institutional_agents' in AgentSpec format,
        or None if coverage is insufficient (caller should fall back to LLM).
    """
    db = get_db()

    if not country:
        country = infer_country(brief)
    if not country:
        logger.info("Could not infer country from brief — no graph lookup")
        return None

    topic_tags = infer_topic_tags(brief, domain)
    logger.info(f"Stakeholder graph: country={country}, topics={topic_tags}")

    result = db.query_for_scenario(
        country=country,
        topic_tags=topic_tags,
        n_elite=n_elite,
        n_institutional=n_institutional,
    )

    n_found = len(result["elite_agents"])
    if n_found < min_coverage:
        logger.info(
            f"Stakeholder graph: only {n_found} elite agents found "
            f"(need {min_coverage}) — falling back to LLM"
        )
        return None

    logger.info(
        f"Stakeholder graph: returning {n_found} elite + "
        f"{len(result['institutional_agents'])} institutional agents"
    )
    return result


def enrich_seed_data(brief: str, country: str = "", topic_tag: str = "") -> list[dict]:
    """Generate VerifiedStakeholder-format dicts for seed_data injection.

    Use this to populate the seed_data_bundle that feeds into the existing
    agent_generator.py pipeline (enriches LLM generation rather than replacing it).
    """
    db = get_db()

    if not country:
        country = infer_country(brief)
    if not country:
        return []

    if not topic_tag:
        tags = infer_topic_tags(brief, "")
        topic_tag = tags[0] if tags else ""

    stakeholders = db.query(
        country=country,
        topic_tag=topic_tag if topic_tag else None,
        min_influence=0.4,
        min_tier=1,
        limit=15,
    )

    return [s.to_seed_format(topic_tag) for s in stakeholders]
