"""Semantic Agent Retriever — picks the perfect 12-15 agents for any crisis.

Given a crisis briefing, retrieves stakeholders from the 744-entry graph
by scoring them across three dimensions:
  1. Thematic relevance — does this stakeholder care about this topic?
  2. Geographic relevance — is this stakeholder local to the crisis?
  3. Institutional reach — can this stakeholder amplify the crisis?

The retriever produces an ordered activation list grouped into tiers:
  - Immediate responders (round 1): local press, sector unions, mayors
  - Secondary wave (round 2-3): national media, party leaders, ministers
  - Tertiary wave (round 4+): PM, president, international actors

This replaces the old "activate everyone every round" pattern.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from stakeholder_graph.db import StakeholderDB
from stakeholder_graph.schema import Stakeholder

logger = logging.getLogger(__name__)


# ── Geographic keywords → regions/cities ────────────────────────────────────

ITALIAN_GEO_KEYWORDS: dict[str, list[str]] = {
    # Regions
    "lombardia": ["lombardia", "milano", "brescia", "bergamo", "monza", "varese"],
    "piemonte": ["piemonte", "torino", "novara", "cuneo", "asti", "alessandria"],
    "veneto": ["veneto", "venezia", "verona", "padova", "vicenza", "treviso"],
    "emilia_romagna": ["emilia", "romagna", "bologna", "modena", "parma", "reggio", "rimini", "ravenna", "ferrara"],
    "toscana": ["toscana", "firenze", "pisa", "siena", "livorno", "arezzo", "lucca"],
    "lazio": ["lazio", "roma", "viterbo", "latina", "frosinone"],
    "campania": ["campania", "napoli", "salerno", "caserta", "avellino", "benevento"],
    "puglia": ["puglia", "bari", "lecce", "taranto", "foggia", "brindisi"],
    "sicilia": ["sicilia", "palermo", "catania", "messina", "siracusa", "ragusa"],
    "sardegna": ["sardegna", "cagliari", "sassari", "nuoro", "oristano"],
    "calabria": ["calabria", "cosenza", "catanzaro", "reggio calabria", "crotone"],
    "liguria": ["liguria", "genova", "savona", "la spezia", "imperia"],
    "friuli": ["friuli", "trieste", "udine", "pordenone", "gorizia"],
    "marche": ["marche", "ancona", "pesaro", "fermo", "macerata"],
    "abruzzo": ["abruzzo", "l'aquila", "pescara", "chieti", "teramo"],
    "umbria": ["umbria", "perugia", "terni", "assisi"],
    "basilicata": ["basilicata", "potenza", "matera", "melfi"],
    "molise": ["molise", "campobasso", "isernia"],
    "trentino": ["trentino", "trento", "bolzano", "alto adige", "südtirol"],
    "valle_aosta": ["valle d'aosta", "aosta"],
}

# Sector keywords → categories/archetypes
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "automotive": ["stellantis", "fiat", "auto", "automotive", "stabilimento auto", "produzione auto", "ferrari", "maserati", "alfa romeo", "tavares"],
    "energy": ["energia", "gas", "petrolio", "rinnovabili", "nucleare", "eni", "enel", "a2a", "snam", "terna"],
    "banking": ["banca", "credito", "finanza", "bce", "spread", "unicredit", "intesa", "mediobanca", "monte paschi"],
    "telecom": ["telecomunicazioni", "rete", "5g", "tim", "fibra", "digitale"],
    "defense": ["difesa", "armi", "nato", "militare", "esercito", "marina", "leonardo"],
    "healthcare": ["sanità", "ospedali", "medici", "infermieri", "ssn", "farmaceutica", "vaccin"],
    "education": ["scuola", "università", "studenti", "docenti", "ricerca", "istruzione", "rettore"],
    "labor": ["lavoro", "licenziamento", "cassa integrazione", "sciopero", "contratto", "stipendi", "salario", "occupazione", "disoccupazione"],
    "immigration": ["immigrazione", "migranti", "sbarchi", "accoglienza", "frontex", "lampedusa", "cpr"],
    "justice": ["giustizia", "magistratura", "riforma", "csm", "processo", "tribunale", "separazione carriere"],
    "environment": ["ambiente", "clima", "green", "transizione ecologica", "emissioni", "rifiuti", "inquinamento"],
    "media": ["rai", "mediaset", "stampa", "giornalismo", "editoria", "censura", "par condicio"],
    "agriculture": ["agricoltura", "pac", "agroalimentare", "coldiretti", "trattori", "cibo"],
    "real_estate": ["casa", "affitti", "edilizia", "immobiliare", "mutui"],
    "tech": ["tecnologia", "startup", "intelligenza artificiale", "cybersecurity", "big tech", "silicon valley"],
    "transport": ["trasporti", "ferrovie", "autostrade", "porti", "aeroporti", "logistica"],
    "culture": ["cultura", "cinema", "teatro", "musica", "arte", "patrimonio"],
    "sport": ["calcio", "serie a", "figc", "coni", "olimpiadi", "doping", "stadio"],
    "religion": ["vaticano", "papa", "chiesa", "cei", "vescovi", "cattolico"],
}


@dataclass
class RelevanceScore:
    """Multi-dimensional relevance score for a stakeholder."""
    stakeholder_id: str
    thematic: float = 0.0       # 0-1: how relevant is the topic to them
    geographic: float = 0.0     # 0-1: how close geographically
    institutional: float = 0.0  # 0-1: how much reach/authority
    total: float = 0.0          # weighted combination

    # Activation metadata
    activation_tier: int = 1    # 1=immediate, 2=secondary, 3=tertiary
    activation_reason: str = ""
    matched_sectors: list[str] = field(default_factory=list)
    matched_regions: list[str] = field(default_factory=list)


@dataclass
class ActivationPlan:
    """Ordered plan of which agents activate in which round."""
    briefing_summary: str = ""
    detected_topics: list[str] = field(default_factory=list)
    detected_regions: list[str] = field(default_factory=list)
    detected_sectors: list[str] = field(default_factory=list)
    country: str = "IT"

    # Agents by activation wave
    wave_1: list[RelevanceScore] = field(default_factory=list)  # Round 1: immediate
    wave_2: list[RelevanceScore] = field(default_factory=list)  # Round 2-3: secondary
    wave_3: list[RelevanceScore] = field(default_factory=list)  # Round 4+: tertiary
    reserve: list[RelevanceScore] = field(default_factory=list)  # Only if crisis escalates

    @property
    def total_agents(self) -> int:
        return len(self.wave_1) + len(self.wave_2) + len(self.wave_3)

    @property
    def all_waves(self) -> list[list[RelevanceScore]]:
        return [self.wave_1, self.wave_2, self.wave_3]

    def agents_for_round(self, round_num: int, engagement_score: float = 0.0) -> list[RelevanceScore]:
        """Get agents that should be active in a given round.

        Args:
            round_num: Current round (1-based)
            engagement_score: 0-1 score of how viral/engaged the crisis is.
                             Higher = more agents activate earlier.
        """
        active = list(self.wave_1)  # Wave 1 always active

        # Wave 2: activate from round 2 OR if engagement is high in round 1
        if round_num >= 2 or engagement_score > 0.6:
            active.extend(self.wave_2)

        # Wave 3: activate from round 4 OR if engagement is very high
        if round_num >= 4 or engagement_score > 0.8:
            active.extend(self.wave_3)

        # Reserve: only if crisis has truly exploded
        if engagement_score > 0.9:
            active.extend(self.reserve)

        return active

    def to_dict(self) -> dict:
        """Serialize for logging/reporting."""
        def _scores(scores: list[RelevanceScore]) -> list[dict]:
            return [
                {
                    "id": s.stakeholder_id,
                    "total": round(s.total, 3),
                    "thematic": round(s.thematic, 3),
                    "geographic": round(s.geographic, 3),
                    "institutional": round(s.institutional, 3),
                    "reason": s.activation_reason,
                }
                for s in scores
            ]

        return {
            "briefing_summary": self.briefing_summary,
            "detected_topics": self.detected_topics,
            "detected_regions": self.detected_regions,
            "detected_sectors": self.detected_sectors,
            "wave_1": _scores(self.wave_1),
            "wave_2": _scores(self.wave_2),
            "wave_3": _scores(self.wave_3),
            "reserve": _scores(self.reserve),
            "total_agents": self.total_agents,
        }


class SemanticRetriever:
    """Retrieves and ranks stakeholders for a crisis briefing.

    Scores each stakeholder across three axes:
    - Thematic: Does this person care about this topic? (position exists, sector match)
    - Geographic: Is this person local? (region/city match, or national figure)
    - Institutional: How much reach? (influence × tier weight)

    Then assigns activation waves based on combined score + role logic.
    """

    # Weights for combining the three relevance dimensions
    W_THEMATIC = 0.45
    W_GEOGRAPHIC = 0.25
    W_INSTITUTIONAL = 0.30

    # Wave thresholds (total relevance score)
    WAVE_1_THRESHOLD = 0.55  # High relevance → immediate activation
    WAVE_2_THRESHOLD = 0.35  # Medium → secondary wave
    WAVE_3_THRESHOLD = 0.20  # Low but non-zero → tertiary

    # Max agents per wave
    MAX_WAVE_1 = 8
    MAX_WAVE_2 = 10
    MAX_WAVE_3 = 8
    MAX_RESERVE = 6

    def __init__(self, db: StakeholderDB):
        self.db = db

    def retrieve(
        self,
        brief: str,
        country: str = "IT",
        max_total: int = 30,
        llm_topics: Optional[list[str]] = None,
    ) -> ActivationPlan:
        """Build an activation plan from a crisis briefing.

        Args:
            brief: Free-text crisis description
            country: Primary country to focus on
            max_total: Max total agents across all waves
            llm_topics: Optional pre-extracted topic tags (from scaffold step)

        Returns:
            ActivationPlan with agents assigned to waves.
        """
        brief_lower = brief.lower()

        # Step 1: Detect topics, regions, sectors from brief
        topics = llm_topics or self._detect_topics(brief_lower)
        regions = self._detect_regions(brief_lower)
        sectors = self._detect_sectors(brief_lower)

        logger.info(f"Retriever: topics={topics}, regions={regions}, sectors={sectors}")

        # Step 2: Get all candidate stakeholders
        candidates = self.db.query(country=country, active_only=True)

        # Also include high-influence international figures
        if country != "US":
            us_heavyweights = self.db.query(country="US", min_influence=0.7, min_tier=1)
            candidates.extend(us_heavyweights)

        # Step 3: Score each candidate
        scores: list[RelevanceScore] = []
        for s in candidates:
            score = self._score_stakeholder(s, topics, regions, sectors, brief_lower)
            if score.total > 0.05:  # skip completely irrelevant
                scores.append(score)

        # Step 4: Sort by total score
        scores.sort(key=lambda s: s.total, reverse=True)

        # Step 5: Assign to waves
        plan = ActivationPlan(
            briefing_summary=brief[:200],
            detected_topics=topics,
            detected_regions=regions,
            detected_sectors=sectors,
            country=country,
        )

        for score in scores:
            self._assign_wave(score, plan)

        # Enforce caps
        plan.wave_1 = plan.wave_1[:self.MAX_WAVE_1]
        plan.wave_2 = plan.wave_2[:self.MAX_WAVE_2]
        plan.wave_3 = plan.wave_3[:self.MAX_WAVE_3]
        plan.reserve = plan.reserve[:self.MAX_RESERVE]

        logger.info(
            f"Activation plan: {len(plan.wave_1)} immediate, "
            f"{len(plan.wave_2)} secondary, {len(plan.wave_3)} tertiary, "
            f"{len(plan.reserve)} reserve"
        )
        return plan

    def _score_stakeholder(
        self,
        s: Stakeholder,
        topics: list[str],
        regions: list[str],
        sectors: list[str],
        brief_lower: str,
    ) -> RelevanceScore:
        """Score a single stakeholder's relevance to the crisis."""
        score = RelevanceScore(stakeholder_id=s.id)

        # ── Thematic relevance ──────────────────────────────────────
        # Direct position match
        for p in s.positions:
            if p.topic_tag in topics:
                score.thematic = max(score.thematic, 0.8)
                break
            if p.topic_tag == "general_left_right":
                score.thematic = max(score.thematic, 0.1)  # everyone has a political lean

        # Sector match via category/role
        for sector in sectors:
            sector_kw = SECTOR_KEYWORDS.get(sector, [])
            role_lower = f"{s.role} {s.party_or_org} {s.bio}".lower()
            if any(kw in role_lower for kw in sector_kw):
                score.thematic = max(score.thematic, 0.9)
                score.matched_sectors.append(sector)
            # Direct name mention in brief
            if s.name.lower() in brief_lower:
                score.thematic = 1.0

        # Category bonus for natural responders
        category_topic_affinity = {
            "journalist": 0.4,   # journalists cover everything
            "politician": 0.3,   # politicians react to everything
            "union_leader": 0.2 if "labor" in sectors else 0.05,
            "ceo": 0.2 if any(sec in sectors for sec in ["automotive", "energy", "banking", "telecom"]) else 0.05,
            "magistrate": 0.5 if "justice" in sectors else 0.05,
            "academic": 0.2,
            "religious": 0.3 if "religion" in sectors or "immigration" in topics else 0.05,
            "military": 0.5 if "defense" in sectors else 0.05,
        }
        cat_bonus = category_topic_affinity.get(s.category, 0.1)
        score.thematic = max(score.thematic, cat_bonus)

        # ── Geographic relevance ────────────────────────────────────
        if not regions:
            # No specific region → national story, everyone relevant
            score.geographic = 0.5
        else:
            # Check if stakeholder is connected to the crisis region
            role_bio = f"{s.role} {s.bio} {s.party_or_org}".lower()
            for region in regions:
                region_kw = ITALIAN_GEO_KEYWORDS.get(region, [region])
                if any(kw in role_bio for kw in region_kw):
                    score.geographic = 0.9
                    score.matched_regions.append(region)
                    break

            # National figures always somewhat relevant
            if score.geographic < 0.3 and s.tier == 1:
                score.geographic = 0.3
            elif score.geographic < 0.2 and s.influence > 0.5:
                score.geographic = 0.2

        # ── Institutional reach ─────────────────────────────────────
        # Combines influence and tier
        tier_weight = {1: 1.0, 2: 0.6, 3: 0.3}.get(s.tier, 0.3)
        score.institutional = s.influence * tier_weight

        # ── Combine ─────────────────────────────────────────────────
        score.total = (
            self.W_THEMATIC * score.thematic +
            self.W_GEOGRAPHIC * score.geographic +
            self.W_INSTITUTIONAL * score.institutional
        )

        # Build activation reason
        reasons = []
        if score.matched_sectors:
            reasons.append(f"sector:{','.join(score.matched_sectors)}")
        if score.matched_regions:
            reasons.append(f"region:{','.join(score.matched_regions)}")
        if score.thematic >= 0.8:
            reasons.append("topic_expert")
        if score.institutional >= 0.5:
            reasons.append("high_influence")
        score.activation_reason = "; ".join(reasons) or "general_relevance"

        return score

    def _assign_wave(self, score: RelevanceScore, plan: ActivationPlan):
        """Assign a scored stakeholder to the appropriate activation wave."""
        # Override: certain categories are always early responders
        # Local press, local politicians, sector unions → wave 1
        # National media, ministers → wave 2
        # PM, president, international → wave 3

        s = self.db.get(score.stakeholder_id)
        if not s:
            return

        # Force wave 1: local actors with high thematic relevance
        if score.geographic >= 0.8 and score.thematic >= 0.5:
            score.activation_tier = 1
            plan.wave_1.append(score)
            return

        # Force wave 1: sector-specific unions and trade bodies
        if s.category == "union_leader" and score.thematic >= 0.6:
            score.activation_tier = 1
            plan.wave_1.append(score)
            return

        # Force wave 1: journalists (they report first)
        if s.category == "journalist" and score.total >= self.WAVE_2_THRESHOLD:
            score.activation_tier = 1
            plan.wave_1.append(score)
            return

        # Heads of state → always wave 3 (they don't react to everything)
        if s.tier == 1 and s.influence >= 0.7 and s.category == "politician":
            role_lower = s.role.lower()
            if any(kw in role_lower for kw in [
                "presidente del consiglio", "premier", "president", "pm",
                "kanzler", "chancellor", "primer ministro",
                "presidente della repubblica",
            ]):
                score.activation_tier = 3
                plan.wave_3.append(score)
                return

        # Score-based assignment
        if score.total >= self.WAVE_1_THRESHOLD:
            score.activation_tier = 1
            plan.wave_1.append(score)
        elif score.total >= self.WAVE_2_THRESHOLD:
            score.activation_tier = 2
            plan.wave_2.append(score)
        elif score.total >= self.WAVE_3_THRESHOLD:
            score.activation_tier = 3
            plan.wave_3.append(score)
        else:
            plan.reserve.append(score)

    def _detect_topics(self, brief_lower: str) -> list[str]:
        """Extract topic_tags from brief text via keyword matching."""
        from stakeholder_graph.integration import infer_topic_tags
        return infer_topic_tags(brief_lower, "")

    def _detect_regions(self, brief_lower: str) -> list[str]:
        """Detect geographic regions mentioned in the brief."""
        found = []
        for region, keywords in ITALIAN_GEO_KEYWORDS.items():
            if any(kw in brief_lower for kw in keywords):
                found.append(region)
        return found

    def _detect_sectors(self, brief_lower: str) -> list[str]:
        """Detect industry sectors mentioned in the brief."""
        found = []
        for sector, keywords in SECTOR_KEYWORDS.items():
            for kw in keywords:
                # Use word-boundary matching for short keywords (< 4 chars)
                if len(kw) < 4:
                    if re.search(rf"\b{re.escape(kw)}\b", brief_lower):
                        found.append(sector)
                        break
                elif kw in brief_lower:
                    found.append(sector)
                    break
        return found
