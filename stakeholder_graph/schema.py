"""Pydantic models for the Global Stakeholder Graph."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import date


class Position(BaseModel):
    """A stakeholder's stance on a specific topic."""
    topic_tag: str                      # e.g. "judiciary_reform", "eu_integration"
    value: float = Field(ge=-1.0, le=1.0)
    source: Literal[
        "voting_record", "public_statement", "institutional_statement",
        "interview", "social_media", "party_line", "inferred",
    ] = "public_statement"
    confidence: Literal["high", "medium", "low"] = "high"
    date_observed: Optional[str] = None  # ISO date or "2024-Q1"
    quotes: list[str] = Field(default_factory=list)
    notes: str = ""


class Relationship(BaseModel):
    """Directed relationship between two stakeholders."""
    target_id: str
    type: Literal[
        "ally", "rival", "member_of", "leads", "reports_to",
        "coalition", "opposition", "mentor", "protege",
    ] = "ally"
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    context: str = ""  # e.g. "same coalition since 2022"


class Stakeholder(BaseModel):
    """A real-world public figure in the stakeholder graph."""
    id: str                             # unique slug: "giorgia_meloni"
    name: str                           # "Giorgia Meloni"
    role: str                           # "Presidente del Consiglio"
    country: str = "IT"                 # ISO 3166-1 alpha-2
    category: Literal[
        "politician", "journalist", "union_leader", "ceo",
        "magistrate", "academic", "activist", "military",
        "religious", "institutional", "diplomat",
        # Sprint 9+: finance domain categories
        "industry_association", "consumer_advocacy", "regulator",
        "central_bank", "competitor_bank", "ceo_business",
        "government",
    ] = "politician"
    party_or_org: str = ""              # "Fratelli d'Italia", "CGIL", "ENI"
    archetype: str = ""                 # maps to domain archetypes
    tier: Literal[1, 2, 3] = 1         # 1=elite, 2=institutional, 3=minor
    influence: float = Field(default=0.5, ge=0.0, le=1.0)
    rigidity: float = Field(default=0.5, ge=0.0, le=1.0)
    bio: str = ""
    communication_style: str = ""
    key_traits: list[str] = Field(default_factory=list)
    platform_primary: str = ""          # "twitter", "tv", "parliament"
    platform_secondary: str = ""

    # Topic-dependent positions
    positions: list[Position] = Field(default_factory=list)

    # Graph edges
    relationships: list[Relationship] = Field(default_factory=list)

    # Financial links (optional — used by FinancialImpactScorer)
    affiliated_tickers: list[str] = Field(default_factory=list)  # e.g. ["ENI.MI", "ENI"]

    # Metadata
    active: bool = True
    last_updated: str = ""              # ISO date
    data_sources: list[str] = Field(default_factory=list)  # "openparlamento", "wikidata", etc.
    wikidata_qid: str = ""              # Q-identifier for linking

    def get_position(self, topic_tag: str, default: float = 0.0) -> float:
        """Get position on a specific topic, falling back to default."""
        for p in self.positions:
            if p.topic_tag == topic_tag:
                return p.value
        # Try party_line positions
        for p in self.positions:
            if p.topic_tag == "general_left_right":
                return p.value
        return default

    def to_agent_spec(self, topic_tag: str = "") -> dict:
        """Convert to AgentSpec-compatible dict for the simulator."""
        import re
        position = self.get_position(topic_tag) if topic_tag else 0.0
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "archetype": self.archetype or self.category,
            "position": max(-1.0, min(1.0, position)),
            "influence": self.influence,
            "rigidity": self.rigidity,
            "tier": self.tier,
            "bio": self.bio,
            "communication_style": self.communication_style,
            "key_traits": self.key_traits,
            "platform_primary": self.platform_primary,
            "platform_secondary": self.platform_secondary,
            "_stakeholder_graph": True,
            "_country": self.country,
            "_party": self.party_or_org,
            "_data_sources": self.data_sources,
        }

    def to_seed_format(self, topic_tag: str = "") -> dict:
        """Convert to VerifiedStakeholder-compatible dict for seed_data."""
        pos = next((p for p in self.positions if p.topic_tag == topic_tag), None)
        return {
            "name": self.name,
            "role": self.role,
            "known_position": pos.value if pos else 0.0,
            "position_source": pos.source if pos else "inferred",
            "bio_verified": self.bio,
            "key_quotes": pos.quotes if pos else [],
            "archetype": self.archetype or self.category,
            "communication_style": self.communication_style,
            "influence": self.influence,
            "rigidity": self.rigidity,
        }
