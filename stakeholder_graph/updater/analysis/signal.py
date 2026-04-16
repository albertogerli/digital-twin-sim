"""Signal types — structured output from LLM article analysis."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class PositionSignal(BaseModel):
    """A signal that a stakeholder's position on a topic may have shifted."""
    stakeholder_id: str
    topic_tag: str
    direction: float = Field(ge=-1.0, le=1.0)  # what the article suggests
    strength: Literal["strong", "moderate", "weak"] = "moderate"
    evidence: str = ""          # quote or paraphrase from article
    source_url: str = ""
    source_name: str = ""
    published: Optional[datetime] = None
    is_new_topic: bool = False  # true if stakeholder had no prior position here


class QuoteSignal(BaseModel):
    """An exact quote from a stakeholder found in an article."""
    stakeholder_id: str
    quote: str
    topic_tag: str = ""
    source_url: str = ""
    source_name: str = ""
    published: Optional[datetime] = None


class InfluenceSignal(BaseModel):
    """A signal that a stakeholder's influence/visibility has changed."""
    stakeholder_id: str
    direction: Literal["up", "down", "stable"] = "stable"
    magnitude: float = Field(default=0.0, ge=0.0, le=1.0)  # how big the shift
    reason: str = ""
    source_url: str = ""


class RelationshipSignal(BaseModel):
    """A signal about a relationship between two stakeholders."""
    source_id: str
    target_name: str  # raw name from article, resolved later
    relation_type: Literal["ally", "rival", "coalition", "opposition", "neutral"] = "neutral"
    evidence: str = ""
    source_url: str = ""


class ArticleAnalysis(BaseModel):
    """Complete LLM analysis of one article for one stakeholder."""
    relevant: bool = False
    position_signals: list[PositionSignal] = Field(default_factory=list)
    quotes: list[QuoteSignal] = Field(default_factory=list)
    influence_signal: Optional[InfluenceSignal] = None
    relationship_signals: list[RelationshipSignal] = Field(default_factory=list)
