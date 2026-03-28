"""Tier 3: Citizen clusters — demographic segments simulated as aggregates."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..llm.base_client import BaseLLMClient
from ..llm.json_parser import JSONParseError

logger = logging.getLogger(__name__)


@dataclass
class CitizenCluster:
    """A demographic cluster representing a population segment."""
    id: str
    name: str
    description: str
    size: int = 1000
    info_channel: str = ""
    position: float = 0.0
    original_position: float = 0.0
    engagement_level: float = 0.5
    trust_institutions: float = 0.5
    dominant_sentiment: str = "indifferent"
    emergent_narrative: str = ""
    sentiment_distribution: dict = field(default_factory=lambda: {
        "positive": 25, "negative": 25, "indifferent": 25, "confused": 25
    })
    key_concerns: list[str] = field(default_factory=list)
    round_history: list[dict] = field(default_factory=list)
    demographic_attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.original_position == 0.0 and self.position != 0.0:
            self.original_position = self.position

    def get_description(self) -> str:
        parts = [self.description]
        for key, val in self.demographic_attributes.items():
            if val:
                parts.append(f"{key.replace('_', ' ').title()}: {val}")
        return "\n".join(parts)

    def get_previous_state(self) -> str:
        if not self.round_history:
            return f"Initial position: {self.position:+.2f}. No previous data."
        last = self.round_history[-1]
        return (
            f"Position last period: {last.get('position', self.position):+.2f}\n"
            f"Dominant sentiment: {last.get('dominant_sentiment', 'indifferent')}\n"
            f"Engagement: {last.get('engagement_level', 0.5):.1f}\n"
            f"Emerging narrative: {last.get('emergent_narrative', 'none')}\n"
            f"Concerns: {', '.join(last.get('key_concerns', []))}"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "info_channel": self.info_channel,
            "position": self.position,
            "original_position": self.original_position,
            "engagement_level": self.engagement_level,
            "trust_institutions": self.trust_institutions,
            "dominant_sentiment": self.dominant_sentiment,
            "sentiment_distribution": self.sentiment_distribution,
            "emergent_narrative": self.emergent_narrative,
            "key_concerns": self.key_concerns,
        }

    @classmethod
    def from_spec(cls, spec) -> "CitizenCluster":
        if hasattr(spec, 'model_dump'):
            d = spec.model_dump()
        else:
            d = spec
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            size=d.get("size", 1000),
            info_channel=d.get("info_channel", ""),
            position=d.get("position", 0.0),
            original_position=d.get("position", 0.0),
            engagement_level=d.get("engagement_base", 0.5),
            demographic_attributes=d.get("demographic_attributes", {}),
        )
