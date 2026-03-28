"""Domain-agnostic base agent class for all three tiers."""

from dataclasses import dataclass, field
from typing import Any

from .agent_memory import AgentMemory


@dataclass
class BaseAgent:
    """Base class for all agent types. Position is on a [-1, +1] axis
    whose meaning is defined by the domain plugin."""
    id: str
    name: str
    role: str
    archetype: str
    position: float  # -1 to +1, meaning defined by domain
    original_position: float = 0.0
    influence: float = 0.5
    rigidity: float = 0.5
    tier: int = 1
    memory: AgentMemory = field(default_factory=AgentMemory)
    system_prompt: str = ""
    emotional_state: str = "neutral"
    engagement_level: float = 0.5
    tolerance: float = 0.4  # Bounded confidence threshold
    domain_attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.original_position == 0.0 and self.position != 0.0:
            self.original_position = self.position
        if not self.tolerance:
            self.tolerance = 0.3 + (1.0 - self.rigidity) * 0.4

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "archetype": self.archetype,
            "position": self.position,
            "original_position": self.original_position,
            "influence": self.influence,
            "rigidity": self.rigidity,
            "tier": self.tier,
            "emotional_state": self.emotional_state,
            "engagement_level": self.engagement_level,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaseAgent":
        memory = AgentMemory()
        return cls(
            id=data["id"],
            name=data["name"],
            role=data["role"],
            archetype=data.get("archetype", "unknown"),
            position=data["position"],
            original_position=data.get("original_position", data["position"]),
            influence=data.get("influence", 0.5),
            rigidity=data.get("rigidity", 0.5),
            tier=data.get("tier", 1),
            memory=memory,
            system_prompt=data.get("system_prompt", ""),
            emotional_state=data.get("emotional_state", "neutral"),
            engagement_level=data.get("engagement_level", 0.5),
        )
