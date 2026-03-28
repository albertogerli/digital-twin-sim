"""Pydantic models for scenario configuration."""

from pydantic import BaseModel, Field
from typing import Any, Literal, Optional


class AxisConfig(BaseModel):
    """Defines what the [-1, +1] position axis means in this domain."""
    negative_label: str  # e.g., "Against reform" or "Brand rejection"
    positive_label: str  # e.g., "Pro reform" or "Brand adoption"
    neutral_label: str = "Neutral"


class ChannelConfig(BaseModel):
    """A communication channel in the simulation."""
    id: str                  # e.g., "social", "review_site", "xsim"
    description: str = ""
    max_length: int = 280
    channel_type: str = "short_form"  # "short_form", "long_form", "official"


class AgentSpec(BaseModel):
    """Specification for creating an agent."""
    id: str
    name: str
    role: str
    archetype: str
    position: float = Field(ge=-1.0, le=1.0)
    influence: float = Field(default=0.5, ge=0.0, le=1.0)
    rigidity: float = Field(default=0.5, ge=0.0, le=1.0)
    tier: Literal[1, 2, 3] = 1
    bio: str = ""
    communication_style: str = ""
    key_traits: list[str] = Field(default_factory=list)
    key_trait: str = ""  # For institutional agents
    category: str = ""
    domain_attributes: dict[str, Any] = Field(default_factory=dict)
    platform_primary: str = ""
    platform_secondary: str = ""


class ClusterSpec(BaseModel):
    """Specification for creating a citizen cluster."""
    id: str
    name: str
    description: str = ""
    size: int = 1000
    position: float = Field(default=0.0, ge=-1.0, le=1.0)
    engagement_base: float = Field(default=0.5, ge=0.0, le=1.0)
    info_channel: str = ""
    demographic_attributes: dict[str, Any] = Field(default_factory=dict)


class ScenarioConfig(BaseModel):
    """Complete configuration for a simulation scenario."""
    name: str
    description: str
    domain: str  # "political", "commercial", "marketing", etc.
    language: str = "en"
    num_rounds: int = Field(default=9, ge=1, le=50)
    timeline_unit: str = "month"  # "day", "week", "month", "quarter"
    timeline_labels: list[str] = Field(default_factory=list)
    position_axis: AxisConfig
    channels: list[ChannelConfig] = Field(default_factory=list)
    elite_agents: list[AgentSpec] = Field(default_factory=list)
    institutional_agents: list[AgentSpec] = Field(default_factory=list)
    citizen_clusters: list[ClusterSpec] = Field(default_factory=list)
    initial_event: str = ""
    scenario_context: str = ""
    metrics_to_track: list[str] = Field(default_factory=list)
    budget_usd: float = 5.0

    # Optional: seed data path for grounding in real-world data
    seed_data_path: str = ""

    # Optional: pre-defined events per round (round 1 is always initial_event)
    round_events: dict[int, dict] = Field(default_factory=dict)
