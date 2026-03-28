"""Pydantic models for verified seed data."""

from pydantic import BaseModel, Field


class VerifiedStakeholder(BaseModel):
    """A real-world stakeholder with verified data for grounding agent generation."""
    name: str
    role: str
    known_position: float = Field(ge=-1.0, le=1.0)
    position_source: str = ""  # "polling", "public_statement", "voting_record"
    bio_verified: str = ""
    key_quotes: list[str] = Field(default_factory=list)
    archetype: str = ""  # "politician", "journalist", etc.
    communication_style: str = ""
    influence: float = Field(default=0.5, ge=0.0, le=1.0)
    rigidity: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: str = "high"  # "high" (manual/verified), "medium" (entity_research), "low" (llm_estimate)


class VerifiedDemographic(BaseModel):
    """A real-world demographic segment with verified data."""
    name: str
    description: str = ""
    population_share: float = Field(default=0.1, ge=0.0, le=1.0)
    known_position: float = Field(default=0.0, ge=-1.0, le=1.0)
    position_source: str = ""
    key_concerns: list[str] = Field(default_factory=list)
    info_channel: str = ""
    demographic_attributes: dict = Field(default_factory=dict)
    confidence: str = "high"  # "high" (manual/verified), "medium" (entity_research), "low" (llm_estimate)


class SeedDataBundle(BaseModel):
    """Complete seed data for grounding a simulation scenario."""
    context_text: str = ""          # Injected into scenario_context
    stakeholders: list[VerifiedStakeholder] = Field(default_factory=list)
    demographics: list[VerifiedDemographic] = Field(default_factory=list)
    historical_text: str = ""       # Injected into event generation
    known_events: list[dict] = Field(default_factory=list)  # Real events to anchor timeline

    def format_stakeholders_for_prompt(self) -> str:
        """Format stakeholders as prompt text for the briefing analyzer."""
        if not self.stakeholders:
            return ""
        lines = ["VERIFIED REAL-WORLD STAKEHOLDERS (use these as agents — do NOT invent alternatives):"]
        for s in self.stakeholders:
            pos_label = "PRO" if s.known_position > 0.2 else ("AGAINST" if s.known_position < -0.2 else "NEUTRAL")
            lines.append(
                f"- {s.name} ({s.role}): position {s.known_position:+.2f} ({pos_label}), "
                f"source: {s.position_source}"
            )
            if s.bio_verified:
                lines.append(f"  Bio: {s.bio_verified}")
            if s.key_quotes:
                lines.append(f'  Quote: "{s.key_quotes[0]}"')
        return "\n".join(lines)

    def format_demographics_for_prompt(self) -> str:
        """Format demographics as prompt text."""
        if not self.demographics:
            return ""
        lines = ["VERIFIED DEMOGRAPHIC SEGMENTS (use these as citizen clusters):"]
        for d in self.demographics:
            lines.append(
                f"- {d.name}: position {d.known_position:+.2f}, "
                f"{d.description}"
            )
            if d.key_concerns:
                lines.append(f"  Concerns: {', '.join(d.key_concerns)}")
        return "\n".join(lines)

    def format_historical_context(self) -> str:
        """Format historical context for event generation."""
        parts = []
        if self.historical_text:
            parts.append(f"HISTORICAL PRECEDENTS:\n{self.historical_text}")
        if self.known_events:
            parts.append("KNOWN REAL-WORLD EVENTS (use as anchors):")
            for evt in self.known_events:
                parts.append(f"- [{evt.get('date', '?')}] {evt.get('description', '')}")
        return "\n\n".join(parts) if parts else ""
