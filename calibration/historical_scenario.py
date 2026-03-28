"""Ground truth data models for historical calibration scenarios."""

from pydantic import BaseModel, Field


class PollingDataPoint(BaseModel):
    """A single polling data point at a specific time."""
    round_equivalent: int  # Which simulation round this maps to
    pro_pct: float         # % supporting the policy
    against_pct: float     # % opposing
    undecided_pct: float = 0.0


class GroundTruth(BaseModel):
    """Complete ground truth for a historical scenario."""
    scenario_name: str
    description: str = ""
    final_outcome_pro_pct: float   # Actual result: % who voted YES/PRO
    final_outcome_against_pct: float
    final_turnout_pct: float = 0.0
    polling_trajectory: list[PollingDataPoint] = Field(default_factory=list)
    key_events: list[dict] = Field(default_factory=list)  # Real events that happened
    calibration_notes: str = ""
