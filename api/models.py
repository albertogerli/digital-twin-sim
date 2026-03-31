"""Pydantic models for the API."""

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel


class SimulationRequest(BaseModel):
    brief: str
    provider: str = "gemini"
    model: Optional[str] = None
    domain: Optional[str] = None
    rounds: Optional[int] = None
    budget: float = 5.0
    elite_only: bool = False
    monte_carlo: bool = False
    monte_carlo_runs: int = 10
    monte_carlo_perturbation: float = 0.15
    online_mode: bool = False  # If True, enables EnKF data assimilation


class SimulationStatus(BaseModel):
    id: str
    status: Literal[
        "queued", "analyzing", "configuring", "running",
        "exporting", "completed", "failed", "cancelled"
    ]
    brief: str
    scenario_name: Optional[str] = None
    scenario_id: Optional[str] = None
    domain: Optional[str] = None
    current_round: int = 0
    total_rounds: int = 0
    cost: float = 0.0
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    agents_count: int = 0


class BranchRequest(BaseModel):
    """What-If branch: fork a completed scenario from a specific round."""
    parent_scenario_id: str
    branch_round: int                           # which round to branch from (1..N)
    what_if: str = ""                           # free-text "what if" description
    event_override: Optional[str] = None        # force a specific event text
    shock_override: Optional[float] = None      # override shock magnitude (0-1)
    agent_overrides: dict = {}                  # {agent_id: {position: 0.5, ...}}
    rounds_to_run: Optional[int] = None         # how many rounds from branch point (default: remaining)
    provider: str = "gemini"
    model: Optional[str] = None
    budget: float = 5.0


class ProgressEvent(BaseModel):
    type: str
    message: str = ""
    round: Optional[int] = None
    phase: Optional[str] = None
    data: dict = {}
    confidence_interval: Optional[dict] = None
    regime_info: Optional[dict] = None


class ObservationInput(BaseModel):
    """Observation for EnKF online data assimilation."""
    type: Literal["polling", "sentiment", "official_result"]
    pro_pct: float
    sample_size: Optional[int] = None
    confidence: Optional[float] = None
    round: Optional[int] = None
