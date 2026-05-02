"""Pydantic models for the API."""

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


# Upload constraints
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per file
MAX_UPLOAD_FILES = 5


class SimulationRequest(BaseModel):
    brief: str = Field(..., max_length=10000)
    provider: str = "gemini"
    model: Optional[str] = None
    domain: Optional[str] = None
    rounds: Optional[int] = Field(default=None, ge=1, le=30)
    budget: float = Field(default=5.0, ge=0.1, le=50.0)
    elite_only: bool = False
    monte_carlo: bool = False
    monte_carlo_runs: int = Field(default=10, ge=1, le=100)
    monte_carlo_perturbation: float = Field(default=0.15, ge=0.0, le=1.0)
    online_mode: bool = False  # If True, enables EnKF data assimilation
    wargame_mode: bool = False  # If True, pauses after each round for human intervention
    player_role: str = Field(default="", max_length=200)
    metrics_to_track: list[str] = []  # User-selected quantitative KPIs to monitor


class KBInjectDoc(BaseModel):
    """Document to be injected mid-simulation into the agent KB."""
    title: str = Field(..., max_length=200)        # Display name (filename or URL slug)
    text: str = Field(..., max_length=200_000)     # Raw text body to chunk + embed
    source: str = Field(default="wargame_inject", max_length=64)  # provenance


class WargameIntervention(BaseModel):
    """Human player's counter-move during a wargame simulation."""
    action_text: str = Field(..., max_length=5000)
    action_type: str = "press_release"         # press_release, internal_memo, social_post, policy_announcement, inject_kb
    shock_magnitude: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    target_audience: str = Field(default="", max_length=500)
    skip: bool = False                         # Skip intervention, let simulation auto-generate
    kb_doc: Optional[KBInjectDoc] = None       # If action_type == "inject_kb", payload to ingest into RAG store


class SimulationStatus(BaseModel):
    id: str
    status: Literal[
        "queued", "analyzing", "configuring", "running",
        "awaiting_player",  # Wargame: paused, waiting for human intervention
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
