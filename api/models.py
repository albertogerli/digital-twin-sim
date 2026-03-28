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


class ProgressEvent(BaseModel):
    type: str
    message: str = ""
    round: Optional[int] = None
    phase: Optional[str] = None
    data: dict = {}
