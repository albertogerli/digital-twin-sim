"""Insurance sub-module — InsuranceTwin (Solvency II flavour).

Sister architecture to banking.FinancialTwin: stateful engine that steps
P&C / Life insurance KPIs in lockstep with the opinion sim. Defaults
calibrated to European insurance market 2024-2025 (EIOPA stats).

Status: v0.1 — minimal viable. Provides:
  - InsuranceState snapshot (combined ratio, premium income, claims paid,
    technical provisions, Solvency II ratio)
  - InsuranceTwin engine (stateful, step per round)
  - default_eu_insurer_params() with EIOPA reference values

Coupling with opinion layer: same pattern as banking — state exposed via
context, FeedbackSignals back. NOT yet wired into round_manager (only
banking is auto-attached today). Use directly in custom scenarios.
"""

from .twin import (
    InsuranceState,
    InsuranceTwin,
    default_eu_insurer_params,
)

__all__ = ["InsuranceState", "InsuranceTwin", "default_eu_insurer_params"]
