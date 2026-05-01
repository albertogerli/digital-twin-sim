"""Financial Twin layer — ALM (asset-liability management) state for the
banking domain. Distinct from core.orchestrator.financial_impact, which
handles equity-market side (tickers, pair trades, betas).

This module provides:
- FinancialState: immutable snapshot of a bank's balance sheet at a round
- FinancialTwin: stateful engine that steps the balance sheet given an
  event + opinion aggregate, applying ALM constraints (deposit beta,
  loan elasticity, NIM compression, regulatory floors).
- default_italian_bank_params(): literature-grounded defaults derived from
  ECB / EBA / Banca d'Italia 2025 benchmarks. Override per scenario.

Design: weak coupling — the twin runs in lockstep with the opinion
simulation but does not directly modify agent positions. Agents read the
current state via context strings; in v0.6 they also receive light
balance-sheet snippets (financial_state field).
"""

from .exposure import (
    aggregate_opinion_by_exposure,
    default_exposure,
    infer_financial_exposure,
)
from .twin import (
    FeedbackSignals,
    FinancialState,
    FinancialTwin,
    default_italian_bank_params,
)

__all__ = [
    "FeedbackSignals",
    "FinancialState",
    "FinancialTwin",
    "default_italian_bank_params",
    "aggregate_opinion_by_exposure",
    "default_exposure",
    "infer_financial_exposure",
]
