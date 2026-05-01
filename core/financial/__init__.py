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
from .country_params import (
    default_dutch_bank_params,
    default_french_bank_params,
    default_german_bank_params,
    default_italian_bank_params_v2,
    default_spanish_bank_params,
    default_uk_bank_params,
    default_us_bank_params,
    select_country_params,
    supported_countries,
)
from .market_data import (
    fetch_all_anchors,
    fetch_country_anchors,
    get_boe_bank_rate_pct,
    get_btp_bund_spread_bps,
    get_ecb_dfr_pct,
    get_euribor_3m_pct,
    get_fed_funds_pct,
    get_uk_10y_gilt_pct,
    get_us_10y_treasury_pct,
)
from .rates import (
    CIRRateProcess,
    EBAStressTemplate,
    eba_adverse_2025_template,
    eba_baseline_2025_template,
    get_stress_template,
    list_stress_templates,
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
    "CIRRateProcess",
    "EBAStressTemplate",
    "eba_adverse_2025_template",
    "eba_baseline_2025_template",
    "get_stress_template",
    "list_stress_templates",
    "fetch_all_anchors",
    "get_btp_bund_spread_bps",
    "get_ecb_dfr_pct",
    "get_euribor_3m_pct",
    "default_italian_bank_params_v2",
    "default_german_bank_params",
    "default_french_bank_params",
    "default_spanish_bank_params",
    "default_dutch_bank_params",
    "default_us_bank_params",
    "default_uk_bank_params",
    "select_country_params",
    "supported_countries",
    "fetch_country_anchors",
    "get_fed_funds_pct",
    "get_us_10y_treasury_pct",
    "get_boe_bank_rate_pct",
    "get_uk_10y_gilt_pct",
]
