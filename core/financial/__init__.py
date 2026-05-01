"""Financial domain — ALM/Solvency/Asset-Mgmt twins for digital-twin sims.

Three sub-namespaces, each a stateful engine that steps in lockstep with
the opinion simulation:

- `core.financial.banking` — bank balance sheet (NIM, CET1, LCR, deposit
  β, loan elasticity, hedging P&L). Default IT/DE/FR/ES/NL/US/UK with
  live ECB/FRED/BoE refresh. Auto-attached to round_manager when
  domain_id == "financial".
- `core.financial.insurance` — insurer P&C / Life (combined ratio,
  Solvency II ratio, lapse, technical provisions). v0.1 minimal.
- `core.financial.asset_mgmt` — asset manager (AUM, fee revenue, net
  flows, market beta). v0.1 minimal.

Backward-compat: top-level imports (`from core.financial import
FinancialTwin`) keep working for the banking case. New code should
prefer the sub-namespace paths for clarity.
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
