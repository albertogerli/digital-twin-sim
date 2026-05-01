"""Banking sub-module — namespace alias for the canonical FinancialTwin
and ALM primitives.

History: until v0.6 these lived directly under `core/financial/`. As the
financial domain expanded to insurance and asset management (v0.7), we
introduced sub-namespaces so that imports read like
`core.financial.banking.FinancialTwin` for clarity.

This is a **re-export only**. The actual implementations stay where they
are (no file moves), so existing code that imports from `core.financial`
keeps working unchanged. New code can use either namespace.
"""

from ..country_params import (
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
from ..exposure import (
    aggregate_opinion_by_exposure,
    default_exposure,
    infer_financial_exposure,
)
from ..market_data import (
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
from ..rates import (
    CIRRateProcess,
    EBAStressTemplate,
    eba_adverse_2025_template,
    eba_baseline_2025_template,
    get_stress_template,
    list_stress_templates,
)
from ..twin import (
    FeedbackSignals,
    FinancialState,
    FinancialTwin,
    default_italian_bank_params,
)

__all__ = [
    "FinancialTwin", "FinancialState", "FeedbackSignals",
    "default_italian_bank_params",
    "default_italian_bank_params_v2",
    "default_german_bank_params",
    "default_french_bank_params",
    "default_spanish_bank_params",
    "default_dutch_bank_params",
    "default_us_bank_params",
    "default_uk_bank_params",
    "select_country_params", "supported_countries",
    "infer_financial_exposure", "default_exposure",
    "aggregate_opinion_by_exposure",
    "CIRRateProcess", "EBAStressTemplate",
    "eba_baseline_2025_template", "eba_adverse_2025_template",
    "get_stress_template", "list_stress_templates",
    "fetch_all_anchors", "fetch_country_anchors",
    "get_euribor_3m_pct", "get_ecb_dfr_pct", "get_btp_bund_spread_bps",
    "get_fed_funds_pct", "get_us_10y_treasury_pct",
    "get_boe_bank_rate_pct", "get_uk_10y_gilt_pct",
]
