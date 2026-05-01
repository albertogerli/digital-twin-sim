"""Asset Management sub-module — AssetMgmtTwin.

Models AUM (assets under management), fee revenue, market beta, and
redemption rate for an asset manager / fund company. Sister design to
banking.FinancialTwin and insurance.InsuranceTwin.

Status: v0.1 minimal — provides the shape, defaults, and step()
mechanics. Not yet wired into round_manager (banking only).
"""

from .twin import (
    AssetMgmtState,
    AssetMgmtTwin,
    default_eu_asset_mgr_params,
)

__all__ = ["AssetMgmtState", "AssetMgmtTwin", "default_eu_asset_mgr_params"]
