"""AssetMgmtTwin — minimal twin for asset managers / fund companies.

Models the core revenue + AUM dynamics that drive any asset-management
scenario:

  - **AUM (assets under management)** — moves with market returns + flows
  - **Fee revenue** = AUM × fee rate (mgmt fee + perf fee)
  - **Net flows** = inflows - redemptions (sensitive to performance + reputation)
  - **Market beta** of the book — exposure to equity/bond returns
  - **Cost/income ratio** — operational efficiency

Coupling: opinion → redemption surge (negative reputation), market shock
→ AUM mark-to-market drop, performance feeds back into next-round flows.

Defaults: European mid-size active manager (Generali Investments / Amundi
Italy / Eurizon profile). Override per scenario.

Status: v0.1, minimal — sufficient as scaffold, not yet wired into the
pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def default_eu_asset_mgr_params() -> dict:
    """Reference parameters for a EU mid-size active asset manager."""
    return {
        # Initial state (normalised; absolute scale via scale_eur_bn)
        "scale_eur_bn": 50.0,                # AUM scale
        "aum": 1.00,                          # normalised AUM
        "equity_share": 0.40,                 # share of AUM in equity
        "bond_share": 0.45,
        "alt_share": 0.15,
        # Fees (basis points on AUM, annual)
        "mgmt_fee_bps": 65.0,                 # 65bp baseline (active EU avg)
        "perf_fee_share": 0.02,               # 2% of positive performance
        # Cost
        "fixed_cost_per_round": 0.0008,       # normalised
        # Sensitivities
        "redemption_to_opinion_coef": 0.06,   # opinion drag on flows
        "inflow_to_perf_coef": 0.40,          # 1pp positive perf → +0.4pp inflow
        "market_beta": 0.80,                  # avg beta of equity sleeve
        # Caps
        "redemption_max_per_round": 0.05,     # 5% AUM/round physiological max
        # Behaviour
        "perf_lookback_months": 3,
    }


@dataclass(frozen=True)
class AssetMgmtState:
    round: int
    aum: float
    fee_revenue_round: float
    net_flow_pct: float           # net flow this round, % of AUM
    redemption_round_pct: float
    cost_income_ratio_pct: float
    market_return_pct: float      # equity market return this round
    breaches: tuple = field(default_factory=tuple)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["breaches"] = list(self.breaches)
        return d

    def to_compact_str(self) -> str:
        return (
            f"AUM {self.aum:.3f} | net_flow {self.net_flow_pct:+.2f}% | "
            f"red {self.redemption_round_pct*100:.1f}%/r | "
            f"C/I {self.cost_income_ratio_pct:.0f}% | "
            f"mkt {self.market_return_pct:+.2f}%"
        )


class AssetMgmtTwin:
    """Stateful asset-mgmt twin.

    Use:
        twin = AssetMgmtTwin()
        s = twin.step(round_num=1, opinion_aggregate=-0.2,
                      market_return_pct=-3.0, narrative="market drawdown")
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = default_eu_asset_mgr_params()
        if params:
            self.params.update(params)
        self.history: list[AssetMgmtState] = []
        self.history.append(self._build_baseline_state())

    def current_state(self) -> AssetMgmtState:
        return self.history[-1]

    def step(
        self,
        round_num: int,
        opinion_aggregate: float = 0.0,
        market_return_pct: float = 0.0,
        polarization: float = 5.0,
        narrative: str = "",
    ) -> AssetMgmtState:
        prev = self.current_state()
        p = self.params

        # 1. Mark-to-market: AUM moves with weighted market return
        mkt_effect_pct = (
            p["equity_share"] * market_return_pct * p["market_beta"]
            + p["bond_share"] * (market_return_pct * 0.20)  # bonds less correlated
            + p["alt_share"] * (market_return_pct * 0.30)
        )

        # 2. Net flows
        # Inflows correlate with recent performance; outflows with bad opinion
        perf_inflow_pct = mkt_effect_pct * p["inflow_to_perf_coef"] / 100.0  # convert % to fraction
        opinion_outflow_pct = max(0.0, -opinion_aggregate) * p["redemption_to_opinion_coef"]
        # Polarization amplifier (panic redemption)
        if polarization > 7.0 and opinion_aggregate < -0.5:
            opinion_outflow_pct += 0.02
        red_pct = min(p["redemption_max_per_round"] + 0.02, opinion_outflow_pct)
        net_flow_pct = perf_inflow_pct - red_pct

        # 3. New AUM
        new_aum = prev.aum * (1 + mkt_effect_pct / 100.0) * (1 + net_flow_pct)

        # 4. Fees this round (per-round = annual / 12 if rounds are months)
        mgmt_fee = new_aum * (p["mgmt_fee_bps"] / 10000.0) / 12.0
        perf_fee = max(0.0, mkt_effect_pct / 100.0) * new_aum * p["perf_fee_share"]
        fee_revenue = mgmt_fee + perf_fee

        # 5. Cost/income
        cost = p["fixed_cost_per_round"]
        ci_pct = 100.0 * cost / max(fee_revenue, 1e-6) if fee_revenue > 0 else 999.0

        # 6. Breach detection
        breaches = []
        if red_pct > p["redemption_max_per_round"] + 0.01:
            breaches.append("REDEMPTION_SURGE")
        if ci_pct > 100.0:
            breaches.append("LOSS")

        new_state = AssetMgmtState(
            round=round_num,
            aum=round(new_aum, 4),
            fee_revenue_round=round(fee_revenue, 5),
            net_flow_pct=round(net_flow_pct * 100, 3),
            redemption_round_pct=round(red_pct, 4),
            cost_income_ratio_pct=round(ci_pct, 1),
            market_return_pct=round(market_return_pct, 2),
            breaches=tuple(breaches),
            notes=narrative[:160] if narrative else "",
        )
        self.history.append(new_state)
        return new_state

    def _build_baseline_state(self) -> AssetMgmtState:
        p = self.params
        baseline_fee = p["aum"] * (p["mgmt_fee_bps"] / 10000.0) / 12.0
        ci = 100.0 * p["fixed_cost_per_round"] / max(baseline_fee, 1e-6)
        return AssetMgmtState(
            round=0,
            aum=p["aum"],
            fee_revenue_round=round(baseline_fee, 5),
            net_flow_pct=0.0,
            redemption_round_pct=0.0,
            cost_income_ratio_pct=round(ci, 1),
            market_return_pct=0.0,
            breaches=(),
            notes="baseline",
        )
