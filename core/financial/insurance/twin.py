"""InsuranceTwin — minimal Solvency II-flavour twin for P&C / Life insurers.

Models the core balance sheet + technical metrics that show up in any
insurance scenario:

  - **Combined ratio** = (claims + expenses) / premium earned. <100 = profit.
    EU P&C avg 94-96% (EIOPA 2024). Life uses different metrics (lapse,
    surplus) but combined ratio still meaningful for non-life book.
  - **Solvency II ratio** = own funds / SCR (Solvency Capital Requirement).
    Regulatory floor 100%, EU avg ~225% (EIOPA Q4 2024).
  - **Premium written** trend
  - **Claims paid** + technical provisions
  - **Lapse rate** (Life specific) — how many policyholders surrender

Coupling: opinion → premium growth (negative reputation = lower new
business), opinion → lapse acceleration, big claims event → ratio shock.

Defaults: EU non-life insurer mid-size (Generali / Allianz Italia /
Mapfre profile). Override per scenario.

Status: v0.1, minimal — sufficient for demo, not yet wired into the
round_manager pipeline. Use programmatically.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Default params (EU mid-size non-life insurer, 2024 EIOPA reference) ────

def default_eu_insurer_params() -> dict:
    """Reference parameters for an EU non-life insurer.

    Sources:
      EIOPA Insurance Statistics Q4 2024 — combined ratio EU avg 94-96%
      EIOPA Solvency II Ratio — EU avg 225% Q4 2024
      ANIA (IT) — bulk premium, claims, mix of business
    """
    return {
        # Initial state (normalised; absolute scale via scale_eur_bn)
        "scale_eur_bn": 3.0,                 # mid-size insurer book
        "premium_written": 1.00,             # normalised yearly premium
        "claims_paid": 0.65,                 # claims/premium baseline
        "expenses": 0.30,                    # expense/premium ratio
        "technical_provisions": 1.50,        # provisions / annual premium
        "own_funds": 0.45,                   # SII own funds, normalised
        "scr": 0.20,                         # Solvency Capital Requirement
        "investment_assets": 1.80,           # investment portfolio
        # Sensitivities
        "premium_to_opinion_coef": 0.12,     # opinion drag on new business
        "lapse_to_opinion_coef": 0.08,       # opinion → lapse acceleration
        "shock_event_loss_coef": 0.40,       # claims surge per unit shock_magnitude
        "investment_yield_pct": 2.20,        # avg yield on bond portfolio
        # Behaviour caps
        "lapse_rate_max_per_round": 0.04,    # 4%/round physiological max
        "premium_drop_max_per_round": 0.06,  # 6%/round max contraction
        # Regulatory
        "scr_ratio_min_pct": 100.0,
        "scr_ratio_alarm_pct": 130.0,
        # Life-only fields (used only if is_life=True at construction)
        "is_life": False,
        "duration_liabilities_yrs": 4.5,
    }


# ── State snapshot ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InsuranceState:
    round: int
    premium_written: float
    claims_paid: float
    expenses: float
    technical_provisions: float
    own_funds: float
    scr: float
    investment_assets: float
    # KPIs
    combined_ratio_pct: float       # (claims + expenses) / premium, %
    sii_ratio_pct: float            # own_funds / scr, %
    lapse_rate_round_pct: float
    investment_pnl: float
    # Flags
    breaches: tuple = field(default_factory=tuple)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["breaches"] = list(self.breaches)
        return d

    def to_compact_str(self) -> str:
        bits = [
            f"CR {self.combined_ratio_pct:.1f}%",
            f"SII {self.sii_ratio_pct:.0f}%",
            f"premium {self.premium_written:.3f}",
            f"lapse {self.lapse_rate_round_pct*100:.1f}%/r",
        ]
        if self.breaches:
            bits.append(f"BREACH:{','.join(self.breaches)}")
        return " | ".join(bits)


# ── The twin engine ─────────────────────────────────────────────────────────

class InsuranceTwin:
    """Stateful Solvency II-flavour twin.

    Use:
        twin = InsuranceTwin()
        s1 = twin.step(round_num=1, opinion_aggregate=-0.2,
                       shock_magnitude=0.3, narrative="Cat event")
    """

    def __init__(self, params: Optional[dict] = None, *, is_life: bool = False):
        self.params = default_eu_insurer_params()
        if params:
            self.params.update(params)
        self.params["is_life"] = is_life
        self.history: list[InsuranceState] = []
        self.history.append(self._build_baseline_state())

    def current_state(self) -> InsuranceState:
        return self.history[-1]

    def step(
        self,
        round_num: int,
        opinion_aggregate: float = 0.0,
        shock_magnitude: float = 0.0,
        polarization: float = 5.0,
        narrative: str = "",
    ) -> InsuranceState:
        prev = self.current_state()
        p = self.params

        # ── 1. Premium written ─────────────────────────────────────────
        # Negative opinion drags new-business premium; baseline grows ~+1%/r
        baseline_growth = 0.01
        opinion_drag = max(0.0, -opinion_aggregate) * p["premium_to_opinion_coef"]
        delta_premium = baseline_growth - min(p["premium_drop_max_per_round"], opinion_drag)
        new_premium = prev.premium_written * (1 + delta_premium)

        # ── 2. Claims paid ─────────────────────────────────────────────
        # Baseline claims/premium ≈ 0.65; shock event temporarily spikes it
        loss_shock = abs(shock_magnitude) * p["shock_event_loss_coef"]
        new_claims = prev.claims_paid * (1 + 0.005) + loss_shock * new_premium

        # ── 3. Expenses ─────────────────────────────────────────────
        new_expenses = prev.expenses * (1 + 0.002)

        # ── 4. Combined ratio ─────────────────────────────────────────
        cr_pct = (new_claims + new_expenses) / max(new_premium, 1e-6) * 100.0

        # ── 5. Technical provisions ─────────────────────────────────────
        # Provisions grow with premium and partially with claims
        new_provisions = prev.technical_provisions * 1.005 + (new_claims - prev.claims_paid) * 0.5

        # ── 6. Investment P&L ─────────────────────────────────────────
        # Yield on bond portfolio, simple model
        inv_yield = p["investment_yield_pct"] / 100.0 / 12.0
        inv_pnl = prev.investment_assets * inv_yield
        new_investments = prev.investment_assets + inv_pnl

        # ── 7. Own funds: capital eroded by net loss ─────────────────────
        net_underwriting = new_premium - new_claims - new_expenses
        new_own_funds = prev.own_funds + net_underwriting + inv_pnl
        # SCR scales mildly with provisions
        new_scr = max(0.05, p["scr"] * (new_provisions / max(p["technical_provisions"], 1e-6)) ** 0.5)
        sii_pct = 100.0 * new_own_funds / max(new_scr, 1e-6)

        # ── 8. Lapse rate (Life-relevant; for non-life still useful as
        # surrender / cancellation rate) ────────────────────────────────
        opinion_lapse = max(0.0, -opinion_aggregate) * p["lapse_to_opinion_coef"]
        polar_amp = 0.0
        if polarization > 6.5 and opinion_aggregate < -0.4:
            polar_amp = 0.015
        lapse_pct = min(p["lapse_rate_max_per_round"] + polar_amp, opinion_lapse + polar_amp)
        # Apply lapse to provisions (simplified: lapse releases provisions)
        new_provisions *= (1 - lapse_pct * 0.5)

        # ── 9. Breach detection ───────────────────────────────────────
        breaches = []
        if sii_pct < p["scr_ratio_min_pct"]:
            breaches.append("SII<min")
        elif sii_pct < p["scr_ratio_alarm_pct"]:
            breaches.append("SII≈alarm")
        if cr_pct > 110.0:
            breaches.append("CR>110")
        if lapse_pct > p["lapse_rate_max_per_round"] + 0.005:
            breaches.append("LAPSE_SURGE")

        new_state = InsuranceState(
            round=round_num,
            premium_written=round(new_premium, 4),
            claims_paid=round(new_claims, 4),
            expenses=round(new_expenses, 4),
            technical_provisions=round(new_provisions, 4),
            own_funds=round(new_own_funds, 4),
            scr=round(new_scr, 4),
            investment_assets=round(new_investments, 4),
            combined_ratio_pct=round(cr_pct, 2),
            sii_ratio_pct=round(sii_pct, 1),
            lapse_rate_round_pct=round(lapse_pct, 4),
            investment_pnl=round(inv_pnl, 5),
            breaches=tuple(breaches),
            notes=narrative[:160] if narrative else "",
        )
        self.history.append(new_state)
        return new_state

    def _build_baseline_state(self) -> InsuranceState:
        p = self.params
        cr = (p["claims_paid"] + p["expenses"]) / max(p["premium_written"], 1e-6) * 100.0
        sii = 100.0 * p["own_funds"] / max(p["scr"], 1e-6)
        return InsuranceState(
            round=0,
            premium_written=p["premium_written"],
            claims_paid=p["claims_paid"],
            expenses=p["expenses"],
            technical_provisions=p["technical_provisions"],
            own_funds=p["own_funds"],
            scr=p["scr"],
            investment_assets=p["investment_assets"],
            combined_ratio_pct=round(cr, 2),
            sii_ratio_pct=round(sii, 1),
            lapse_rate_round_pct=0.0,
            investment_pnl=0.0,
            breaches=(),
            notes="baseline",
        )
