"""FinancialTwin — domain-agnostic ALM (asset-liability management) engine
for banking-style scenarios.

Why this exists
---------------
The opinion-only sim produces emergent KPIs that are LLM-narrative driven
(e.g. "deposit runoff 92%" on a +2% rate hike, which is a bank-run number,
not a repricing). Real bank balance sheets respond to rate shocks within
well-defined ALM bounds:

    deposit pass-through (beta)        ECB 2025: IT sight β≈0.45, term β≈0.75
    consumer loan demand elasticity    Bonaccorsi/Magri IT micro: median -2.14
    NIM (net interest margin)          EBA Q2 2025: EU avg 1.58%
    CET1 ratio                         EBA Q3 2025: EU avg 16.3%
    LCR (liquidity coverage)           EBA Q3 2025: EU avg 160.7%
    deposit runoff cap (no panic)      empirical IT: 3-5% / month max

This twin enforces those bounds. It does NOT model an individual bank's
real balance sheet — it parameterises a *generic* commercial-bank ALM
profile that any scenario can override via `params` at construction.

Coupling model (v0.6 weak)
--------------------------
For each round the twin receives:
  - shock_magnitude, shock_direction (from event_injector)
  - opinion_aggregate (mean position over agents, ∈ [-1, +1])
  - polarization (current 0-10)

It produces a new `FinancialState` that the round_manager attaches to
`round_result["financial_twin"]` and exposes (compressed) to agent prompts.

The twin does NOT mutate agent positions directly — opinion drift remains
the LLM/opinion_dynamics responsibility. In v0.7 we'll close the loop:
worsening NIM/CET1 will inject negative anchor pressure on retail and
positive social pressure on critics.

Pure Python, no JAX dependency. Differentiability is a v0.7 concern —
adding JAX here would over-couple this module to the calibration layer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

logger = logging.getLogger(__name__)


# ── Defaults (Italian commercial-bank reference, 2025 benchmarks) ──────────

def default_italian_bank_params() -> dict:
    """Reference parameter set derived from public 2025 EU/IT data.

    All numbers cited here are from public sources (EBA Risk Dashboard,
    ECB Economic Bulletin, Banca d'Italia, peer-reviewed research). They
    describe a *generic mid-size Italian commercial bank* with consumer-
    credit + retail-mortgage focus. Override any field via constructor.
    """
    return {
        # ── Initial state (normalised; absolute scale defined by `scale_eur_bn`)
        "scale_eur_bn": 5.0,        # bank size proxy; KPIs scale linearly
        "deposit_balance": 1.00,    # normalised; absolute = scale * 1.00 * 5 ≈ 5B
        "loan_balance": 0.85,       # loan/deposit ratio ≈ 0.85 (IT median)
        "hqla_balance": 0.085,      # HQLA / deposits — calibrated so baseline LCR≈170% (EBA Q3 2025 EU avg 160.7%)
        "rwa_density": 0.55,        # RWA / total assets — calibrated so baseline CET1≈16% (EBA Q3 2025 EU avg 16.3%)
        "tier1_capital": 0.083,     # CET1 capital — calibrated so baseline CET1≈16% (EBA Q3 2025 EU avg 16.3%)
        # ── ALM sensitivity coefficients (IT 2025 benchmarks)
        "deposit_beta_sight": 0.45,    # ECB 2025: IT sight deposits pass-through
        "deposit_beta_term": 0.75,     # term deposits, faster pass-through
        "sight_share": 0.65,           # share of deposits that are demand
        "loan_repricing_speed": 0.35,  # share of loan book that reprices each rate cycle
        "consumer_loan_elasticity": -1.7,  # Bonaccorsi/Magri IT median -2.14, conservative -1.7
        "mortgage_var_share_stock": 0.50,  # stock var/fix ≈ 50/50 IT
        "mortgage_var_share_new": 0.20,    # new mortgages predominantly fixed
        "duration_gap_yrs": 1.5,           # asset minus liability duration
        "hedge_ratio": 0.40,               # share of duration gap hedged via IRS
        # ── Stress / behaviour caps
        "deposit_runoff_max_per_round": 0.04,  # 4% physiological max (no panic)
        "deposit_runoff_panic_extra": 0.06,    # extra cap when polarization > 7 AND trust collapsed
        "loan_demand_floor": 0.55,             # demand floor as share of baseline (IT recession lower bound)
        # ── Regulatory thresholds (Basel III / CRR3 IT)
        "cet1_min_pct": 11.5,        # SREP requirement + buffer typical IT
        "lcr_min_pct": 100.0,        # regulatory floor
        "cet1_alarm_pct": 12.5,      # internal early-warning
        # ── Initial macro context (override per scenario)
        "policy_rate_pct": 2.40,     # ECB DFR Apr 2026
        "btp_bund_spread_bps": 95,   # IT 10Y vs DE 10Y, recent
        # ── Opinion-coupling sensitivities (how much opinion shifts affect balance)
        "opinion_to_runoff_coef": 0.025,  # extra runoff per unit of negative opinion drift
        "opinion_to_loan_demand_coef": 0.18,  # demand drag per unit of negative opinion
        "trust_to_repricing_speed": 0.10,  # how much accelerated runoff of high-rate term deposits
    }


# ── State snapshot ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeedbackSignals:
    """Per-round signals the twin sends back to the opinion layer to close
    the loop. Each signal is normalised to [0, 1] = "intensity of stress".
    Agents with relevant exposure read these in their prompt context and
    can choose to update positions accordingly (LLM-driven, not forced).
    """
    nim_anxiety: float          # market sees NIM compression → retail/PMI worry about lending costs
    cet1_alarm: float           # capital ratio nearing regulatory floor → all stakeholders react
    runoff_panic: float         # deposit outflow above physiological → trust crisis spiral risk
    competitor_pressure: float  # peer banks gain market share → opportunity narrative for rivals
    rate_pressure: float        # cumulative rate change → mortgage / consumer-loan holders feel rate increase

    def to_dict(self) -> dict:
        return asdict(self)

    def to_compact_str(self) -> str:
        """Human-readable, low-token summary for agent prompts. Returns
        an empty string if all signals are below noise floor (< 0.1)."""
        bits = []
        if self.cet1_alarm >= 0.5:
            bits.append(f"CET1 vicino al limite regolatorio (alarm {self.cet1_alarm:.2f})")
        if self.runoff_panic >= 0.4:
            bits.append(f"deposit outflow oltre soglia fisiologica (panic {self.runoff_panic:.2f})")
        if self.nim_anxiety >= 0.4:
            bits.append(f"compressione NIM percepita (anxiety {self.nim_anxiety:.2f})")
        if self.competitor_pressure >= 0.4:
            bits.append(f"competitor approfittano del momento (pressure {self.competitor_pressure:.2f})")
        if self.rate_pressure >= 0.5:
            bits.append(f"rate path cumulato significativo ({self.rate_pressure:.2f})")
        return " · ".join(bits)


@dataclass(frozen=True)
class FinancialState:
    """Immutable snapshot of the bank's ALM state at a single round.

    All balance figures are in the normalised scale defined by
    `params['scale_eur_bn']` (so deposit_balance = 1.0 means
    `scale_eur_bn` €B). KPIs (NIM, LCR, CET1) are reported in
    market-standard units (% or pp).
    """
    round: int
    # Balance sheet (normalised)
    deposit_balance: float
    loan_balance: float
    hqla_balance: float
    tier1_capital: float
    # Income (per-round)
    nim_pct: float                 # net interest margin, %
    nii_per_round: float           # net interest income for this round, normalised
    hedging_pnl: float             # mark-to-market on IRS hedge book
    # Risk ratios
    cet1_pct: float                # CET1 / RWA, %
    lcr_pct: float                 # LCR, %
    duration_gap_yrs: float        # asset - liability duration
    # Behavioural / market
    deposit_runoff_round_pct: float    # this round's deposit outflow rate
    loan_demand_index: float           # 1.0 = baseline, <1.0 = below baseline
    deposit_beta_effective: float      # blended sight+term beta
    # Macro context
    policy_rate_pct: float
    btp_bund_spread_bps: float
    # Flags / annotations
    breaches: tuple = field(default_factory=tuple)  # ("CET1<min", "LCR<100", ...)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["breaches"] = list(self.breaches)
        return d

    def to_compact_str(self) -> str:
        """One-line agent-prompt-ready summary. Cheap on tokens."""
        bits = [
            f"NIM {self.nim_pct:.2f}%",
            f"CET1 {self.cet1_pct:.1f}%",
            f"LCR {self.lcr_pct:.0f}%",
            f"deposit_runoff {self.deposit_runoff_round_pct*100:.1f}%/r",
            f"loan_demand_idx {self.loan_demand_index:.2f}",
        ]
        if self.breaches:
            bits.append(f"BREACH:{','.join(self.breaches)}")
        return " | ".join(bits)


# ── The twin engine ─────────────────────────────────────────────────────────

class FinancialTwin:
    """Stateful ALM engine. One instance per simulation.

    Use:
        twin = FinancialTwin()
        s0 = twin.current_state()             # initial state
        s1 = twin.step(round_num=1,
                       rate_change_bps=200,    # +200bps event
                       opinion_aggregate=-0.10,
                       polarization=3.5,
                       narrative="rate hike announcement")
        # s1 is a FinancialState; twin.history is the list of all snapshots
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        *,
        stress_template_name: Optional[str] = None,
        cir_rate_process=None,
    ):
        """Build the twin.

        Args:
            params: override of default_italian_bank_params() keys.
            stress_template_name: optional EBA-style template name
                ("baseline" | "adverse"). Applies twin_overrides + sets up a
                CIR process with the template's cir_overrides if no
                cir_rate_process is provided. See core.financial.rates.
            cir_rate_process: optional CIRRateProcess instance. If set, the
                twin will draw rate increments from it each round (overriding
                manual rate_change_bps). Pass None to keep manual control.
        """
        self.params = default_italian_bank_params()
        if params:
            self.params.update(params)
        # Sprint 3: stress template overlay
        self.stress_template = None
        if stress_template_name:
            from .rates import get_stress_template, CIRRateProcess
            self.stress_template = get_stress_template(stress_template_name)
            if self.stress_template:
                self.params.update(self.stress_template.twin_overrides)
                if cir_rate_process is None:
                    self.cir = CIRRateProcess(**self.stress_template.cir_overrides)
                else:
                    self.cir = cir_rate_process
            else:
                self.cir = cir_rate_process
        else:
            self.cir = cir_rate_process
        self.history: list[FinancialState] = []
        # Internal trackers (not exposed in state snapshot)
        self._cum_rate_change_bps = 0.0
        self._opinion_anchor = 0.0
        self._consecutive_negative_rounds = 0
        # Track baseline rate for CIR delta computation
        self._cir_baseline_rate = self.cir.r0 if self.cir is not None else None
        # Sprint 2: feedback signals history (parallel to state history)
        self.feedback_history: list[FeedbackSignals] = []
        # Build round-0 baseline state
        self.history.append(self._build_baseline_state())
        self.feedback_history.append(FeedbackSignals(0.0, 0.0, 0.0, 0.0, 0.0))

    # ── Public API ──────────────────────────────────────────────────────

    def current_state(self) -> FinancialState:
        return self.history[-1]

    def step(
        self,
        round_num: int,
        rate_change_bps: float = 0.0,
        opinion_aggregate: float = 0.0,
        polarization: float = 5.0,
        narrative: str = "",
        # ── Sprint 2: closed-loop coupling inputs ─────────────────────
        opinion_by_exposure: Optional[dict] = None,
    ) -> FinancialState:
        """Advance the twin one round.

        Args:
            round_num: 1-based round index.
            rate_change_bps: change in policy / market rate this round.
                For "+2% rate hike" pass 200; "-25bp cut" pass -25. Combined
                with cumulative tracker so the twin sees the full path.
            opinion_aggregate: mean agent position ∈ [-1, +1]. Negative =
                public sentiment turning against the bank.
            polarization: 0-10 polarization index from the opinion model.
                High polarization + negative opinion compounds runoff.
            narrative: optional free-text label of what drove the change
                (logged into the snapshot for auditability).
            opinion_by_exposure: dict with exposure-weighted opinion aggregates.
                Keys (all optional, default 0.0):
                  - "depositors_negative": mean of -opinion weighted by deposit
                    balance share (drives runoff intensity)
                  - "borrowers_negative": mean of -opinion weighted by loan
                    balance share (drives loan demand contraction)
                  - "competitors_negative_to_us": mean opinion of competitor
                    bank agents toward THIS bank (high = predator behaviour)
                Falls back to opinion_aggregate-based defaults if absent.

        Returns the new FinancialState (also appended to self.history).
        Use `latest_feedback()` after calling step to get the FeedbackSignals.
        """
        prev = self.current_state()
        p = self.params

        # Sprint 3: if a CIR process is attached, draw the round's rate from it
        # and override the manual rate_change_bps. Round 1 also applies any
        # initial_rate_shock_bps from a stress template (one-shot).
        if self.cir is not None:
            new_rate = self.cir.step()
            inferred_rate_bps = (new_rate - self._cir_baseline_rate) * 10000.0
            # Apply initial stress shock once at round 1
            if (round_num == 1 and self.stress_template is not None
                    and self.stress_template.initial_rate_shock_bps != 0):
                inferred_rate_bps += self.stress_template.initial_rate_shock_bps
            rate_change_bps = inferred_rate_bps
            self._cir_baseline_rate = new_rate
        self._cum_rate_change_bps += rate_change_bps

        # Sprint 2: exposure-weighted opinion (defaults to flat opinion_aggregate)
        ex = opinion_by_exposure or {}
        depositors_neg = float(ex.get("depositors_negative", max(0.0, -opinion_aggregate)))
        borrowers_neg = float(ex.get("borrowers_negative", max(0.0, -opinion_aggregate)))
        competitors_neg = float(ex.get("competitors_negative_to_us", 0.0))

        # ── 1. Deposit dynamics ─────────────────────────────────────────
        # Effective beta = blended sight + term, weighted by share
        beta_eff = (
            p["sight_share"] * p["deposit_beta_sight"]
            + (1 - p["sight_share"]) * p["deposit_beta_term"]
        )
        # Runoff rate: a fraction of deposits leaves each round when:
        #   (a) rates rise but our beta is < competitor beta (price war), and
        #   (b) opinion turns negative (trust erosion).
        rate_pressure = max(0.0, rate_change_bps / 100.0) * (1 - beta_eff) * 0.015
        # Sprint 2: use depositor-weighted negative opinion if available
        opinion_pressure = depositors_neg * p["opinion_to_runoff_coef"]
        # Competitor pressure adds incremental runoff (deposits flowing to peers)
        competitor_drain = competitors_neg * 0.012
        # Polarization amplifier — only kicks in past the panic threshold
        panic_amp = 0.0
        if polarization > 7.0 and opinion_aggregate < -0.4:
            panic_amp = p["deposit_runoff_panic_extra"] * min(
                1.0, (polarization - 7.0) / 3.0
            )
        runoff_pct = rate_pressure + opinion_pressure + competitor_drain + panic_amp
        # Cap at the physiological max (no LLM-driven bank-run unless panic_amp explicit)
        runoff_pct = min(runoff_pct, p["deposit_runoff_max_per_round"] + panic_amp)
        runoff_pct = max(0.0, runoff_pct)
        new_deposits = prev.deposit_balance * (1 - runoff_pct)

        # ── 2. Loan demand ──────────────────────────────────────────────
        # Elasticity model: % change in demand ≈ elasticity * (% change in rate)
        # Baseline = 1.0; demand drops as rates rise, demand drops with bad opinion.
        rate_pct_change = self._cum_rate_change_bps / 10000.0  # bps → fraction
        demand_from_rate = 1.0 + p["consumer_loan_elasticity"] * rate_pct_change
        # Sprint 2: borrower-weighted opinion drag (consumers who actually
        # need loans react more strongly to bank's reputation than the average)
        demand_from_opinion = 1.0 - borrowers_neg * p["opinion_to_loan_demand_coef"]
        loan_demand_index = max(p["loan_demand_floor"], demand_from_rate * demand_from_opinion)
        # Loan balance is sticky (stock vs flow): lerp toward demand
        new_loans = prev.loan_balance * 0.92 + (loan_demand_index * 0.85) * 0.08

        # ── 3. NIM (net interest margin) ────────────────────────────────
        # Δ NIM ≈ (asset_repricing_share - liab_repricing_share) * rate_change / 100
        # Liability repricing share ≈ beta_eff; asset repricing share ≈ loan_repricing_speed
        asset_repricing = p["loan_repricing_speed"] * (1 - p["mortgage_var_share_stock"]) * 0.5 + p["mortgage_var_share_stock"]
        liab_repricing = beta_eff
        nim_delta_pct = (asset_repricing - liab_repricing) * (self._cum_rate_change_bps / 10000.0)
        # Initial NIM derives from deposit balance + tier1 + duration carry
        baseline_nim = self._initial_nim_pct()
        new_nim_pct = baseline_nim + nim_delta_pct * 100  # convert to %
        # Negative opinion erodes NIM via cross-selling losses (small effect)
        new_nim_pct -= max(0.0, -opinion_aggregate) * 0.05
        new_nim_pct = max(0.30, new_nim_pct)  # floor at 30bp; below that bank is unviable

        nii = (new_loans + new_deposits * 0.10) * (new_nim_pct / 100.0) / 12.0  # roughly per-month

        # ── 4. Hedging P&L ─────────────────────────────────────────────
        # Bank receives floating, pays fixed on hedge ratio of duration gap.
        # If rates rise unexpectedly, hedge gains. Approximate DV01 effect.
        hedged_duration = p["duration_gap_yrs"] * p["hedge_ratio"]
        hedge_pnl = hedged_duration * (rate_change_bps / 10000.0) * prev.deposit_balance * 0.01
        # Apply PnL to capital
        new_capital = prev.tier1_capital + nii + hedge_pnl

        # ── 5. Liquidity (LCR) ─────────────────────────────────────────
        # Outflows over 30 days proxy: this round's runoff projected forward
        outflow_30d = runoff_pct * 4.0 * new_deposits  # if rounds ≈ weeks
        outflow_30d = max(outflow_30d, 0.05 * new_deposits)  # min 5% reg outflow factor
        new_hqla = prev.hqla_balance * 0.98 + (new_capital - prev.tier1_capital) * 0.5
        lcr_pct = 100.0 * (new_hqla / max(outflow_30d, 1e-6))

        # ── 6. CET1 ────────────────────────────────────────────────────
        # RWA proxy: loan_balance × rwa_density (deposits don't drive RWA materially for retail).
        rwa = new_loans * p["rwa_density"] + new_deposits * 0.05
        cet1_pct = 100.0 * (new_capital / max(rwa, 1e-6))

        # ── 7. Breach detection ────────────────────────────────────────
        breaches = []
        if cet1_pct < p["cet1_min_pct"]:
            breaches.append("CET1<min")
        elif cet1_pct < p["cet1_alarm_pct"]:
            breaches.append("CET1≈alarm")
        if lcr_pct < p["lcr_min_pct"]:
            breaches.append("LCR<100")
        if runoff_pct > p["deposit_runoff_max_per_round"] + 0.01:
            breaches.append("DEPOSIT_RUN")

        # ── 8. Update internal state ───────────────────────────────────
        if opinion_aggregate < -0.2:
            self._consecutive_negative_rounds += 1
        else:
            self._consecutive_negative_rounds = max(0, self._consecutive_negative_rounds - 1)

        new_state = FinancialState(
            round=round_num,
            deposit_balance=round(new_deposits, 4),
            loan_balance=round(new_loans, 4),
            hqla_balance=round(new_hqla, 4),
            tier1_capital=round(new_capital, 5),
            nim_pct=round(new_nim_pct, 3),
            nii_per_round=round(nii, 5),
            hedging_pnl=round(hedge_pnl, 5),
            cet1_pct=round(cet1_pct, 2),
            lcr_pct=round(lcr_pct, 1),
            duration_gap_yrs=round(p["duration_gap_yrs"], 2),
            deposit_runoff_round_pct=round(runoff_pct, 4),
            loan_demand_index=round(loan_demand_index, 3),
            deposit_beta_effective=round(beta_eff, 3),
            policy_rate_pct=round(p["policy_rate_pct"] + self._cum_rate_change_bps / 100.0, 3),
            btp_bund_spread_bps=round(p["btp_bund_spread_bps"] + max(0, -opinion_aggregate) * 30, 1),
            breaches=tuple(breaches),
            notes=narrative[:160] if narrative else "",
        )
        self.history.append(new_state)

        # ── Sprint 2: feedback signals to opinion layer ───────────────────
        # Each signal is squashed into [0, 1] = "intensity of stress".
        # Agents read these via prompt context and may update positions.
        baseline_nim = self._initial_nim_pct()
        nim_delta_pct = max(0.0, baseline_nim - new_nim_pct) / max(baseline_nim, 0.5)
        cet1_distance = max(0.0, p["cet1_alarm_pct"] - cet1_pct) / max(
            p["cet1_alarm_pct"] - p["cet1_min_pct"], 0.5
        )
        runoff_excess = max(0.0, runoff_pct - p["deposit_runoff_max_per_round"]) / max(
            p["deposit_runoff_panic_extra"], 0.001
        )
        rate_change_norm = abs(self._cum_rate_change_bps) / 400.0  # normalise: 400bps = full

        feedback = FeedbackSignals(
            nim_anxiety=round(min(1.0, nim_delta_pct * 3.0), 3),
            cet1_alarm=round(min(1.0, cet1_distance), 3),
            runoff_panic=round(min(1.0, runoff_excess + max(0.0, runoff_pct / 0.05 - 0.6)), 3),
            competitor_pressure=round(min(1.0, competitors_neg), 3),
            rate_pressure=round(min(1.0, rate_change_norm), 3),
        )
        self.feedback_history.append(feedback)
        return new_state

    def latest_feedback(self) -> FeedbackSignals:
        """Most recent FeedbackSignals (parallel to current_state)."""
        return self.feedback_history[-1]

    def refresh_market_anchors(self, use_cache: bool = True) -> dict:
        """Sprint 4: pull live anchors (ECB DFR, BTP-Bund, Euribor 3M) and
        update twin params before round 1. Idempotent if cache fresh.

        Returns the fetched anchors dict (also applied to self.params and
        baseline state). Safe to call any time but normally invoked once
        right after FinancialTwin construction.
        """
        from .market_data import fetch_all_anchors
        anchors = fetch_all_anchors(use_cache=use_cache)
        # Only override the keys the twin actually uses (skip private _ ones)
        for k, v in anchors.items():
            if k in self.params and not k.startswith("_"):
                self.params[k] = v
        # Rebuild baseline state if no rounds simulated yet, so the new
        # anchors reflect in the round-0 snapshot.
        if len(self.history) == 1:
            self.history[0] = self._build_baseline_state()
        logger.info(
            f"FinancialTwin: refreshed market anchors → "
            f"policy_rate={self.params['policy_rate_pct']:.3f}%, "
            f"BTP-Bund={self.params['btp_bund_spread_bps']:.0f}bp"
        )
        return anchors

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_baseline_state(self) -> FinancialState:
        p = self.params
        beta_eff = p["sight_share"] * p["deposit_beta_sight"] + (1 - p["sight_share"]) * p["deposit_beta_term"]
        rwa = p["loan_balance"] * p["rwa_density"] + p["deposit_balance"] * 0.05
        cet1_pct = 100.0 * p["tier1_capital"] / max(rwa, 1e-6)
        lcr_pct = 100.0 * p["hqla_balance"] / max(0.05 * p["deposit_balance"], 1e-6)
        return FinancialState(
            round=0,
            deposit_balance=p["deposit_balance"],
            loan_balance=p["loan_balance"],
            hqla_balance=p["hqla_balance"],
            tier1_capital=p["tier1_capital"],
            nim_pct=self._initial_nim_pct(),
            nii_per_round=0.0,
            hedging_pnl=0.0,
            cet1_pct=round(cet1_pct, 2),
            lcr_pct=round(lcr_pct, 1),
            duration_gap_yrs=p["duration_gap_yrs"],
            deposit_runoff_round_pct=0.0,
            loan_demand_index=1.0,
            deposit_beta_effective=round(beta_eff, 3),
            policy_rate_pct=p["policy_rate_pct"],
            btp_bund_spread_bps=p["btp_bund_spread_bps"],
            breaches=(),
            notes="baseline (round 0)",
        )

    def _initial_nim_pct(self) -> float:
        """Anchor NIM around EBA 2025 EU average (1.58%) with small bias from
        the bank's loan/deposit mix and duration gap. Conservative."""
        p = self.params
        base = 1.58  # EBA Q2 2025 EU avg
        # Banks with longer asset duration earn more carry → +duration_gap * 8bp
        carry_bps = p["duration_gap_yrs"] * 8.0
        # Higher loan/deposit ratio → slightly higher NIM (more interest assets)
        ldr_bonus = (p["loan_balance"] / p["deposit_balance"] - 0.85) * 30.0
        return round(base + (carry_bps + ldr_bonus) / 100.0, 3)
