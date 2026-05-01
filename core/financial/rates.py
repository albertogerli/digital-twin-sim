"""Stochastic short-rate processes for the financial twin.

Provides:
- CIRRateProcess: Cox-Ingersoll-Ross 1-factor mean-reverting square-root.
  drift always pulls toward θ, σ scales with √r so r stays positive
  (Feller condition: 2κθ > σ² recommended).
- HW1FRateProcess: stub for Hull-White 1-factor (deferred to Sprint 4
  when we have the BTP zero curve loaded).
- EBAStressTemplate: parameterised "+200bps adverse + recession" scenario
  that shifts CIR parameters and twin state to mirror a stress test cycle.

CIR is the right model for a single-rate ALM driver when you don't yet
have a calibrated term-structure model. Discretised via Euler-Maruyama;
clamped at zero to avoid sqrt of negative residuals (rare with sane params).

Default parameters anchored to Euribor 3M area euro 2024-2026:
  κ = 0.30  (mean-reversion speed → half-life ≈ 2.3 yr)
  θ = 2.5%  (long-term mean post-tightening cycle)
  σ = 1.5%  (annualised volatility)
  r0 = 2.4% (current ECB DFR Apr 2026)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ── CIR 1-factor process ────────────────────────────────────────────────────

@dataclass
class CIRRateProcess:
    """Cox-Ingersoll-Ross 1-factor short-rate.

        dr = κ(θ - r) dt + σ √r dW

    Default params from Euribor 3M area euro 2024-2026 (CB tightening cycle
    ending, mean reversion toward neutral). All rates in fractional units
    (e.g. 0.024 = 2.4%).
    """
    kappa: float = 0.30
    theta: float = 0.025
    sigma: float = 0.015
    r0: float = 0.024
    dt: float = 1.0 / 12.0      # 1 round = 1 month
    seed: int = 42
    # State
    r_current: float = 0.0
    history: list[float] = field(default_factory=list)
    rng: Optional[random.Random] = None

    def __post_init__(self):
        self.r_current = self.r0
        self.history = [self.r0]
        self.rng = random.Random(self.seed)

    def step(self) -> float:
        """Advance one round, return new short rate (fractional)."""
        r = max(0.0, self.r_current)
        drift = self.kappa * (self.theta - r) * self.dt
        diffusion = self.sigma * math.sqrt(r * self.dt) * self.rng.gauss(0.0, 1.0)
        new_r = max(0.0001, r + drift + diffusion)  # floor at 1bp
        self.r_current = new_r
        self.history.append(new_r)
        return new_r

    def reset(self, seed: Optional[int] = None):
        self.r_current = self.r0
        self.history = [self.r0]
        self.rng = random.Random(seed if seed is not None else self.seed)

    def trajectory_summary(self) -> dict:
        if len(self.history) < 2:
            return {"min_pct": self.r0 * 100, "max_pct": self.r0 * 100,
                    "final_pct": self.r0 * 100, "n_steps": 0}
        return {
            "min_pct": round(min(self.history) * 100, 3),
            "max_pct": round(max(self.history) * 100, 3),
            "final_pct": round(self.history[-1] * 100, 3),
            "n_steps": len(self.history) - 1,
        }


# ── EBA-style stress templates ──────────────────────────────────────────────

@dataclass(frozen=True)
class EBAStressTemplate:
    """Parameter overlay for a single regulatory stress scenario.

    Apply via FinancialTwin(params=template.twin_overrides()) and pass
    `cir_rate_process` with template.cir_overrides() applied.

    Templates are inspired by EBA 2025 EU-wide stress test methodology
    (adverse + recession + market shock). They are NOT verbatim EBA
    parameters — they are illustrative defaults for sales demos. Bank-
    specific calibration should override these in production.
    """
    name: str
    description: str
    # Initial twin overrides
    twin_overrides: dict
    # CIR overrides (kappa/theta/sigma/r0 if set)
    cir_overrides: dict
    # Forced cumulative rate path (bps applied at round 1) — used when caller
    # wants a deterministic shock rather than CIR diffusion
    initial_rate_shock_bps: float = 0.0
    # Notes for report rendering
    expected_narrative: str = ""


def eba_adverse_2025_template() -> EBAStressTemplate:
    """Approximation of EBA 2025 Adverse Scenario for a mid-size IT bank.

    Key features (illustrative, not verbatim):
      - +250bps cumulative rate shock by end-horizon
      - Sharp deposit beta increase (depositors switch to term)
      - Loan elasticity worsens (consumers pull back)
      - CET1 capital starting buffer compressed
      - Credit losses (NIM erosion via provisions, simplified here)
    """
    return EBAStressTemplate(
        name="EBA 2025 Adverse — IT mid-size bank",
        description=(
            "Scenario stress regolatorio EBA 2025 adverse adattato per banca "
            "commerciale italiana media: shock tassi cumulativo +250bps, "
            "compressione buffer capitale, deposit beta in salita rapida, "
            "elasticità credito al consumo peggiorata."
        ),
        twin_overrides={
            "cet1_alarm_pct": 13.0,        # alarm più alto durante stress
            "cet1_min_pct": 11.5,
            "deposit_beta_sight": 0.55,    # passthrough più rapido sotto stress
            "deposit_beta_term": 0.85,
            "consumer_loan_elasticity": -2.1,  # consumatori più sensibili
            "deposit_runoff_max_per_round": 0.045,
        },
        cir_overrides={
            "kappa": 0.20,    # più lento mean-reversion (shock persistente)
            "theta": 0.040,   # long-term mean elevato (tassi alti più a lungo)
            "sigma": 0.025,   # vol amplificata
            "r0": 0.024,
        },
        initial_rate_shock_bps=250.0,
        expected_narrative=(
            "Stress regolatorio: rate shock +250bps cumulativo, deposit beta "
            "compressa, loan demand contratta, credit losses crescenti. "
            "CET1 sotto pressione ma sopra il floor regolamentare se la "
            "banca parte da una posizione capital-strong (16%+)."
        ),
    )


def eba_baseline_2025_template() -> EBAStressTemplate:
    """Mirror of the baseline scenario (no shock, just normal cycle).
    Useful as control for delta-vs-adverse comparison runs."""
    return EBAStressTemplate(
        name="EBA 2025 Baseline — IT mid-size bank",
        description=(
            "Scenario baseline regolatorio: ciclo normale, nessuno shock "
            "esogeno, parametri ALM standard banca commerciale italiana 2025."
        ),
        twin_overrides={},
        cir_overrides={
            "kappa": 0.30, "theta": 0.025, "sigma": 0.012, "r0": 0.024,
        },
        initial_rate_shock_bps=0.0,
        expected_narrative=(
            "Scenario baseline: tassi che vagano vicino al livello neutrale, "
            "depositi stabili, domanda credito allineata al trend macro."
        ),
    )


_TEMPLATES = {
    "baseline": eba_baseline_2025_template,
    "adverse": eba_adverse_2025_template,
    "eba_baseline": eba_baseline_2025_template,
    "eba_adverse": eba_adverse_2025_template,
}


def get_stress_template(name: str) -> Optional[EBAStressTemplate]:
    """Return template by name (case-insensitive). None if unknown."""
    if not name:
        return None
    fn = _TEMPLATES.get(name.strip().lower())
    return fn() if fn else None


def list_stress_templates() -> list[str]:
    return ["baseline", "adverse"]
