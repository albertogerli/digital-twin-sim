"""Regime switching extension for opinion dynamics.

Adds a hidden Markov-like regime switching mechanism to the base simulator.
Two regimes:
    Regime 0 — "Normal dynamics": standard force-based opinion evolution.
    Regime 1 — "Crisis/collapse": amplified herd/event forces, suppressed anchoring,
               larger step sizes. Models discontinuous trust collapse.

The switching is SOFT (sigmoid-based) for JAX differentiability and jit-compatibility.
No discrete branching — parameters are interpolated by regime probability.

Key insight: financial crisis scenarios (WeWork δ=-1.38, SVB δ=-0.84) show
systematic over-prediction because the current model can't produce rapid drops.
Regime switching adds this capability without changing normal dynamics.

Usage:
    from src.dynamics.regime_switching import RegimeSwitchingSimulator

    rs = RegimeSwitchingSimulator()
    result = rs.simulate(params, scenario_data, institutional_trust=0.3)
    # result has same keys as simulate_scenario + regime_probs, regime_sequence
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.nn import sigmoid

from .opinion_dynamics_jax import (
    ScenarioData,
    N_FORCES,
    compute_forces,
    ema_standardize,
    smooth_clamp,
)
from .param_utils import get_default_frozen_params


# ── Crisis parameter defaults ──────────────────────────────────

CRISIS_DEFAULTS = {
    "lambda_multiplier": 3.0,       # λ_crisis = λ_normal × multiplier
    "anchor_suppression": 0.1,      # anchor force multiplied by this (near 0)
    "event_amplification": 2.5,     # event force multiplied by this
    "herd_amplification": 2.0,      # herd force multiplied by this
    "contagion_speed": 0.4,         # direct crisis push strength (tunable via SVI in v3)
}

TRANSITION_DEFAULTS = {
    "shock_trigger_threshold": 0.5,     # shock_magnitude above which crisis triggers
    "velocity_trigger_threshold": 0.1,  # |Δp_mean| above which crisis triggers
    "trust_sensitivity": 1.5,           # positive: low trust → higher crisis probability
    "crisis_duration_mean": 2.5,        # expected rounds in crisis before recovery
    "recovery_rate": 0.4,              # per-round probability of exiting crisis
}


# ── Regime carry state ─────────────────────────────────────────

class RegimeCarry(NamedTuple):
    """State carried through scan for regime switching."""
    positions: jnp.ndarray           # [n_agents]
    original_positions: jnp.ndarray  # [n_agents]
    ema_mu: jnp.ndarray              # [5]
    ema_sigma: jnp.ndarray           # [5]
    regime_prob: jnp.ndarray         # scalar — P(crisis) for current round
    rounds_in_crisis: jnp.ndarray    # scalar — soft count of rounds in crisis
    prev_mean_velocity: jnp.ndarray  # scalar — |Δp_mean| from previous round


# ── Regime detection ──────────────────────────────────────────

def compute_regime_prob(
    shock_mag: jnp.ndarray,
    mean_velocity: jnp.ndarray,
    institutional_trust: jnp.ndarray,
    rounds_in_crisis: jnp.ndarray,
    prev_regime_prob: jnp.ndarray,
    transition_params: dict,
) -> jnp.ndarray:
    """Compute P(crisis) for the current round.

    Combines multiple signals via logistic regression:
        logit(P) = w_shock * (|shock| - threshold)
                 + w_vel * (velocity - vel_threshold)
                 + w_trust * (1 - institutional_trust)
                 + w_recovery * (-rounds_in_crisis / duration_mean)
                 + w_momentum * logit(prev_regime_prob)

    All weights are positive, thresholds center the activation.
    Returns a value in (0, 1).
    """
    shock_thresh = transition_params["shock_trigger_threshold"]
    vel_thresh = transition_params["velocity_trigger_threshold"]
    trust_sens = transition_params["trust_sensitivity"]
    duration_mean = transition_params["crisis_duration_mean"]

    # Activation terms — each contributes to logit(P(crisis))

    # Shock: main trigger. Large shocks (> threshold) push strongly into crisis.
    shock_term = 8.0 * (shock_mag - shock_thresh)

    # Velocity: rapid position changes indicate crisis dynamics already underway.
    vel_term = 4.0 * (mean_velocity - vel_thresh)

    # Trust: low institutional trust lowers the crisis threshold.
    # trust_sens is positive: (1 - trust) is high when trust is low.
    trust_term = trust_sens * (1.0 - institutional_trust)

    # Recovery: the longer in crisis, the more likely to exit (mean reversion).
    recovery_term = -1.5 * rounds_in_crisis / duration_mean

    # Momentum: crisis is sticky (additive, avoids logit-of-zero issue).
    # Maps prev_regime_prob ∈ [0,1] to [-0.5, 2.5] — mild hysteresis.
    momentum_term = 3.0 * prev_regime_prob - 0.5

    # Base bias: default to normal regime
    base_bias = -1.5

    logit_p = base_bias + shock_term + vel_term + trust_term + recovery_term + momentum_term
    return sigmoid(logit_p)


# ── Modified single-round step ─────────────────────────────────

def step_round_with_regime(
    positions: jnp.ndarray,
    original_positions: jnp.ndarray,
    ema_mu: jnp.ndarray,
    ema_sigma: jnp.ndarray,
    regime_prob: jnp.ndarray,
    params: dict,
    crisis_params: dict,
    scenario: ScenarioData,
    round_idx: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Execute one round with soft regime switching.

    Parameters are interpolated:
        θ_eff = (1 - regime_prob) * θ_normal + regime_prob * θ_crisis

    This is differentiable and avoids branching.

    Returns: (new_positions, new_original, new_ema_mu, new_ema_sigma)
    """
    p = regime_prob  # shorthand

    # Extract normal constrained params
    lambda_elite_n = jnp.exp(params["log_lambda_elite"])
    lambda_citizen_n = jnp.exp(params["log_lambda_citizen"])
    herd_threshold = sigmoid(params["logit_herd_threshold"])
    anchor_drift_n = sigmoid(params["logit_anchor_drift"])

    # Crisis: amplified step sizes
    lambda_mult = crisis_params["lambda_multiplier"]
    lambda_elite_c = lambda_elite_n * lambda_mult
    lambda_citizen_c = lambda_citizen_n * lambda_mult

    # Interpolated step sizes
    lambda_elite = (1.0 - p) * lambda_elite_n + p * lambda_elite_c
    lambda_citizen = (1.0 - p) * lambda_citizen_n + p * lambda_citizen_c

    # Interpolated anchor drift (suppressed in crisis)
    anchor_drift_c = anchor_drift_n * crisis_params["anchor_suppression"]
    anchor_drift = (1.0 - p) * anchor_drift_n + p * anchor_drift_c

    lambda_per_agent = jnp.where(
        scenario.agent_types == 0, lambda_elite, lambda_citizen
    )
    cap_per_agent = jnp.where(scenario.agent_types == 0, 0.15, 0.25)
    # In crisis, relax caps
    cap_crisis = cap_per_agent * lambda_mult
    cap_effective = (1.0 - p) * cap_per_agent + p * cap_crisis

    # Event and round data
    shock_mag = scenario.events[round_idx, 0]
    shock_dir = scenario.events[round_idx, 1]
    llm_shift = scenario.llm_shifts[round_idx]

    # Compute raw forces
    raw_forces = compute_forces(
        positions, original_positions,
        scenario.agent_rigidities, scenario.agent_tolerances,
        scenario.interaction_matrix, llm_shift,
        shock_mag, shock_dir, herd_threshold,
    )

    # EMA standardization
    std_forces, new_ema_mu, new_ema_sigma = ema_standardize(
        raw_forces, ema_mu, ema_sigma,
    )

    # Softmax mixing weights — regime-dependent
    # Normal weights
    alpha_herd_n = params["alpha_herd"]
    alpha_anchor_n = params["alpha_anchor"]
    alpha_social_n = params["alpha_social"]
    alpha_event_n = params["alpha_event"]

    # Crisis weights: amplify herd and event, suppress anchor
    alpha_herd_c = alpha_herd_n + jnp.log(crisis_params["herd_amplification"])
    alpha_anchor_c = alpha_anchor_n - 2.0  # suppress anchor (large negative shift)
    alpha_social_c = alpha_social_n
    alpha_event_c = alpha_event_n + jnp.log(crisis_params["event_amplification"])

    # Interpolated alphas
    alpha_vec = jnp.array([
        0.0,  # direct — gauge-fixed
        (1.0 - p) * alpha_herd_n + p * alpha_herd_c,
        (1.0 - p) * alpha_anchor_n + p * alpha_anchor_c,
        (1.0 - p) * alpha_social_n + p * alpha_social_c,
        (1.0 - p) * alpha_event_n + p * alpha_event_c,
    ])
    pi = jax.nn.softmax(alpha_vec)

    combined = jnp.sum(pi[None, :] * std_forces, axis=-1)
    delta_p = smooth_clamp(lambda_per_agent * combined, cap_effective)

    # Crisis-specific direct push: bypass force mixing entirely.
    # In a trust collapse, agents are pushed directly in the shock direction,
    # proportional to their susceptibility (1 - rigidity).
    # This produces the discontinuous jump that linear forces can't.
    shock_mag = scenario.events[round_idx, 0]
    shock_dir = scenario.events[round_idx, 1]
    susceptibility = 1.0 - scenario.agent_rigidities
    crisis_push = (
        p * crisis_params["contagion_speed"]
        * shock_mag * shock_dir * susceptibility
    )

    new_positions = jnp.clip(positions + delta_p + crisis_push, -1.0, 1.0)
    new_original = original_positions + anchor_drift * (new_positions - original_positions)

    return new_positions, new_original, new_ema_mu, new_ema_sigma


# ── Scan body with regime ────────────────────────────────────

def _make_scan_body(params, crisis_params, transition_params,
                    scenario, institutional_trust):
    """Create a scan body closure with regime switching.

    Returns a function compatible with jax.lax.scan.
    """
    def scan_body(carry: RegimeCarry, round_idx):
        # Unpack carry
        positions = carry.positions
        original_positions = carry.original_positions
        ema_mu = carry.ema_mu
        ema_sigma = carry.ema_sigma
        prev_regime_prob = carry.regime_prob
        rounds_in_crisis = carry.rounds_in_crisis
        prev_velocity = carry.prev_mean_velocity

        # Compute current shock magnitude
        shock_mag = scenario.events[round_idx, 0]

        # Compute regime probability
        regime_prob = compute_regime_prob(
            shock_mag, prev_velocity, institutional_trust,
            rounds_in_crisis, prev_regime_prob, transition_params,
        )

        # Step with regime-interpolated params
        new_pos, new_orig, new_ema_mu, new_ema_sigma = step_round_with_regime(
            positions, original_positions, ema_mu, ema_sigma,
            regime_prob, params, crisis_params, scenario, round_idx,
        )

        # Track velocity for next round's regime detection
        mean_velocity = jnp.mean(jnp.abs(new_pos - positions))

        # Soft count of rounds in crisis (accumulate regime_prob)
        new_rounds_in_crisis = rounds_in_crisis * 0.9 + regime_prob  # decayed accumulator

        new_carry = RegimeCarry(
            positions=new_pos,
            original_positions=new_orig,
            ema_mu=new_ema_mu,
            ema_sigma=new_ema_sigma,
            regime_prob=regime_prob,
            rounds_in_crisis=new_rounds_in_crisis,
            prev_mean_velocity=mean_velocity,
        )

        # Output: positions + regime_prob per round
        return new_carry, (new_pos, regime_prob)

    return scan_body


# ── Main simulator class ──────────────────────────────────────

class RegimeSwitchingSimulator:
    """Wraps the base simulator with regime switching.

    Normal scenarios (low shocks, stable dynamics) produce regime_prob ≈ 0
    and results identical to the base simulator. Crisis scenarios (large shocks,
    rapid movement) trigger regime_prob > 0, enabling faster dynamics.
    """

    def __init__(
        self,
        crisis_params: Optional[dict] = None,
        transition_params: Optional[dict] = None,
    ):
        self.crisis_params = {**CRISIS_DEFAULTS, **(crisis_params or {})}
        self.transition_params = {**TRANSITION_DEFAULTS, **(transition_params or {})}

    def simulate(
        self,
        params: dict,
        scenario_data: ScenarioData,
        institutional_trust: float = 0.5,
    ) -> dict:
        """Simulate with regime switching.

        Args:
            params: Full parameter dict (calibrable + frozen).
            scenario_data: ScenarioData for the scenario.
            institutional_trust: Scenario-level covariable (0-1).

        Returns:
            Dict with standard keys + regime info:
                trajectories: [n_rounds, n_agents]
                pro_fraction: [n_rounds]
                final_pro_pct: scalar (0-100)
                mix_weights: [5] (from last round's regime-adjusted weights)
                regime_probs: [n_rounds] — P(crisis) per round
                regime_sequence: [n_rounds] — 1 if regime_prob > 0.5, else 0
        """
        init_pos = scenario_data.initial_positions
        n_rounds = scenario_data.events.shape[0]

        init_carry = RegimeCarry(
            positions=init_pos,
            original_positions=init_pos,
            ema_mu=jnp.zeros(N_FORCES),
            ema_sigma=jnp.ones(N_FORCES),
            regime_prob=jnp.array(0.0),
            rounds_in_crisis=jnp.array(0.0),
            prev_mean_velocity=jnp.array(0.0),
        )

        scan_body = _make_scan_body(
            params, self.crisis_params, self.transition_params,
            scenario_data, jnp.array(institutional_trust),
        )

        _, (trajectories, regime_probs) = jax.lax.scan(
            scan_body, init_carry, jnp.arange(n_rounds),
        )
        # trajectories: [n_rounds, n_agents]
        # regime_probs: [n_rounds]

        # Pro fraction — same readout as base simulator
        tau_pro = 0.02
        soft_pro = sigmoid((trajectories - 0.05) / tau_pro)
        soft_against = sigmoid((-trajectories - 0.05) / tau_pro)
        soft_decided = soft_pro + soft_against
        decided_count = jnp.sum(soft_decided, axis=1) + 1e-8
        pro_fraction = jnp.sum(soft_pro, axis=1) / decided_count

        # Mix weights from final round (approximate — uses normal params)
        alpha_vec = jnp.array([
            0.0, params["alpha_herd"], params["alpha_anchor"],
            params["alpha_social"], params["alpha_event"],
        ])
        mix_weights = jax.nn.softmax(alpha_vec)

        final_pro_pct = pro_fraction[-1] * 100.0
        regime_sequence = (regime_probs > 0.5).astype(jnp.int32)

        return {
            "trajectories": trajectories,
            "pro_fraction": pro_fraction,
            "mix_weights": mix_weights,
            "final_pro_pct": final_pro_pct,
            "regime_probs": regime_probs,
            "regime_sequence": regime_sequence,
        }


# JIT-compiled version for inference
def simulate_with_regimes_jit(
    params: dict,
    scenario_data: ScenarioData,
    crisis_params: dict,
    transition_params: dict,
    institutional_trust: float = 0.5,
) -> dict:
    """Functional interface for jit compilation.

    Avoids class method overhead — suitable for jax.jit wrapping.
    """
    sim = RegimeSwitchingSimulator(crisis_params, transition_params)
    return sim.simulate(params, scenario_data, institutional_trust)
