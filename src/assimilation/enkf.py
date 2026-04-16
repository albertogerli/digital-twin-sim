"""Ensemble Kalman Filter for online data assimilation.

Bridges the offline hierarchical calibration (v2) with real-time scenario updates.
The ensemble is initialized from the posterior of the offline calibration, then
forecast/update cycles assimilate incoming observations (polls, sentiment, results).

State vector per ensemble member:
    x = [θ, z]
    θ = [alpha_herd, alpha_anchor, alpha_social, alpha_event]  (4 params)
    z = agent_positions  (n_agents positions in [-1, +1])

θ evolves via random walk: θ_{k+1} = θ_k + η,  η ~ N(0, Q_θ)
z evolves via the JAX simulator: z_{k+1} = step_round(z_k, θ, event_k)

The EnKF update assimilates observations using the stochastic EnKF algorithm
with multiplicative inflation to prevent ensemble collapse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
from jax.nn import sigmoid

from ..dynamics.opinion_dynamics_jax import (
    ScenarioData,
    N_FORCES,
    compute_forces,
    ema_standardize,
    smooth_clamp,
)
from ..dynamics.param_utils import get_default_frozen_params, CALIBRABLE_PARAMS


# ── State container ────────────────────────────────────────────

@dataclass
class EnKFState:
    """Ensemble state at a given time step.

    Attributes:
        params_ensemble: (n_ensemble, 4) — calibrable params per member.
        positions_ensemble: (n_ensemble, n_agents) — agent positions per member.
        original_positions_ensemble: (n_ensemble, n_agents) — anchor positions.
        ema_mu_ensemble: (n_ensemble, 5) — EMA force means.
        ema_sigma_ensemble: (n_ensemble, 5) — EMA force stds.
        step: Current round index (0-based).
        log: History of predictions and updates.
    """

    params_ensemble: jnp.ndarray        # (n_ensemble, 4)
    positions_ensemble: jnp.ndarray     # (n_ensemble, n_agents)
    original_positions_ensemble: jnp.ndarray  # (n_ensemble, n_agents)
    ema_mu_ensemble: jnp.ndarray        # (n_ensemble, 5)
    ema_sigma_ensemble: jnp.ndarray     # (n_ensemble, 5)
    step: int = 0
    log: list[dict] = field(default_factory=list)


# ── Readout function ──────────────────────────────────────────

def readout_pro_pct(positions: jnp.ndarray) -> jnp.ndarray:
    """Convert agent positions to pro percentage (0-100 scale).

    Uses the same smooth sigmoid readout as the simulator:
    soft_pro = σ((p - 0.05) / 0.02), soft_against = σ((-p - 0.05) / 0.02)
    pro_pct = 100 × Σ soft_pro / Σ (soft_pro + soft_against)
    """
    tau = 0.02
    soft_pro = sigmoid((positions - 0.05) / tau)
    soft_against = sigmoid((-positions - 0.05) / tau)
    decided = soft_pro + soft_against
    return jnp.sum(soft_pro) / (jnp.sum(decided) + 1e-8) * 100.0


def readout_pro_pct_batched(positions_batch: jnp.ndarray) -> jnp.ndarray:
    """Vectorized readout: (n_ensemble, n_agents) → (n_ensemble,)."""
    return jax.vmap(readout_pro_pct)(positions_batch)


# ── Single-member forecast step ──────────────────────────────

def _forecast_member(
    params_vec: jnp.ndarray,       # (4,) — calibrable params
    positions: jnp.ndarray,        # (n_agents,)
    original_positions: jnp.ndarray,
    ema_mu: jnp.ndarray,           # (5,)
    ema_sigma: jnp.ndarray,        # (5,)
    # Static scenario data (not vmapped):
    agent_types: jnp.ndarray,
    agent_rigidities: jnp.ndarray,
    agent_tolerances: jnp.ndarray,
    interaction_matrix: jnp.ndarray,
    # Round-specific data:
    event_mag: jnp.ndarray,        # scalar
    event_dir: jnp.ndarray,        # scalar
    llm_shift: jnp.ndarray,        # (n_agents,)
    # Frozen params:
    frozen_vec: jnp.ndarray,       # (4,) — [log_lam_e, log_lam_c, logit_ht, logit_ad]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Propagate one ensemble member through one simulator round.

    Mirrors step_round() from opinion_dynamics_jax but takes flat arrays
    instead of a params dict + ScenarioData, making it vmap-friendly.

    Returns: (new_positions, new_original, new_ema_mu, new_ema_sigma)
    """
    # Unpack frozen
    lambda_elite = jnp.exp(frozen_vec[0])
    lambda_citizen = jnp.exp(frozen_vec[1])
    herd_threshold = sigmoid(frozen_vec[2])
    anchor_drift = sigmoid(frozen_vec[3])

    # Per-agent step size and caps
    lambda_per_agent = jnp.where(agent_types == 0, lambda_elite, lambda_citizen)
    cap_per_agent = jnp.where(agent_types == 0, 0.15, 0.25)

    # Compute forces
    raw_forces = compute_forces(
        positions, original_positions,
        agent_rigidities, agent_tolerances,
        interaction_matrix, llm_shift,
        event_mag, event_dir, herd_threshold,
    )

    # EMA standardization
    std_forces, new_ema_mu, new_ema_sigma = ema_standardize(
        raw_forces, ema_mu, ema_sigma,
    )

    # Softmax mix
    alpha_vec = jnp.array([
        0.0,  # direct — gauge-fixed
        params_vec[0],  # alpha_herd
        params_vec[1],  # alpha_anchor
        params_vec[2],  # alpha_social
        params_vec[3],  # alpha_event
    ])
    pi = jax.nn.softmax(alpha_vec)

    combined = jnp.sum(pi[None, :] * std_forces, axis=-1)
    delta_p = smooth_clamp(lambda_per_agent * combined, cap_per_agent)

    new_positions = jnp.clip(positions + delta_p, -1.0, 1.0)
    new_original = original_positions + anchor_drift * (new_positions - original_positions)

    return new_positions, new_original, new_ema_mu, new_ema_sigma


# Vectorize over ensemble dimension (first arg of each array)
_forecast_ensemble = jax.vmap(
    _forecast_member,
    in_axes=(
        0,     # params_vec: (E, 4)
        0,     # positions: (E, A)
        0,     # original_positions: (E, A)
        0,     # ema_mu: (E, 5)
        0,     # ema_sigma: (E, 5)
        None,  # agent_types: (A,) — shared
        None,  # agent_rigidities: (A,) — shared
        None,  # agent_tolerances: (A,) — shared
        None,  # interaction_matrix: (A, A) — shared
        None,  # event_mag: scalar — shared
        None,  # event_dir: scalar — shared
        None,  # llm_shift: (A,) — shared
        None,  # frozen_vec: (4,) — shared
    ),
)


# ── EnKF class ────────────────────────────────────────────────

class EnsembleKalmanFilter:
    """Ensemble Kalman Filter for DigitalTwinSim online assimilation.

    Usage:
        1. Build from offline posterior: enkf.initialize_from_posterior(...)
        2. Per round: state = enkf.forecast(state, event)
        3. When observation arrives: state = enkf.update(state, obs_value, obs_var)
        4. Read prediction: enkf.get_prediction(state)
    """

    def __init__(
        self,
        scenario_data: ScenarioData,
        n_ensemble: int = 50,
        process_noise_params: float = 0.01,
        process_noise_state: float = 0.005,
        inflation_factor: float = 1.02,
        key: Optional[jax.Array] = None,
    ):
        """
        Args:
            scenario_data: Static scenario (agent types, interaction matrix, events).
            n_ensemble: Number of ensemble members.
            process_noise_params: Std of random walk on θ (Q_θ diagonal).
            process_noise_state: Std of additive noise on positions (Q_z diagonal).
            inflation_factor: Multiplicative inflation (1.0 = off, 1.02-1.05 typical).
            key: JAX PRNG key. If None, uses key(0).
        """
        self.scenario_data = scenario_data
        self.n_ensemble = n_ensemble
        self.n_agents = scenario_data.initial_positions.shape[0]
        self.n_params = len(CALIBRABLE_PARAMS)
        self.process_noise_params = process_noise_params
        self.process_noise_state = process_noise_state
        self.inflation_factor = inflation_factor
        self.key = key if key is not None else jax.random.PRNGKey(0)

        # Frozen params as a flat vector
        frozen = get_default_frozen_params()
        self.frozen_vec = jnp.array([
            frozen["log_lambda_elite"],
            frozen["log_lambda_citizen"],
            frozen["logit_herd_threshold"],
            frozen["logit_anchor_drift"],
        ])

    def initialize_from_posterior(
        self,
        posterior_samples: dict,
        initial_positions: Optional[jnp.ndarray] = None,
    ) -> EnKFState:
        """Initialize the ensemble from the offline calibration posterior.

        This is the BRIDGE between offline Bayesian calibration and online
        assimilation. Each ensemble member samples θ from the posterior.

        Args:
            posterior_samples: Output of calibration v2. Expected format:
                Either the full posteriors_v2.json dict (with "global" key)
                or a dict with "theta_s" key containing (n_samples, 4) array.
            initial_positions: Optional (n_agents,) initial positions.
                If None, uses scenario_data.initial_positions.
        """
        self.key, key_params, key_pos = jax.random.split(self.key, 3)

        # Extract parameter samples from posterior
        if "global" in posterior_samples:
            # posteriors_v2.json format: use global mu/sigma to sample
            mu = jnp.array(posterior_samples["global"]["mu_global"]["mean"])
            # Approximate sigma from CI width
            ci_lo = jnp.array(posterior_samples["global"]["mu_global"]["ci95_lo"])
            ci_hi = jnp.array(posterior_samples["global"]["mu_global"]["ci95_hi"])
            sigma = (ci_hi - ci_lo) / (2 * 1.96)
            params_ens = mu[None, :] + sigma[None, :] * jax.random.normal(
                key_params, (self.n_ensemble, self.n_params)
            )
        elif "theta_s" in posterior_samples:
            # Raw samples: (n_samples, 4)
            raw = jnp.array(posterior_samples["theta_s"])
            # Subsample or bootstrap to n_ensemble
            indices = jax.random.choice(
                key_params, raw.shape[0], (self.n_ensemble,), replace=True
            )
            params_ens = raw[indices]
        else:
            raise ValueError(
                "posterior_samples must have 'global' or 'theta_s' key"
            )

        # Initial positions with small perturbation
        if initial_positions is None:
            initial_positions = self.scenario_data.initial_positions
        pos_noise = self.process_noise_state * jax.random.normal(
            key_pos, (self.n_ensemble, self.n_agents)
        )
        positions_ens = jnp.clip(
            initial_positions[None, :] + pos_noise, -1.0, 1.0
        )

        return EnKFState(
            params_ensemble=params_ens,
            positions_ensemble=positions_ens,
            original_positions_ensemble=positions_ens.copy(),
            ema_mu_ensemble=jnp.zeros((self.n_ensemble, N_FORCES)),
            ema_sigma_ensemble=jnp.ones((self.n_ensemble, N_FORCES)),
            step=0,
            log=[],
        )

    def forecast(self, state: EnKFState, round_idx: Optional[int] = None) -> EnKFState:
        """Propagate the ensemble by one time step through the simulator.

        Uses the event data for the current round from scenario_data.

        Args:
            state: Current EnKF state.
            round_idx: Override round index (default: state.step).

        Returns:
            Updated EnKFState after forecast.
        """
        self.key, key_params, key_state = jax.random.split(self.key, 3)

        r = round_idx if round_idx is not None else state.step
        n_rounds = self.scenario_data.events.shape[0]
        # Clamp to valid round range
        r = min(r, n_rounds - 1)

        event_mag = self.scenario_data.events[r, 0]
        event_dir = self.scenario_data.events[r, 1]
        llm_shift = self.scenario_data.llm_shifts[r]

        # Add process noise to params (random walk)
        param_noise = self.process_noise_params * jax.random.normal(
            key_params, state.params_ensemble.shape
        )
        noisy_params = state.params_ensemble + param_noise

        # Propagate each ensemble member through the simulator
        new_pos, new_orig, new_ema_mu, new_ema_sigma = _forecast_ensemble(
            noisy_params,
            state.positions_ensemble,
            state.original_positions_ensemble,
            state.ema_mu_ensemble,
            state.ema_sigma_ensemble,
            self.scenario_data.agent_types,
            self.scenario_data.agent_rigidities,
            self.scenario_data.agent_tolerances,
            self.scenario_data.interaction_matrix,
            event_mag,
            event_dir,
            llm_shift,
            self.frozen_vec,
        )

        # Add process noise to positions
        pos_noise = self.process_noise_state * jax.random.normal(
            key_state, new_pos.shape
        )
        new_pos = jnp.clip(new_pos + pos_noise, -1.0, 1.0)

        return EnKFState(
            params_ensemble=noisy_params,
            positions_ensemble=new_pos,
            original_positions_ensemble=new_orig,
            ema_mu_ensemble=new_ema_mu,
            ema_sigma_ensemble=new_ema_sigma,
            step=state.step + 1,
            log=state.log,
        )

    def update(
        self,
        state: EnKFState,
        observation: float,
        obs_variance: float,
        obs_type: str = "polling",
    ) -> EnKFState:
        """Assimilate an observation using the stochastic EnKF update.

        The full state vector is x = [θ, z]. The observation operator H maps
        positions to pro_pct: y = h(z) + ε, ε ~ N(0, R).

        Steps:
            1. Compute predicted observations H(x_j) for each member.
            2. Compute ensemble anomaly covariance.
            3. Kalman gain K = P_xH^T (H P_xH^T + R)^{-1}.
            4. Stochastic update: x^a_j = x^f_j + K(y + ε_j - H(x^f_j)).
            5. Multiplicative inflation on ensemble spread.
            6. Clamp θ and z to physical ranges.

        Args:
            state: Current (forecast) EnKF state.
            observation: Observed pro_pct (0-100).
            obs_variance: Observation variance in pct² units.
            obs_type: Type tag for logging.

        Returns:
            Updated EnKFState after assimilation.
        """
        self.key, key_perturb = jax.random.split(self.key)

        E = self.n_ensemble
        A = self.n_agents

        # 1. Predicted observations: H(x_j) = readout(positions_j)
        y_pred = readout_pro_pct_batched(state.positions_ensemble)  # (E,)

        # 2. Build full state matrix: [θ | z]  → (E, 4+A)
        X = jnp.concatenate(
            [state.params_ensemble, state.positions_ensemble], axis=1
        )  # (E, 4+A)

        # Ensemble mean and anomalies
        X_mean = jnp.mean(X, axis=0)                    # (4+A,)
        X_anom = X - X_mean[None, :]                     # (E, 4+A)

        y_mean = jnp.mean(y_pred)                         # scalar
        y_anom = y_pred - y_mean                           # (E,)

        # 3. Covariances (using ensemble formulas)
        # P_xy = (1/(E-1)) X_anom^T y_anom → (4+A,)
        P_xy = jnp.dot(X_anom.T, y_anom) / (E - 1)       # (4+A,)
        # P_yy = (1/(E-1)) y_anom^T y_anom → scalar
        P_yy = jnp.dot(y_anom, y_anom) / (E - 1)          # scalar

        # Kalman gain: K = P_xy / (P_yy + R)
        R = obs_variance
        K = P_xy / (P_yy + R + 1e-10)                     # (4+A,)

        # 4. Stochastic update: perturb observation for each member
        obs_perturbations = jnp.sqrt(R) * jax.random.normal(key_perturb, (E,))
        innovations = (observation + obs_perturbations) - y_pred  # (E,)

        # Update: X^a = X^f + K * innovation
        X_updated = X + K[None, :] * innovations[:, None]  # (E, 4+A)

        # 5. Multiplicative inflation around ensemble mean
        if self.inflation_factor > 1.0:
            X_mean_updated = jnp.mean(X_updated, axis=0)
            X_updated = X_mean_updated[None, :] + self.inflation_factor * (
                X_updated - X_mean_updated[None, :]
            )

        # 6. Split back and clamp
        params_updated = X_updated[:, :self.n_params]
        positions_updated = jnp.clip(X_updated[:, self.n_params:], -1.0, 1.0)

        # Log the update
        pred = self.get_prediction(state)
        log_entry = {
            "step": state.step,
            "action": "update",
            "obs_type": obs_type,
            "observation": float(observation),
            "obs_variance": float(obs_variance),
            "pre_update_mean": float(pred["pro_pct_mean"]),
            "pre_update_std": float(pred["pro_pct_std"]),
        }

        return EnKFState(
            params_ensemble=params_updated,
            positions_ensemble=positions_updated,
            original_positions_ensemble=state.original_positions_ensemble,
            ema_mu_ensemble=state.ema_mu_ensemble,
            ema_sigma_ensemble=state.ema_sigma_ensemble,
            step=state.step,
            log=state.log + [log_entry],
        )

    def get_prediction(self, state: EnKFState) -> dict:
        """Extract current prediction with uncertainty from the ensemble.

        Returns:
            dict with pro_pct_mean, pro_pct_std, pro_pct_ci95,
            params_mean, params_std, ensemble_spread.
        """
        # Pro percentage from each member
        y_ens = readout_pro_pct_batched(state.positions_ensemble)  # (E,)
        pro_mean = float(jnp.mean(y_ens))
        pro_std = float(jnp.std(y_ens))
        pro_lo = float(jnp.percentile(y_ens, 5.0))
        pro_hi = float(jnp.percentile(y_ens, 95.0))

        # Parameter statistics
        p_mean = jnp.mean(state.params_ensemble, axis=0)  # (4,)
        p_std = jnp.std(state.params_ensemble, axis=0)    # (4,)

        params_mean = {
            name: float(p_mean[i]) for i, name in enumerate(CALIBRABLE_PARAMS)
        }
        params_std = {
            name: float(p_std[i]) for i, name in enumerate(CALIBRABLE_PARAMS)
        }

        # Ensemble spread diagnostic: mean std of positions across agents
        pos_std_per_agent = jnp.std(state.positions_ensemble, axis=0)  # (A,)
        ensemble_spread = float(jnp.mean(pos_std_per_agent))

        return {
            "pro_pct_mean": pro_mean,
            "pro_pct_std": pro_std,
            "pro_pct_ci90": (pro_lo, pro_hi),
            "params_mean": params_mean,
            "params_std": params_std,
            "ensemble_spread": ensemble_spread,
        }
