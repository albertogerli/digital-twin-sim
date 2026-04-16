"""Online assimilation runner — orchestrates EnKF with the simulator.

Connects the EnsembleKalmanFilter to scenario data and observation streams,
running the forecast/update cycle round by round.
"""

from __future__ import annotations

import json
from typing import Optional

import jax.numpy as jnp

from .enkf import EnsembleKalmanFilter, EnKFState
from .data_sources import ObservationSource
from ..dynamics.opinion_dynamics_jax import ScenarioData
from ..observation.observation_model import (
    build_scenario_data_from_json,
    load_scenario_observations,
)


class OnlineAssimilationRunner:
    """Run a scenario with real-time data assimilation.

    Usage:
        runner = OnlineAssimilationRunner(enkf, posterior_samples, scenario_config)
        results = runner.run_with_observations(observations)
    """

    def __init__(
        self,
        enkf: EnsembleKalmanFilter,
        posterior_samples: dict,
        scenario_config: Optional[dict] = None,
    ):
        """
        Args:
            enkf: Configured EnsembleKalmanFilter instance.
            posterior_samples: Offline calibration posterior (posteriors_v2.json format
                or raw theta_s samples).
            scenario_config: Optional scenario JSON dict (for metadata/logging).
        """
        self.enkf = enkf
        self.posterior_samples = posterior_samples
        self.scenario_config = scenario_config

    def run_with_observations(
        self,
        observations: list[tuple[int, ObservationSource]],
        n_rounds: Optional[int] = None,
    ) -> list[dict]:
        """Run the simulation round by round, assimilating observations.

        For each round:
            1. Forecast: propagate ensemble through simulator.
            2. If observation available for this round: update (assimilate).
            3. Record prediction with CI.

        Args:
            observations: List of (round_number, ObservationSource) pairs.
                round_number is 1-based (matching scenario convention).
            n_rounds: Total rounds to simulate. If None, inferred from
                scenario_data.events.shape[0].

        Returns:
            List of prediction dicts per round, each containing:
                round, pro_pct_mean, pro_pct_std, pro_pct_ci90,
                params_mean, ensemble_spread, had_observation,
                observation_value (if applicable).
        """
        if n_rounds is None:
            n_rounds = self.enkf.scenario_data.events.shape[0]

        # Build observation lookup: round_idx (0-based) → ObservationSource
        obs_by_round = {}
        for r_1based, obs in observations:
            obs_by_round[r_1based - 1] = obs

        # Initialize ensemble from posterior
        state = self.enkf.initialize_from_posterior(self.posterior_samples)

        # Record initial (prior) prediction
        prior = self.enkf.get_prediction(state)

        results = []
        for r in range(n_rounds):
            # 1. Forecast
            state = self.enkf.forecast(state, round_idx=r)

            # 2. Assimilate observation if available
            had_obs = False
            obs_val = None
            if r in obs_by_round:
                value, variance, obs_type = obs_by_round[r].to_observation()
                state = self.enkf.update(state, value, variance, obs_type)
                had_obs = True
                obs_val = value

            # 3. Record prediction
            pred = self.enkf.get_prediction(state)
            results.append({
                "round": r + 1,
                **pred,
                "had_observation": had_obs,
                "observation_value": obs_val,
            })

        return results

    @classmethod
    def from_scenario_path(
        cls,
        scenario_path: str,
        posterior_path: str,
        n_ensemble: int = 50,
        process_noise_params: float = 0.01,
        process_noise_state: float = 0.005,
        inflation_factor: float = 1.02,
        seed: int = 42,
    ) -> OnlineAssimilationRunner:
        """Convenience constructor: load scenario and posterior from file paths.

        Args:
            scenario_path: Path to empirical scenario .json file.
            posterior_path: Path to posteriors_v2.json.
            n_ensemble: EnKF ensemble size.
            process_noise_params: θ random walk std.
            process_noise_state: Position noise std.
            inflation_factor: Multiplicative inflation.
            seed: Random seed for interaction matrix.
        """
        import jax

        scenario_dict, _ = load_scenario_observations(scenario_path)
        scenario_data = build_scenario_data_from_json(scenario_dict, seed=seed)

        with open(posterior_path, "r") as f:
            posterior = json.load(f)

        enkf = EnsembleKalmanFilter(
            scenario_data=scenario_data,
            n_ensemble=n_ensemble,
            process_noise_params=process_noise_params,
            process_noise_state=process_noise_state,
            inflation_factor=inflation_factor,
            key=jax.random.PRNGKey(seed),
        )

        return cls(
            enkf=enkf,
            posterior_samples=posterior,
            scenario_config=scenario_dict,
        )
