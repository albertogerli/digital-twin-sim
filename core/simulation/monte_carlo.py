"""Monte Carlo simulation runner — N parallel runs with parameter perturbation.

Runs the full LLM simulation multiple times with slightly different parameters
around the calibrated values, producing confidence intervals on outcomes.

Usage:
    engine = MonteCarloEngine(base_config, domain, n_runs=20)
    results = await engine.run(llm, progress_callback)
    # results.confidence_intervals, results.outcome_distribution, etc.
"""

import asyncio
import copy
import logging
import random
from dataclasses import dataclass, field
from statistics import mean, stdev

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloRoundResult:
    """Aggregated results for a single round across all runs."""
    round: int
    polarization_mean: float
    polarization_std: float
    polarization_ci_low: float
    polarization_ci_high: float
    avg_position_mean: float
    avg_position_std: float
    sentiment_positive_mean: float
    sentiment_neutral_mean: float
    sentiment_negative_mean: float
    coalition_sizes: list[dict]  # [{label, mean_size, std_size}]


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo analysis result."""
    n_runs: int
    n_completed: int
    rounds: list[MonteCarloRoundResult]
    final_polarization_mean: float
    final_polarization_std: float
    final_polarization_ci: tuple[float, float]
    final_position_mean: float
    final_position_std: float
    final_position_ci: tuple[float, float]
    outcome_pro_pct: float  # % of runs where avg_position > 0
    outcome_against_pct: float
    parameter_sets: list[dict]
    per_run_summaries: list[dict]


def perturb_params(
    base_params: dict,
    perturbation_pct: float = 0.15,
    seed: int = None,
) -> dict:
    """Perturb calibrated parameters by ±perturbation_pct.

    Uses uniform distribution around each parameter value.
    Clamps to reasonable ranges.
    """
    rng = random.Random(seed)
    perturbed = {}

    param_ranges = {
        "anchor_weight": (0.01, 0.30),
        "social_weight": (0.03, 0.35),
        "event_weight": (0.01, 0.20),
        "herd_weight": (0.01, 0.15),
        "herd_threshold": (0.05, 0.50),
        "direct_shift_weight": (0.1, 0.8),
        "anchor_drift_rate": (0.05, 0.50),
    }

    for key, value in base_params.items():
        if key in param_ranges:
            lo, hi = param_ranges[key]
            delta = value * perturbation_pct
            perturbed_val = rng.uniform(value - delta, value + delta)
            perturbed[key] = max(lo, min(hi, perturbed_val))
        else:
            perturbed[key] = value

    return perturbed


def _compute_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval using normal approximation."""
    if len(values) < 2:
        m = values[0] if values else 0
        return (m, m)

    m = mean(values)
    s = stdev(values)
    # z-score for 95% CI
    z = 1.96 if confidence == 0.95 else 1.645
    margin = z * s / (len(values) ** 0.5)
    return (m - margin, m + margin)


class MonteCarloEngine:
    """Runs N simulations with perturbed parameters and aggregates results."""

    def __init__(
        self,
        n_runs: int = 20,
        perturbation_pct: float = 0.15,
        base_seed: int = 42,
    ):
        self.n_runs = n_runs
        self.perturbation_pct = perturbation_pct
        self.base_seed = base_seed

    def generate_parameter_sets(self, base_params: dict) -> list[dict]:
        """Generate N parameter sets: first is base, rest are perturbed."""
        param_sets = [base_params.copy()]  # Run 0 = unperturbed baseline
        for i in range(1, self.n_runs):
            param_sets.append(perturb_params(
                base_params,
                perturbation_pct=self.perturbation_pct,
                seed=self.base_seed + i,
            ))
        return param_sets

    def aggregate_results(
        self,
        all_runs: list[dict],
        parameter_sets: list[dict],
    ) -> MonteCarloResult:
        """Aggregate results from N simulation runs.

        Each run is a dict with:
            - rounds: [{round, polarization, avg_position, sentiment, coalitions}]
            - final_polarization: float
            - final_avg_position: float
        """
        if not all_runs:
            return MonteCarloResult(
                n_runs=self.n_runs, n_completed=0, rounds=[],
                final_polarization_mean=0, final_polarization_std=0,
                final_polarization_ci=(0, 0),
                final_position_mean=0, final_position_std=0,
                final_position_ci=(0, 0),
                outcome_pro_pct=50, outcome_against_pct=50,
                parameter_sets=parameter_sets, per_run_summaries=[],
            )

        # Aggregate per-round
        max_rounds = max(len(r.get("rounds", [])) for r in all_runs)
        round_results = []

        for rd in range(max_rounds):
            pols = []
            positions = []
            sent_pos = []
            sent_neu = []
            sent_neg = []

            for run in all_runs:
                rounds = run.get("rounds", [])
                if rd < len(rounds):
                    r = rounds[rd]
                    pols.append(r.get("polarization", 0))
                    positions.append(r.get("avg_position", 0))
                    s = r.get("sentiment", {})
                    sent_pos.append(s.get("positive", 0))
                    sent_neu.append(s.get("neutral", 0))
                    sent_neg.append(s.get("negative", 0))

            pol_ci = _compute_ci(pols)
            round_results.append(MonteCarloRoundResult(
                round=rd + 1,
                polarization_mean=mean(pols) if pols else 0,
                polarization_std=stdev(pols) if len(pols) > 1 else 0,
                polarization_ci_low=pol_ci[0],
                polarization_ci_high=pol_ci[1],
                avg_position_mean=mean(positions) if positions else 0,
                avg_position_std=stdev(positions) if len(positions) > 1 else 0,
                sentiment_positive_mean=mean(sent_pos) if sent_pos else 0,
                sentiment_neutral_mean=mean(sent_neu) if sent_neu else 0,
                sentiment_negative_mean=mean(sent_neg) if sent_neg else 0,
                coalition_sizes=[],
            ))

        # Final metrics
        final_pols = [r.get("final_polarization", 0) for r in all_runs]
        final_positions = [r.get("final_avg_position", 0) for r in all_runs]

        pro_count = sum(1 for p in final_positions if p > 0.05)
        against_count = sum(1 for p in final_positions if p < -0.05)
        decided = pro_count + against_count or 1

        return MonteCarloResult(
            n_runs=self.n_runs,
            n_completed=len(all_runs),
            rounds=round_results,
            final_polarization_mean=mean(final_pols) if final_pols else 0,
            final_polarization_std=stdev(final_pols) if len(final_pols) > 1 else 0,
            final_polarization_ci=_compute_ci(final_pols),
            final_position_mean=mean(final_positions) if final_positions else 0,
            final_position_std=stdev(final_positions) if len(final_positions) > 1 else 0,
            final_position_ci=_compute_ci(final_positions),
            outcome_pro_pct=round(pro_count / decided * 100, 1),
            outcome_against_pct=round(against_count / decided * 100, 1),
            parameter_sets=parameter_sets,
            per_run_summaries=[
                {
                    "run": i,
                    "final_polarization": r.get("final_polarization", 0),
                    "final_avg_position": r.get("final_avg_position", 0),
                    "params": parameter_sets[i] if i < len(parameter_sets) else {},
                }
                for i, r in enumerate(all_runs)
            ],
        )

    def result_to_dict(self, result: MonteCarloResult) -> dict:
        """Convert MonteCarloResult to JSON-serializable dict."""
        return {
            "n_runs": result.n_runs,
            "n_completed": result.n_completed,
            "final_polarization": {
                "mean": round(result.final_polarization_mean, 2),
                "std": round(result.final_polarization_std, 2),
                "ci_low": round(result.final_polarization_ci[0], 2),
                "ci_high": round(result.final_polarization_ci[1], 2),
            },
            "final_position": {
                "mean": round(result.final_position_mean, 3),
                "std": round(result.final_position_std, 3),
                "ci_low": round(result.final_position_ci[0], 3),
                "ci_high": round(result.final_position_ci[1], 3),
            },
            "outcome_probability": {
                "pro_pct": result.outcome_pro_pct,
                "against_pct": result.outcome_against_pct,
            },
            "rounds": [
                {
                    "round": r.round,
                    "polarization": {
                        "mean": round(r.polarization_mean, 2),
                        "std": round(r.polarization_std, 2),
                        "ci_low": round(r.polarization_ci_low, 2),
                        "ci_high": round(r.polarization_ci_high, 2),
                    },
                    "avg_position": {
                        "mean": round(r.avg_position_mean, 3),
                        "std": round(r.avg_position_std, 3),
                    },
                    "sentiment": {
                        "positive": round(r.sentiment_positive_mean, 2),
                        "neutral": round(r.sentiment_neutral_mean, 2),
                        "negative": round(r.sentiment_negative_mean, 2),
                    },
                }
                for r in result.rounds
            ],
            "per_run": result.per_run_summaries,
        }
