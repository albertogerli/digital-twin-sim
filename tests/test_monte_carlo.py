"""Tests for Monte Carlo simulation engine — perturbation, aggregation, CI."""

import os
import sys
from statistics import mean, stdev

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulation.monte_carlo import (
    MonteCarloEngine,
    MonteCarloResult,
    _compute_ci,
    perturb_params,
)

# ── Base parameters for testing ──────────────────────────────────────────────

BASE_PARAMS = {
    "anchor_weight": 0.10,
    "social_weight": 0.15,
    "event_weight": 0.05,
    "herd_weight": 0.05,
    "herd_threshold": 0.20,
    "direct_shift_weight": 0.40,
    "anchor_drift_rate": 0.20,
}


# ── Tests: parameter perturbation ────────────────────────────────────────────


class TestPerturbParams:
    def test_returns_all_keys(self):
        """Perturbed params should contain all original keys."""
        perturbed = perturb_params(BASE_PARAMS, perturbation_pct=0.15, seed=42)
        assert set(perturbed.keys()) == set(BASE_PARAMS.keys())

    def test_perturbed_values_differ_from_base(self):
        """With non-zero perturbation, at least some values should change."""
        perturbed = perturb_params(BASE_PARAMS, perturbation_pct=0.15, seed=42)
        differences = sum(
            1 for k in BASE_PARAMS
            if abs(perturbed[k] - BASE_PARAMS[k]) > 1e-10
        )
        assert differences > 0

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical perturbations."""
        p1 = perturb_params(BASE_PARAMS, perturbation_pct=0.15, seed=99)
        p2 = perturb_params(BASE_PARAMS, perturbation_pct=0.15, seed=99)
        for k in BASE_PARAMS:
            assert p1[k] == p2[k]

    def test_different_seeds_produce_different_params(self):
        """Different seeds should usually produce different values."""
        p1 = perturb_params(BASE_PARAMS, perturbation_pct=0.15, seed=1)
        p2 = perturb_params(BASE_PARAMS, perturbation_pct=0.15, seed=2)
        differ = any(abs(p1[k] - p2[k]) > 1e-10 for k in BASE_PARAMS)
        assert differ

    def test_values_within_valid_ranges(self):
        """Perturbed values must stay within defined min/max ranges."""
        param_ranges = {
            "anchor_weight": (0.01, 0.30),
            "social_weight": (0.03, 0.35),
            "event_weight": (0.01, 0.20),
            "herd_weight": (0.01, 0.15),
            "herd_threshold": (0.05, 0.50),
            "direct_shift_weight": (0.1, 0.8),
            "anchor_drift_rate": (0.05, 0.50),
        }
        for seed in range(20):
            perturbed = perturb_params(BASE_PARAMS, perturbation_pct=0.30, seed=seed)
            for k, (lo, hi) in param_ranges.items():
                assert lo <= perturbed[k] <= hi, (
                    f"seed={seed}, {k}={perturbed[k]} out of [{lo}, {hi}]"
                )

    def test_unknown_keys_pass_through(self):
        """Keys not in param_ranges should pass through unchanged."""
        params = {**BASE_PARAMS, "custom_param": 42}
        perturbed = perturb_params(params, perturbation_pct=0.15, seed=7)
        assert perturbed["custom_param"] == 42

    def test_zero_perturbation_returns_base(self):
        """With perturbation_pct=0, values should equal base."""
        perturbed = perturb_params(BASE_PARAMS, perturbation_pct=0.0, seed=1)
        for k in BASE_PARAMS:
            assert abs(perturbed[k] - BASE_PARAMS[k]) < 1e-10


# ── Tests: generate_parameter_sets ───────────────────────────────────────────


class TestGenerateParameterSets:
    def test_correct_count(self):
        engine = MonteCarloEngine(n_runs=10)
        sets = engine.generate_parameter_sets(BASE_PARAMS)
        assert len(sets) == 10

    def test_first_set_is_base(self):
        """Run 0 should be the unperturbed baseline."""
        engine = MonteCarloEngine(n_runs=5)
        sets = engine.generate_parameter_sets(BASE_PARAMS)
        for k in BASE_PARAMS:
            assert sets[0][k] == BASE_PARAMS[k]

    def test_remaining_sets_are_perturbed(self):
        """Runs 1..N-1 should differ from the base."""
        engine = MonteCarloEngine(n_runs=5)
        sets = engine.generate_parameter_sets(BASE_PARAMS)
        for i in range(1, 5):
            differ = any(
                abs(sets[i][k] - BASE_PARAMS[k]) > 1e-10 for k in BASE_PARAMS
            )
            assert differ, f"Run {i} is identical to base"


# ── Tests: _compute_ci ───────────────────────────────────────────────────────


class TestComputeCI:
    def test_single_value(self):
        """Single value => CI is (value, value)."""
        lo, hi = _compute_ci([3.14])
        assert lo == hi == 3.14

    def test_empty_list(self):
        lo, hi = _compute_ci([])
        assert lo == hi == 0

    def test_ci_contains_mean(self):
        """CI should contain the sample mean."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = _compute_ci(values)
        m = mean(values)
        assert lo <= m <= hi

    def test_wider_ci_with_more_variance(self):
        """Higher variance should produce wider CI."""
        narrow = [5.0, 5.1, 4.9, 5.0, 5.0]
        wide = [1.0, 9.0, 2.0, 8.0, 5.0]
        lo_n, hi_n = _compute_ci(narrow)
        lo_w, hi_w = _compute_ci(wide)
        assert (hi_w - lo_w) > (hi_n - lo_n)

    def test_ci_symmetric_around_mean(self):
        """CI margins should be symmetric around the mean."""
        values = [2.0, 4.0, 6.0, 8.0, 10.0]
        lo, hi = _compute_ci(values)
        m = mean(values)
        assert abs((m - lo) - (hi - m)) < 1e-10


# ── Tests: aggregation ───────────────────────────────────────────────────────


class TestAggregation:
    def _make_runs(self, n=3, rounds_per_run=2):
        runs = []
        for i in range(n):
            rounds = []
            for r in range(rounds_per_run):
                rounds.append({
                    "polarization": 3.0 + i * 0.5 + r * 0.1,
                    "avg_position": 0.1 * i - 0.05 * r,
                    "sentiment": {
                        "positive": 0.3, "neutral": 0.4, "negative": 0.3,
                    },
                })
            runs.append({
                "rounds": rounds,
                "final_polarization": rounds[-1]["polarization"],
                "final_avg_position": rounds[-1]["avg_position"],
            })
        return runs

    def test_aggregate_correct_n_completed(self):
        engine = MonteCarloEngine(n_runs=3)
        runs = self._make_runs(n=3)
        param_sets = engine.generate_parameter_sets(BASE_PARAMS)[:3]
        result = engine.aggregate_results(runs, param_sets)
        assert result.n_completed == 3

    def test_aggregate_correct_round_count(self):
        engine = MonteCarloEngine(n_runs=3)
        runs = self._make_runs(n=3, rounds_per_run=4)
        param_sets = engine.generate_parameter_sets(BASE_PARAMS)[:3]
        result = engine.aggregate_results(runs, param_sets)
        assert len(result.rounds) == 4

    def test_aggregate_mean_computation(self):
        engine = MonteCarloEngine(n_runs=3)
        runs = self._make_runs(n=3, rounds_per_run=1)
        param_sets = engine.generate_parameter_sets(BASE_PARAMS)[:3]
        result = engine.aggregate_results(runs, param_sets)

        # Manually compute expected final polarization mean
        final_pols = [r["final_polarization"] for r in runs]
        expected_mean = mean(final_pols)
        assert abs(result.final_polarization_mean - expected_mean) < 1e-10

    def test_aggregate_ci_contains_mean(self):
        engine = MonteCarloEngine(n_runs=5)
        runs = self._make_runs(n=5, rounds_per_run=2)
        param_sets = engine.generate_parameter_sets(BASE_PARAMS)[:5]
        result = engine.aggregate_results(runs, param_sets)
        lo, hi = result.final_polarization_ci
        assert lo <= result.final_polarization_mean <= hi

    def test_aggregate_empty_runs(self):
        engine = MonteCarloEngine(n_runs=5)
        result = engine.aggregate_results([], [BASE_PARAMS])
        assert result.n_completed == 0
        assert result.final_polarization_mean == 0

    def test_minimal_1_round_2_runs(self):
        """Minimal case: 1 round, 2 runs."""
        engine = MonteCarloEngine(n_runs=2)
        runs = self._make_runs(n=2, rounds_per_run=1)
        param_sets = engine.generate_parameter_sets(BASE_PARAMS)[:2]
        result = engine.aggregate_results(runs, param_sets)
        assert result.n_completed == 2
        assert len(result.rounds) == 1


# ── Tests: result_to_dict serialization ──────────────────────────────────────


class TestResultToDict:
    def test_serializable(self):
        """result_to_dict output should be JSON-serializable."""
        import json

        engine = MonteCarloEngine(n_runs=3)
        runs = [
            {"rounds": [{"polarization": 3.0, "avg_position": 0.1,
                          "sentiment": {"positive": 0.3, "neutral": 0.4, "negative": 0.3}}],
             "final_polarization": 3.0, "final_avg_position": 0.1},
            {"rounds": [{"polarization": 4.0, "avg_position": -0.1,
                          "sentiment": {"positive": 0.2, "neutral": 0.5, "negative": 0.3}}],
             "final_polarization": 4.0, "final_avg_position": -0.1},
            {"rounds": [{"polarization": 3.5, "avg_position": 0.0,
                          "sentiment": {"positive": 0.25, "neutral": 0.45, "negative": 0.3}}],
             "final_polarization": 3.5, "final_avg_position": 0.0},
        ]
        param_sets = engine.generate_parameter_sets(BASE_PARAMS)[:3]
        result = engine.aggregate_results(runs, param_sets)
        d = engine.result_to_dict(result)

        # Should not raise
        serialized = json.dumps(d)
        assert "final_polarization" in serialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
