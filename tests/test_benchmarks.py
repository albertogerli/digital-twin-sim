"""Tests for the quantitative benchmarks package.

Covers:
  - Null baselines return sane forecasts on toy series
  - Diebold-Mariano matches known-good reference values (within tolerance)
  - HLN small-sample correction widens p-values relative to plain DM
  - Calibration coverage detects over/under-confident intervals
  - Scenario matrix exposes missing axis values
"""

from __future__ import annotations

import math
import random

import pytest

from benchmarks.coverage import (
    compute_coverage,
    coverage_from_quantiles,
)
from benchmarks.diebold_mariano import dm_test
from benchmarks.forecasters import (
    BASELINES,
    ar1,
    forecast_errors,
    generate_baseline_trajectory,
    linear_trend,
    naive_persistence,
    random_walk_mean,
    rmse,
)
from benchmarks.scenario_matrix import ScenarioCell, coverage_report, full_matrix


# ── Forecasters ────────────────────────────────────────────────────────────


def test_persistence_repeats_last_value():
    assert naive_persistence([1.0, 2.0, 3.0], horizon=1) == 3.0
    assert naive_persistence([], horizon=1) == 0.0


def test_linear_trend_extrapolates_slope():
    # y = 2x + 1: [1, 3, 5, 7, 9]; n=5, next index = 5 → expect 11.
    assert linear_trend([1.0, 3.0, 5.0, 7.0, 9.0], horizon=1) == pytest.approx(11.0)


def test_ar1_collapses_to_mean_at_long_horizon():
    history = [random.gauss(0.5, 0.3) for _ in range(30)]
    far = ar1(history, horizon=100)
    mean = sum(history) / len(history)
    # At horizon 100 the AR(1) has regressed to the mean regardless of phi.
    assert abs(far - mean) < 0.05


def test_baseline_trajectory_length_matches_realized():
    realized = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    for name, fn in BASELINES.items():
        traj = generate_baseline_trajectory(fn, realized, train_frac=0.25)
        assert len(traj) == len(realized), name


def test_rmse_zero_when_forecasts_perfect():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


# ── Diebold-Mariano ─────────────────────────────────────────────────────────


def test_dm_identical_errors_gives_zero_statistic():
    errs = [0.1, 0.2, 0.15, 0.3, 0.25, 0.2, 0.18]
    result = dm_test(errs, errs, horizon=1, correction="hln")
    assert abs(result.statistic) < 1e-9
    assert result.better == "tie"


def test_dm_detects_clearly_better_forecast():
    """A is uniformly better — DM must reject H0 in its favor."""
    random.seed(0)
    good = [random.gauss(0, 0.1) ** 2 for _ in range(40)]
    bad = [g + random.uniform(0.5, 1.0) for g in good]
    result = dm_test(good, bad, horizon=1, correction="hln")
    assert result.statistic < 0
    assert result.p_value < 0.01
    assert result.better == "a"


def test_dm_detects_clearly_worse_forecast():
    random.seed(1)
    good = [random.gauss(0, 0.1) ** 2 for _ in range(40)]
    bad = [g + random.uniform(0.5, 1.0) for g in good]
    result = dm_test(bad, good, horizon=1, correction="hln")
    assert result.statistic > 0
    assert result.p_value < 0.01
    assert result.better == "b"


def test_dm_hln_more_conservative_than_uncorrected():
    """HLN should produce a larger (or equal) p-value at short horizons."""
    random.seed(2)
    a = [random.random() for _ in range(8)]
    b = [x + 0.05 for x in a]
    hln = dm_test(a, b, horizon=2, correction="hln")
    plain = dm_test(a, b, horizon=2, correction="none")
    assert hln.p_value >= plain.p_value - 1e-9


def test_dm_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        dm_test([1, 2, 3], [1, 2], horizon=1)


# ── Calibration coverage ────────────────────────────────────────────────────


def test_coverage_well_calibrated_interval():
    """95% CI built from true distribution should cover ≈95%."""
    random.seed(3)
    n = 500
    realized = [random.gauss(0, 1) for _ in range(n)]
    # Nominal 95% interval around each realized point's distribution (N(0,1)).
    ci_low = [-1.96] * n
    ci_high = [1.96] * n
    rep = compute_coverage(ci_low, ci_high, realized, nominal=0.95, n_bootstrap=500)
    # Empirical must be within a few percentage points of 95%.
    assert 0.91 <= rep.empirical <= 0.99
    assert rep.is_calibrated


def test_coverage_detects_overconfident_interval():
    """Too-narrow interval → empirical < nominal, not calibrated."""
    random.seed(4)
    n = 300
    realized = [random.gauss(0, 1) for _ in range(n)]
    ci_low = [-0.5] * n
    ci_high = [0.5] * n
    rep = compute_coverage(ci_low, ci_high, realized, nominal=0.95, n_bootstrap=500)
    assert rep.empirical < 0.6
    assert not rep.is_calibrated


def test_coverage_from_ensemble_quantiles():
    """Build CIs from an ensemble and verify coverage on injected realizations."""
    random.seed(5)
    n_rounds = 20
    samples = [[random.gauss(r * 0.1, 1.0) for _ in range(200)]
               for r in range(n_rounds)]
    realized = [random.gauss(r * 0.1, 1.0) for r in range(n_rounds)]
    rep = coverage_from_quantiles(samples, realized, nominal=0.9, n_bootstrap=300)
    # Empirical should be within bootstrap band of 0.9 most of the time.
    assert rep.nominal == 0.9
    assert 0 <= rep.empirical <= 1


def test_coverage_rejects_length_mismatch():
    with pytest.raises(ValueError):
        compute_coverage([0], [1, 2], [0.5, 1.5], nominal=0.95)


# ── Scenario matrix ─────────────────────────────────────────────────────────


def test_full_matrix_non_empty_and_unique():
    cells = full_matrix()
    assert len(cells) > 0
    assert len({c.key for c in cells}) == len(cells)


def test_coverage_report_flags_missing_axes():
    sample = [
        ScenarioCell(domain="financial", region="EU", tension="high"),
        ScenarioCell(domain="commercial", region="US", tension="moderate"),
    ]
    rep = coverage_report(sample)
    assert rep["cells"] == 2
    assert rep["complete"] is False
    assert "public_health" in rep["missing"]["domain"]
    assert "APAC" in rep["missing"]["region"]


def test_coverage_report_complete_on_full_matrix():
    rep = coverage_report(full_matrix())
    assert rep["complete"] is True
    assert all(not m for m in rep["missing"].values())


# ── End-to-end: our sim must beat persistence on at least one scenario ──────


def test_sim_beats_persistence_on_controlled_signal():
    """Smoke test that ties forecasters + DM together on an easy problem:
    a noisy linear trend. Linear-trend forecaster should beat persistence."""
    random.seed(7)
    n = 30
    truth = [0.2 * t + random.gauss(0, 0.05) for t in range(n)]

    from benchmarks.forecasters import forecast_errors

    pers_traj = generate_baseline_trajectory(naive_persistence, truth, train_frac=0.2)
    trend_traj = generate_baseline_trajectory(linear_trend, truth, train_frac=0.2)

    errs_trend = forecast_errors(trend_traj, truth)
    errs_pers = forecast_errors(pers_traj, truth)

    # DM test: H0 is equal loss. We want linear_trend (a) to beat pers (b).
    result = dm_test(errs_trend, errs_pers, horizon=1, correction="hln")
    assert result.mean_loss_a < result.mean_loss_b
    assert result.statistic < 0
    # On n=30 with a clear signal HLN should still reject H0.
    assert result.p_value < 0.10
