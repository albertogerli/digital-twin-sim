"""Stress tests: stability of the benchmark pipeline across the full scenario
matrix and under randomized / adversarial inputs.

These don't run the live sim (too expensive for CI). They exercise the
*scoring* layer with synthesized series that stand in for sim outputs across
every axis cell, plus edge-case signals (flat, explosive, alternating,
heavy-tailed). The goal is to fail loudly if any bucket produces NaN/Inf,
negative variance, or sign-inverted DM decisions under well-understood
distributions.
"""

from __future__ import annotations

import math
import random

import pytest

from benchmarks.coverage import compute_coverage, coverage_from_quantiles
from benchmarks.diebold_mariano import dm_test
from benchmarks.forecasters import (
    BASELINES,
    ar1,
    forecast_errors,
    generate_baseline_trajectory,
    linear_trend,
    naive_persistence,
    random_walk_mean,
)
from benchmarks.residual_ci import residual_bootstrap_intervals
from benchmarks.scenario_matrix import (
    DOMAINS,
    REGIONS,
    TENSION_LEVELS,
    ScenarioCell,
    full_matrix,
)


pytestmark = pytest.mark.stress


# ── Matrix exhaustiveness ───────────────────────────────────────────────────


def test_full_matrix_size_matches_cartesian_product():
    cells = full_matrix()
    assert len(cells) == len(DOMAINS) * len(REGIONS) * len(TENSION_LEVELS)


def test_every_axis_value_appears_in_full_matrix():
    cells = full_matrix()
    seen_domains = {c.domain for c in cells}
    seen_regions = {c.region for c in cells}
    seen_tensions = {c.tension for c in cells}
    assert seen_domains == set(DOMAINS)
    assert seen_regions == set(REGIONS)
    assert seen_tensions == set(TENSION_LEVELS)


# ── Forecasters: stability across adversarial signals ───────────────────────


def _finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


ADVERSARIAL_SIGNALS: dict[str, list[float]] = {
    "flat": [3.0] * 20,
    "monotone_up": [0.1 * i for i in range(20)],
    "monotone_down": [-0.1 * i for i in range(20)],
    "alternating": [1.0 if i % 2 == 0 else -1.0 for i in range(20)],
    "spike": [0.0] * 10 + [99.0] + [0.0] * 9,
    "heavy_tail": [
        0.0, 0.1, -0.1, 0.2, -50.0, 0.1, 0.3, 0.05, 40.0, -0.1,
        0.2, 0.4, -0.3, 0.15, 0.25, -0.5, 0.1, 0.3, -0.2, 0.05,
    ],
    "near_constant_noise": [1.0 + 1e-9 * random.random() for _ in range(20)],
}


@pytest.mark.parametrize("name,series", list(ADVERSARIAL_SIGNALS.items()))
@pytest.mark.parametrize("baseline_name", list(BASELINES.keys()))
def test_baseline_trajectories_are_finite_on_adversarial_signals(
    name: str, series: list[float], baseline_name: str
):
    random.seed(hash((name, baseline_name)) & 0xFFFF)
    fn = BASELINES[baseline_name]
    traj = generate_baseline_trajectory(fn, series, train_frac=0.2)
    assert len(traj) == len(series), (name, baseline_name)
    assert all(_finite(x) for x in traj), (name, baseline_name, traj)


# ── DM: never returns NaN / Inf; decision sign is consistent ────────────────


def test_dm_never_returns_nan_or_inf_on_random_inputs():
    for seed in range(50):
        rng = random.Random(seed)
        n = rng.randint(3, 40)
        a = [rng.random() for _ in range(n)]
        b = [rng.random() for _ in range(n)]
        r = dm_test(a, b, horizon=1, correction="hln")
        assert _finite(r.statistic), (seed, r)
        assert 0.0 <= r.p_value <= 1.0, (seed, r)
        assert r.better in ("a", "b", "tie")


def test_dm_sign_consistent_under_swap():
    """Swapping a↔b must flip the statistic sign and preserve the p-value."""
    rng = random.Random(99)
    a = [rng.random() for _ in range(30)]
    b = [x + rng.gauss(0.0, 0.1) for x in a]
    r_ab = dm_test(a, b, horizon=1, correction="hln")
    r_ba = dm_test(b, a, horizon=1, correction="hln")
    assert r_ab.statistic == pytest.approx(-r_ba.statistic, abs=1e-9)
    assert r_ab.p_value == pytest.approx(r_ba.p_value, abs=1e-9)
    if r_ab.better == "a":
        assert r_ba.better == "b"
    elif r_ab.better == "b":
        assert r_ba.better == "a"


def test_dm_survives_constant_series():
    """Zero-variance diff must not divide-by-zero — guard returns tie."""
    r = dm_test([0.5] * 10, [0.5] * 10, horizon=1, correction="hln")
    assert _finite(r.statistic)
    assert r.better == "tie"


# ── Coverage: empirical always in [0, 1]; bootstrap band brackets it ────────


def test_coverage_always_bounded_and_bracketed():
    rng = random.Random(7)
    for trial in range(20):
        n = rng.randint(5, 100)
        realized = [rng.gauss(0, 1) for _ in range(n)]
        width = rng.uniform(0.1, 4.0)
        lo = [-width / 2] * n
        hi = [+width / 2] * n
        rep = compute_coverage(lo, hi, realized, nominal=0.9, n_bootstrap=200)
        assert 0.0 <= rep.empirical <= 1.0
        assert rep.ci_low <= rep.empirical <= rep.ci_high + 1e-9
        assert rep.miss_below + rep.miss_above + int(rep.empirical * n) <= n + 1


def test_coverage_from_quantiles_handles_ragged_ensemble_sizes():
    rng = random.Random(8)
    samples = [[rng.gauss(0, 1) for _ in range(rng.randint(50, 400))] for _ in range(15)]
    realized = [rng.gauss(0, 1) for _ in range(15)]
    rep = coverage_from_quantiles(samples, realized, nominal=0.9, n_bootstrap=200)
    assert rep.nominal == 0.9
    assert 0.0 <= rep.empirical <= 1.0


# ── Cross-axis smoke: every scenario cell can drive the pipeline end-to-end ─


@pytest.mark.parametrize("cell", full_matrix()[:20])  # sample first 20 to stay fast
def test_pipeline_runs_per_cell_without_crashing(cell: ScenarioCell):
    rng = random.Random(hash(cell.key) & 0xFFFF)
    n = 9
    realized = [0.1 * i + rng.gauss(0, 0.1) for i in range(n)]
    traj = generate_baseline_trajectory(naive_persistence, realized, train_frac=0.2)
    errs_sim = forecast_errors(realized, realized)  # sim = truth for this mock
    errs_pers = forecast_errors(traj, realized)
    r = dm_test(errs_sim, errs_pers, horizon=1, correction="hln")
    assert _finite(r.statistic)
    lo, hi = residual_bootstrap_intervals(traj, realized, nominal=0.9, n_samples=200)
    cov = compute_coverage(lo, hi, realized, nominal=0.9, n_bootstrap=200)
    assert 0.0 <= cov.empirical <= 1.0


# ── Property: linear_trend beats persistence on strong linear signal ────────


def test_linear_trend_dominates_persistence_on_linear_signal():
    random.seed(13)
    n = 40
    truth = [0.3 * t + random.gauss(0, 0.05) for t in range(n)]
    pers = generate_baseline_trajectory(naive_persistence, truth, train_frac=0.2)
    trend = generate_baseline_trajectory(linear_trend, truth, train_frac=0.2)
    err_p = forecast_errors(pers, truth)
    err_t = forecast_errors(trend, truth)
    r = dm_test(err_t, err_p, horizon=1, correction="hln")
    assert r.mean_loss_a < r.mean_loss_b
    assert r.better == "a"
    assert r.p_value < 0.05


# ── Property: mean baseline beats persistence on white noise ────────────────


def test_mean_beats_persistence_on_white_noise_series():
    random.seed(17)
    n = 80
    truth = [random.gauss(0, 1) for _ in range(n)]
    pers = generate_baseline_trajectory(naive_persistence, truth, train_frac=0.2)
    mean_traj = generate_baseline_trajectory(random_walk_mean, truth, train_frac=0.2)
    err_p = forecast_errors(pers, truth)
    err_m = forecast_errors(mean_traj, truth)
    r = dm_test(err_m, err_p, horizon=1, correction="hln")
    assert r.mean_loss_a <= r.mean_loss_b
