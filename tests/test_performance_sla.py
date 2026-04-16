"""Performance SLA tests for the benchmarks package.

These protect the hot paths from accidental O(n²) regressions. The thresholds
are intentionally generous — they're not microbenchmarks, they're guard rails.
Every threshold is overridable via env var so slow CI boxes don't flake:

    DTS_SLA_DM=5.0            # max seconds for 1_000 DM tests
    DTS_SLA_COVERAGE=5.0      # max seconds for 500 coverage + bootstrap runs
    DTS_SLA_BASELINES=2.0     # max seconds for 500 baseline trajectories
    DTS_SLA_RUNNER=3.0        # max seconds for runner on a 20-scenario tree

Marked `perf` so they can be deselected with `-m "not perf"` in coverage runs
(where instrumentation multiplies wall time by 3-10×).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import pytest

from benchmarks.coverage import compute_coverage
from benchmarks.diebold_mariano import dm_test
from benchmarks.forecasters import (
    BASELINES,
    generate_baseline_trajectory,
    linear_trend,
    naive_persistence,
)
from benchmarks.runner import run_benchmark


pytestmark = pytest.mark.perf


def _budget(env_key: str, default: float) -> float:
    return float(os.getenv(env_key, default))


# ── DM throughput ──────────────────────────────────────────────────────────


def test_dm_test_throughput():
    random.seed(0)
    errs_a = [random.random() for _ in range(50)]
    errs_b = [x + 0.01 for x in errs_a]
    budget = _budget("DTS_SLA_DM", 5.0)
    t0 = time.perf_counter()
    for _ in range(1_000):
        dm_test(errs_a, errs_b, horizon=1, correction="hln")
    elapsed = time.perf_counter() - t0
    assert elapsed < budget, f"DM: 1000 runs took {elapsed:.2f}s > {budget}s"


# ── Coverage + bootstrap throughput ─────────────────────────────────────────


def test_coverage_bootstrap_throughput():
    random.seed(1)
    n = 100
    ci_lo = [-1.96] * n
    ci_hi = [1.96] * n
    realized = [random.gauss(0, 1) for _ in range(n)]
    budget = _budget("DTS_SLA_COVERAGE", 5.0)
    t0 = time.perf_counter()
    for _ in range(500):
        compute_coverage(ci_lo, ci_hi, realized, nominal=0.95, n_bootstrap=200)
    elapsed = time.perf_counter() - t0
    assert elapsed < budget, f"coverage: 500 runs took {elapsed:.2f}s > {budget}s"


# ── Baseline forecaster throughput ──────────────────────────────────────────


def test_baseline_trajectory_throughput():
    random.seed(2)
    realized = [random.gauss(0, 1) for _ in range(50)]
    budget = _budget("DTS_SLA_BASELINES", 2.0)
    t0 = time.perf_counter()
    for _ in range(500):
        for fn in BASELINES.values():
            generate_baseline_trajectory(fn, realized, train_frac=0.2)
    elapsed = time.perf_counter() - t0
    assert elapsed < budget, (
        f"baselines: 500 × {len(BASELINES)} trajectories took {elapsed:.2f}s > {budget}s"
    )


# ── Runner throughput on a synthetic 20-scenario tree ───────────────────────


def test_runner_throughput(tmp_path: Path):
    random.seed(3)
    for i in range(20):
        scen = tmp_path / f"scenario_synthetic_{i:02d}"
        scen.mkdir()
        rounds = [
            {
                "round": r + 1,
                "polarization": 3.0 + 0.1 * r + random.gauss(0, 0.05),
                "avg_position": -0.3 + 0.05 * r + random.gauss(0, 0.02),
                "num_agents": 24,
            }
            for r in range(9)
        ]
        (scen / "polarization.json").write_text(json.dumps(rounds))
        (scen / "metadata.json").write_text(json.dumps({
            "scenario_name": f"synthetic_{i}",
            "num_rounds": 9,
            "domain": ["financial", "commercial", "political"][i % 3],
            "region": ["EU", "US", "APAC"][i % 3],
            "tension": ["low", "moderate", "high"][i % 3],
        }))
    budget = _budget("DTS_SLA_RUNNER", 3.0)
    t0 = time.perf_counter()
    report = run_benchmark(tmp_path)
    elapsed = time.perf_counter() - t0
    assert elapsed < budget, f"runner: 20 scenarios took {elapsed:.2f}s > {budget}s"
    assert report["n_scenarios"] == 20
    assert report["n_metric_evaluations"] == 40


# ── Memory-style guard: DM stays stable with mild n ─────────────────────────


def test_dm_scales_linearly_ish():
    """Quadratic blowup in autocov would make horizon=5 explode past horizon=1.
    We don't measure memory, but we check wall-time doesn't go nonlinear."""
    random.seed(4)
    errs = [random.random() for _ in range(500)]
    other = [x + 0.01 for x in errs]
    t0 = time.perf_counter()
    dm_test(errs, other, horizon=1, correction="hln")
    t_h1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    dm_test(errs, other, horizon=5, correction="hln")
    t_h5 = time.perf_counter() - t0
    # h=5 does 4 extra autocov passes; expect <10× (plenty of margin).
    assert t_h5 < max(t_h1 * 10.0, 0.5), f"h=1 {t_h1:.3f}s, h=5 {t_h5:.3f}s"
