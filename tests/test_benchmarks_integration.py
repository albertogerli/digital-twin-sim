"""End-to-end integration tests for the benchmark runner.

These exercise the full pipeline (discover → forecast → DM → coverage → report)
against synthesized scenario data so the test is hermetic and fast. A separate
smoke test optionally hits the real `frontend/public/data` tree when present.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from benchmarks.coverage import compute_coverage
from benchmarks.residual_ci import residual_bootstrap_intervals, residual_summary
from benchmarks.runner import (
    DEFAULT_ROOT,
    METRICS,
    benchmark_scenario,
    build_matrix_coverage,
    render_markdown,
    run_benchmark,
)


def _make_scenario(
    tmp: Path,
    name: str,
    polarizations: list[float],
    avg_positions: list[float] | None = None,
    domain: str = "financial",
    region: str = "EU",
    tension: str = "moderate",
) -> Path:
    scen = tmp / f"scenario_{name}"
    scen.mkdir(parents=True)
    if avg_positions is None:
        avg_positions = [p * 0.1 - 0.5 for p in polarizations]
    rounds = [
        {
            "round": i + 1,
            "polarization": polarizations[i],
            "avg_position": avg_positions[i],
            "num_agents": 24,
        }
        for i in range(len(polarizations))
    ]
    (scen / "polarization.json").write_text(json.dumps(rounds))
    (scen / "metadata.json").write_text(json.dumps({
        "scenario_id": name,
        "scenario_name": name.replace("_", " "),
        "num_rounds": len(polarizations),
        "domain": domain,
        "region": region,
        "tension": tension,
    }))
    return scen


# ── Integration: synthesized scenarios ──────────────────────────────────────


def test_runner_discovers_scenarios_and_reports(tmp_path: Path):
    _make_scenario(tmp_path, "A", [4.0, 4.5, 4.8, 5.1, 5.4], domain="financial")
    _make_scenario(tmp_path, "B", [2.0, 2.1, 2.0, 2.2, 2.1, 2.3], domain="political")
    _make_scenario(tmp_path, "C", [1.0, 1.5], domain="commercial")  # too short — skipped

    report = run_benchmark(tmp_path)
    assert report["n_scenarios"] == 3
    # A and B produce two metric rows each; C skipped.
    assert report["n_metric_evaluations"] == 4
    assert {r["scenario"] for r in report["scenarios"]} == {"A", "B"}


def test_runner_matrix_coverage_flags_missing_axes(tmp_path: Path):
    _make_scenario(tmp_path, "A", [1, 2, 3, 4, 5], domain="financial", region="EU")
    report = run_benchmark(tmp_path)
    mc = report["matrix_coverage"]
    assert mc["cells"] == 1
    assert mc["complete"] is False
    assert "US" in mc["missing"]["region"]


def test_benchmark_scenario_honors_metric_filter(tmp_path: Path):
    scen = _make_scenario(tmp_path, "A", [1.0, 2.0, 3.0, 4.0, 5.0])
    only_pol = benchmark_scenario(scen, metrics=("polarization",))
    assert len(only_pol) == 1
    assert only_pol[0].metric == "polarization"


def test_markdown_report_is_well_formed(tmp_path: Path):
    _make_scenario(tmp_path, "A", [1.0, 1.1, 1.2, 1.3, 1.4])
    report = run_benchmark(tmp_path)
    md = render_markdown(report)
    assert md.startswith("# Quantitative Benchmark Report")
    assert "Beats persistence" in md
    assert "Scenario-matrix coverage" in md


# ── Residual-bootstrap coverage on synthetic data ────────────────────────────


def test_residual_ci_well_calibrated_on_iid_noise():
    """If residuals are iid, bootstrap intervals should cover near-nominal."""
    random.seed(11)
    n = 200
    realized = [random.gauss(0, 1.0) for _ in range(n)]
    point = [0.0] * n  # deliberately biased forecast
    lo, hi = residual_bootstrap_intervals(point, realized, nominal=0.9, n_samples=2000)
    cov = compute_coverage(lo, hi, realized, nominal=0.9, n_bootstrap=500)
    assert 0.82 <= cov.empirical <= 0.96


def test_residual_summary_captures_bias():
    point = [0.0, 0.0, 0.0, 0.0]
    realized = [1.0, 1.0, 1.0, 1.0]
    s = residual_summary(point, realized)
    assert s["bias"] == pytest.approx(1.0)
    assert s["max_abs"] == pytest.approx(1.0)


def test_residual_ci_rejects_length_mismatch():
    with pytest.raises(ValueError):
        residual_bootstrap_intervals([1.0, 2.0], [1.0, 2.0, 3.0])


# ── Smoke test against real data (skipped when unavailable) ─────────────────


@pytest.mark.skipif(
    not DEFAULT_ROOT.is_dir() or not any(DEFAULT_ROOT.glob("scenario_*")),
    reason="no real scenario outputs available",
)
def test_runner_against_real_scenarios_produces_report():
    report = run_benchmark(DEFAULT_ROOT)
    assert report["n_scenarios"] >= 1
    # Every scenario row has a persistence baseline and a DM result.
    for row in report["scenarios"]:
        assert "persistence" in row["baselines"]
        assert "dm_vs_sim" in row["baselines"]["persistence"]
