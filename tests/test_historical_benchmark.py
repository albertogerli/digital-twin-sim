"""Tests for the empirical ground-truth loader and historical benchmark runner.

Uses synthesized on-disk fixtures for unit tests, plus a smoke test against the
real corpus under `calibration/empirical/` when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.historical import (
    COUNTRY_TO_REGION,
    DOMAIN_ALIAS,
    DEFAULT_V1,
    DEFAULT_V22,
    EmpiricalScenario,
    load_empirical_scenarios,
    summarize,
    _infer_tension,
)
from benchmarks.historical_runner import (
    METRICS,
    render_markdown,
    run_historical_benchmark,
)
from benchmarks.scenario_matrix import DOMAINS, REGIONS, TENSION_LEVELS


# ── Fixtures ────────────────────────────────────────────────────────────────


def _write_scenario(
    path: Path,
    *,
    name: str,
    domain: str = "political",
    country: str = "IT",
    trajectory: list[tuple[float, float, float]] | None = None,
    gt_pro: float | None = 0.55,
):
    trajectory = trajectory or [
        (40.0, 30.0, 30.0),
        (42.0, 32.0, 26.0),
        (45.0, 35.0, 20.0),
        (50.0, 35.0, 15.0),
        (52.0, 38.0, 10.0),
        (55.0, 40.0, 5.0),
    ]
    traj = [
        {
            "round": i + 1,
            "date": f"2024-0{i+1}-01",
            "pro_pct": p,
            "against_pct": a,
            "undecided_pct": u,
            "sample_size": 1000,
            "source": "synthetic",
            "pollster": "test",
        }
        for i, (p, a, u) in enumerate(trajectory)
    ]
    gt = None if gt_pro is None else {"pro_pct": gt_pro * 100.0, "source": "test", "type": "test"}
    path.write_text(json.dumps({
        "id": name,
        "domain": domain,
        "title": name,
        "country": country,
        "date_start": "2024-01-01",
        "date_end": "2024-06-01",
        "n_rounds": len(traj),
        "round_duration_days": 30,
        "ground_truth_outcome": gt,
        "polling_trajectory": traj,
        "events": [],
        "agents": [],
        "covariates": {},
        "notes": "",
    }))


# ── Unit: tension inference ──────────────────────────────────────────────────


def test_tension_low_on_flat_series():
    assert _infer_tension([0.1, 0.1, 0.1, 0.1]) == "low"


def test_tension_critical_on_big_swings():
    # round-on-round deltas ≈ 0.3, stdev ≈ 0.3 → critical
    assert _infer_tension([0.0, 0.3, 0.0, 0.3, 0.0]) == "critical"


def test_tension_scales_with_volatility():
    low = _infer_tension([0.0, 0.01, 0.02, 0.01])
    mod = _infer_tension([0.0, 0.04, 0.08, 0.04])
    high = _infer_tension([0.0, 0.08, 0.0, 0.08, 0.0])
    crit = _infer_tension([0.0, 0.2, 0.0, 0.2, 0.0])
    assert (low, mod, high, crit) == ("low", "moderate", "high", "critical")


# ── Unit: loader normalizes trajectories ────────────────────────────────────


def test_loader_normalizes_polling(tmp_path: Path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v22 = tmp_path / "v22"; v22.mkdir()
    _write_scenario(v1 / "A.json", name="A", domain="political", country="IT")
    scenarios = load_empirical_scenarios(v1_dir=v1, v22_dir=v22)
    assert len(scenarios) == 1
    s = scenarios[0]
    # pro_pct / 100
    assert s.support[0] == pytest.approx(0.40)
    # (pro - against) / 100
    assert s.signed_position[0] == pytest.approx(0.10)
    # region derived from country
    assert s.region == "EU"
    assert s.domain in DOMAINS
    assert s.tension in TENSION_LEVELS
    assert s.ground_truth_support == pytest.approx(0.55)


def test_loader_prefers_v22_on_duplicate(tmp_path: Path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v22 = tmp_path / "v22"; v22.mkdir()
    _write_scenario(v1 / "DUP.json", name="DUP_V1", country="IT")
    _write_scenario(v22 / "DUP.json", name="DUP_V22", country="IT")
    scenarios = load_empirical_scenarios(v1_dir=v1, v22_dir=v22)
    assert len(scenarios) == 1
    assert scenarios[0].id == "DUP_V22"
    assert scenarios[0].version == "v2.2"


def test_loader_skips_short_trajectories(tmp_path: Path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v22 = tmp_path / "v22"; v22.mkdir()
    _write_scenario(v1 / "A.json", name="A", trajectory=[(40.0, 30.0, 30.0)])
    scenarios = load_empirical_scenarios(v1_dir=v1, v22_dir=v22)
    assert scenarios == []


def test_loader_handles_none_fields(tmp_path: Path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v22 = tmp_path / "v22"; v22.mkdir()
    path = v1 / "A.json"
    path.write_text(json.dumps({
        "id": "A", "domain": "political", "title": "A", "country": "IT",
        "n_rounds": 3, "ground_truth_outcome": None,
        "polling_trajectory": [
            {"round": 1, "pro_pct": 40.0, "against_pct": None, "undecided_pct": None},
            {"round": 2, "pro_pct": 45.0, "against_pct": 30.0, "undecided_pct": 25.0},
            {"round": 3, "pro_pct": 50.0, "against_pct": 35.0, "undecided_pct": 15.0},
        ],
    }))
    scenarios = load_empirical_scenarios(v1_dir=v1, v22_dir=v22)
    assert len(scenarios) == 1
    assert scenarios[0].ground_truth_support is None


def test_loader_maps_unknown_domain_to_canonical(tmp_path: Path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v22 = tmp_path / "v22"; v22.mkdir()
    _write_scenario(v1 / "X.json", name="X", domain="technology", country="US")
    # "technology" should collapse to "commercial" via DOMAIN_ALIAS
    scenarios = load_empirical_scenarios(v1_dir=v1, v22_dir=v22)
    assert scenarios[0].domain == DOMAIN_ALIAS["technology"]
    assert scenarios[0].domain in DOMAINS


def test_loader_unknown_country_falls_back_to_global(tmp_path: Path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v22 = tmp_path / "v22"; v22.mkdir()
    _write_scenario(v1 / "Y.json", name="Y", country="ZZ")
    scenarios = load_empirical_scenarios(v1_dir=v1, v22_dir=v22)
    assert scenarios[0].region == "GLOBAL"
    assert scenarios[0].region in REGIONS


# ── Unit: runner on synthetic corpus ────────────────────────────────────────


def test_runner_produces_complete_report(tmp_path: Path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v22 = tmp_path / "v22"; v22.mkdir()
    _write_scenario(v1 / "A.json", name="A", domain="political", country="IT")
    _write_scenario(v1 / "B.json", name="B", domain="financial", country="US")
    _write_scenario(v1 / "C.json", name="C", domain="labor", country="BR")
    scenarios = load_empirical_scenarios(v1_dir=v1, v22_dir=v22)
    report = run_historical_benchmark(scenarios=scenarios)

    assert report["corpus"]["n"] == 3
    # Two metrics × three scenarios
    assert len(report["scenarios"]) == 6
    # All four baselines scored on every row
    for row in report["scenarios"]:
        assert set(row["baselines"].keys()) == {"persistence", "mean", "linear_trend", "ar1"}
    # Aggregates exist for both metrics
    for metric in METRICS:
        assert metric in report["per_metric"]
        assert "global" in report["per_metric"][metric]
        assert "persistence" in report["per_metric"][metric]["global"]


def test_runner_terminal_error_uses_ground_truth(tmp_path: Path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v22 = tmp_path / "v22"; v22.mkdir()
    _write_scenario(
        v1 / "A.json", name="A", country="IT",
        trajectory=[(50.0, 30.0, 20.0)] * 6, gt_pro=0.60,
    )
    scenarios = load_empirical_scenarios(v1_dir=v1, v22_dir=v22)
    report = run_historical_benchmark(scenarios=scenarios)
    support_row = next(r for r in report["scenarios"] if r["metric"] == "support")
    pers = support_row["baselines"]["persistence"]
    # Persistence emits last observed support=0.5; gt=0.6 → terminal_error=0.1
    assert pers["terminal_error"] == pytest.approx(0.1, abs=1e-6)


def test_runner_metric_filter(tmp_path: Path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v22 = tmp_path / "v22"; v22.mkdir()
    _write_scenario(v1 / "A.json", name="A", country="IT")
    scenarios = load_empirical_scenarios(v1_dir=v1, v22_dir=v22)
    report = run_historical_benchmark(scenarios=scenarios, metrics=("support",))
    assert all(r["metric"] == "support" for r in report["scenarios"])
    assert "signed_position" not in report["per_metric"]


def test_markdown_renders_all_sections(tmp_path: Path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v22 = tmp_path / "v22"; v22.mkdir()
    _write_scenario(v1 / "A.json", name="A", country="IT")
    _write_scenario(v1 / "B.json", name="B", country="US")
    scenarios = load_empirical_scenarios(v1_dir=v1, v22_dir=v22)
    report = run_historical_benchmark(scenarios=scenarios)
    md = render_markdown(report)
    assert "Historical Benchmark Report" in md
    assert "Corpus" in md
    assert "Scenario-matrix coverage" in md
    assert "Skill on `support`" in md
    assert "Skill on `signed_position`" in md
    assert "By domain" in md


# ── Aggregation correctness ─────────────────────────────────────────────────


def test_aggregate_groups_by_axes(tmp_path: Path):
    v1 = tmp_path / "v1"; v1.mkdir()
    v22 = tmp_path / "v22"; v22.mkdir()
    _write_scenario(v1 / "A.json", name="A", domain="political", country="IT")
    _write_scenario(v1 / "B.json", name="B", domain="political", country="US")
    _write_scenario(v1 / "C.json", name="C", domain="financial", country="BR")
    scenarios = load_empirical_scenarios(v1_dir=v1, v22_dir=v22)
    report = run_historical_benchmark(scenarios=scenarios)

    per_dom = report["per_metric"]["support"]["by_domain"]
    per_reg = report["per_metric"]["support"]["by_region"]
    assert per_dom["political"]["persistence"]["n_scenarios"] == 2
    assert per_dom["financial"]["persistence"]["n_scenarios"] == 1
    assert per_reg["EU"]["persistence"]["n_scenarios"] == 1
    assert per_reg["US"]["persistence"]["n_scenarios"] == 1
    assert per_reg["LATAM"]["persistence"]["n_scenarios"] == 1


# ── Smoke test on real corpus (skipped when files missing) ──────────────────


@pytest.mark.skipif(
    not DEFAULT_V1.is_dir() or not any(DEFAULT_V1.glob("*.json")),
    reason="real empirical corpus not available",
)
def test_real_corpus_produces_complete_matrix():
    scenarios = load_empirical_scenarios()
    assert len(scenarios) >= 30
    report = run_historical_benchmark(scenarios=scenarios)

    # Every scenario contributes a support and signed_position row.
    assert report["corpus"]["n"] == len(scenarios)
    assert len(report["scenarios"]) == 2 * len(scenarios)

    # Persistence RMSE on support should be small (polling is sticky) but not zero.
    g = report["per_metric"]["support"]["global"]["persistence"]
    assert g["mean_rmse"] > 0
    assert g["mean_rmse"] < 0.2  # <20pp RMSE on normalized support — sanity

    # Matrix coverage should be complete across all 3 axes (we verified this on
    # first run; fail loudly if the corpus loses a bucket later).
    mc = report["matrix_coverage"]
    for axis, missing in mc["missing"].items():
        assert not missing, f"missing {axis} values: {missing}"
