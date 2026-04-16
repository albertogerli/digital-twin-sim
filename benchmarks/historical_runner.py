"""Historical benchmark runner: evaluates null baselines against 40+ real
polling trajectories (Scottish indyref, Boeing MAX, SVB, Dieselgate, …) and
reports which baselines carry signal where, establishing a "skill floor" for
future sim-vs-real comparisons.

Two output layers:
  1. Per-scenario: RMSE + terminal-round error of each baseline against the
     empirical trajectory, plus DM tests of `linear_trend` / `ar1` / `mean`
     against `persistence`.
  2. Aggregates: pooled skill scores by domain, region, tension bucket; plus
     a `scenario_matrix.coverage_report` so we can see which (domain, region,
     tension) cells the empirical corpus actually covers.

When our sim eventually produces a matching trajectory per empirical scenario,
wire it in via `sim_forecasts_for(scenario_id)` and this same runner will
emit sim-vs-each-baseline DM rows automatically.

Usage:
    python -m benchmarks.historical_runner
    python -m benchmarks.historical_runner --out outputs/historical.json \\
        --markdown outputs/historical.md
    python -m benchmarks.historical_runner --metric signed_position
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .diebold_mariano import dm_test
from .forecasters import (
    BASELINES,
    forecast_errors,
    generate_baseline_trajectory,
    naive_persistence,
    rmse,
)
from .historical import EmpiricalScenario, load_empirical_scenarios, summarize
from .residual_ci import residual_bootstrap_intervals
from .coverage import compute_coverage
from .scenario_matrix import coverage_report


METRICS = ("support", "signed_position")


@dataclass
class BaselineScore:
    name: str
    rmse: float
    terminal_error: float | None  # |forecast - ground_truth| at terminal round
    dm_vs_persistence_p: float
    dm_vs_persistence_better: str  # "a", "b", "tie" (a=this baseline, b=persistence)


@dataclass
class ScenarioScoreRow:
    id: str
    title: str
    domain: str
    region: str
    tension: str
    metric: str
    n_rounds: int
    ground_truth: float | None
    baselines: dict[str, BaselineScore]
    coverage_nominal: float
    coverage_empirical: float
    coverage_calibrated: bool


def _score_scenario(scen: EmpiricalScenario, metric: str) -> ScenarioScoreRow | None:
    series = getattr(scen, metric)
    if len(series) < 3:
        return None

    persistence_traj = generate_baseline_trajectory(
        naive_persistence, series, train_frac=0.2
    )
    errs_pers = forecast_errors(persistence_traj, series)

    gt: float | None = None
    if metric == "support":
        gt = scen.ground_truth_support
    elif metric == "signed_position" and scen.ground_truth_support is not None:
        gt = 2.0 * scen.ground_truth_support - 1.0

    results: dict[str, BaselineScore] = {}
    for name, fn in BASELINES.items():
        traj = generate_baseline_trajectory(fn, series, train_frac=0.2)
        errs = forecast_errors(traj, series)
        baseline_rmse = rmse(traj, series)

        if name == "persistence":
            dm_p, dm_better = 1.0, "tie"
        else:
            try:
                r = dm_test(errs, errs_pers, horizon=1, correction="hln")
                dm_p, dm_better = r.p_value, r.better
            except ValueError:
                dm_p, dm_better = 1.0, "tie"

        term_err: float | None = None
        if gt is not None:
            # Terminal forecast = last fitted value (what we'd have predicted
            # for the final observed round).
            term_err = abs(traj[-1] - gt)

        results[name] = BaselineScore(
            name=name,
            rmse=round(baseline_rmse, 4),
            terminal_error=round(term_err, 4) if term_err is not None else None,
            dm_vs_persistence_p=round(dm_p, 4),
            dm_vs_persistence_better=dm_better,
        )

    # Residual-bootstrap coverage of the persistence baseline on this scenario.
    lo, hi = residual_bootstrap_intervals(
        persistence_traj, series, nominal=0.9, n_samples=500
    )
    try:
        cov = compute_coverage(lo, hi, series, nominal=0.9, n_bootstrap=400)
        cov_emp, cov_cal = cov.empirical, cov.is_calibrated
    except ValueError:
        cov_emp, cov_cal = 0.0, False

    return ScenarioScoreRow(
        id=scen.id,
        title=scen.title,
        domain=scen.domain,
        region=scen.region,
        tension=scen.tension,
        metric=metric,
        n_rounds=scen.n_rounds,
        ground_truth=gt,
        baselines=results,
        coverage_nominal=0.9,
        coverage_empirical=round(cov_emp, 3),
        coverage_calibrated=cov_cal,
    )


def _aggregate(rows: list[ScenarioScoreRow]) -> dict:
    """Pooled skill per baseline, globally and by axis."""
    out: dict[str, dict] = {"global": {}, "by_domain": {}, "by_region": {}, "by_tension": {}}
    baseline_names = list(BASELINES.keys())

    def _fold(subset: list[ScenarioScoreRow]) -> dict:
        if not subset:
            return {}
        res: dict[str, dict] = {}
        for name in baseline_names:
            rmses = [r.baselines[name].rmse for r in subset if name in r.baselines]
            term_errs = [
                r.baselines[name].terminal_error
                for r in subset
                if r.baselines[name].terminal_error is not None
            ]
            significant_wins = sum(
                1
                for r in subset
                if name in r.baselines
                and r.baselines[name].dm_vs_persistence_better == "a"
                and r.baselines[name].dm_vs_persistence_p < 0.05
            )
            res[name] = {
                "mean_rmse": round(statistics.mean(rmses), 4) if rmses else None,
                "median_rmse": round(statistics.median(rmses), 4) if rmses else None,
                "mean_terminal_error": round(statistics.mean(term_errs), 4) if term_errs else None,
                "n_scenarios": len(subset),
                "n_significant_beats_persistence": significant_wins,
            }
        return res

    out["global"] = _fold(rows)
    axes = {"by_domain": "domain", "by_region": "region", "by_tension": "tension"}
    for out_key, field in axes.items():
        buckets: dict[str, list[ScenarioScoreRow]] = {}
        for r in rows:
            buckets.setdefault(getattr(r, field), []).append(r)
        out[out_key] = {k: _fold(v) for k, v in buckets.items()}
    return out


def run_historical_benchmark(
    scenarios: list[EmpiricalScenario] | None = None,
    metrics: Sequence[str] = METRICS,
) -> dict:
    if scenarios is None:
        scenarios = load_empirical_scenarios()

    all_rows: list[ScenarioScoreRow] = []
    for scen in scenarios:
        for metric in metrics:
            row = _score_scenario(scen, metric)
            if row is not None:
                all_rows.append(row)

    cells = [s.cell for s in scenarios]
    matrix_cov = coverage_report(cells)
    corpus = summarize(scenarios)

    # Per-metric aggregates
    per_metric: dict[str, dict] = {}
    for metric in metrics:
        per_metric[metric] = _aggregate([r for r in all_rows if r.metric == metric])

    return {
        "corpus": corpus,
        "matrix_coverage": matrix_cov,
        "per_metric": per_metric,
        "scenarios": [_row_to_dict(r) for r in all_rows],
    }


def _row_to_dict(r: ScenarioScoreRow) -> dict:
    d = asdict(r)
    d["baselines"] = {name: asdict(bs) for name, bs in r.baselines.items()}
    return d


def render_markdown(report: dict) -> str:
    corpus = report["corpus"]
    lines = [
        "# Historical Benchmark Report",
        "",
        "Null-baseline forecasters evaluated against real polling trajectories.",
        "",
        "## Corpus",
        "",
        f"- **Scenarios**: {corpus['n']}",
        f"- **Avg rounds**: {corpus['avg_rounds']:.1f}",
        f"- **With verified ground truth**: {corpus['with_ground_truth']}/{corpus['n']}",
        f"- **Domains**: {', '.join(f'{k}={v}' for k,v in sorted(corpus['by_domain'].items()))}",
        f"- **Regions**: {', '.join(f'{k}={v}' for k,v in sorted(corpus['by_region'].items()))}",
        f"- **Tension**: {', '.join(f'{k}={v}' for k,v in sorted(corpus['by_tension'].items()))}",
        "",
        "## Scenario-matrix coverage",
        "",
    ]
    mc = report["matrix_coverage"]
    lines.append(f"- Cells observed: {mc['cells']}, unique: {mc['unique_cells']}")
    lines.append(f"- Complete matrix: **{mc['complete']}**")
    for axis, missing in mc["missing"].items():
        if missing:
            lines.append(f"- Missing **{axis}**: {', '.join(missing)}")

    for metric, agg in report["per_metric"].items():
        lines += ["", f"## Skill on `{metric}`", ""]
        g = agg.get("global", {})
        lines.append("| Baseline | Mean RMSE | Median RMSE | Mean terminal err | Beats pers (sig) |")
        lines.append("|----------|-----------|-------------|-------------------|------------------|")
        for name, stats in g.items():
            lines.append(
                f"| {name} | {stats.get('mean_rmse', '?')} | "
                f"{stats.get('median_rmse', '?')} | "
                f"{stats.get('mean_terminal_error', '?')} | "
                f"{stats.get('n_significant_beats_persistence', 0)}"
                f"/{stats.get('n_scenarios', 0)} |"
            )
        lines.append("")
        lines.append("### By domain (mean RMSE of linear_trend)")
        lines.append("")
        lines.append("| Domain | n | linear_trend | ar1 | mean | persistence |")
        lines.append("|--------|---|--------------|-----|------|-------------|")
        by_dom = agg.get("by_domain", {})
        for dom, stats in sorted(by_dom.items()):
            def _rmse(name: str) -> str:
                x = stats.get(name, {}).get("mean_rmse")
                return f"{x}" if x is not None else "?"
            lines.append(
                f"| {dom} | {stats.get('persistence', {}).get('n_scenarios', 0)} | "
                f"{_rmse('linear_trend')} | {_rmse('ar1')} | {_rmse('mean')} | "
                f"{_rmse('persistence')} |"
            )

    lines += ["", "## Per-scenario highlights (support metric)", ""]
    lines.append("| Scenario | Domain/Region/Tension | n | GT | pers RMSE | best baseline | DM p |")
    lines.append("|----------|-----------------------|---|----|-----------|----------------|------|")
    for row in report["scenarios"]:
        if row["metric"] != "support":
            continue
        pers = row["baselines"].get("persistence", {})
        # Choose the baseline with lowest RMSE
        best_name, best = min(
            row["baselines"].items(), key=lambda kv: kv[1].get("rmse", float("inf"))
        )
        gt = row["ground_truth"]
        gt_str = f"{gt:.2f}" if isinstance(gt, (int, float)) else "-"
        lines.append(
            f"| {row['title'][:38]} | {row['domain']}/{row['region']}/{row['tension']} | "
            f"{row['n_rounds']} | {gt_str} | {pers.get('rmse', '?')} | "
            f"{best_name} ({best.get('rmse', '?')}) | "
            f"{best.get('dm_vs_persistence_p', '?')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run historical benchmarks on empirical corpus.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--markdown", type=Path, default=None)
    p.add_argument(
        "--metric", choices=list(METRICS) + ["all"], default="all",
        help="Restrict to a single metric",
    )
    args = p.parse_args(argv)

    metrics = METRICS if args.metric == "all" else (args.metric,)
    report = run_historical_benchmark(metrics=metrics)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    if not args.out and not args.markdown:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
