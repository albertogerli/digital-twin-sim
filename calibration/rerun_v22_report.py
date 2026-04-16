#!/usr/bin/env python3
"""Re-run v2.2 steps 3+4 (compare + report) from saved results."""

import json
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Paths
SCENARIOS_ORIG_DIR = ROOT / "calibration" / "empirical" / "scenarios"
SCENARIOS_V22_DIR = ROOT / "calibration" / "empirical" / "scenarios_v2.2"
POSTERIORS_V2_PATH = ROOT / "calibration" / "results" / "hierarchical_calibration" / "v2_discrepancy" / "posteriors_v2.json"
V22_OUTPUT_DIR = ROOT / "calibration" / "results" / "hierarchical_calibration" / "v2.2_hybrid"

GROUND_DOMAINS = {"financial", "corporate", "energy", "public_health"}

# Load v2.2 results from disk
with open(V22_OUTPUT_DIR / "posteriors_v2.json") as f:
    posteriors_v22 = json.load(f)
with open(V22_OUTPUT_DIR / "validation_results_v2.json") as f:
    results_per_scenario = json.load(f)

# Load v2 posteriors
with open(POSTERIORS_V2_PATH) as f:
    posteriors_v2 = json.load(f)

# Identify hybrid scenarios
hybrid_scenarios = []
for p in sorted(SCENARIOS_V22_DIR.glob("*.json")):
    if p.name == "manifest.json" or p.name.endswith(".meta.json"):
        continue
    with open(p) as f:
        d = json.load(f)
    if d.get("_grounding_v2.2"):
        hybrid_scenarios.append(p.stem)

print(f"Hybrid scenarios: {len(hybrid_scenarios)}")

# Compute v2.2 metrics from validation results
train_results = [r for r in results_per_scenario if r["group"] == "train"]
test_results = [r for r in results_per_scenario if r["group"] == "test"]

def compute_agg(results):
    if not results:
        return {}
    maes = [r["abs_error"] for r in results]
    errors = [r["error"] for r in results]
    cov90 = [r["in_90"] for r in results]
    cov50 = [r["in_50"] for r in results]
    crps = [r["crps"] for r in results]
    return {
        "mae": np.mean(maes),
        "rmse": np.sqrt(np.mean(np.array(errors)**2)),
        "coverage_90": np.mean(cov90),
        "coverage_50": np.mean(cov50),
        "mean_crps": np.mean(crps),
        "n": len(results),
    }

train_agg = compute_agg(train_results)
test_agg = compute_agg(test_results)

# v2.1 metrics (from previous run)
v21_mae_test = 18.9
v21_mae_train = 15.1
v21_rmse_test = 21.6
v21_cov90_train = 2.9

# ── Table A: Headline Metrics ──
print("\n" + "=" * 70)
print("COMPARISON: v2 vs v2.1 vs v2.2")
print("=" * 70)

print("\n── Table A: Headline Metrics ──")
print(f"{'Metric':<30} {'v2':>10} {'v2.1':>10} {'v2.2':>10}")
print("-" * 65)

rows = [
    ("MAE test",        19.2, v21_mae_test,   test_agg.get("mae")),
    ("MAE train",       14.3, v21_mae_train,  train_agg.get("mae")),
    ("RMSE test",       26.6, v21_rmse_test,  test_agg.get("rmse")),
    ("Coverage 90% tr", 79.4, v21_cov90_train, train_agg.get("coverage_90", 0) * 100),
    ("Coverage 50% tr", None, None, train_agg.get("coverage_50", 0) * 100),
    ("Mean CRPS train", None, None, train_agg.get("mean_crps")),
]

for name, v2_val, v21_val, v22_val in rows:
    v2_str = f"{v2_val:>8.1f}pp" if v2_val is not None else f"{'':>10}"
    v21_str = f"{v21_val:>8.1f}pp" if v21_val is not None else f"{'':>10}"
    v22_str = f"{v22_val:>8.1f}pp" if v22_val is not None else f"{'?':>10}"
    print(f"  {name:<28} {v2_str} {v21_str} {v22_str}")

# ── Table B: Discrepancy ──
v2_disc = posteriors_v2.get("discrepancy", {})
v22_disc = posteriors_v22.get("discrepancy", {})

print("\n── Table B: Discrepancy ──")
print(f"{'Metric':<25} {'v2':>12} {'v2.2':>12} {'Δ':>10}")
print("-" * 62)

for label, key in [("σ_b,between", "sigma_delta_between"), ("σ_b,within", "sigma_delta_within")]:
    v2_val = v2_disc.get(key, {}).get("mean", 0)
    v22_val = v22_disc.get(key, {}).get("mean", None)
    if v22_val is not None:
        delta = v22_val - v2_val
        print(f"  {label:<23} {v2_val:>10.3f} {v22_val:>10.3f} {delta:>+10.3f}")
    else:
        print(f"  {label:<23} {v2_val:>10.3f} {'?':>12} {'?':>10}")

# ── Table C: Per-scenario detail ──
v2_scenarios = posteriors_v2.get("scenarios", {})
v22_scenarios = posteriors_v22.get("scenarios", {})

print("\n── Table C: Per-Scenario Results (ALL) ──")
print(f"{'Scenario':<45} {'GT%':>5} {'Sim%':>6} {'Err':>7} {'|δ_s|v2':>8} {'|δ_s|v22':>9} {'90%CI':>7}")
print("-" * 95)

for r in sorted(results_per_scenario, key=lambda x: -abs(x["error"])):
    sid = r["id"]
    v2_ds = v2_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
    v22_ds = v22_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
    ci_str = "✓" if r["in_90"] else "✗"
    grounded = "H" if sid in hybrid_scenarios else " "
    print(f"{grounded} {sid[:43]:<43} {r['gt']:>5.1f} {r['sim_mean']:>6.1f} {r['error']:>+7.1f} {abs(v2_ds):>8.3f} {abs(v22_ds):>9.3f} {ci_str:>5}")

# ── Per-scenario grounded only ──
print("\n── Table D: Grounded Scenarios δ_s Comparison ──")
print(f"{'Scenario':<45} {'GT%':>5} {'v2 δ_s':>8} {'v22 δ_s':>9} {'|Δ|δ_s|':>8} {'Sim%':>6} {'Err':>7}")
print("-" * 95)

improved = 0
for sid in sorted(hybrid_scenarios):
    v2_ds = v2_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
    v22_ds = v22_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
    delta_abs = abs(v2_ds) - abs(v22_ds)
    better = abs(v22_ds) < abs(v2_ds)
    if better:
        improved += 1
    # Find this scenario in results
    res = next((r for r in results_per_scenario if r["id"] == sid), None)
    gt = res["gt"] if res else 0
    sim = res["sim_mean"] if res else 0
    err = res["error"] if res else 0
    mark = "▼" if better else "▲"
    print(f"  {sid[:43]:<43} {gt:>5.1f} {v2_ds:>+8.3f} {v22_ds:>+9.3f} {delta_abs:>+8.3f}{mark} {sim:>6.1f} {err:>+7.1f}")

print(f"\n  Improved δ_s: {improved}/{len(hybrid_scenarios)} grounded scenarios")
mean_v2 = np.mean([abs(v2_scenarios.get(s, {}).get("delta_s", {}).get("mean", 0)) for s in hybrid_scenarios])
mean_v22 = np.mean([abs(v22_scenarios.get(s, {}).get("delta_s", {}).get("mean", 0)) for s in hybrid_scenarios])
print(f"  Mean |δ_s| grounded: v2={mean_v2:.3f} → v2.2={mean_v22:.3f} ({mean_v22-mean_v2:+.3f})")

# ── Generate Report ──
lines = [
    "# Grounding v2.2: Hybrid Grounding + Discrepancy Model",
    "",
    "## Strategy",
    "",
    "- **Agents**: Keep original (human-calibrated positions, influence, rigidity)",
    "- **Events**: Replace with grounded verified events from Google Search",
    "- **Model**: hierarchical_model_v2 with discrepancy δ_s (not transfer model)",
    "",
    f"- Hybrid scenarios: {len(hybrid_scenarios)}",
    f"- Total scenarios: {len(results_per_scenario)}",
    f"- Train: {len(train_results)}, Test: {len(test_results)}",
    "",
    "## Headline Metrics",
    "",
    "| Metric | v2 | v2.1 | v2.2 | Best? |",
    "|---|---|---|---|---|",
]

v22_mae_test = test_agg["mae"]
v22_mae_train = train_agg["mae"]
v22_rmse_test = test_agg["rmse"]
v22_cov90 = train_agg["coverage_90"] * 100

best_mae = "v2.2" if v22_mae_test < min(19.2, 18.9) else ("v2.1" if 18.9 < 19.2 else "v2")
best_rmse = "v2.2" if v22_rmse_test < min(26.6, 21.6) else ("v2.1" if 21.6 < 26.6 else "v2")
best_cov = "v2.2" if v22_cov90 > max(79.4, 2.9) else ("v2" if 79.4 > 2.9 else "v2.1")

lines.append(f"| MAE test | 19.2pp | 18.9pp | {v22_mae_test:.1f}pp | {best_mae} |")
lines.append(f"| MAE train | 14.3pp | 15.1pp | {v22_mae_train:.1f}pp | |")
lines.append(f"| RMSE test | 26.6pp | 21.6pp | {v22_rmse_test:.1f}pp | {best_rmse} |")
lines.append(f"| Coverage 90% train | 79.4% | 2.9% | {v22_cov90:.1f}% | {best_cov} |")

# Discrepancy
sigma_bw_v2 = v2_disc.get("sigma_delta_within", {}).get("mean", 0)
sigma_bw_v22 = v22_disc.get("sigma_delta_within", {}).get("mean", 0)
sigma_bb_v2 = v2_disc.get("sigma_delta_between", {}).get("mean", 0)
sigma_bb_v22 = v22_disc.get("sigma_delta_between", {}).get("mean", 0)

lines.extend([
    "", "## Discrepancy", "",
    "| Metric | v2 | v2.2 | Δ |",
    "|---|---|---|---|",
    f"| σ_b,between | {sigma_bb_v2:.3f} | {sigma_bb_v22:.3f} | {sigma_bb_v22-sigma_bb_v2:+.3f} |",
    f"| σ_b,within | {sigma_bw_v2:.3f} | {sigma_bw_v22:.3f} | {sigma_bw_v22-sigma_bw_v2:+.3f} |",
])

# Per-scenario detail
lines.extend(["", "## Per-Scenario Results", ""])
lines.append("| Scenario | Group | GT% | Sim% | Err | δ_s v2 | δ_s v2.2 | in 90%CI |")
lines.append("|---|---|---|---|---|---|---|---|")

for r in sorted(results_per_scenario, key=lambda x: x["id"]):
    sid = r["id"]
    v2_ds = v2_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
    v22_ds = v22_scenarios.get(sid, {}).get("delta_s", {}).get("mean", 0)
    ci_str = "YES" if r["in_90"] else "no"
    grounded = " (H)" if sid in hybrid_scenarios else ""
    lines.append(f"| {sid[:40]}{grounded} | {r['group']} | {r['gt']:.1f} | {r['sim_mean']:.1f} | {r['error']:+.1f} | {v2_ds:+.3f} | {v22_ds:+.3f} | {ci_str} |")

# Verdict
lines.extend(["", "## Verdict", ""])

verdict_parts = []
if v22_mae_test < 18.9:
    verdict_parts.append(f"MAE test improved: 19.2→{v22_mae_test:.1f}pp (better than v2.1's 18.9)")
elif v22_mae_test < 19.2:
    verdict_parts.append(f"MAE test: {v22_mae_test:.1f}pp (between v2 and v2.1)")
else:
    verdict_parts.append(f"MAE test: {v22_mae_test:.1f}pp (regression vs v2's 19.2)")

if v22_cov90 > 50:
    verdict_parts.append(f"Coverage 90% restored: {v22_cov90:.1f}% (v2=79.4%, v2.1=2.9%)")
elif v22_cov90 > 10:
    verdict_parts.append(f"Coverage 90% partially recovered: {v22_cov90:.1f}%")
else:
    verdict_parts.append(f"Coverage 90% still low: {v22_cov90:.1f}%")

if sigma_bw_v22 < sigma_bw_v2:
    verdict_parts.append(f"σ_b,within reduced: {sigma_bw_v2:.3f}→{sigma_bw_v22:.3f}")

verdict_parts.append(f"Improved δ_s: {improved}/{len(hybrid_scenarios)} grounded scenarios")
verdict_parts.append(f"Mean |δ_s| grounded: v2={mean_v2:.3f} → v2.2={mean_v22:.3f} ({mean_v22-mean_v2:+.3f})")

for v in verdict_parts:
    lines.append(f"- {v}")

lines.extend(["", "---", f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*"])

report_path = V22_OUTPUT_DIR / "calibration_report_v2.2.md"
with open(report_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nReport saved: {report_path}")
