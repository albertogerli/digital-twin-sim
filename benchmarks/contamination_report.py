"""Render the JSON output of contamination_probe.py as a human-readable
markdown report suitable for a methodology section / pitch artifact.

Usage:
    python -m benchmarks.contamination_report
    python -m benchmarks.contamination_report --input outputs/contamination_probe.json \
        --output outputs/contamination_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _bucket(idx: float) -> str:
    if idx >= 0.60:
        return "high"
    if idx >= 0.35:
        return "medium"
    return "low"


def render(data: dict) -> str:
    summary = data.get("summary", {})
    probes = data.get("probes", [])

    n = summary.get("n_scenarios", 0)
    mean_idx = summary.get("mean_index", 0.0)
    by_axis = summary.get("by_axis_mean", {})

    buckets = {"high": [], "medium": [], "low": []}
    for p in probes:
        buckets[_bucket(p["index"])].append(p)
    for b in buckets.values():
        b.sort(key=lambda p: -p["index"])

    lines: list[str] = []
    lines.append("# Contamination Probe — LLM prior-knowledge audit")
    lines.append("")
    lines.append(
        f"Probe run on **{n} empirical scenarios** using the same LLM (Gemini "
        f"flash-lite) we use inside the simulation. Each scenario is queried "
        f"on four independent axes (outcome, trajectory, events, actors) with "
        f"schema-constrained JSON. Each axis is graded 0-1 against the "
        f"ground-truth file; the scenario index is a weighted mean "
        f"(outcome ×2, others ×1) scaled by the model's stated confidence."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Corpus mean contamination index: **{mean_idx:.3f}** "
                 f"({_bucket(mean_idx)})")
    lines.append(f"- Scenarios with high leakage (index ≥ 0.60): "
                 f"**{len(buckets['high'])} / {n}**")
    lines.append(f"- Scenarios with low leakage (index < 0.35): "
                 f"**{len(buckets['low'])} / {n}**")
    lines.append("")
    lines.append("## By axis")
    lines.append("")
    lines.append("| axis | mean leak score | read |")
    lines.append("|------|-----------------|------|")
    reads = {
        "outcome": "model recalls the final result / %s",
        "trajectory": "model recalls the shape of the polling movement",
        "events": "model recalls specific campaign events",
        "actors": "model recalls named individuals / organizations on each side",
    }
    for axis in ("outcome", "trajectory", "events", "actors"):
        v = by_axis.get(axis, 0.0)
        lines.append(f"| {axis} | {v:.3f} | {reads[axis]} |")
    lines.append("")

    lines.append("## High-contamination scenarios (unusable without blinding)")
    lines.append("")
    lines.append("Running the simulator on these *as-is* means the LLM has "
                 "non-trivial odds of simply regurgitating memorized ground "
                 "truth rather than simulating. These should only be used in "
                 "a **blinded** configuration — agents see stripped/renamed "
                 "entities, dates, and geography.")
    lines.append("")
    lines.append("| index | id | title |")
    lines.append("|-------|----|-------|")
    for p in buckets["high"]:
        title = (p["title"] or "")[:70]
        lines.append(f"| {p['index']:.3f} | `{p['id']}` | {title} |")
    lines.append("")

    lines.append("## Low-contamination scenarios (candidate clean benchmarks)")
    lines.append("")
    lines.append("These show weak prior knowledge across axes. Usable in "
                 "contaminated (naked) configuration as a sanity check, though "
                 "blinding is still recommended for the formal benchmark.")
    lines.append("")
    lines.append("| index | id | title |")
    lines.append("|-------|----|-------|")
    for p in buckets["low"]:
        title = (p["title"] or "")[:70]
        lines.append(f"| {p['index']:.3f} | `{p['id']}` | {title} |")
    lines.append("")

    lines.append("## Medium-contamination (grey zone)")
    lines.append("")
    lines.append("| index | id | title |")
    lines.append("|-------|----|-------|")
    for p in buckets["medium"]:
        title = (p["title"] or "")[:70]
        lines.append(f"| {p['index']:.3f} | `{p['id']}` | {title} |")
    lines.append("")

    # -- caveats derived from inspecting the data --
    lines.append("## Known grader caveats")
    lines.append("")
    lines.append(
        "- **Pro/against semantic flip**. Some ballot questions invert the "
        "`pro` label relative to common usage (e.g. for the Brexit scenario, "
        "`pro` = Leave in the dataset, but colloquially `pro-Brexit` = Leave "
        "= `pro`). When the model answers using the opposite convention its "
        "answers are graded as *wrong* even though it knows the underlying "
        "facts. Brexit's reported index (~0.17) understates its true "
        "contamination for this reason; the five English-speaking blue-chip "
        "referendums in the high bucket are not affected because their "
        "framing is unambiguous."
    )
    lines.append(
        "- **Confidence scaling**. The final score is multiplied by the "
        "model's own stated confidence (floored at 0.3). This rewards honest "
        "\"I don't know\" answers and penalizes overconfident wrong answers, "
        "but it also means a model that is *right but unsure* looks less "
        "contaminated than it actually is."
    )
    lines.append(
        "- **Actor matching** uses word-token intersection, so partial name "
        "overlaps score. This tends to be lenient."
    )
    lines.append("")

    lines.append("## Implications for the retrospective benchmark")
    lines.append("")
    lines.append(
        "1. **No naked benchmark on high-leak scenarios.** Running the sim "
        "on Scottish indyref '14, Italian constitutional '16, French '17, "
        "US '20 etc. without blinding produces uninterpretable numbers — we "
        "cannot distinguish simulation skill from LLM memory."
    )
    lines.append(
        "2. **Blinding is the load-bearing design choice.** Because "
        "`outcome` and `events` are the leakiest axes, the blinding protocol "
        "must (a) rename named events/actors, (b) strip absolute dates in "
        "favour of relative timestamps (T-30d, T-0), (c) use generic "
        "country/institution labels when the demographics themselves aren't "
        "needed for agent behaviour."
    )
    lines.append(
        "3. **Measure blinding lift.** For the pitch, the decisive experiment "
        "is: take one high-leak scenario, run it *contaminated* and *blinded*, "
        "and show the blinded sim trajectories still match ground truth "
        "(DTW / terminal error) at comparable accuracy to the contaminated "
        "run — ideally better than the baseline forecasters from "
        "`historical_runner`. That's the single slide that earns a ≥€20M "
        "valuation."
    )
    lines.append(
        "4. **Clean-benchmark candidates**: prefer `ARCHEGOS 2021`, "
        "`ELEZIONI EUROPEE 2019 ITALIA`, `MASKING MANDATE USA 2021`, "
        "`MONKEYPOX 2022`, `BOEING MAX return`, `CATALUNYA 2017`, and "
        "`ASTRAZENECA HESITANCY` as primary reporting set — these are where "
        "contaminated and blinded runs should converge."
    )
    lines.append("")

    stats = summary.get("stats", {})
    if stats:
        lines.append("## Cost")
        lines.append("")
        lines.append(f"- Total LLM cost: **${stats.get('total_cost', 0):.4f}**")
        lines.append(f"- Calls: {stats.get('calls', 0)}, "
                     f"errors: {stats.get('errors', 0)}")
        lines.append(f"- Tokens: {stats.get('in_tokens', 0):,} in / "
                     f"{stats.get('out_tokens', 0):,} out")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("outputs/contamination_probe.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/contamination_report.md"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    md = render(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"wrote {args.output} ({len(md):,} chars)")


if __name__ == "__main__":
    main()
