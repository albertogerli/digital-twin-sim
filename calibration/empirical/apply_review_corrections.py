"""Apply reviewer corrections to empirical scenarios.

Reads all scenario JSONs and .meta.json sidecars, applies corrections
from the external review, recalculates quality scores with review
penalties, updates manifest, and produces a review summary.

Usage:
    python -m calibration.empirical.apply_review_corrections
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from copy import deepcopy

from calibration.empirical.validate_scenario import (
    load_schema, validate_schema, check_consistency,
    check_sources, compute_quality_score,
)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
REVIEW_DIR = Path(__file__).parent / "review"

# ── Correction definitions ─────────────────────────────────────────

CORRECTIONS = {
    "FIN-2021-GAMESTOP": {
        "type": "note_update",
        "description": "Add verification uncertainty note to ground truth",
        "actions": [
            {
                "field": "notes",
                "action": "append",
                "value": (
                    " REVIEWER NOTE: The 72% ground truth figure is approximate; "
                    "Morning Consult polls from this period show high but variable "
                    "support for retail investors depending on question wording. "
                    "Treat as +/-5pp uncertainty."
                ),
            },
        ],
        "meta_updates": {
            "ground_truth_outcome.pro_pct": "partially_verified",
        },
    },
    "LAB-2015-UBER_VS_TAXI_PROTESTS_FRANCE_2": {
        "type": "shock_direction_fix",
        "description": "R1 taxi strike shock was pro-Uber (sympathy effect), not anti-Uber",
        "actions": [
            {
                "field": "events",
                "action": "fix_event",
                "match": {"round": 1, "date": "2015-01-26"},
                "updates": {"shock_direction": 0.3},
            },
        ],
    },
    "POL-2015-GREEK_BAILOUT_REFERENDUM_GREF": {
        "type": "date_correction",
        "description": "Tsipras announced referendum early hours of June 27, not June 26",
        "actions": [
            {
                "field": "events",
                "action": "fix_event",
                "match": {"round": 1, "date": "2015-06-26"},
                "updates": {"date": "2015-06-27"},
            },
        ],
    },
    "POL-2014-SCOTTISH_INDEPENDENCE_REFEREND": {
        "type": "date_correction",
        "description": (
            "Famous YouGov Yes-lead poll was Sep 6-7 2014, not Aug 25. "
            "Sep 6 falls in R7 (Aug 30 - Sep 18), so move event from R6 to R7."
        ),
        "actions": [
            {
                "field": "events",
                "action": "fix_event",
                "match": {"round": 6, "date": "2014-08-25"},
                "updates": {"date": "2014-09-06", "round": 7},
            },
        ],
    },
}

# Scenarios with documented overlaps (non-independent pairs)
OVERLAPS = [
    {
        "scenarios": ["CORP-2019-BOEING_MAX", "CORP-2020-BOEING_737_MAX_RETURN_TO_SERVI"],
        "reason": (
            "Cover different phases of the same Boeing 737 MAX crisis. "
            "If used together in calibration, they are not independent. "
            "Options: (a) use only one, (b) treat as multi-phase scenario, "
            "(c) document dependency and use both consciously."
        ),
    },
    {
        "scenarios": ["TECH-2018-GDPR_ADOPTION_AND_ACCEPTANCE_E", "TECH-2018-FACEBOOK_CAMBRIDGE_ANALYTICA_S"],
        "reason": (
            "Share the Cambridge Analytica scandal (March 17, 2018) as a "
            "common shock event. Not independent for calibration purposes."
        ),
    },
]

# Scenarios with interpolated undecided percentages
UNDECIDED_INTERPOLATED = [
    "CORP-2017-UBER_LONDON_LICENSE_BATTLE_201",
    "TECH-2018-FACEBOOK_CAMBRIDGE_ANALYTICA_S",
]

# Scenarios with monotonic (too smooth) trajectories
MONOTONIC_TRAJECTORIES = [
    "ENV-2018-GRETA_THUNBERG_CLIMATE_STRIKES",
]


# ── Helpers ────────────────────────────────────────────────────────

def load_scenario(scenario_id: str) -> tuple[dict, dict | None, Path]:
    """Load scenario JSON and optional meta. Returns (scenario, meta, path)."""
    path = SCENARIOS_DIR / f"{scenario_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario not found: {path}")
    with open(path) as f:
        scenario = json.load(f)

    meta_path = path.with_suffix("").with_suffix(".meta.json")
    meta = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return scenario, meta, path


def save_scenario(scenario: dict, path: Path):
    """Save scenario JSON."""
    with open(path, "w") as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)


def save_meta(meta: dict, scenario_id: str):
    """Save meta JSON."""
    meta_path = SCENARIOS_DIR / f"{scenario_id}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def apply_correction(scenario: dict, correction: dict) -> list[str]:
    """Apply a single correction to a scenario. Returns list of changes made."""
    changes = []
    for action in correction["actions"]:
        if action["action"] == "append" and action["field"] == "notes":
            old_notes = scenario.get("notes") or ""
            scenario["notes"] = old_notes + action["value"]
            changes.append(f"Appended reviewer note to notes field")

        elif action["action"] == "fix_event":
            for ev in scenario.get("events", []):
                match = action["match"]
                if all(ev.get(k) == v for k, v in match.items()):
                    for k, v in action["updates"].items():
                        old_val = ev.get(k)
                        ev[k] = v
                        changes.append(f"Event R{match['round']}: {k} {old_val} -> {v}")
                    break
            else:
                changes.append(f"WARNING: could not find event matching {action['match']}")

    return changes


def compute_reviewed_quality(scenario: dict, scenario_id: str, meta: dict | None) -> dict:
    """Compute quality score with review penalties."""
    base = compute_quality_score(scenario)
    penalties = []

    # Penalty for partially_verified ground truth
    if meta:
        fp = meta.get("field_provenance", {})
        if fp.get("ground_truth_outcome.pro_pct") == "partially_verified":
            base["score"] -= 5
            penalties.append("partially_verified ground truth: -5")

    # Penalty for interpolated undecided
    if scenario_id in UNDECIDED_INTERPOLATED:
        base["score"] -= 3
        penalties.append("undecided_interpolated: -3")

    # Penalty for overlap with other scenarios
    for overlap in OVERLAPS:
        if scenario_id in overlap["scenarios"]:
            base["score"] -= 10
            penalties.append("overlap with another scenario: -10")
            break

    base["score"] = max(0, min(100, base["score"]))
    if penalties:
        base["review_penalties"] = penalties
    # Recompute grade
    s = base["score"]
    base["grade"] = (
        "A" if s >= 85 else "B" if s >= 70 else
        "C" if s >= 55 else "D" if s >= 40 else "F"
    )
    return base


# ── Main pipeline ──────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  APPLYING REVIEW CORRECTIONS")
    print("=" * 70)

    # Collect all scenarios
    all_paths = sorted(
        p for p in SCENARIOS_DIR.glob("*.json")
        if not p.name.endswith(".meta.json") and p.name != "manifest.json"
    )

    results = []  # (id, verdict, corrections_list, quality_before, quality_after)
    corrections_applied = {}

    for path in all_paths:
        with open(path) as f:
            scenario = json.load(f)
        sid = scenario["id"]

        meta_path = path.with_suffix("").with_suffix(".meta.json")
        meta = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        # Quality before
        quality_before = compute_quality_score(scenario)
        # Also apply consistency penalties like the validator does
        consistency = check_consistency(scenario)
        if consistency:
            quality_before["score"] = max(0, quality_before["score"] - len(consistency) * 5)

        # Apply corrections if any
        changes = []
        verdict = "PASS"
        if sid in CORRECTIONS:
            corr = CORRECTIONS[sid]
            changes = apply_correction(scenario, corr)
            verdict = "CORRECTIONS"

            # Apply meta updates if specified
            if "meta_updates" in corr:
                if meta is None:
                    # Create meta for hand-crafted scenarios
                    meta = {
                        "generation_timestamp": "hand_crafted",
                        "generator": "hand_crafted",
                        "field_provenance": {},
                        "verification_summary": {},
                    }
                for field, prov in corr["meta_updates"].items():
                    meta["field_provenance"][field] = prov
                    changes.append(f"Meta: {field} -> {prov}")

            # Save corrected scenario
            save_scenario(scenario, path)
            if meta is not None:
                save_meta(meta, sid)

            corrections_applied[sid] = {
                "description": corr["description"],
                "changes": changes,
            }
            print(f"  [{sid}] CORRECTIONS applied: {len(changes)} changes")
        else:
            print(f"  [{sid}] PASS (no corrections needed)")

        # Mark undecided_interpolated in meta
        if sid in UNDECIDED_INTERPOLATED:
            if meta is None:
                meta = {
                    "generation_timestamp": "hand_crafted",
                    "generator": "hand_crafted",
                    "field_provenance": {},
                    "verification_summary": {},
                }
            meta["undecided_interpolated"] = True
            save_meta(meta, sid)
            changes.append("Meta: undecided_interpolated = true")

        # Quality after (with review penalties)
        quality_after = compute_reviewed_quality(scenario, sid, meta)
        # Also apply consistency penalties
        consistency = check_consistency(scenario)
        if consistency:
            quality_after["score"] = max(0, quality_after["score"] - len(consistency) * 5)

        results.append({
            "id": sid,
            "verdict": verdict,
            "corrections": changes,
            "quality_before": quality_before["score"],
            "quality_after": quality_after["score"],
            "grade_before": quality_before["grade"],
            "grade_after": quality_after["grade"],
            "review_penalties": quality_after.get("review_penalties", []),
        })

    # ── Update manifest ────────────────────────────────────────────
    manifest_path = SCENARIOS_DIR / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    manifest["review_status"] = "reviewed"
    manifest["review_date"] = "2026-03-29"
    manifest["corrections_applied"] = corrections_applied

    manifest["reviewer_notes"] = [
        {
            "id": "overlap_boeing",
            "severity": "warning",
            "scenarios": ["CORP-2019-BOEING_MAX", "CORP-2020-BOEING_737_MAX_RETURN_TO_SERVI"],
            "note": OVERLAPS[0]["reason"],
        },
        {
            "id": "overlap_gdpr_facebook",
            "severity": "warning",
            "scenarios": ["TECH-2018-GDPR_ADOPTION_AND_ACCEPTANCE_E", "TECH-2018-FACEBOOK_CAMBRIDGE_ANALYTICA_S"],
            "note": OVERLAPS[1]["reason"],
        },
        {
            "id": "undecided_interpolated",
            "severity": "info",
            "scenarios": UNDECIDED_INTERPOLATED,
            "note": (
                "Undecided percentages are clearly interpolated by the LLM "
                "(constant 10% across all rounds). Marked in .meta.json."
            ),
        },
        {
            "id": "monotonic_trajectory",
            "severity": "info",
            "scenarios": MONOTONIC_TRAJECTORIES,
            "note": (
                "ENV-2018-GRETA_THUNBERG has monotonically increasing trajectory "
                "with no dips. Real data showed backlash phases. Usable for trend "
                "calibration but not for oscillatory dynamics."
            ),
        },
    ]

    # Update per-scenario review status in manifest
    for entry in manifest.get("scenarios", []):
        entry["review_status"] = "reviewed"
        entry["review_date"] = "2026-03-29"
        for r in results:
            if r["id"] == entry["id"]:
                entry["quality_score_reviewed"] = r["quality_after"]
                entry["quality_grade_reviewed"] = r["grade_after"]
                entry["review_verdict"] = r["verdict"]
                break

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n  Manifest updated: {manifest_path}")

    # ── Generate review_summary.md ─────────────────────────────────
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REVIEW_DIR / "review_summary.md"

    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    n_corr = sum(1 for r in results if r["verdict"] == "CORRECTIONS")
    n_reject = sum(1 for r in results if r["verdict"] == "REJECT")

    lines = [
        "# Review Summary — Empirical Calibration Scenarios",
        "",
        f"**Review date:** 2026-03-29",
        f"**Total scenarios:** {len(results)}",
        f"**Verdicts:** {n_pass} PASS, {n_corr} CORRECTIONS, {n_reject} REJECT",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Verdict | Corrections | Quality Before | Quality After |",
        "|----------|---------|-------------|---------------|--------------|",
    ]

    for r in results:
        corr_str = "; ".join(r["corrections"]) if r["corrections"] else "-"
        if len(corr_str) > 60:
            corr_str = corr_str[:57] + "..."
        penalties = ""
        if r["review_penalties"]:
            penalties = " (" + ", ".join(r["review_penalties"]) + ")"
        lines.append(
            f"| {r['id'][:40]} | {r['verdict']} | {corr_str} | "
            f"{r['quality_before']}/{r['grade_before']} | "
            f"{r['quality_after']}/{r['grade_after']}{penalties} |"
        )

    lines.extend([
        "",
        "## Scenario Overlaps (Non-Independent Pairs)",
        "",
    ])
    for ov in OVERLAPS:
        lines.append(f"- **{' + '.join(ov['scenarios'])}**")
        lines.append(f"  {ov['reason']}")
        lines.append("")

    lines.extend([
        "## Data Quality Warnings",
        "",
        f"- **Interpolated undecided:** {', '.join(UNDECIDED_INTERPOLATED)}",
        f"- **Monotonic trajectory:** {', '.join(MONOTONIC_TRAJECTORIES)}",
        "",
        "## Independence Recommendation",
        "",
        "For calibration requiring independent scenarios, exclude one from each overlap pair:",
        "",
        "- **Recommended exclusion set A** (conservative, 22 scenarios):",
        "  - Drop `CORP-2020-BOEING_737_MAX_RETURN_TO_SERVI` (keep original crisis)",
        "  - Drop `TECH-2018-FACEBOOK_CAMBRIDGE_ANALYTICA_S` (keep GDPR, broader scope)",
        "",
        "- **Recommended exclusion set B** (aggressive, 20 scenarios):",
        "  - Same as A, plus drop `ENV-2018-GRETA_THUNBERG_CLIMATE_STRIKES` (monotonic)",
        "  - Plus drop `CORP-2017-UBER_LONDON_LICENSE_BATTLE_201` (flat undecided)",
        "",
    ])

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Review summary: {summary_path}")

    # ── Print summary ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  REVIEW COMPLETE")
    print(f"  {n_pass} PASS | {n_corr} CORRECTIONS | {n_reject} REJECT")
    avg_before = sum(r["quality_before"] for r in results) / len(results)
    avg_after = sum(r["quality_after"] for r in results) / len(results)
    print(f"  Avg quality: {avg_before:.1f} -> {avg_after:.1f}")

    # Show scenarios with penalties
    penalized = [r for r in results if r["review_penalties"]]
    if penalized:
        print(f"\n  Scenarios with review penalties:")
        for r in penalized:
            print(f"    {r['id']}: {r['quality_before']} -> {r['quality_after']} ({', '.join(r['review_penalties'])})")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
