"""Validator for empirical calibration scenarios.

Validates against JSON schema, checks internal consistency, verifies
source URLs, and computes a data quality score.

Usage:
    python -m calibration.empirical.validate_scenario [path ...]
    python -m calibration.empirical.validate_scenario --all
"""

import json
import re
import sys
from datetime import datetime, date
from pathlib import Path
from statistics import stdev
from urllib.parse import urlparse

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    print("jsonschema not installed. Run: pip install jsonschema")
    sys.exit(1)


SCHEMA_PATH = Path(__file__).parent / "schema_empirical_scenario.json"
SCENARIOS_DIR = Path(__file__).parent / "scenarios"


# ── Schema Validation ───────────────────────────────────────────────

def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_schema(scenario: dict, schema: dict) -> list[str]:
    """Validate scenario against JSON schema. Returns list of errors."""
    errors = []
    try:
        validate(instance=scenario, schema=schema)
    except ValidationError as e:
        errors.append(f"Schema: {e.message} (path: {'/'.join(str(p) for p in e.absolute_path)})")
        # Collect all errors
        validator = jsonschema.Draft202012Validator(schema)
        for err in validator.iter_errors(scenario):
            if err.message != e.message:
                path = "/".join(str(p) for p in err.absolute_path)
                errors.append(f"Schema: {err.message} (path: {path})")
    return errors


# ── Consistency Checks ──────────────────────────────────────────────

def check_consistency(scenario: dict) -> list[str]:
    """Check internal consistency. Returns list of warnings/errors."""
    issues = []

    # 1. n_rounds matches polling_trajectory length
    n_rounds = scenario["n_rounds"]
    n_polling = len(scenario["polling_trajectory"])
    if n_polling != n_rounds:
        issues.append(
            f"Consistency: n_rounds={n_rounds} but polling_trajectory "
            f"has {n_polling} entries"
        )

    # 2. Polling rounds are sequential 1..n
    rounds_seen = [p["round"] for p in scenario["polling_trajectory"]]
    expected = list(range(1, n_rounds + 1))
    if rounds_seen != expected:
        issues.append(
            f"Consistency: polling rounds {rounds_seen} != expected {expected}"
        )

    # 3. Dates are ordered
    try:
        start = datetime.strptime(scenario["date_start"], "%Y-%m-%d").date()
        end = datetime.strptime(scenario["date_end"], "%Y-%m-%d").date()
        if end <= start:
            issues.append(f"Consistency: date_end ({end}) <= date_start ({start})")

        # Check polling dates are within range and ordered
        prev_date = None
        for p in scenario["polling_trajectory"]:
            d = datetime.strptime(p["date"], "%Y-%m-%d").date()
            if d < start or d > end:
                issues.append(
                    f"Consistency: polling date {d} outside "
                    f"[{start}, {end}] (round {p['round']})"
                )
            if prev_date and d < prev_date:
                issues.append(
                    f"Consistency: polling dates not ordered: "
                    f"{d} < {prev_date} (round {p['round']})"
                )
            prev_date = d
    except (ValueError, KeyError) as e:
        issues.append(f"Consistency: date parsing error: {e}")

    # 4. Event rounds are within [1, n_rounds]
    for ev in scenario.get("events", []):
        if ev["round"] < 1 or ev["round"] > n_rounds:
            issues.append(
                f"Consistency: event round {ev['round']} "
                f"outside [1, {n_rounds}]"
            )

    # 5. Ground truth pro_pct is plausible given final polling
    gt_pro = scenario["ground_truth_outcome"]["pro_pct"]
    last_poll = scenario["polling_trajectory"][-1]
    if last_poll["pro_pct"] is not None:
        delta = abs(gt_pro - last_poll["pro_pct"])
        if delta > 15:
            issues.append(
                f"Consistency: ground truth ({gt_pro:.1f}%) diverges "
                f"from final poll ({last_poll['pro_pct']:.1f}%) by {delta:.1f}pp"
            )

    # 6. Agent types: at least one elite and one citizen_cluster
    types = {a["type"] for a in scenario["agents"]}
    if "elite" not in types:
        issues.append("Consistency: no elite agents defined")
    if "citizen_cluster" not in types:
        issues.append("Consistency: no citizen_cluster agents defined")

    # 7. Polling percentages should be plausible
    for p in scenario["polling_trajectory"]:
        if p["pro_pct"] is not None:
            against = p.get("against_pct")
            undecided = p.get("undecided_pct")
            if against is not None and undecided is not None:
                total = p["pro_pct"] + against + undecided
                if abs(total - 100.0) > 2.0:
                    issues.append(
                        f"Consistency: round {p['round']} percentages "
                        f"sum to {total:.1f}%, expected ~100%"
                    )

    # 8. Covariates: initial_polarization should match agent positions
    positions = [a["initial_position"] for a in scenario["agents"]]
    if len(positions) >= 2:
        actual_std = stdev(positions)
        declared = scenario["covariates"]["initial_polarization"]
        if abs(actual_std - declared) > 0.15:
            issues.append(
                f"Consistency: declared initial_polarization={declared:.3f} "
                f"but agent position std={actual_std:.3f}"
            )

    return issues


# ── Source Validation ───────────────────────────────────────────────

def check_sources(scenario: dict) -> list[str]:
    """Check that sources look like valid URLs or citations."""
    issues = []

    def _check_source(source: str, context: str):
        if source is None:
            return
        # Accept URLs or citations (at least 10 chars with author-like pattern)
        parsed = urlparse(source)
        is_url = parsed.scheme in ("http", "https") and parsed.netloc
        is_citation = len(source) >= 10 and (
            "," in source or "et al" in source or "(" in source
            or source.startswith("doi:") or "/" in source
        )
        if not is_url and not is_citation:
            issues.append(f"Source: {context} — not a valid URL or citation: '{source[:60]}'")

    _check_source(scenario["ground_truth_outcome"]["source"], "ground_truth")

    for i, p in enumerate(scenario["polling_trajectory"]):
        _check_source(p["source"], f"polling[{i}]")

    for i, ev in enumerate(scenario.get("events", [])):
        if ev.get("source"):
            _check_source(ev["source"], f"events[{i}]")

    return issues


# ── Quality Score ───────────────────────────────────────────────────

def compute_quality_score(scenario: dict) -> dict:
    """Compute data quality score (0-100).

    Scoring:
      Base: 50 points (valid schema + consistency)
      +5  per polling round with non-null pro_pct (max 30)
      +3  per polling round with sample_size (max 18)
      +2  per event with source (max 10)
      +5  per covariate with non-null value (max 15)
      +5  if ground_truth source is URL
      -5  per consistency issue
      -3  per null pro_pct in polling
      -2  per null source in events
    """
    score = 50.0
    breakdown = {}

    # Polling completeness
    n_polls = len(scenario["polling_trajectory"])
    non_null_polls = sum(1 for p in scenario["polling_trajectory"] if p["pro_pct"] is not None)
    null_polls = n_polls - non_null_polls
    polls_bonus = min(30, non_null_polls * 5)
    polls_penalty = null_polls * 3
    score += polls_bonus - polls_penalty
    breakdown["polling_data"] = f"+{polls_bonus} ({non_null_polls}/{n_polls} non-null)"
    if null_polls:
        breakdown["polling_nulls"] = f"-{polls_penalty} ({null_polls} null pro_pct)"

    # Sample sizes
    has_sample = sum(1 for p in scenario["polling_trajectory"] if p.get("sample_size") is not None)
    sample_bonus = min(18, has_sample * 3)
    score += sample_bonus
    breakdown["sample_sizes"] = f"+{sample_bonus} ({has_sample}/{n_polls} with sample_size)"

    # Event sources
    events = scenario.get("events", [])
    sourced_events = sum(1 for e in events if e.get("source"))
    event_bonus = min(10, sourced_events * 2)
    unsourced = len(events) - sourced_events
    event_penalty = unsourced * 2
    score += event_bonus - event_penalty
    breakdown["event_sources"] = f"+{event_bonus} ({sourced_events}/{len(events)} sourced)"

    # Covariates
    covs = scenario.get("covariates", {})
    non_null_covs = sum(1 for v in covs.values() if v is not None)
    cov_bonus = min(15, non_null_covs * 5)
    score += cov_bonus
    breakdown["covariates"] = f"+{cov_bonus} ({non_null_covs}/5 non-null)"

    # Ground truth source quality
    gt_source = scenario["ground_truth_outcome"]["source"]
    parsed = urlparse(gt_source)
    if parsed.scheme in ("http", "https"):
        score += 5
        breakdown["gt_source"] = "+5 (URL)"
    else:
        breakdown["gt_source"] = "+0 (citation, not URL)"

    score = max(0, min(100, score))

    return {
        "score": round(score, 1),
        "grade": (
            "A" if score >= 85 else
            "B" if score >= 70 else
            "C" if score >= 55 else
            "D" if score >= 40 else
            "F"
        ),
        "breakdown": breakdown,
    }


# ── Full Validation ─────────────────────────────────────────────────

def validate_scenario(path: str) -> dict:
    """Full validation pipeline for a scenario file.

    Returns dict with errors, warnings, quality score, and pass/fail.
    """
    with open(path) as f:
        scenario = json.load(f)

    schema = load_schema()

    schema_errors = validate_schema(scenario, schema)
    consistency_issues = check_consistency(scenario) if not schema_errors else []
    source_issues = check_sources(scenario) if not schema_errors else []
    quality = compute_quality_score(scenario) if not schema_errors else {"score": 0, "grade": "F", "breakdown": {}}

    # Penalize quality for consistency issues
    if consistency_issues:
        quality["score"] = max(0, quality["score"] - len(consistency_issues) * 5)

    errors = schema_errors  # blocking
    warnings = consistency_issues + source_issues  # non-blocking

    return {
        "file": str(path),
        "id": scenario.get("id", "?"),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "quality": quality,
    }


def print_validation(result: dict):
    """Pretty-print validation result."""
    status = "\033[92m PASS \033[0m" if result["valid"] else "\033[91m FAIL \033[0m"
    grade = result["quality"]["grade"]
    score = result["quality"]["score"]

    print(f"\n{'─' * 60}")
    print(f"  {result['id']}  [{status}]  Quality: {grade} ({score}/100)")
    print(f"  {result['file']}")
    print(f"{'─' * 60}")

    if result["errors"]:
        print(f"  ERRORS ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"    \033[91m✗\033[0m {e}")

    if result["warnings"]:
        print(f"  WARNINGS ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"    \033[93m⚠\033[0m {w}")

    if result["quality"]["breakdown"]:
        print(f"  QUALITY BREAKDOWN:")
        for k, v in result["quality"]["breakdown"].items():
            print(f"    {k}: {v}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate empirical scenarios")
    parser.add_argument("paths", nargs="*", help="Scenario JSON files to validate")
    parser.add_argument("--all", action="store_true", help="Validate all in scenarios/")
    args = parser.parse_args()

    if args.all:
        paths = sorted(
            p for p in SCENARIOS_DIR.glob("*.json")
            if not p.name.endswith(".meta.json") and p.name != "manifest.json"
        )
    elif args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        print("Usage: validate_scenario.py [--all | file1.json file2.json ...]")
        sys.exit(1)

    if not paths:
        print(f"No scenario files found in {SCENARIOS_DIR}")
        sys.exit(1)

    results = []
    for p in paths:
        r = validate_scenario(str(p))
        print_validation(r)
        results.append(r)

    # Summary
    n_pass = sum(1 for r in results if r["valid"])
    n_fail = len(results) - n_pass
    avg_quality = sum(r["quality"]["score"] for r in results) / len(results) if results else 0

    print(f"\n{'═' * 60}")
    print(f"  SUMMARY: {n_pass} passed, {n_fail} failed, "
          f"avg quality {avg_quality:.0f}/100")
    print(f"{'═' * 60}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
