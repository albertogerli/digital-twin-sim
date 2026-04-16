"""LLM-assisted empirical scenario generator.

Takes a CSV of historical events and uses Gemini to draft full scenario JSONs
following the empirical schema. Each field is tagged as 'verified' (from CSV)
or 'llm_estimated' in a metadata layer. Outputs validated scenarios + manifest.

Usage:
    python -m calibration.empirical.scenario_generator_assisted \
        --csv events.csv [--output-dir calibration/empirical/scenarios] \
        [--max-concurrent 3] [--dry-run]

CSV format (required columns):
    title, domain, country, date_start, date_end, outcome_pro_pct,
    outcome_source, outcome_type

Optional CSV columns:
    id, n_rounds, round_duration_days, notes
"""

import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from statistics import stdev
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.llm.gemini_client import GeminiClient
from calibration.empirical.validate_scenario import (
    load_schema, validate_schema, check_consistency, check_sources,
    compute_quality_score,
)

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema_empirical_scenario.json"
SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# Domains from schema
VALID_DOMAINS = [
    "political", "financial", "commercial", "public_health",
    "corporate", "environmental", "technology", "labor",
    "social", "marketing", "sport", "energy", "legal",
]

OUTCOME_TYPES = [
    "referendum", "election", "poll_final", "market_close",
    "survey", "regulatory_outcome",
]

# ── CSV Parsing ────────────────────────────────────────────────────

def parse_csv(csv_path: str) -> list[dict]:
    """Parse input CSV into list of event specs."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"title", "domain", "country", "date_start", "date_end",
                     "outcome_pro_pct", "outcome_source", "outcome_type"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"CSV missing required columns: {missing}")

        for i, row in enumerate(reader, 1):
            # Validate domain
            domain = row["domain"].strip().lower()
            if domain not in VALID_DOMAINS:
                logger.warning(f"Row {i}: invalid domain '{domain}', skipping")
                continue

            # Validate country code
            country = row["country"].strip().upper()
            if not re.match(r"^[A-Z]{2}$", country):
                logger.warning(f"Row {i}: invalid country '{country}', skipping")
                continue

            # Parse dates
            try:
                date_start = datetime.strptime(row["date_start"].strip(), "%Y-%m-%d")
                date_end = datetime.strptime(row["date_end"].strip(), "%Y-%m-%d")
                if date_end <= date_start:
                    logger.warning(f"Row {i}: date_end <= date_start, skipping")
                    continue
            except ValueError:
                logger.warning(f"Row {i}: invalid dates, skipping")
                continue

            # Parse outcome
            try:
                outcome_pro_pct = float(row["outcome_pro_pct"].strip())
            except ValueError:
                logger.warning(f"Row {i}: invalid outcome_pro_pct, skipping")
                continue

            outcome_type = row["outcome_type"].strip().lower()
            if outcome_type not in OUTCOME_TYPES:
                logger.warning(f"Row {i}: invalid outcome_type '{outcome_type}', defaulting to 'survey'")
                outcome_type = "survey"

            # Generate ID if not provided
            if row.get("id", "").strip():
                scenario_id = row["id"].strip().upper()
            else:
                domain_prefix = domain[:3].upper()
                if domain == "public_health":
                    domain_prefix = "PH"
                elif domain == "commercial":
                    domain_prefix = "COM"
                elif domain == "political":
                    domain_prefix = "POL"
                elif domain == "financial":
                    domain_prefix = "FIN"
                elif domain == "corporate":
                    domain_prefix = "CORP"
                elif domain == "environmental":
                    domain_prefix = "ENV"
                elif domain == "technology":
                    domain_prefix = "TECH"

                year = date_start.year
                slug = re.sub(r"[^A-Z0-9]", "_", row["title"].strip().upper()[:30])
                slug = re.sub(r"_+", "_", slug).strip("_")
                scenario_id = f"{domain_prefix}-{year}-{slug}"

            # Optional fields
            n_rounds = int(row.get("n_rounds", "0").strip() or "0")
            round_duration = int(row.get("round_duration_days", "0").strip() or "0")

            # Auto-compute if not provided
            total_days = (date_end - date_start).days
            if n_rounds == 0:
                # Heuristic: ~5-8 rounds
                if total_days <= 30:
                    n_rounds = 5
                elif total_days <= 90:
                    n_rounds = 6
                elif total_days <= 180:
                    n_rounds = 7
                else:
                    n_rounds = min(9, max(5, total_days // 60))
            if round_duration == 0:
                round_duration = max(1, total_days // n_rounds)

            rows.append({
                "id": scenario_id,
                "title": row["title"].strip(),
                "domain": domain,
                "country": country,
                "date_start": row["date_start"].strip(),
                "date_end": row["date_end"].strip(),
                "n_rounds": n_rounds,
                "round_duration_days": round_duration,
                "outcome_pro_pct": outcome_pro_pct,
                "outcome_source": row["outcome_source"].strip(),
                "outcome_type": outcome_type,
                "notes": row.get("notes", "").strip(),
            })

    logger.info(f"Parsed {len(rows)} valid events from CSV")
    return rows


# ── LLM Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert researcher generating structured empirical scenario data
for calibrating an opinion dynamics simulation model. You must produce accurate,
well-sourced JSON following a strict schema.

CRITICAL RULES:
1. Use REAL data when you know it. Mark estimates clearly in the source field.
2. For polling_trajectory: use actual survey data when known, otherwise provide
   reasonable estimates with "Estimated:" prefix in source field.
3. For events: use real historical events with actual dates. Include URLs to
   reputable sources (BBC, Reuters, NYT, gov sites, academic papers).
4. For agents: use real people/organizations involved. Set positions on [-1, +1]
   where +1 = pro-side, -1 = against-side.
5. shock_magnitude [0,1]: how big the event's impact was on public opinion.
   shock_direction [-1,+1]: which direction it pushed opinion.
6. Covariates must be realistic floats in [0,1].
7. All dates must be YYYY-MM-DD format.
8. Polling percentages: pro_pct + against_pct + undecided_pct should sum to ~100%.
9. Include 3-6 events, 6-12 agents (mix of elite, institutional, citizen_cluster).
10. sample_size should be null unless you know the actual survey sample size."""


def build_generation_prompt(event: dict) -> str:
    """Build the LLM prompt for scenario generation."""
    # Compute round dates for reference
    start = datetime.strptime(event["date_start"], "%Y-%m-%d")
    round_dates = []
    for r in range(event["n_rounds"]):
        d = start + timedelta(days=r * event["round_duration_days"])
        round_dates.append(d.strftime("%Y-%m-%d"))

    return f"""Generate a complete empirical scenario JSON for calibrating an opinion dynamics model.

EVENT: {event["title"]}
DOMAIN: {event["domain"]}
COUNTRY: {event["country"]}
PERIOD: {event["date_start"]} to {event["date_end"]}
ROUNDS: {event["n_rounds"]} rounds, each ~{event["round_duration_days"]} days
ROUND DATES (approximate): {", ".join(round_dates)}
GROUND TRUTH OUTCOME: {event["outcome_pro_pct"]}% pro-side
OUTCOME SOURCE: {event["outcome_source"]}
OUTCOME TYPE: {event["outcome_type"]}
{f"NOTES: {event['notes']}" if event.get("notes") else ""}

The "pro" side is defined as the position that ended at {event["outcome_pro_pct"]}%.
Clearly state what "pro" and "against" mean in the notes field.

Generate the COMPLETE JSON object with these exact fields:
- id: "{event["id"]}"
- domain: "{event["domain"]}"
- title: (descriptive, 10-200 chars)
- country: "{event["country"]}"
- date_start: "{event["date_start"]}"
- date_end: "{event["date_end"]}"
- n_rounds: {event["n_rounds"]}
- round_duration_days: {event["round_duration_days"]}
- ground_truth_outcome: {{pro_pct: {event["outcome_pro_pct"]}, source: "{event["outcome_source"]}", type: "{event["outcome_type"]}"}}
- polling_trajectory: array of {event["n_rounds"]} objects, one per round:
  {{round: int, date: "YYYY-MM-DD", pro_pct: float|null, against_pct: float|null, undecided_pct: float|null, sample_size: int|null, source: "string", pollster: string|null}}
- events: array of 3-6 key events:
  {{round: int, date: "YYYY-MM-DD", description: "string (>10 chars)", shock_magnitude: float [0,1], shock_direction: float [-1,+1], source: "URL or null"}}
- agents: array of 6-12 agents:
  {{name: "string", type: "elite"|"institutional"|"citizen_cluster", initial_position: float [-1,+1], influence: float [0,1], description: "string"}}
- covariates: {{initial_polarization: float, event_volatility: float, elite_concentration: float|null, institutional_trust: float|null, undecided_share: float|null}}
- notes: "string explaining pro/against definitions and data quality caveats"

For each polling round source field:
- If you know actual survey data, cite it with URL
- If estimating, prefix with "Estimated:" and explain basis

Return ONLY the JSON object, no markdown fences, no explanation."""


# ── Metadata Layer ─────────────────────────────────────────────────

def compute_metadata(event_spec: dict, scenario: dict) -> dict:
    """Tag each field as 'verified' (from CSV) or 'llm_estimated'."""
    metadata = {
        "generation_timestamp": datetime.now(tz=None).astimezone().isoformat(),
        "generator": "scenario_generator_assisted.py",
        "llm_model": "gemini-3.1-flash-lite-preview",
        "field_provenance": {},
    }

    # Fields that come directly from CSV = verified
    verified_fields = [
        "id", "domain", "country", "date_start", "date_end",
        "n_rounds", "round_duration_days",
        "ground_truth_outcome.pro_pct",
        "ground_truth_outcome.source",
        "ground_truth_outcome.type",
    ]
    for f in verified_fields:
        metadata["field_provenance"][f] = "verified_csv"

    # Title: verified if unchanged from CSV, otherwise llm_refined
    if scenario.get("title", "").strip() == event_spec["title"].strip():
        metadata["field_provenance"]["title"] = "verified_csv"
    else:
        metadata["field_provenance"]["title"] = "llm_refined"

    # Polling trajectory: check each round
    polling_provenance = []
    for p in scenario.get("polling_trajectory", []):
        source = p.get("source", "")
        if source.startswith("Estimated"):
            polling_provenance.append("llm_estimated")
        elif "http" in source:
            polling_provenance.append("verified_url")
        else:
            polling_provenance.append("llm_estimated_with_citation")
    metadata["field_provenance"]["polling_trajectory"] = polling_provenance

    # Events: check sources
    event_provenance = []
    for ev in scenario.get("events", []):
        src = ev.get("source")
        if src and src.startswith("http"):
            event_provenance.append("verified_url")
        elif src:
            event_provenance.append("llm_estimated_with_citation")
        else:
            event_provenance.append("llm_estimated")
    metadata["field_provenance"]["events"] = event_provenance

    # Agents and covariates are always LLM-generated
    metadata["field_provenance"]["agents"] = "llm_estimated"
    metadata["field_provenance"]["covariates"] = "llm_estimated"
    metadata["field_provenance"]["notes"] = "llm_generated"

    # Summary stats
    total_polls = len(polling_provenance)
    verified_polls = sum(1 for p in polling_provenance if "verified" in p)
    total_events = len(event_provenance)
    verified_events = sum(1 for e in event_provenance if "verified" in e)

    metadata["verification_summary"] = {
        "polling_verified_pct": round(100 * verified_polls / total_polls, 1) if total_polls else 0,
        "events_verified_pct": round(100 * verified_events / total_events, 1) if total_events else 0,
        "needs_human_review": verified_polls < total_polls * 0.5,
    }

    return metadata


# ── Post-processing & Repair ──────────────────────────────────────

def repair_scenario(scenario: dict, event_spec: dict) -> dict:
    """Fix common LLM output issues to pass schema validation."""
    # Force verified fields from CSV
    scenario["id"] = event_spec["id"]
    scenario["domain"] = event_spec["domain"]
    scenario["country"] = event_spec["country"]
    scenario["date_start"] = event_spec["date_start"]
    scenario["date_end"] = event_spec["date_end"]
    scenario["n_rounds"] = event_spec["n_rounds"]
    scenario["round_duration_days"] = event_spec["round_duration_days"]

    # Force ground truth from CSV
    scenario["ground_truth_outcome"] = {
        "pro_pct": event_spec["outcome_pro_pct"],
        "source": event_spec["outcome_source"],
        "type": event_spec["outcome_type"],
    }

    # Fix polling trajectory
    polls = scenario.get("polling_trajectory", [])
    start = datetime.strptime(event_spec["date_start"], "%Y-%m-%d")
    dur = event_spec["round_duration_days"]
    n_rounds = event_spec["n_rounds"]

    # Ensure correct number of rounds
    if len(polls) < n_rounds:
        for r in range(len(polls) + 1, n_rounds + 1):
            d = start + timedelta(days=(r - 1) * dur)
            polls.append({
                "round": r,
                "date": d.strftime("%Y-%m-%d"),
                "pro_pct": None,
                "against_pct": None,
                "undecided_pct": None,
                "sample_size": None,
                "source": "Estimated: no data available for this round",
                "pollster": None,
            })
    elif len(polls) > n_rounds:
        polls = polls[:n_rounds]

    # Fix each poll entry
    for i, p in enumerate(polls):
        p["round"] = i + 1
        # Ensure date exists and is valid
        if "date" not in p or not p["date"]:
            d = start + timedelta(days=i * dur)
            p["date"] = d.strftime("%Y-%m-%d")
        # Clamp percentages
        for key in ("pro_pct", "against_pct", "undecided_pct"):
            if p.get(key) is not None:
                p[key] = max(0.0, min(100.0, float(p[key])))
        # Ensure required fields
        if "source" not in p or not p["source"]:
            p["source"] = "Estimated: LLM-generated placeholder"
        if "sample_size" not in p:
            p["sample_size"] = None
        if "pollster" not in p:
            p["pollster"] = None

    scenario["polling_trajectory"] = polls

    # Fix events
    for ev in scenario.get("events", []):
        ev["shock_magnitude"] = max(0.0, min(1.0, float(ev.get("shock_magnitude", 0.3))))
        ev["shock_direction"] = max(-1.0, min(1.0, float(ev.get("shock_direction", 0.0))))
        if ev.get("round", 0) < 1:
            ev["round"] = 1
        if ev.get("round", 999) > n_rounds:
            ev["round"] = n_rounds
        if not ev.get("description") or len(ev["description"]) < 10:
            ev["description"] = "Event description placeholder (needs human review)"
        if "source" not in ev:
            ev["source"] = None

    # Fix agents
    for ag in scenario.get("agents", []):
        ag["initial_position"] = max(-1.0, min(1.0, float(ag.get("initial_position", 0.0))))
        ag["influence"] = max(0.0, min(1.0, float(ag.get("influence", 0.3))))
        if ag.get("type") not in ("elite", "institutional", "citizen_cluster"):
            ag["type"] = "citizen_cluster"
        if not ag.get("name") or len(ag["name"]) < 2:
            ag["name"] = "Unknown Agent"
        if "description" not in ag:
            ag["description"] = None

    # Fix covariates
    covs = scenario.get("covariates", {})
    for key in ("initial_polarization", "event_volatility"):
        if key not in covs or covs[key] is None:
            covs[key] = 0.5
        covs[key] = max(0.0, min(1.0, float(covs[key])))
    for key in ("elite_concentration", "institutional_trust", "undecided_share"):
        if key in covs and covs[key] is not None:
            covs[key] = max(0.0, min(1.0, float(covs[key])))
        elif key not in covs:
            covs[key] = None
    scenario["covariates"] = covs

    # Recompute initial_polarization from agents if possible
    positions = [a["initial_position"] for a in scenario.get("agents", []) if a.get("initial_position") is not None]
    if len(positions) >= 2:
        covs["initial_polarization"] = round(stdev(positions), 3)

    # Ensure notes
    if "notes" not in scenario:
        scenario["notes"] = None

    return scenario


# ── Generation Pipeline ───────────────────────────────────────────

async def generate_one_scenario(
    client: GeminiClient,
    event_spec: dict,
    max_retries: int = 2,
) -> dict:
    """Generate a single scenario JSON via LLM, validate, and return result."""
    prompt = build_generation_prompt(event_spec)
    scenario_id = event_spec["id"]

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"[{scenario_id}] Generating (attempt {attempt + 1})...")
            scenario = await client.generate_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=4096,
                component=f"scenario_gen_{scenario_id}",
            )

            # Repair common issues
            scenario = repair_scenario(scenario, event_spec)

            # Validate against schema
            schema = load_schema()
            schema_errors = validate_schema(scenario, schema)
            if schema_errors:
                logger.warning(f"[{scenario_id}] Schema errors (attempt {attempt + 1}): {schema_errors[:3]}")
                if attempt < max_retries:
                    continue
                # On last attempt, return with errors noted
                return {
                    "id": scenario_id,
                    "status": "schema_error",
                    "errors": schema_errors,
                    "scenario": scenario,
                    "metadata": compute_metadata(event_spec, scenario),
                    "quality": {"score": 0, "grade": "F"},
                }

            # Consistency + source checks
            consistency = check_consistency(scenario)
            source_issues = check_sources(scenario)
            quality = compute_quality_score(scenario)

            # Penalize for consistency issues
            if consistency:
                quality["score"] = max(0, quality["score"] - len(consistency) * 5)

            metadata = compute_metadata(event_spec, scenario)

            logger.info(
                f"[{scenario_id}] Generated: quality={quality['score']}/100 "
                f"({quality['grade']}), {len(consistency)} warnings"
            )

            return {
                "id": scenario_id,
                "status": "ok",
                "errors": [],
                "warnings": consistency + source_issues,
                "scenario": scenario,
                "metadata": metadata,
                "quality": quality,
            }

        except Exception as e:
            logger.error(f"[{scenario_id}] Error (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                return {
                    "id": scenario_id,
                    "status": "error",
                    "errors": [str(e)],
                    "scenario": None,
                    "metadata": None,
                    "quality": {"score": 0, "grade": "F"},
                }


async def generate_all(
    csv_path: str,
    output_dir: str,
    max_concurrent: int = 3,
    dry_run: bool = False,
) -> dict:
    """Main pipeline: parse CSV, generate scenarios, save results."""
    events = parse_csv(csv_path)
    if not events:
        logger.error("No valid events in CSV")
        return {"total": 0, "ok": 0, "failed": 0}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"\n{'=' * 60}")
        print(f"  DRY RUN — {len(events)} scenarios to generate")
        print(f"{'=' * 60}")
        for ev in events:
            print(f"  {ev['id']}: {ev['title'][:60]}")
            print(f"    {ev['domain']} | {ev['country']} | {ev['n_rounds']}r × {ev['round_duration_days']}d")
            print(f"    outcome: {ev['outcome_pro_pct']}% ({ev['outcome_type']})")
        return {"total": len(events), "ok": 0, "failed": 0, "dry_run": True}

    # Initialize Gemini client
    client = GeminiClient(budget=10.0)

    # Generate with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def gen_with_limit(event_spec):
        async with semaphore:
            return await generate_one_scenario(client, event_spec)

    tasks = [gen_with_limit(ev) for ev in events]
    results = await asyncio.gather(*tasks)

    # Save results
    manifest = {
        "generated_at": datetime.now(tz=None).astimezone().isoformat(),
        "generator": "scenario_generator_assisted.py",
        "llm_model": "gemini-3.1-flash-lite-preview",
        "total": len(results),
        "ok": 0,
        "failed": 0,
        "scenarios": [],
        "coverage": {},
        "usage": {},
    }

    for result in results:
        entry = {
            "id": result["id"],
            "status": result["status"],
            "quality_score": result["quality"]["score"],
            "quality_grade": result["quality"]["grade"],
        }

        if result["status"] == "ok" and result["scenario"]:
            # Save scenario JSON
            scenario_path = output_path / f"{result['id']}.json"
            with open(scenario_path, "w") as f:
                json.dump(result["scenario"], f, indent=2, ensure_ascii=False)

            # Save metadata sidecar
            meta_path = output_path / f"{result['id']}.meta.json"
            with open(meta_path, "w") as f:
                json.dump(result["metadata"], f, indent=2, ensure_ascii=False)

            entry["file"] = str(scenario_path.name)
            entry["meta_file"] = str(meta_path.name)
            entry["warnings"] = result.get("warnings", [])
            entry["verification"] = result["metadata"]["verification_summary"]
            manifest["ok"] += 1
        else:
            entry["errors"] = result.get("errors", [])
            manifest["failed"] += 1

        manifest["scenarios"].append(entry)

    # Coverage report
    domain_counts = {}
    quality_by_domain = {}
    for result in results:
        if result["status"] == "ok" and result["scenario"]:
            d = result["scenario"]["domain"]
            domain_counts[d] = domain_counts.get(d, 0) + 1
            quality_by_domain.setdefault(d, []).append(result["quality"]["score"])

    manifest["coverage"] = {
        "by_domain": {
            d: {
                "count": domain_counts.get(d, 0),
                "avg_quality": round(
                    sum(quality_by_domain.get(d, [0])) / len(quality_by_domain.get(d, [1])), 1
                ) if d in quality_by_domain else None,
            }
            for d in VALID_DOMAINS
        },
        "domains_covered": len(domain_counts),
        "domains_total": len(VALID_DOMAINS),
        "gap_domains": [d for d in VALID_DOMAINS if d not in domain_counts],
    }

    # Overall stats
    ok_scores = [r["quality"]["score"] for r in results if r["status"] == "ok"]
    manifest["summary"] = {
        "avg_quality": round(sum(ok_scores) / len(ok_scores), 1) if ok_scores else 0,
        "min_quality": min(ok_scores) if ok_scores else 0,
        "max_quality": max(ok_scores) if ok_scores else 0,
        "needs_review": sum(
            1 for r in results
            if r["status"] == "ok" and r.get("metadata", {}).get("verification_summary", {}).get("needs_human_review")
        ),
    }

    # Usage stats from client
    manifest["usage"] = {
        "total_cost": f"${client.stats.total_cost:.4f}",
        "total_tokens_in": client.stats.total_input_tokens,
        "total_tokens_out": client.stats.total_output_tokens,
    }

    # Save manifest
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Print report
    print_report(manifest)

    return manifest


def print_report(manifest: dict):
    """Print a human-readable generation report."""
    print(f"\n{'=' * 70}")
    print(f"  SCENARIO GENERATION REPORT")
    print(f"  Generated: {manifest['generated_at']}")
    print(f"{'=' * 70}")
    print(f"  Total: {manifest['total']} | OK: {manifest['ok']} | Failed: {manifest['failed']}")
    print(f"  Avg Quality: {manifest['summary']['avg_quality']}/100")
    print(f"  Needs Human Review: {manifest['summary']['needs_review']}")
    print(f"  LLM Cost: {manifest['usage'].get('total_cost', 'N/A')}")
    print()

    # Per-scenario results
    print(f"  {'ID':<35} {'Status':<12} {'Quality':<10} {'Review?'}")
    print(f"  {'─' * 35} {'─' * 12} {'─' * 10} {'─' * 7}")
    for s in manifest["scenarios"]:
        status = "\033[92mOK\033[0m" if s["status"] == "ok" else f"\033[91m{s['status']}\033[0m"
        quality = f"{s['quality_score']}/100 ({s['quality_grade']})"
        needs_review = "YES" if s.get("verification", {}).get("needs_human_review") else "no"
        print(f"  {s['id']:<35} {status:<20} {quality:<10} {needs_review}")

    # Coverage
    print(f"\n  DOMAIN COVERAGE ({manifest['coverage']['domains_covered']}/{manifest['coverage']['domains_total']})")
    for d, info in manifest["coverage"]["by_domain"].items():
        if info["count"] > 0:
            print(f"    {d:<20} {info['count']} scenarios, avg quality {info['avg_quality']}")

    gaps = manifest["coverage"]["gap_domains"]
    if gaps:
        print(f"\n  GAPS (no scenarios): {', '.join(gaps)}")
    print(f"\n{'=' * 70}")


# ── CLI ────────────────────────────────────────────────────────────

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="LLM-assisted empirical scenario generator"
    )
    parser.add_argument("--csv", required=True, help="Input CSV file with events")
    parser.add_argument(
        "--output-dir",
        default=str(SCENARIOS_DIR),
        help="Output directory for generated scenarios",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent LLM calls (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse CSV and show what would be generated, without calling LLM",
    )
    args = parser.parse_args()

    asyncio.run(generate_all(
        csv_path=args.csv,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
