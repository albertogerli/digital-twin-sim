"""Generate 1000+ historical calibration scenarios using LLM.

Each scenario is a real-world event with known outcome and polling trajectory.
The LLM generates plausible polling trajectories based on the known final outcome.

Usage:
    python -m calibration.generate_scenarios --target 1000 --batch 20
    python -m calibration.generate_scenarios --domain political --target 200
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Domains and their scenario archetypes for generation
DOMAIN_ARCHETYPES = {
    "political": {
        "count": 200,
        "archetypes": [
            "national referendum or plebiscite",
            "presidential or parliamentary election",
            "constitutional amendment or reform vote",
            "independence or separatism movement",
            "policy change (immigration, taxation, defense)",
            "impeachment or censure motion",
            "trade agreement ratification",
            "EU membership or integration vote",
        ],
    },
    "commercial": {
        "count": 150,
        "archetypes": [
            "product launch success or failure",
            "brand crisis and recovery (boycott, scandal)",
            "competitor disruption (new entrant, pivot)",
            "pricing controversy or backlash",
            "acquisition or merger announcement",
            "influencer partnership gone wrong",
            "product recall or safety issue",
        ],
    },
    "corporate": {
        "count": 120,
        "archetypes": [
            "CEO replacement or leadership change",
            "major layoff or restructuring announcement",
            "merger or acquisition integration",
            "whistleblower or fraud revelation",
            "workplace culture controversy",
            "strategic pivot (new market, tech shift)",
            "IPO or going private",
        ],
    },
    "technology": {
        "count": 120,
        "archetypes": [
            "AI regulation debate",
            "privacy controversy (data breach, surveillance)",
            "platform policy change (moderation, algorithm)",
            "open source vs proprietary battle",
            "right to repair or antitrust action",
            "cryptocurrency regulation or crash",
            "social media platform migration",
        ],
    },
    "public_health": {
        "count": 100,
        "archetypes": [
            "vaccine mandate or campaign",
            "pandemic response policy",
            "drug approval controversy",
            "healthcare system reform",
            "food safety scare",
            "mental health policy debate",
            "environmental health crisis (pollution, contamination)",
        ],
    },
    "financial": {
        "count": 100,
        "archetypes": [
            "central bank rate decision",
            "stock market crash or correction",
            "cryptocurrency boom or bust",
            "banking crisis or bailout",
            "inflation policy debate",
            "pension reform controversy",
            "tax reform impact",
        ],
    },
    "marketing": {
        "count": 80,
        "archetypes": [
            "viral campaign success or backlash",
            "rebranding effort (logo, name, identity)",
            "celebrity endorsement controversy",
            "Super Bowl / major event ad reaction",
            "cause marketing authenticity debate",
            "user-generated content campaign",
        ],
    },
    "environmental": {
        "count": 60,
        "archetypes": [
            "climate policy vote or agreement",
            "pipeline or energy project protest",
            "deforestation or conservation battle",
            "plastic ban or regulation",
            "carbon tax implementation",
            "renewable energy transition debate",
        ],
    },
    "labor": {
        "count": 40,
        "archetypes": [
            "major strike or union action",
            "minimum wage debate",
            "gig economy regulation",
            "remote work policy battle",
            "automation and job displacement",
        ],
    },
    "social": {
        "count": 30,
        "archetypes": [
            "civil rights movement or protest",
            "education reform debate",
            "housing affordability crisis",
            "immigration policy controversy",
        ],
    },
}

GENERATION_PROMPT = """You are a historian and data analyst. Generate {batch_size} DISTINCT historical calibration scenarios for the domain "{domain}".

Each scenario must be a REAL event that actually happened (or a highly plausible composite based on real patterns). Include real dates, real organizations, real outcomes.

Archetype to focus on: {archetype}

CRITICAL RULES:
- Each scenario MUST have a clear binary outcome (pro vs against, success vs failure, support vs opposition)
- The polling_trajectory must have exactly 9 data points (round_equivalent 1 through 9)
- pro_pct + against_pct + undecided_pct = 100 for each round
- The trajectory should show realistic evolution (not just linear, include momentum shifts)
- final_outcome_pro_pct and final_outcome_against_pct must sum to 100 (decided voters only)
- key_events should have 3-5 real events at specific rounds
- Each scenario_name must be unique and use snake_case (e.g., "brexit_2016")
- DO NOT repeat scenarios that already exist: {existing_names}
- Write in English

Geographic diversity: include events from different continents/countries.
Time diversity: include events from 2000-2025.

Return a JSON array of {batch_size} scenarios:
[
  {{
    "scenario_name": "unique_snake_case_name",
    "domain": "{domain}",
    "description": "2-3 sentence description with real details, dates, outcomes",
    "final_outcome_pro_pct": 55.0,
    "final_outcome_against_pct": 45.0,
    "final_turnout_pct": 65.0,
    "polling_trajectory": [
      {{"round_equivalent": 1, "pro_pct": 50, "against_pct": 35, "undecided_pct": 15}},
      ...9 total data points...
      {{"round_equivalent": 9, "pro_pct": 54, "against_pct": 43, "undecided_pct": 3}}
    ],
    "key_events": [
      {{"round_equivalent": 2, "description": "Real event description"}},
      ...3-5 events...
    ],
    "calibration_notes": "What makes this scenario useful for calibration"
  }}
]"""


async def generate_batch(
    llm,
    domain: str,
    archetype: str,
    batch_size: int,
    existing_names: set[str],
) -> list[dict]:
    """Generate a batch of scenarios using LLM."""
    prompt = GENERATION_PROMPT.format(
        batch_size=batch_size,
        domain=domain,
        archetype=archetype,
        existing_names=", ".join(sorted(existing_names)[:50]) or "none",
    )

    try:
        result = await llm.generate_json(
            prompt=prompt,
            temperature=0.9,
            max_output_tokens=8000,
            component="scenario_generation",
        )

        if isinstance(result, dict) and "scenarios" in result:
            result = result["scenarios"]
        if not isinstance(result, list):
            result = [result]

        validated = []
        for s in result:
            if _validate_scenario(s, existing_names):
                s["domain"] = domain
                validated.append(s)
                existing_names.add(s["scenario_name"])
            else:
                logger.warning(f"Invalid scenario skipped: {s.get('scenario_name', '?')}")

        return validated
    except Exception as e:
        logger.error(f"Batch generation failed for {domain}/{archetype}: {e}")
        return []


def _validate_scenario(s: dict, existing_names: set[str]) -> bool:
    """Validate a generated scenario meets all requirements."""
    name = s.get("scenario_name", "")
    if not name or name in existing_names:
        return False

    # Check required fields
    for field in ["description", "final_outcome_pro_pct", "final_outcome_against_pct", "polling_trajectory"]:
        if field not in s:
            return False

    # Check trajectory has 9 points
    traj = s.get("polling_trajectory", [])
    if len(traj) != 9:
        return False

    # Check each polling point sums correctly
    for p in traj:
        total = p.get("pro_pct", 0) + p.get("against_pct", 0) + p.get("undecided_pct", 0)
        if abs(total - 100) > 2:
            return False
        if p.get("round_equivalent", 0) < 1 or p.get("round_equivalent", 0) > 9:
            return False

    # Check final outcome sums to ~100
    final_sum = s.get("final_outcome_pro_pct", 0) + s.get("final_outcome_against_pct", 0)
    if abs(final_sum - 100) > 2:
        return False

    # Check key_events exist
    if len(s.get("key_events", [])) < 2:
        return False

    return True


async def generate_all_scenarios(
    target: int = 1000,
    batch_size: int = 10,
    domain_filter: str = None,
    provider: str = "gemini",
    model: str = None,
):
    """Generate scenarios across all domains up to target count."""
    # Setup LLM
    if provider == "gemini":
        from core.llm.gemini_client import GeminiClient
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        llm = GeminiClient(api_key=api_key, model=model or "gemini-2.5-flash-preview-05-20")
    else:
        from core.llm.openai_client import OpenAIClient
        api_key = os.environ.get("OPENAI_API_KEY", "")
        llm = OpenAIClient(api_key=api_key, model=model or "gpt-4o-mini")

    scenarios_dir = Path(__file__).parent / "scenarios"
    scenarios_dir.mkdir(exist_ok=True)

    # Load existing scenarios
    existing_names = set()
    for f in scenarios_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            existing_names.add(data.get("scenario_name", f.stem))
        except (json.JSONDecodeError, KeyError):
            pass

    print(f"Existing scenarios: {len(existing_names)}")
    print(f"Target: {target}")

    domains = DOMAIN_ARCHETYPES
    if domain_filter:
        domains = {k: v for k, v in domains.items() if k == domain_filter}

    # Calculate how many to generate per domain
    remaining = target - len(existing_names)
    if remaining <= 0:
        print(f"Already at target ({len(existing_names)} >= {target})")
        return

    total_weight = sum(d["count"] for d in domains.values())
    generated_total = 0

    for domain, config in domains.items():
        domain_target = int(remaining * config["count"] / total_weight)
        if domain_target <= 0:
            continue

        archetypes = config["archetypes"]
        per_archetype = max(1, domain_target // len(archetypes))

        print(f"\n--- {domain.upper()} (target: {domain_target}) ---")

        for archetype in archetypes:
            batches_needed = (per_archetype + batch_size - 1) // batch_size
            for batch_i in range(batches_needed):
                this_batch = min(batch_size, per_archetype - batch_i * batch_size)
                if this_batch <= 0:
                    break

                print(f"  Generating {this_batch} scenarios: {archetype}...", end=" ", flush=True)
                scenarios = await generate_batch(llm, domain, archetype, this_batch, existing_names)

                for s in scenarios:
                    filepath = scenarios_dir / f"{s['scenario_name']}.json"
                    filepath.write_text(json.dumps(s, indent=2, ensure_ascii=False))

                generated_total += len(scenarios)
                print(f"✓ {len(scenarios)} saved (total: {len(existing_names)})")

                if len(existing_names) >= target:
                    break

                # Rate limiting
                await asyncio.sleep(0.5)

            if len(existing_names) >= target:
                break

        if len(existing_names) >= target:
            break

    print(f"\n=== Done! Generated {generated_total} new scenarios. Total: {len(existing_names)} ===")
    print(f"Cost: ${llm.stats.total_cost:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Generate historical calibration scenarios")
    parser.add_argument("--target", type=int, default=1000, help="Target number of scenarios")
    parser.add_argument("--batch", type=int, default=10, help="Scenarios per LLM call")
    parser.add_argument("--domain", type=str, default=None, help="Filter to single domain")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "openai"])
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(generate_all_scenarios(
        target=args.target,
        batch_size=args.batch,
        domain_filter=args.domain,
        provider=args.provider,
        model=args.model,
    ))


if __name__ == "__main__":
    # Ensure project root on path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
