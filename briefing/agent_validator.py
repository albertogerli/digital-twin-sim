"""Agent validation and balancing — programmatic checks + optional LLM critic.

Phase 4: Validates generated agents for:
1. Position distribution — at least 1 agent per quadrant
2. Archetype coverage vs domain required_archetypes
3. Influence variance — flag if std < 0.1
4. Name deduplication
5. Position-role coherence heuristics
6. Optional LLM critic for qualitative review
"""

import logging
import math
from typing import Optional

from core.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class ValidationResult:
    """Aggregates validation issues and auto-fixes."""

    def __init__(self):
        self.issues: list[dict] = []  # {"severity": "warning"|"error", "check": str, "message": str}
        self.fixes_applied: list[str] = []

    @property
    def has_errors(self) -> bool:
        return any(i["severity"] == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i["severity"] == "warning" for i in self.issues)

    def add(self, severity: str, check: str, message: str):
        self.issues.append({"severity": severity, "check": check, "message": message})
        logger.info(f"Validation [{severity}] {check}: {message}")

    def summary(self) -> str:
        if not self.issues:
            return "All checks passed."
        lines = []
        for i in self.issues:
            icon = "!!" if i["severity"] == "error" else "  "
            lines.append(f"  {icon} [{i['check']}] {i['message']}")
        if self.fixes_applied:
            lines.append("  Auto-fixes applied:")
            for f in self.fixes_applied:
                lines.append(f"    + {f}")
        return "\n".join(lines)


def validate_agents(
    analysis: dict,
    domain_guidance: dict = None,
) -> tuple[dict, ValidationResult]:
    """Run programmatic validation on the generated agent roster.

    Args:
        analysis: The merged output from agent_generator (scaffold + agents + clusters)
        domain_guidance: Output of domain_plugin.get_agent_generation_guidance()

    Returns:
        (possibly-fixed analysis dict, ValidationResult)
    """
    result = ValidationResult()
    domain_guidance = domain_guidance or {}

    elite = analysis.get("suggested_elite_agents", [])
    institutional = analysis.get("suggested_institutional_agents", [])
    clusters = analysis.get("suggested_citizen_clusters", [])

    # ── Check 1: Position distribution ─────────────────────────────────────
    all_positions = (
        [a.get("position", 0) for a in elite]
        + [a.get("position", 0) for a in institutional]
    )
    if all_positions:
        quadrants = {
            "strong_neg (< -0.5)": [p for p in all_positions if p < -0.5],
            "mod_neg (-0.5 to 0)": [p for p in all_positions if -0.5 <= p < 0],
            "mod_pos (0 to 0.5)": [p for p in all_positions if 0 <= p < 0.5],
            "strong_pos (>= 0.5)": [p for p in all_positions if p >= 0.5],
        }
        empty_quadrants = [k for k, v in quadrants.items() if not v]
        if empty_quadrants:
            result.add("warning", "position_distribution",
                        f"Empty quadrants: {', '.join(empty_quadrants)}")

        # Check if all positions are clustered
        avg = sum(all_positions) / len(all_positions)
        std = math.sqrt(sum((p - avg) ** 2 for p in all_positions) / len(all_positions))
        if std < 0.2:
            result.add("warning", "position_spread",
                        f"Position std={std:.3f} is very low — agents are too clustered around {avg:+.2f}")

    # ── Check 2: Archetype coverage ────────────────────────────────────────
    required = set(domain_guidance.get("required_archetypes", []))
    if required:
        present = set(a.get("archetype", "") for a in elite)
        missing = required - present
        if missing:
            result.add("warning", "archetype_coverage",
                        f"Missing required archetypes: {', '.join(missing)}")

    # ── Check 3: Influence variance ────────────────────────────────────────
    influences = [a.get("influence", 0.5) for a in elite]
    if len(influences) > 2:
        inf_avg = sum(influences) / len(influences)
        inf_std = math.sqrt(sum((i - inf_avg) ** 2 for i in influences) / len(influences))
        if inf_std < 0.1:
            result.add("warning", "influence_variance",
                        f"Influence std={inf_std:.3f} — agents have very similar influence levels")

    # ── Check 4: Name deduplication ────────────────────────────────────────
    all_names = [a.get("name", "") for a in elite + institutional]
    seen = {}
    for i, name in enumerate(all_names):
        name_lower = name.lower().strip()
        if name_lower in seen:
            result.add("error", "duplicate_name",
                        f"Duplicate name: '{name}' (indices {seen[name_lower]} and {i})")
            # Auto-fix: append suffix to the later one
            if i < len(elite):
                elite[i]["name"] = f"{name} (2)"
                elite[i]["id"] = f"{elite[i].get('id', '')}__2"
            else:
                idx = i - len(elite)
                institutional[idx]["name"] = f"{name} (2)"
                institutional[idx]["id"] = f"{institutional[idx].get('id', '')}__2"
            result.fixes_applied.append(f"Renamed duplicate '{name}' with suffix")
        else:
            seen[name_lower] = i

    # ── Check 5: Position-role coherence heuristics ────────────────────────
    for a in elite:
        archetype = a.get("archetype", "")
        pos = a.get("position", 0)
        # Activists rarely neutral
        if archetype == "activist" and abs(pos) < 0.15:
            result.add("warning", "coherence",
                        f"Activist '{a.get('name', '')}' has near-neutral position ({pos:+.2f})")
        # Regulators/judges typically moderate
        if archetype in ("judge", "magistrate", "regulator") and abs(pos) > 0.85:
            result.add("warning", "coherence",
                        f"{archetype.title()} '{a.get('name', '')}' has extreme position ({pos:+.2f})")

    # ── Check 6: Minimum counts ────────────────────────────────────────────
    elite_min = domain_guidance.get("elite_count_range", (8, 14))[0]
    inst_min = domain_guidance.get("institutional_count_range", (6, 10))[0]
    cluster_min = domain_guidance.get("cluster_count_range", (5, 8))[0]

    if len(elite) < elite_min:
        result.add("warning", "count", f"Only {len(elite)} elite agents (min {elite_min})")
    if len(institutional) < inst_min:
        result.add("warning", "count", f"Only {len(institutional)} institutional agents (min {inst_min})")
    if len(clusters) < cluster_min:
        result.add("warning", "count", f"Only {len(clusters)} citizen clusters (min {cluster_min})")

    # Write back possibly fixed data
    analysis["suggested_elite_agents"] = elite
    analysis["suggested_institutional_agents"] = institutional
    analysis["suggested_citizen_clusters"] = clusters

    return analysis, result


LLM_CRITIC_PROMPT = """You are a simulation quality reviewer. Evaluate this agent roster for a digital-twin simulation
and suggest at most 3 concrete fixes.

SCENARIO: {scenario_name}
AXIS: [{neg_label}] <--> [{pos_label}]

PROGRAMMATIC ISSUES FOUND:
{validation_issues}

ELITE AGENTS:
{elite_summary}

INSTITUTIONAL AGENTS:
{inst_summary}

CITIZEN CLUSTERS:
{cluster_summary}

Respond with JSON:
{{
  "quality_score": 7,
  "issues": [
    {{
      "agent_id": "id of problematic agent or null",
      "problem": "description of the issue",
      "suggested_fix": {{
        "field": "position|influence|rigidity|archetype|name",
        "old_value": "current value",
        "new_value": "suggested value"
      }}
    }}
  ],
  "overall_assessment": "1-2 sentence assessment"
}}

Only suggest fixes that meaningfully improve realism. Max 3 fixes."""


async def critic_review(
    analysis: dict,
    validation_result: ValidationResult,
    llm: BaseLLMClient,
) -> dict:
    """Optional LLM critic call to review and suggest fixes for the agent roster.

    Returns the analysis dict with critic fixes applied (if any).
    """
    try:
        axis = analysis.get("position_axis", {})

        elite_summary = "\n".join(
            f"- [{a.get('id')}] {a.get('name')} ({a.get('archetype')}) pos={a.get('position', 0):+.2f} inf={a.get('influence', 0.5):.1f}"
            for a in analysis.get("suggested_elite_agents", [])
        )
        inst_summary = "\n".join(
            f"- [{a.get('id')}] {a.get('name')} ({a.get('category')}) pos={a.get('position', 0):+.2f}"
            for a in analysis.get("suggested_institutional_agents", [])
        )
        cluster_summary = "\n".join(
            f"- [{c.get('id')}] {c.get('name')} size={c.get('size', 0)} pos={c.get('position', 0):+.2f}"
            for c in analysis.get("suggested_citizen_clusters", [])
        )

        review = await llm.generate_json(
            prompt=LLM_CRITIC_PROMPT.format(
                scenario_name=analysis.get("scenario_name", ""),
                neg_label=axis.get("negative_label", "Against"),
                pos_label=axis.get("positive_label", "In favor"),
                validation_issues=validation_result.summary(),
                elite_summary=elite_summary,
                inst_summary=inst_summary,
                cluster_summary=cluster_summary,
            ),
            temperature=0.3,
            max_output_tokens=1000,
            component="agent_critic",
        )

        quality_score = review.get("quality_score", 7)
        issues = review.get("issues", [])
        logger.info(f"Critic review: score={quality_score}, fixes={len(issues)}")

        # Apply fixes
        if issues:
            agent_index = {}
            for a in analysis.get("suggested_elite_agents", []):
                agent_index[a.get("id", "")] = a
            for a in analysis.get("suggested_institutional_agents", []):
                agent_index[a.get("id", "")] = a
            for c in analysis.get("suggested_citizen_clusters", []):
                agent_index[c.get("id", "")] = c

            for issue in issues[:3]:
                agent_id = issue.get("agent_id")
                fix = issue.get("suggested_fix", {})
                if not agent_id or not fix or agent_id not in agent_index:
                    continue

                field = fix.get("field", "")
                new_value = fix.get("new_value")
                if field and new_value is not None:
                    agent = agent_index[agent_id]
                    # Type coerce for numeric fields
                    if field in ("position", "influence", "rigidity"):
                        try:
                            new_value = float(new_value)
                            new_value = max(-1.0, min(1.0, new_value))
                        except (ValueError, TypeError):
                            continue
                    old = agent.get(field)
                    agent[field] = new_value
                    logger.info(f"Critic fix: {agent_id}.{field}: {old} -> {new_value}")

        return analysis

    except Exception as e:
        logger.warning(f"Critic review failed (non-fatal): {e}")
        return analysis
