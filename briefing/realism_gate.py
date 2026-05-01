"""Layer 0 — post-generation realism gate.

Given a `BriefScope` and a generated agent roster, scores every elite /
institutional agent on *scope plausibility*: "would this person / organization
credibly comment on this brief at this scope?". Phase 1 is OBSERVABILITY ONLY:
it logs a `RealismReport` and does not mutate the roster.

Later phases (Phase 4) can switch this to enforcement by re-generating rejected
agents when the rejection rate exceeds a threshold.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from core.llm.base_client import BaseLLMClient
from .brief_scope import BriefScope

logger = logging.getLogger(__name__)


Verdict = Literal["accept", "reject", "uncertain"]


@dataclass
class AgentVerdict:
    agent_id: str
    name: str
    archetype: str
    tier: Literal["elite", "institutional"]
    verdict: Verdict
    rationale: str


@dataclass
class RealismReport:
    scope: BriefScope
    verdicts: list[AgentVerdict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def rejected(self) -> list[AgentVerdict]:
        return [v for v in self.verdicts if v.verdict == "reject"]

    @property
    def accepted(self) -> list[AgentVerdict]:
        return [v for v in self.verdicts if v.verdict == "accept"]

    @property
    def uncertain(self) -> list[AgentVerdict]:
        return [v for v in self.verdicts if v.verdict == "uncertain"]

    @property
    def rejection_rate(self) -> float:
        return len(self.rejected) / self.total if self.total else 0.0

    @property
    def realism_score(self) -> float:
        """0.0-1.0. Uncertain counts as half-credit."""
        if not self.total:
            return 0.0
        return (len(self.accepted) + 0.5 * len(self.uncertain)) / self.total

    def to_dict(self) -> dict:
        return {
            "scope": self.scope.to_dict(),
            "realism_score": round(self.realism_score, 3),
            "rejection_rate": round(self.rejection_rate, 3),
            "total": self.total,
            "accepted": len(self.accepted),
            "uncertain": len(self.uncertain),
            "rejected": len(self.rejected),
            "rejected_agents": [
                {"id": v.agent_id, "name": v.name, "archetype": v.archetype,
                 "tier": v.tier, "reason": v.rationale}
                for v in self.rejected
            ],
            "uncertain_agents": [
                {"id": v.agent_id, "name": v.name, "archetype": v.archetype,
                 "tier": v.tier, "reason": v.rationale}
                for v in self.uncertain
            ],
        }

    def summary(self) -> str:
        return (
            f"realism={self.realism_score:.2f} "
            f"({len(self.accepted)}✓ {len(self.uncertain)}? {len(self.rejected)}✗ "
            f"of {self.total}) — scope={self.scope.summary()}"
        )


REALISM_CHECK_PROMPT = """You are a realism auditor for digital-twin simulation agent rosters.
For each listed agent, decide whether they would CREDIBLY COMMENT PUBLICLY on this
scenario at the given scope. The test is scope plausibility, not archetype taxonomy.

SCENARIO SCOPE:
- Sector: {sector}{sub_sector_line}
- Geography: {geography}
- Scope tier: {scope_tier}
- Brief (verbatim): {brief_excerpt}
- Archetype whitelist (ADVISORY — typical commenters; not exhaustive): {allowed_archetypes}
- Archetype denylist (HARD — these are out-of-scope): {excluded_archetypes}

AGENTS TO AUDIT (name, archetype, role, tier):
{agent_list}

DECISION RULES (apply in order, first match wins):

1. REJECT if any of the following is true:
   - Agent's archetype is on the DENYLIST (hard block).
   - Agent's known public role is clearly OUTSIDE the sector (e.g. pharma CEO on a fashion brief)
     AND they are not explicitly named in the brief.
   - Agent's geography is clearly OUTSIDE the scope (e.g. US-only local politician on an Italy-only brief)
     AND they don't operate internationally in this sector.
   - The scope tier is national/niche and the agent only makes sense at a global tier
     (heads of state, central bank governors, global tech billionaires, world religious leaders),
     UNLESS the brief explicitly invokes them.
   - The name is generic, invented-sounding, or has no obvious public record.

2. ACCEPT if:
   - Agent's public role is plausibly IN the sector AND IN the geography/tier, regardless of
     whether their archetype string is literally on the whitelist.
   - Industry associations, academic institutions, trade bodies, regulators, and media outlets
     that cover the sector count as in-scope even if their archetype label isn't in the whitelist,
     SO LONG AS their sector/geography match.
   - Agent is explicitly named in the brief.

3. UNCERTAIN if:
   - Boundary case: same industry but different vertical, or same country but peripheral role.
   - Insufficient public record to be confident either way.

The archetype whitelist is a hint about typical commenters, NOT an exhaustive filter.
Do not reject on archetype mismatch alone when sector + geography clearly fit.

Respond with JSON:
{{
  "verdicts": [
    {{
      "agent_id": "...",
      "verdict": "accept|reject|uncertain",
      "rationale": "one short sentence — cite WHICH scope dimension matched or failed. Do NOT cite archetype-whitelist mismatch as the sole reason."
    }}
  ]
}}
"""


def _format_agent_list(agents: list[tuple[str, str, str, str, str]]) -> str:
    """agents: list of (id, name, archetype, role, tier)"""
    if not agents:
        return "(none)"
    return "\n".join(
        f"- id={a[0]} | name={a[1]} | archetype={a[2]} | role={a[3]} | tier={a[4]}"
        for a in agents
    )


async def run_realism_gate(
    scope: BriefScope,
    brief_text: str,
    analysis: dict,
    llm: BaseLLMClient,
) -> RealismReport:
    """Audit elite + institutional agents in the LLM analysis dict against scope.

    `analysis` is the dict produced by `generate_agents_multistep`, containing
    "suggested_elite_agents" and "suggested_institutional_agents" keys.
    """

    elites = analysis.get("suggested_elite_agents", []) or []
    insts = analysis.get("suggested_institutional_agents", []) or []

    rows: list[tuple[str, str, str, str, str]] = []
    for a in elites:
        rows.append((
            str(a.get("id", "")),
            str(a.get("name", "")),
            str(a.get("archetype", "")),
            str(a.get("role", "")),
            "elite",
        ))
    for a in insts:
        rows.append((
            str(a.get("id", "")),
            str(a.get("name", "")),
            str(a.get("category", a.get("archetype", ""))),
            str(a.get("role", "")),
            "institutional",
        ))

    report = RealismReport(scope=scope)
    if not rows:
        return report

    # Chunk into batches of 12 agents per LLM call to keep prompt small.
    BATCH = 12
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]

    # We run batches sequentially to keep LLM budget predictable (gate is
    # cheap; each batch is small). If latency becomes a problem, switch to
    # asyncio.gather with a bounded semaphore.
    brief_excerpt = (brief_text or "").strip().replace("\n", " ")
    if len(brief_excerpt) > 600:
        brief_excerpt = brief_excerpt[:600] + "…"

    prompt_base = REALISM_CHECK_PROMPT
    sub_sector_line = f" / {scope.sub_sector}" if scope.sub_sector else ""

    verdict_by_id: dict[str, tuple[Verdict, str]] = {}

    for batch in batches:
        prompt = prompt_base.format(
            sector=scope.sector,
            sub_sector_line=sub_sector_line,
            geography=", ".join(scope.geography),
            scope_tier=scope.scope_tier,
            brief_excerpt=brief_excerpt or "(empty)",
            allowed_archetypes=", ".join(scope.stakeholder_archetypes) or "(none specified)",
            excluded_archetypes=", ".join(scope.excluded_archetypes) or "(none specified)",
            agent_list=_format_agent_list(batch),
        )

        try:
            result = await llm.generate_json(
                prompt=prompt,
                temperature=0.1,
                max_output_tokens=2000,
                component="realism_gate",
            )
        except Exception as exc:
            logger.warning(f"realism_gate LLM call failed for batch of {len(batch)}: {exc}")
            # Mark this whole batch as uncertain so we don't silently accept.
            for row in batch:
                verdict_by_id[row[0]] = ("uncertain", f"gate LLM error: {exc}")
            continue

        for v in result.get("verdicts", []) or []:
            aid = str(v.get("agent_id", "")).strip()
            verdict = _coerce_verdict(v.get("verdict"))
            rationale = str(v.get("rationale", "") or "").strip()
            if aid:
                verdict_by_id[aid] = (verdict, rationale)

    # Assemble report in roster order
    for row in rows:
        aid, name, archetype, _role, tier = row
        verdict, rationale = verdict_by_id.get(aid, ("uncertain", "no gate verdict returned"))
        report.verdicts.append(AgentVerdict(
            agent_id=aid,
            name=name,
            archetype=archetype,
            tier="elite" if tier == "elite" else "institutional",  # type: ignore
            verdict=verdict,
            rationale=rationale,
        ))

    return report


def _coerce_verdict(value) -> Verdict:
    v = str(value or "").strip().lower()
    if v in ("accept", "reject", "uncertain"):
        return v  # type: ignore
    return "uncertain"


# Action / meta keywords the LLM sometimes emits as if they were agent names.
# These slip past the realism audit because they look like valid JSON objects
# (with role / archetype / position fields) but the "name" is actually an
# operation token. Keep this list conservative — real names are very unlikely
# to collide with any of these.
_INVALID_NAME_TOKENS = {
    "remove_agent", "remove", "delete_agent", "delete", "drop", "skip",
    "replace", "noop", "none", "null", "n/a", "tbd", "tba", "placeholder",
    "agent", "agent_1", "agent_2", "unknown", "unnamed", "action",
}


def _is_invalid_agent_name(name: str) -> bool:
    """True if `name` is clearly not a real person's name (action keyword,
    snake_case meta token, empty, etc.)."""
    if not name:
        return True
    s = name.strip().lower()
    if s in _INVALID_NAME_TOKENS:
        return True
    # snake_case-only tokens with no spaces are almost never real names.
    # Exception: single mononyms like "Madonna" exist, so require >=2 chars
    # AND no underscore for a single-word name to pass.
    if "_" in s and " " not in s:
        return True
    return False


def filter_invalid_agents(agents: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop agents whose `name` is an action keyword or otherwise non-name.
    Returns (cleaned_agents, dropped_names). Logs each rejection."""
    cleaned, dropped = [], []
    for a in agents:
        name = str(a.get("name", "") or "").strip()
        if _is_invalid_agent_name(name):
            dropped.append(name or "<empty>")
            logger.warning(
                f"filter_invalid_agents: dropped {name!r} "
                f"(role={a.get('role', '?')!r}, id={a.get('id', '?')!r})"
            )
            continue
        cleaned.append(a)
    return cleaned, dropped


# ── Phase 3: enforcement + regeneration ────────────────────────────────────


REGEN_ELITE_PROMPT = """You previously generated elite agents for a digital-twin simulation.
A realism audit REJECTED some of them as out of scope. Replace the rejected agents with
new, in-scope elite figures. Do NOT repeat any of the kept agents.

{scope_section}

KEPT ELITE AGENTS (do NOT duplicate these; leave them alone):
{kept_summary}

REJECTED ELITE AGENTS (regenerate replacements — address the stated reasons):
{rejected_summary}

Produce exactly {n_needed} NEW elite agents, strictly inside the scope above.
- Use REAL named public figures only (no invented names).
- Cover the same rough position spread as what was rejected.
- Avoid any archetype on the denylist. Prefer archetypes on the whitelist.
- If you cannot think of enough real in-scope figures, return fewer rather than hallucinating.

Respond with JSON:
{{
  "elite_agents": [
    {{
      "id": "unique_snake_case_id",
      "name": "Full Real Name",
      "role": "their role/title",
      "archetype": "from whitelist",
      "position": 0.0,
      "influence": 0.5,
      "rigidity": 0.5,
      "bio": "1-2 sentence bio grounded in public record",
      "communication_style": "how they communicate",
      "key_traits": ["trait1", "trait2"]
    }}
  ]
}}"""


REGEN_INSTITUTIONAL_PROMPT = """You previously generated institutional agents for a digital-twin
simulation. A realism audit REJECTED some as out of scope. Replace them with in-scope orgs.

{scope_section}

KEPT INSTITUTIONAL AGENTS (do NOT duplicate):
{kept_summary}

REJECTED INSTITUTIONAL AGENTS (regenerate replacements — address the stated reasons):
{rejected_summary}

Produce exactly {n_needed} NEW institutional agents inside scope. Real org names only.

Respond with JSON:
{{
  "institutional_agents": [
    {{
      "id": "unique_snake_case_id",
      "name": "Real Organization Name",
      "role": "institutional role",
      "category": "government|media|business|union|ngo|academic|industry_body|regulator",
      "position": 0.0,
      "influence": 0.3,
      "rigidity": 0.5,
      "key_trait": "defining characteristic"
    }}
  ]
}}"""


def _fmt_kept(agents: list[dict], kind: str) -> str:
    if not agents:
        return "(none — first regeneration pass)"
    lines = []
    for a in agents:
        if kind == "elite":
            lines.append(
                f"- {a.get('name', '?')} ({a.get('role', '')}) "
                f"archetype={a.get('archetype', '?')} position={a.get('position', 0):+.2f}"
            )
        else:
            lines.append(
                f"- {a.get('name', '?')} ({a.get('role', '')}) "
                f"category={a.get('category', a.get('archetype', '?'))} position={a.get('position', 0):+.2f}"
            )
    return "\n".join(lines)


def _fmt_rejected(verdicts: list[AgentVerdict]) -> str:
    if not verdicts:
        return "(none)"
    return "\n".join(
        f"- {v.name} (archetype={v.archetype}) — REJECTED because: {v.rationale}"
        for v in verdicts
    )


async def enforce_realism(
    scope: BriefScope,
    brief_text: str,
    analysis: dict,
    llm: BaseLLMClient,
    *,
    rejection_threshold: float = 0.15,
    max_passes: int = 2,
) -> tuple[RealismReport, Optional[RealismReport]]:
    """Audit the roster; if rejection rate exceeds threshold, regenerate
    rejected agents and re-audit. Mutates `analysis` in place by replacing
    rejected entries in `suggested_elite_agents` / `suggested_institutional_agents`.

    Returns (final_report, initial_report_if_regen_happened).

    Designed to be cost-bounded: at most `max_passes` regeneration cycles,
    each touching only the rejected subset (not the whole roster).
    """
    initial_report = await run_realism_gate(scope, brief_text, analysis, llm)

    if initial_report.rejection_rate <= rejection_threshold or initial_report.total == 0:
        logger.info(
            f"enforce_realism: rejection_rate={initial_report.rejection_rate:.2f} ≤ "
            f"threshold={rejection_threshold}, no regeneration needed"
        )
        return initial_report, None

    current_report = initial_report
    for pass_num in range(1, max_passes + 1):
        rejected_elites = [v for v in current_report.rejected if v.tier == "elite"]
        rejected_insts = [v for v in current_report.rejected if v.tier == "institutional"]

        if not rejected_elites and not rejected_insts:
            break

        await _regenerate_rejected(
            scope, analysis, llm, rejected_elites, rejected_insts
        )

        # Re-audit to see if we converged
        current_report = await run_realism_gate(scope, brief_text, analysis, llm)
        logger.info(
            f"enforce_realism pass {pass_num}: rejection_rate={current_report.rejection_rate:.2f}"
        )
        if current_report.rejection_rate <= rejection_threshold:
            break

    return current_report, initial_report


async def _regenerate_rejected(
    scope: BriefScope,
    analysis: dict,
    llm: BaseLLMClient,
    rejected_elites: list[AgentVerdict],
    rejected_insts: list[AgentVerdict],
) -> None:
    """Mutate analysis: drop rejected agents, ask LLM for replacements, append."""
    scope_section = scope.prompt_block()

    # ── Elites ──
    if rejected_elites:
        rejected_ids = {v.agent_id for v in rejected_elites}
        all_elites = analysis.get("suggested_elite_agents", []) or []
        kept = [a for a in all_elites if str(a.get("id", "")) not in rejected_ids]
        n_needed = len(rejected_elites)

        prompt = REGEN_ELITE_PROMPT.format(
            scope_section=scope_section,
            kept_summary=_fmt_kept(kept, "elite"),
            rejected_summary=_fmt_rejected(rejected_elites),
            n_needed=n_needed,
        )
        try:
            result = await llm.generate_json(
                prompt=prompt,
                temperature=0.5,
                max_output_tokens=4000,
                component="realism_regen_elite",
            )
            if isinstance(result, list):
                result = result[0] if result and isinstance(result[0], dict) else {}
            new_agents = result.get("elite_agents", []) or []
            # Filter out LLM-emitted action tokens masquerading as agent names
            # ("remove_agent", "delete_agent", placeholders, snake_case ids).
            new_agents, dropped_names = filter_invalid_agents(new_agents)
            if dropped_names:
                logger.warning(
                    f"regen elites: filtered {len(dropped_names)} invalid names "
                    f"({dropped_names})"
                )
            # Dedupe IDs against kept
            kept_ids = {str(a.get("id", "")) for a in kept}
            deduped = [a for a in new_agents if str(a.get("id", "")) not in kept_ids]
            analysis["suggested_elite_agents"] = kept + deduped
            logger.info(
                f"regen elites: dropped {len(rejected_elites)}, added {len(deduped)}"
            )
        except Exception as exc:
            logger.warning(f"elite regen failed: {exc}")
            # Fall back: keep roster as-is (rejected still present). Next pass won't help.

    # ── Institutional ──
    if rejected_insts:
        rejected_ids = {v.agent_id for v in rejected_insts}
        all_insts = analysis.get("suggested_institutional_agents", []) or []
        kept = [a for a in all_insts if str(a.get("id", "")) not in rejected_ids]
        n_needed = len(rejected_insts)

        prompt = REGEN_INSTITUTIONAL_PROMPT.format(
            scope_section=scope_section,
            kept_summary=_fmt_kept(kept, "institutional"),
            rejected_summary=_fmt_rejected(rejected_insts),
            n_needed=n_needed,
        )
        try:
            result = await llm.generate_json(
                prompt=prompt,
                temperature=0.5,
                max_output_tokens=3000,
                component="realism_regen_inst",
            )
            if isinstance(result, list):
                result = result[0] if result and isinstance(result[0], dict) else {}
            new_agents = result.get("institutional_agents", []) or []
            new_agents, dropped_names = filter_invalid_agents(new_agents)
            if dropped_names:
                logger.warning(
                    f"regen insts: filtered {len(dropped_names)} invalid names "
                    f"({dropped_names})"
                )
            kept_ids = {str(a.get("id", "")) for a in kept}
            deduped = [a for a in new_agents if str(a.get("id", "")) not in kept_ids]
            analysis["suggested_institutional_agents"] = kept + deduped
            logger.info(
                f"regen insts: dropped {len(rejected_insts)}, added {len(deduped)}"
            )
        except Exception as exc:
            logger.warning(f"institutional regen failed: {exc}")
