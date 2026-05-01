"""Sprint 13 — Chain-of-Thought reasoning audit for marginal cases.

Why this exists
---------------
Layer 1 (relevance_score) gives a numeric score; Layer 2 (realism_gate)
gives accept/reject. Neither tells the *cliente* WHY a stakeholder
was included or rejected — they just see numbers or labels.

For an enterprise demo (Banca Sella), the question "perché Carlo Cottarelli
SÌ ma Mario Draghi NO?" must be answerable in 1 sentence with reasoning.

This module provides a focused LLM pass that:
  1. Runs ONLY on the marginal zone (Layer 1 score 0.30-0.60) — the
     stakeholders where the system is least confident.
  2. Asks the LLM to produce a structured reasoning chain for each:
     [step1: identify brief topics, step2: identify stakeholder expertise,
      step3: assess overlap, step4: verdict + confidence]
  3. Stores the trace in `analysis["_realism_reasoning"]` for the report.

Cost
----
Marginal-zone-only filtering keeps cost low: typical sim has ~5-10
marginal stakeholders, batched into 1-2 LLM calls (~$0.005 total).
Confident accept/reject from Layer 1+2 don't need reasoning trace.

Auditability
------------
The reasoning trace is the killer demo feature: enterprise compliance
reviewers want to see chain-of-thought for ANY automated filtering
decision. The trace is verbatim from the LLM, so it's traceable and
reproducible per (stakeholder, brief_hash) cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.llm.base_client import BaseLLMClient
    from .brief_scope import BriefScope

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────

MARGINAL_LOW = 0.30        # below this: clearly drop, no reasoning needed
MARGINAL_HIGH = 0.60       # above this: clearly keep, no reasoning needed
MAX_AGENTS_PER_CALL = 6    # CoT prompts are token-heavy → smaller batches
DEFAULT_TEMPERATURE = 0.1

# In-process reasoning cache (brief_hash, agent_id) → ReasoningTrace
_REASONING_CACHE: dict[tuple[str, str], "ReasoningTrace"] = {}
_CACHE_MAX = 2000


@dataclass
class ReasoningStep:
    label: str
    content: str

    def to_dict(self) -> dict:
        return {"label": self.label, "content": self.content}


@dataclass
class ReasoningTrace:
    agent_id: str
    name: str
    layer1_score: float
    verdict: str               # "accept" | "reject" | "uncertain"
    confidence: float          # 0..1
    steps: list[ReasoningStep] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "layer1_score": self.layer1_score,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary,
        }


def _brief_hash(brief: str) -> str:
    return hashlib.sha1((brief or "").encode("utf-8")).hexdigest()[:12]


# ── Prompt ─────────────────────────────────────────────────────────────────

REASONING_PROMPT = """You are a STRICT realism auditor. For each candidate
agent below, produce a structured chain-of-thought judging whether they
would credibly comment publicly on this specific brief.

BRIEF (verbatim):
{brief_excerpt}

SCOPE:
- Sector: {sector}
- Geography: {geography}
- Tier: {scope_tier}
- Named entities: {named_entities}

CANDIDATES:
{candidates_block}

For EACH candidate, produce reasoning in this exact JSON shape:
{{
  "verdicts": [
    {{
      "agent_id": "<id>",
      "verdict": "accept" | "reject" | "uncertain",
      "confidence": 0.0-1.0,
      "summary": "ONE sentence (≤25 words) executive verdict",
      "steps": [
        {{"label": "topics_in_brief", "content": "1 sentence: what is this brief actually about?"}},
        {{"label": "agent_expertise", "content": "1 sentence: what does this agent publicly cover?"}},
        {{"label": "overlap_assessment", "content": "1 sentence: where does expertise meet brief, or fail to?"}},
        {{"label": "decision_rationale", "content": "1 sentence: why this verdict, with key constraint named (geography? sector? role mismatch? brief mention?)"}}
      ]
    }}
  ]
}}

DECISION RULES (apply during the reasoning, surface in the steps):
- A foreign head-of-state on a non-foreign-policy brief → REJECT, unless
  the brief verbatim names them.
- An in-country political top figure on a NON-political brief → REJECT,
  unless verbatim named.
- A sector-aligned regulator / industry body / direct competitor in the
  geography → ACCEPT if no other red flag.
- A peripheral / boundary case (same industry, different vertical) →
  UNCERTAIN with low confidence.

Be SPECIFIC about WHICH constraint drove the decision — the report
shows this verbatim to enterprise compliance reviewers.
"""


def _format_candidates(verdicts: list) -> str:
    """Render the candidates for the CoT prompt."""
    rows = []
    for v in verdicts:
        # v is a RelevanceVerdict from Layer 1
        comp_str = ", ".join(
            f"{k}={v.components.get(k)}" for k in
            ("country", "sector", "semantic", "brief_mention",
             "global_figure_pen") if v.components.get(k) is not None
        )
        rows.append(
            f"- id={v.stakeholder_id} | name={v.name!r} | "
            f"layer1_score={v.score} | {comp_str}"
        )
    return "\n".join(rows)


# ── Public API ─────────────────────────────────────────────────────────────

async def reason_about_marginal_stakeholders(
    layer1_verdicts: list,           # list[RelevanceVerdict] from Layer 1
    brief: str,
    scope=None,
    llm: Optional["BaseLLMClient"] = None,
    *,
    marginal_low: float = MARGINAL_LOW,
    marginal_high: float = MARGINAL_HIGH,
) -> dict[str, ReasoningTrace]:
    """Run CoT reasoning on the marginal zone of Layer 1 verdicts.

    Returns dict: agent_id → ReasoningTrace. Stakeholders with score
    outside [marginal_low, marginal_high] are NOT processed (their
    decision is confident already).

    Marginal zone defaults [0.30, 0.60] cover most demo cases. The
    typical Sella sim has ~5-10 marginal candidates → 1-2 LLM calls.
    """
    if llm is None:
        return {}

    brief_h = _brief_hash(brief)
    # Filter to marginal zone
    marginals = [v for v in layer1_verdicts
                 if marginal_low <= v.score <= marginal_high]
    if not marginals:
        return {}

    # Pull cached
    out: dict[str, ReasoningTrace] = {}
    to_query = []
    for v in marginals:
        cached = _REASONING_CACHE.get((brief_h, v.stakeholder_id))
        if cached is not None:
            out[v.stakeholder_id] = cached
        else:
            to_query.append(v)

    if not to_query:
        logger.info(f"reasoning_audit: cache hit for all {len(marginals)} marginal verdicts")
        return out

    logger.info(
        f"reasoning_audit: {len(to_query)} marginal candidates to audit "
        f"({len(out)} cache hits)"
    )

    # Batch process
    brief_excerpt = (brief or "").strip().replace("\n", " ")
    if len(brief_excerpt) > 500:
        brief_excerpt = brief_excerpt[:500] + "…"
    sector = (getattr(scope, "sector", "") or "unknown") if scope else "unknown"
    geography = ", ".join(getattr(scope, "geography", []) or []) if scope else ""
    scope_tier = (getattr(scope, "scope_tier", "") or "") if scope else ""
    named_entities = ", ".join((getattr(scope, "named_entities", []) or [])[:8]) if scope else ""

    batches = [to_query[i:i + MAX_AGENTS_PER_CALL]
               for i in range(0, len(to_query), MAX_AGENTS_PER_CALL)]

    for batch in batches:
        prompt = REASONING_PROMPT.format(
            brief_excerpt=brief_excerpt or "(empty)",
            sector=sector,
            geography=geography or "(unknown)",
            scope_tier=scope_tier or "(unknown)",
            named_entities=named_entities or "(none)",
            candidates_block=_format_candidates(batch),
        )
        try:
            result = await llm.generate_json(
                prompt=prompt,
                temperature=DEFAULT_TEMPERATURE,
                max_output_tokens=2500,
                component="reasoning_audit",
            )
        except Exception as exc:
            logger.warning(f"reasoning_audit LLM call failed: {exc}")
            # Mark batch as uncertain so report shows "could not audit"
            for v in batch:
                trace = ReasoningTrace(
                    agent_id=v.stakeholder_id,
                    name=v.name,
                    layer1_score=v.score,
                    verdict="uncertain",
                    confidence=0.0,
                    steps=[ReasoningStep("error", f"LLM call failed: {exc}")],
                    summary="Audit non disponibile (errore LLM).",
                )
                out[v.stakeholder_id] = trace
                _cache_set(brief_h, v.stakeholder_id, trace)
            continue

        for entry in (result.get("verdicts", []) or []):
            aid = str(entry.get("agent_id", "")).strip()
            if not aid:
                continue
            v_match = next((v for v in batch if v.stakeholder_id == aid), None)
            steps = []
            for s in (entry.get("steps", []) or []):
                steps.append(ReasoningStep(
                    label=str(s.get("label", "")),
                    content=str(s.get("content", "")),
                ))
            trace = ReasoningTrace(
                agent_id=aid,
                name=v_match.name if v_match else "?",
                layer1_score=v_match.score if v_match else 0.0,
                verdict=str(entry.get("verdict", "uncertain")).lower(),
                confidence=float(entry.get("confidence", 0.0) or 0.0),
                steps=steps,
                summary=str(entry.get("summary", "") or ""),
            )
            out[aid] = trace
            _cache_set(brief_h, aid, trace)

    return out


def _cache_set(brief_hash: str, agent_id: str, trace: ReasoningTrace):
    if len(_REASONING_CACHE) >= _CACHE_MAX:
        for k in list(_REASONING_CACHE.keys())[: _CACHE_MAX // 10]:
            _REASONING_CACHE.pop(k, None)
    _REASONING_CACHE[(brief_hash, agent_id)] = trace


def serialise_traces(traces: dict[str, ReasoningTrace]) -> list[dict]:
    """Serialise trace dict to list of dicts for JSON storage / report."""
    return [t.to_dict() for t in traces.values()]
