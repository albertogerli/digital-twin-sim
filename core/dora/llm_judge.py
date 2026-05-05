"""DORA economic-impact — Sprint C (LLM judge layer).

STATUS: SCAFFOLDING ONLY. Returns None today; the implementation
will land in Sprint C and use gemini-3.1-pro-preview (per user
2026-05-05: pro variant authorised for analytical-reasoning use
case, while gemini-3.1-flash-lite-preview remains default for
narrative agents).

Concept: a third method on top of (A) anchor and (B) ticker, where
a more capable LLM looks at the brief, the simulated trajectory,
the per-round shock + ticker history, and the K most-similar
historical incidents (RAG-retrieved from the reference table) and
produces:

  { point_eur, low_eur, high_eur, reasoning, similar_incidents,
    confidence_score }

The reasoning chain is explicit: "this looks most like the 2017
MPS recapitalisation crossed with the 2024 CrowdStrike outage; the
sim shows ~1.8 shock-units of escalation but with a fast recovery
arc, so we estimate €X cost with Y% confidence". The operator can
audit the reasoning, and the score becomes a third input to the
combined estimator.

Combined output (Sprint C final):
    point = weighted_average([A, B, C], weights=W)
where W is calibrated on a holdout of past incidents — the method
that historically tracks reality best gets the highest weight.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def estimate_via_llm_judge(
    brief: str,
    sim_trajectory: list[dict],
    ticker_history: list[dict],
    top_k_similar: int = 5,
) -> Optional[dict]:
    """TODO Sprint C — call gemini-3.1-pro-preview with retrieved
    similar incidents and ask for a reasoned cost estimate.

    Plan:
      1. Embed the brief + final trajectory summary into a vector.
      2. Retrieve the top_k most-similar historical incidents from
         shared/dora_reference_incidents.json by:
           similarity(brief_summary, incident.label + sources_text)
         using the existing RAG store.
      3. Build a system prompt explaining the task: "you are a
         senior risk officer analysing this simulated incident
         relative to the K closest historical analogues; estimate
         the realistic euro cost".
      4. Call gemini-3.1-pro-preview with structured JSON output:
         { point_eur, low_eur, high_eur, reasoning,
           per_analog_relevance: [{id, similarity, weight}],
           confidence_score: 0-1 }
      5. Validate the response (point_eur within reasonable bounds)
         and return.

    Until implemented: returns None so the combined() function
    falls back to A/B only.
    """
    logger.debug("LLM judge stub — Sprint C not yet implemented")
    return None


# When implemented:
# from core.llm.gemini_client import GeminiClient
# JUDGE_MODEL = "gemini-3.1-pro-preview"
#
# Prompt template lives at core/dora/llm_judge_prompts.py (TODO).
