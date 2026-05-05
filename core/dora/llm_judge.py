"""DORA economic-impact — Sprint C (LLM judge layer).

Uses gemini-3.1-pro-preview as a third estimator. Looks at the brief,
the simulated trajectory summary, and the K most-similar historical
incidents from shared/dora_reference_incidents.json, and produces a
reasoned point/low/high estimate plus a confidence score and the list
of analogues actually used.

The judge fires ONCE per /api/compliance/dora/preview request, not
per round, not per agent — well outside the per-round narrative LLM
hot path that's restricted to flash-lite.

Combine logic in core/dora/economic_impact.py:combine() takes the
weighted average of A (anchor), B (ticker), C (judge) where w_C
scales with the confidence_score the judge returns. Heuristic
defaults until we have enough holdout to recalibrate weights:

    w_A = 0.30
    w_B = 0.30
    w_C = 0.40 × confidence_score   (so a low-confidence C de-weights)

When the judge returns None (network error, budget exhausted, missing
GOOGLE_API_KEY), combine() falls back to max(A, B) — same behaviour
as before Sprint C.
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INCIDENTS_PATH = REPO_ROOT / "shared" / "dora_reference_incidents.json"

# Authorised model for this judge — see memory feedback_gemini_model.md
JUDGE_MODEL = "gemini-3.1-pro-preview"


_JUDGE_SYSTEM_PROMPT = """You are a senior risk officer at a DORA-regulated financial \
institution. You are reviewing a SIMULATED operational/regulatory incident scenario \
and must estimate the realistic euro cost using only what the simulation produced \
plus the historical analogues you are given. Be conservative and explicit about \
uncertainty.

Output a single JSON object with these keys:
  point_eur:     int  (best-guess realistic cost in EUR)
  low_eur:       int  (90% CI lower bound)
  high_eur:      int  (90% CI upper bound)
  reasoning:     string (3-5 sentences, in English)
  similar_used:  array of incident ids you weighted heavily, max 5
  confidence_score: float in [0, 1] — your confidence in this estimate
                    given the analogue match quality

Hard rules:
- Use ONLY the analogues provided. Don't invent.
- 'point_eur' must lie inside [low_eur, high_eur].
- low_eur >= 0.
- If the analogues are weak matches, lower confidence_score and widen the band.
- No prose outside the JSON."""


def _summarise_trajectory(sim_summary: dict) -> str:
    """Compact trajectory description the judge can reason over."""
    if not sim_summary:
        return "No trajectory metrics available."
    parts = []
    if "n_rounds" in sim_summary:
        parts.append(f"{sim_summary['n_rounds']} rounds simulated")
    if "polarization_peak" in sim_summary:
        parts.append(f"peak polarization {sim_summary['polarization_peak']:.2f}/10")
    if "total_shock_units" in sim_summary:
        parts.append(f"Σ |shock| = {sim_summary['total_shock_units']:.3f} units")
    if "viral_posts_count" in sim_summary:
        parts.append(f"{sim_summary['viral_posts_count']} viral posts")
    if "countries_affected" in sim_summary:
        parts.append(f"{sim_summary['countries_affected']} countries impacted")
    if "ticker_final_pcts" in sim_summary and sim_summary["ticker_final_pcts"]:
        moves = ", ".join(
            f"{tk} {v:+.1f}%" for tk, v in list(sim_summary["ticker_final_pcts"].items())[:6]
        )
        parts.append(f"final ticker moves: {moves}")
    return "; ".join(parts) if parts else "No trajectory metrics available."


def _load_all_incidents() -> list[dict]:
    try:
        return json.loads(INCIDENTS_PATH.read_text()).get("incidents", [])
    except Exception as e:
        logger.warning(f"judge: incidents load failed: {e}")
        return []


def _retrieve_similar_incidents(category: Optional[str], total_shock: float, k: int = 5) -> list[dict]:
    """Pick the K most-similar historical incidents.

    Similarity = (category match? +2.0) + (1 / (1 + |shock_diff|))
    so same-category + similar shock magnitude bubble to the top.
    """
    incidents = _load_all_incidents()
    if not incidents:
        return []
    scored = []
    for inc in incidents:
        score = 0.0
        if category and inc.get("category") == category:
            score += 2.0
        shock_diff = abs(float(inc.get("shock_units", 0) or 0) - total_shock)
        score += 1.0 / (1.0 + shock_diff)
        scored.append((score, inc))
    scored.sort(key=lambda x: -x[0])
    return [inc for _, inc in scored[:k]]


def _format_analogues_block(incidents: list[dict]) -> str:
    if not incidents:
        return "(no historical analogues available)"
    lines = []
    for inc in incidents:
        lines.append(
            f"- id={inc.get('id')}, category={inc.get('category')}, "
            f"shock_units={inc.get('shock_units')}, cost_eur_m={inc.get('cost_eur_m')}, "
            f"label=\"{inc.get('label','')}\""
        )
    return "\n".join(lines)


def estimate_via_llm_judge(
    brief: str,
    sim_summary: dict,
    detected_category: Optional[str] = None,
    total_shock: Optional[float] = None,
    top_k_similar: int = 5,
) -> Optional[dict]:
    """Call gemini-3.1-pro-preview to produce a reasoned third estimate.

    Returns the structured judge dict, or None when the judge can't run
    (no API key, budget exhausted, response unparseable). The caller in
    economic_impact.combine() uses None to fall back to A/B only.
    """
    if not brief:
        return None
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.info("judge: GOOGLE_API_KEY not set — skipping LLM judge")
        return None

    shock = float(total_shock if total_shock is not None else (sim_summary.get("total_shock_units") or 0.0))
    analogues = _retrieve_similar_incidents(detected_category, shock, k=top_k_similar)
    analogues_block = _format_analogues_block(analogues)
    trajectory_block = _summarise_trajectory(sim_summary)

    prompt = f"""SIMULATED INCIDENT BRIEF:
{brief.strip()[:2000]}

SIMULATION TRAJECTORY:
{trajectory_block}

DETECTED CATEGORY: {detected_category or "(none — use heterogeneous analogues)"}

HISTORICAL ANALOGUES (most-similar by category + shock-magnitude proximity):
{analogues_block}

Produce the JSON now.
"""

    try:
        from core.llm.gemini_client import GeminiClient
        # Run sync — judge is one-shot, low-volume; avoid the asyncio
        # ceremony of spinning up an event loop just for a single call.
        client = GeminiClient(model=JUDGE_MODEL, budget=0.50)
        # generate_json is async — invoke via asyncio.run if no loop
        import asyncio as _aio
        try:
            loop = _aio.get_running_loop()
            # Already in an event loop (e.g. FastAPI handler) — schedule
            # via a thread to avoid nested-loop errors.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_aio.run, client.generate_json(
                    prompt=prompt, system_prompt=_JUDGE_SYSTEM_PROMPT,
                    temperature=0.2, max_output_tokens=1200,
                    component="dora_llm_judge",
                ))
                response = fut.result(timeout=45)
        except RuntimeError:
            # No running loop
            response = _aio.run(client.generate_json(
                prompt=prompt, system_prompt=_JUDGE_SYSTEM_PROMPT,
                temperature=0.2, max_output_tokens=1200,
                component="dora_llm_judge",
            ))
    except Exception as e:
        logger.warning(f"judge: LLM call failed: {e}")
        return None

    if not isinstance(response, dict):
        logger.warning(f"judge: unexpected response shape: {type(response)}")
        return None

    # Validate + sanitise
    try:
        point = float(response.get("point_eur", 0))
        low = float(response.get("low_eur", 0))
        high = float(response.get("high_eur", 0))
        conf = float(response.get("confidence_score", 0.5))
    except (TypeError, ValueError) as e:
        logger.warning(f"judge: response parse failed: {e}")
        return None
    if not math.isfinite(point) or point < 0:
        return None
    low = max(0.0, min(low, point))
    high = max(point, high)
    conf = max(0.0, min(1.0, conf))
    similar_used = response.get("similar_used") or []
    if not isinstance(similar_used, list):
        similar_used = []
    return {
        "method": "llm_judge",
        "point_eur": round(point, 2),
        "low_eur": round(low, 2),
        "high_eur": round(high, 2),
        "confidence_score": round(conf, 3),
        "reasoning": str(response.get("reasoning", ""))[:1500],
        "similar_used": similar_used[:5],
        "model": JUDGE_MODEL,
        "n_analogues_retrieved": len(analogues),
        "analogues_provided": [
            {"id": inc.get("id"), "category": inc.get("category"),
             "shock_units": inc.get("shock_units"), "cost_eur_m": inc.get("cost_eur_m")}
            for inc in analogues
        ],
    }
