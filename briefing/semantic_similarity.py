"""Sprint 12 — semantic similarity layer for stakeholder relevance.

Uses Gemini text-embedding-004 (no new dependencies — google-genai is
already required) to compute cosine similarity between:
  - the brief text + sector hints
  - each stakeholder's biographical text (bio + role + party + topic_tags)

This complements Layer 1 deterministic scoring: when stakeholders have
sparse / missing topic_tags in their JSON, the embedding component
captures the semantic alignment that the rule-based score misses.

Caching strategy
----------------
- Stakeholder embeddings are EXPENSIVE-ish to compute (~1 API call each)
  but stable: a stakeholder's bio doesn't change between sims. Cache
  on disk in `outputs/stakeholder_embeddings.json` keyed by stakeholder.id.
- Brief embedding is cheap (1 API call per sim) — no cache needed.
- All-or-nothing failure mode: if Gemini embedding API fails for any
  reason (network, rate limit, missing key), `semantic_similarity()`
  returns 0.5 as a neutral score so the score formula degrades
  gracefully back to Layer 1 deterministic only.

API choice
----------
Gemini text-embedding-004 (768-dim) is in the free tier (1500 req/day),
multilingual (handles Italian + English), and ships with google-genai.
We DON'T use sentence-transformers locally to avoid pulling in 500MB
of torch + model weights.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Cache lives next to other persistence files in the Railway volume
_CACHE_PATH = os.path.join(
    os.environ.get("DTS_OUTPUTS_DIR")
    or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"),
    "stakeholder_embeddings.json",
)
_EMBED_MODEL = "text-embedding-004"
_NEUTRAL_SCORE = 0.5
_DIMENSION = 768  # text-embedding-004 dimension


# ── In-process cache (loaded lazily from disk) ────────────────────────────

_DISK_CACHE: Optional[dict] = None


def _load_disk_cache() -> dict:
    global _DISK_CACHE
    if _DISK_CACHE is not None:
        return _DISK_CACHE
    if not os.path.exists(_CACHE_PATH):
        _DISK_CACHE = {}
        return _DISK_CACHE
    try:
        with open(_CACHE_PATH) as f:
            _DISK_CACHE = json.load(f) or {}
    except Exception as exc:
        logger.warning(f"embedding cache read failed: {exc}")
        _DISK_CACHE = {}
    return _DISK_CACHE


def _save_disk_cache():
    if _DISK_CACHE is None:
        return
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(_DISK_CACHE, f)
    except Exception as exc:
        logger.warning(f"embedding cache write failed: {exc}")


# ── Embedding text builders ───────────────────────────────────────────────

def _stakeholder_text(stakeholder) -> str:
    """Compose the text we want to embed for a stakeholder.

    Uses fields most likely to carry semantic signal about what topics
    they comment on: role + party + bio + key_traits + position topic_tags.
    Concatenated with newlines so the embedder treats them as one document.
    """
    parts = []
    if getattr(stakeholder, "name", ""):
        parts.append(f"Name: {stakeholder.name}")
    if getattr(stakeholder, "role", ""):
        parts.append(f"Role: {stakeholder.role}")
    if getattr(stakeholder, "party_or_org", ""):
        parts.append(f"Organisation: {stakeholder.party_or_org}")
    if getattr(stakeholder, "bio", ""):
        parts.append(f"Bio: {stakeholder.bio}")
    if getattr(stakeholder, "key_traits", []):
        parts.append(f"Traits: {', '.join(stakeholder.key_traits)}")
    # Topic tags from positions
    tag_lines = []
    for pos in (getattr(stakeholder, "positions", []) or []):
        tag = getattr(pos, "topic_tag", None) or (
            pos.get("topic_tag", "") if isinstance(pos, dict) else ""
        )
        if tag:
            tag_lines.append(tag)
    if tag_lines:
        parts.append(f"Topics: {', '.join(tag_lines)}")
    return "\n".join(parts)[:2000]  # cap to avoid blowing API limits


def _brief_text(brief: str, scope=None) -> str:
    """Compose the text we want to embed for the brief query."""
    parts = [brief or ""]
    if scope is not None:
        sector = getattr(scope, "sector", "")
        sub = getattr(scope, "sub_sector", "")
        named = getattr(scope, "named_entities", []) or []
        if sector:
            parts.append(f"Sector: {sector}")
        if sub:
            parts.append(f"Sub-sector: {sub}")
        if named:
            parts.append(f"Named entities: {', '.join(named[:8])}")
    return "\n".join(parts)[:4000]


# ── Gemini embedding client (lazy-loaded) ─────────────────────────────────

_GENAI_CLIENT = None


def _get_genai_client():
    """Lazy-load the Gemini client. Soft-fails if no key / no library."""
    global _GENAI_CLIENT
    if _GENAI_CLIENT is not None:
        return _GENAI_CLIENT if _GENAI_CLIENT is not False else None
    try:
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set; semantic similarity disabled")
            _GENAI_CLIENT = False
            return None
        _GENAI_CLIENT = genai.Client(api_key=api_key)
    except Exception as exc:
        logger.warning(f"genai client init failed: {exc}")
        _GENAI_CLIENT = False
        return None
    return _GENAI_CLIENT


def _embed_text(text: str) -> Optional[list[float]]:
    """Return the Gemini embedding vector for `text`. None on any failure."""
    if not text:
        return None
    client = _get_genai_client()
    if client is None:
        return None
    try:
        result = client.models.embed_content(
            model=_EMBED_MODEL,
            contents=text,
        )
        # SDK returns result.embeddings (list) → first item.values is the vector
        if hasattr(result, "embeddings") and result.embeddings:
            v = result.embeddings[0]
            return list(v.values) if hasattr(v, "values") else list(v)
        return None
    except Exception as exc:
        logger.warning(f"embed_content failed: {exc}")
        return None


# ── Public API ────────────────────────────────────────────────────────────

def get_stakeholder_embedding(stakeholder) -> Optional[list[float]]:
    """Return cached embedding for a stakeholder, computing once if missing."""
    if stakeholder is None:
        return None
    sid = getattr(stakeholder, "id", "")
    if not sid:
        return None
    cache = _load_disk_cache()
    if sid in cache:
        return cache[sid]
    text = _stakeholder_text(stakeholder)
    if not text:
        return None
    vec = _embed_text(text)
    if vec is None:
        return None
    cache[sid] = vec
    _save_disk_cache()
    return vec


def get_brief_embedding(brief: str, scope=None) -> Optional[list[float]]:
    """Compute the brief embedding (no cache — brief changes every sim)."""
    text = _brief_text(brief, scope)
    return _embed_text(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Standard cosine similarity ∈ [-1, 1]. 0 if either vector empty / mismatched."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_similarity(stakeholder, brief_embedding: Optional[list[float]]) -> float:
    """Return semantic similarity ∈ [0, 1] (rescaled from cosine).

    Returns _NEUTRAL_SCORE on any failure so the relevance score formula
    degrades gracefully when the embedding API is unavailable.
    """
    if brief_embedding is None:
        return _NEUTRAL_SCORE
    s_emb = get_stakeholder_embedding(stakeholder)
    if s_emb is None:
        return _NEUTRAL_SCORE
    cos = cosine_similarity(brief_embedding, s_emb)
    # Rescale [-1, 1] → [0, 1]; for text embeddings cos is usually in [0, 1] already
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


def precompute_for_stakeholders(stakeholders: list, max_concurrent: int = 1):
    """Best-effort batch precompute. Sequential to respect free-tier limits.
    Cache hits are skipped automatically. Safe to call multiple times."""
    cache = _load_disk_cache()
    new_count = 0
    for s in stakeholders:
        sid = getattr(s, "id", "")
        if not sid or sid in cache:
            continue
        vec = _embed_text(_stakeholder_text(s))
        if vec is None:
            continue
        cache[sid] = vec
        new_count += 1
    if new_count > 0:
        _save_disk_cache()
        logger.info(f"semantic_similarity: precomputed {new_count} new embeddings")
    return new_count
