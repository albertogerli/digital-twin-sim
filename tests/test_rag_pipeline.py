"""End-to-end tests for the RAG citation pipeline.

Covers the full path:
    upload doc → chunk + embed → store → retrieve → attach to post →
    persist to SQLite → re-hydrate via export → ready for frontend chip render.

Embedding calls are stubbed so the test is fast, deterministic, and free.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Deterministic embedding shim ──────────────────────────────────────────
import hashlib
import re

_DIM = 256  # large enough to keep token-bucket collisions rare


def _stable_bucket(token: str, dim: int = _DIM) -> int:
    """Process-stable hash → bucket index (Python's hash() is randomized
    per-process, so use MD5 to keep CI / local results identical)."""
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "big") % dim


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _fake_embed(text: str):
    """Bag-of-words 256-dim 'embedding' with stable bucketing.

    Texts that share the same content words score high against each other,
    deterministically across runs and platforms.
    """
    if not text:
        return None
    vec = [0.0] * _DIM
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) < 3:
            continue  # drop stop-words / punctuation
        vec[_stable_bucket(tok)] += 1.0
    # L2 normalize
    norm = sum(v * v for v in vec) ** 0.5
    if norm == 0:
        return None
    return [v / norm for v in vec]


@pytest.fixture(autouse=True)
def _patch_embed(monkeypatch):
    """Patch the Gemini embedder used by RAGStore so tests don't hit the network."""
    import api.rag_store as rs
    monkeypatch.setattr(rs, "_embed_text", _fake_embed)


# ── Test 1: RAGStore basic chunk + embed + retrieve ──────────────────────
def test_rag_store_indexes_and_retrieves_relevant_chunks():
    from api.rag_store import RAGStore

    store = RAGStore()

    # Two documents, distinct topics
    crypto_doc = (
        "Italy introduces a five percent transaction tax on cryptocurrency exchanges. "
        "The bill targets DeFi platforms, centralized exchanges, and on-chain swaps. "
        "Industry groups warn of capital flight to Switzerland and Malta. "
    ) * 4  # repeat to clear MIN_CHUNK_WORDS

    energy_doc = (
        "EU Commission proposes Article 14 revision tightening grid investment timelines. "
        "Member states must align with 2030 renewable targets. "
        "Industry lobbies push back, citing infrastructure capacity constraints. "
    ) * 4

    n_crypto = store.add_document("d1", "crypto_tax_brief.txt", crypto_doc)
    n_energy = store.add_document("d2", "eu_energy_package.txt", energy_doc)

    assert n_crypto > 0, "crypto doc should produce at least 1 chunk"
    assert n_energy > 0, "energy doc should produce at least 1 chunk"
    assert store.doc_count == 2
    assert store.chunk_count == n_crypto + n_energy

    # Query about crypto: should retrieve d1 chunks first
    hits = store.retrieve("cryptocurrency tax exchanges", k=3, min_score=0.0)
    assert len(hits) > 0, "retrieval returned nothing"
    assert hits[0].doc_id == "d1", f"expected d1 first, got {hits[0].doc_id}"
    assert all(h.score >= 0.0 for h in hits)

    # Query about EU energy: should retrieve d2 first
    hits = store.retrieve("EU Article 14 grid renewable", k=3, min_score=0.0)
    assert hits[0].doc_id == "d2", f"expected d2 first, got {hits[0].doc_id}"


# ── Test 2: live KB inject (wargame path) ────────────────────────────────
def test_rag_store_supports_live_inject_after_init():
    from api.rag_store import RAGStore

    store = RAGStore()
    store.add_document(
        "d1", "baseline.txt",
        "Routine policy briefing on industrial timelines and grid investment. " * 5,
    )
    initial = store.chunk_count

    # Mid-run inject (matches what wargame_intervene now does)
    new_chunks = store.add_document(
        "inject_r4_wargame",
        "breaking_news_leak.txt",
        "BREAKING: leaked document reveals industry has secretly funded opposition campaigns. " * 5,
    )
    assert new_chunks > 0
    assert store.chunk_count > initial

    # The new content should be retrievable immediately
    hits = store.retrieve("leaked industry funding opposition", k=2, min_score=0.0)
    assert hits[0].doc_id == "inject_r4_wargame", "live-injected doc didn't surface"


# ── Test 3: SQLite round-trip preserves citations ────────────────────────
def test_post_citations_round_trip_through_sqlite(tmp_path):
    """
    Persisting a post with citations through PlatformEngine should survive
    re-hydration via export.get_posts_for_round and arrive as a Python list,
    not a JSON string. This is the pre-condition for the frontend chip to render.
    """
    from core.platform.platform_engine import PlatformEngine
    from export import get_posts_for_round

    db_path = tmp_path / "social_test.db"
    pe = PlatformEngine(str(db_path))

    citations = [
        {"doc_id": "d1", "chunk_id": "d1_c000", "title": "EU_energy_package.pdf",
         "score": 0.84, "snippet": "Article 14 tightens grid investment timelines…"},
        {"doc_id": "d2", "chunk_id": "d2_c003", "title": "industry_position.docx",
         "score": 0.71, "snippet": "Industry lobbies push back, citing capacity…"},
    ]
    post_data = {
        "author_id": "agent_brandt",
        "author_tier": 1,
        "platform": "twitter",
        "text": "The Commission must reconsider Article 14 timelines.",
        "round": 3,
        "timestamp_sim": "2026-04-22T10:00:00",
        "citations": citations,
    }

    post_id = pe.add_post(post_data, round_num=3)
    pe.flush()
    pe.close()

    # Re-open via the export read path (matches what writes replay_round_N.json)
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = get_posts_for_round(conn, 3)
    conn.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["author_id"] == "agent_brandt"
    assert row["round"] == 3

    cits = row["citations"]
    assert isinstance(cits, list), f"citations should be list after round-trip, got {type(cits).__name__}"
    assert len(cits) == 2

    # Field-by-field shape used by the frontend chip
    assert cits[0]["doc_id"] == "d1"
    assert cits[0]["chunk_id"] == "d1_c000"
    assert cits[0]["title"].endswith(".pdf")
    assert 0.0 < cits[0]["score"] <= 1.0
    assert "snippet" in cits[0]


# ── Test 4: post WITHOUT citations stays empty (no false chips) ──────────
def test_post_without_citations_round_trips_empty(tmp_path):
    from core.platform.platform_engine import PlatformEngine
    from export import get_posts_for_round
    import sqlite3

    db_path = tmp_path / "social_no_cite.db"
    pe = PlatformEngine(str(db_path))
    pe.add_post(
        {"author_id": "a", "author_tier": 2, "platform": "x",
         "text": "no kb consulted", "round": 1, "timestamp_sim": ""},
        round_num=1,
    )
    pe.flush()
    pe.close()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = get_posts_for_round(conn, 1)
    conn.close()

    assert rows[0]["citations"] == [], "post without citations should yield []"


# ── Test 5: full chunk → embed → cite → persist → export pipeline ────────
def test_full_pipeline_chunk_embed_cite_persist_export(tmp_path):
    """End-to-end: simulates the per-round flow round_manager runs.

    We deliberately don't import round_manager (PEP 604 union types fail on
    Python 3.9) — instead we exercise the same primitives in sequence:

        1. RAGStore.add_document()  ← what document_processor calls on upload
        2. RAGStore.retrieve()       ← what _rag_setup_round calls per round
        3. attach citations to post  ← what _attach_citations does
        4. PlatformEngine.add_post() ← what each phase calls per agent
        5. export.get_posts_for_round() ← what the JSON export reads back
    """
    from api.rag_store import RAGStore
    from core.platform.platform_engine import PlatformEngine
    from export import get_posts_for_round
    import sqlite3

    # 1. Upload + index
    store = RAGStore()
    store.add_document(
        "d1", "scenario_brief.pdf",
        "EU Commission tabled draft Article 14 revision. Industry groups push back. " * 6,
    )
    assert store.chunk_count > 0

    # 2. Round-time retrieval (what RoundManager._rag_setup_round does)
    round_event = "Commission moves on Article 14, industry mobilizes counter-position"
    hits = store.retrieve(round_event, k=4, min_score=0.0)
    assert hits, "no chunks retrieved for round event"

    # 3. Build post payload + attach citations (what _attach_citations does)
    citations_for_round = [
        {"doc_id": h.doc_id, "chunk_id": h.chunk_id, "title": h.title,
         "score": h.score, "snippet": h.snippet}
        for h in hits
    ]
    post = {
        "author_id": "elite_brandt",
        "author_tier": 1,
        "platform": "twitter",
        "text": "Article 14 timelines need revisiting.",
        "round": 1,
        "timestamp_sim": "",
        "citations": citations_for_round,
    }

    # 4. Persist
    db_path = tmp_path / "social_pipeline.db"
    pe = PlatformEngine(str(db_path))
    pe.add_post(post, round_num=1)
    pe.flush()
    pe.close()

    # 5. Read back via export path
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = get_posts_for_round(conn, 1)
    conn.close()

    assert len(rows) == 1
    out_cits = rows[0]["citations"]
    assert isinstance(out_cits, list) and len(out_cits) == len(citations_for_round)
    # Same chunks come out — frontend chip will render with these
    out_chunks = [c["chunk_id"] for c in out_cits]
    in_chunks = [c["chunk_id"] for c in citations_for_round]
    assert out_chunks == in_chunks
