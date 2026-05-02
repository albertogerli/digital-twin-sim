"""In-memory RAG store: chunks + Gemini embeddings + cosine retrieval.

Keeps things minimal — one store per simulation lives in SimulationState.
No vector DB dependency; for production scale-out, replace with Qdrant or
pgvector behind the same Add/Retrieve interface.

Pipeline:
  add_document(doc_id, title, text)
    └─ chunk into ~300-word windows with 60-word overlap
    └─ embed each chunk via Gemini text-embedding-004 (cached client)
    └─ store {chunk_id, doc_id, title, text, vec}

  retrieve(query, k)
    └─ embed query
    └─ cosine-rank stored chunks
    └─ return top-K with scores
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

from briefing.semantic_similarity import _embed_text  # reuse Gemini client

logger = logging.getLogger(__name__)

CHUNK_WORDS = 300
CHUNK_OVERLAP = 60
MIN_CHUNK_WORDS = 30   # don't bother indexing tiny scraps


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    vec: Optional[List[float]] = None
    char_offset: int = 0


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    snippet: str
    score: float


def _chunk_text(text: str) -> List[str]:
    """Split text into ~CHUNK_WORDS-word windows with CHUNK_OVERLAP overlap."""
    words = text.split()
    if len(words) <= MIN_CHUNK_WORDS:
        return [text] if text.strip() else []
    chunks = []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        window = words[i:i + CHUNK_WORDS]
        if len(window) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(window))
        if i + CHUNK_WORDS >= len(words):
            break
    return chunks


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class RAGStore:
    """In-memory chunk store. Construct empty, then add_document() per file."""
    chunks: List[Chunk] = field(default_factory=list)
    _doc_count: int = 0

    @property
    def doc_count(self) -> int:
        return self._doc_count

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def add_document(self, doc_id: str, title: str, text: str) -> int:
        """Chunk + embed + store. Returns the number of chunks added."""
        if not text or not text.strip():
            logger.debug(f"RAG: skipping empty doc {doc_id}")
            return 0
        windows = _chunk_text(text)
        added = 0
        for i, win in enumerate(windows):
            cid = f"{doc_id}_c{i:03d}"
            vec = _embed_text(win)
            if vec is None:
                logger.warning(f"RAG: embed failed for {cid}, storing without vec")
            self.chunks.append(Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                title=title,
                text=win,
                vec=vec,
                char_offset=i * (CHUNK_WORDS - CHUNK_OVERLAP),
            ))
            added += 1
        self._doc_count += 1
        logger.info(f"RAG: indexed {doc_id} ({title}) → {added} chunks")
        return added

    def retrieve(self, query: str, k: int = 4, min_score: float = 0.30) -> List[RetrievedChunk]:
        """Top-K chunks by cosine similarity to the query embedding."""
        if not query or not self.chunks:
            return []
        qvec = _embed_text(query)
        if qvec is None:
            return []
        scored: List[tuple[float, Chunk]] = []
        for c in self.chunks:
            if c.vec is None:
                continue
            scored.append((_cosine(qvec, c.vec), c))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[RetrievedChunk] = []
        for score, c in scored[:k]:
            if score < min_score:
                break
            out.append(RetrievedChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                title=c.title,
                snippet=c.text[:240] + ("…" if len(c.text) > 240 else ""),
                score=round(score, 4),
            ))
        return out

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None
