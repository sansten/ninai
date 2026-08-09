"""
Two-step reranker for v2 Graph-RAG retrieval results.

Step 1 — Score fusion:
  Qdrant chunks:  final = 0.7 * cosine_sim   + 0.3 * keyword_overlap
  Graph nodes:    final = 0.5 * (weight*recency) + 0.5 * keyword_overlap

Step 2 — Top-K selection (caller decides cutoff).

No external API calls — all scoring is in-process arithmetic.
Typical overhead: ~0.2 ms for 10 Qdrant + 30 graph candidates.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time

logger = logging.getLogger(__name__)
from typing import Any

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "what", "when",
    "where", "who", "how", "why", "which", "that", "this", "these", "those",
    "i", "me", "my", "you", "your", "he", "she", "his", "her", "it", "its",
    "we", "our", "they", "their", "there", "then", "than", "not", "no",
    "if", "as", "so", "about", "after", "before", "between",
})

# Node weight decays to 0.5 after 7 days of no access (half-life)
_RECENCY_HALF_LIFE_MS: float = 7 * 24 * 3600 * 1000.0


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOPWORDS and len(t) > 2}


def _keyword_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    hits = len(query_tokens & text_tokens)
    return hits / len(query_tokens)   # recall-oriented


def _recency_score(created_at_ms: int | float | None) -> float:
    """Exponential decay: 1.0 for brand-new, 0.5 at half-life, never < 0.05."""
    if not created_at_ms:
        return 0.5
    age_ms = max(0.0, time.time() * 1000.0 - float(created_at_ms))
    return max(0.05, math.exp(-age_ms / _RECENCY_HALF_LIFE_MS * math.log(2)))


def rerank_context(
    query: str,
    qdrant_chunks: list[dict[str, Any]],
    graph_nodes: list[dict[str, Any]],
    top_qdrant: int = 10,
    top_graph: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Two-stage reranker for Qdrant + graph retrieval candidates.

    Qdrant: when more candidates than top_qdrant are available (i.e. a larger
    retrieval pool was fetched), reorder by combining cosine-position score
    with keyword overlap so specific-fact queries surface the right chunk from
    the wider candidate pool.  When the pool equals top_qdrant, just trim
    (no-op — cosine order is already optimal).

    Graph: sort by weight × recency.

    Returns (qdrant_chunks, reranked_graph_nodes), each trimmed to top_k.
    """
    # Qdrant two-stage: if we have more candidates than the final budget,
    # reorder using (cosine-position × keyword) to surface the best match.
    n = len(qdrant_chunks)
    if n > top_qdrant:
        query_tokens = _tokenize(query)
        scored: list[tuple[float, dict]] = []
        for rank, chunk in enumerate(qdrant_chunks):
            payload = chunk.get("payload", {})
            text = str(payload.get("text") or payload.get("content") or "")
            cosine_pos = 1.0 - rank / n          # 1.0 for rank-0, ~0 for last
            kw = _keyword_score(query_tokens, text) if query_tokens else 0.0
            scored.append((0.65 * cosine_pos + 0.35 * kw, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked_qdrant = [c for _, c in scored[:top_qdrant]]
    else:
        reranked_qdrant = qdrant_chunks[:top_qdrant]

    # Graph: re-sort by weight × recency (FalkorDB returns weight DESC but
    # doesn't apply recency; this corrects stale high-weight nodes).
    scored_graph: list[tuple[float, dict]] = []
    for node in graph_nodes:
        weight = float(node.get("weight") or 0.5)
        recency = _recency_score(node.get("created_at"))
        scored_graph.append((weight * recency, node))

    scored_graph.sort(key=lambda x: x[0], reverse=True)
    reranked_graph = [n for _, n in scored_graph[:top_graph]]

    return reranked_qdrant, reranked_graph


# ---------------------------------------------------------------------------
# Cross-encoder reranker — high-precision relevance ranking of retrieved chunks.
#
# Replaces the lossy LLM pre-filter (which truncated chunks to ~250 chars, only
# considered the first 20 candidates, and asked an LLM to emit index numbers).
# A passage cross-encoder scores the FULL query<->chunk pair, so the answer chunk
# outranks distractors instead of getting dropped. Model-agnostic / dataset-neutral
# (ms-marco MiniLM is a general web-passage reranker, no LoCoMo tuning). Small and
# CPU-friendly. Lazy-loaded once; gracefully returns None so callers fall back.
# ---------------------------------------------------------------------------

# BGE reranker (conversational/retrieval-fit) rather than ms-marco (web-search-fit):
# bge-reranker-base is the CPU-practical default; set CE_RERANK_MODEL=BAAI/bge-reranker-v2-m3
# for a stronger (heavier) reranker. Scores are relevance logits — we only sort by them.
_CE_MODEL_NAME = os.environ.get("CE_RERANK_MODEL", "BAAI/bge-reranker-base")
_CE_MAX_CHARS = 2000            # full-ish chunk text (vs the old 250); CE truncates to its max tokens
_HIGH_SIG_TYPES = ("segment_gist", "personal_attribute", "temporal_event")
_ce_model = None
_ce_failed = False


def _get_cross_encoder():
    global _ce_model, _ce_failed
    if _ce_model is None and not _ce_failed:
        try:
            from sentence_transformers import CrossEncoder
            _ce_model = CrossEncoder(_CE_MODEL_NAME)
            logger.info("Cross-encoder reranker loaded: %s", _CE_MODEL_NAME)
        except Exception as exc:
            logger.warning("Cross-encoder load failed (%s); reranker disabled", exc)
            _ce_failed = True
    return _ce_model


def _chunk_text(ch: dict[str, Any]) -> str:
    p = ch.get("payload", {}) or {}
    return str(p.get("text") or p.get("content") or p.get("summary") or "")


def _is_high_signal(ch: dict[str, Any]) -> bool:
    p = ch.get("payload", {}) or {}
    return p.get("type") in _HIGH_SIG_TYPES or p.get("entity_type") in _HIGH_SIG_TYPES


def _ce_rerank_sync(query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]] | None:
    model = _get_cross_encoder()
    if model is None or len(chunks) <= top_k:
        return None
    pairs = [(query, _chunk_text(ch)[:_CE_MAX_CHARS]) for ch in chunks]
    scores = model.predict(pairs)
    ranked = [ch for _, ch in sorted(zip(scores, chunks), key=lambda t: float(t[0]), reverse=True)]
    top = ranked[:top_k]
    # Always retain high-signal chunks (gists, entity/temporal facts) even if they
    # ranked below the cutoff — same safety net the LLM pre-filter had.
    top_ids = {id(c) for c in top}
    extra = [c for c in chunks if id(c) not in top_ids and _is_high_signal(c)]
    return top + extra


async def cross_encoder_rerank(
    query: str, chunks: list[dict[str, Any]], top_k: int = 12
) -> list[dict[str, Any]] | None:
    """Rerank retrieved chunks by cross-encoder relevance to the query.

    Returns the top_k most relevant chunks (plus retained high-signal chunks), or
    None if the model is unavailable / there's nothing to trim — so callers fall
    back to the existing LLM pre-filter. The blocking predict runs in a thread so
    it never stalls the event loop.
    """
    try:
        return await asyncio.to_thread(_ce_rerank_sync, query, chunks, top_k)
    except Exception as exc:
        logger.warning("Cross-encoder rerank failed: %s", exc)
        return None
