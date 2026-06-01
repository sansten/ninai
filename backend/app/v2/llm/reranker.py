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

import math
import re
import time
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
