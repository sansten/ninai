"""Multi-Hop Retrieval Service — Phase 92.

Implements ReAct-style iterative retrieval for chain questions:

    "Who reported to the person who approved the budget that delayed the project?"

A single-pass vector search misses 42% of answers here because the answer
chunk is not semantically close to the question — it's two hops away.

Algorithm:
  1. Retrieve with the original query (Hop 0).
  2. Extract "chain anchors" — entities that the question likely pivots on —
     using LLM analysis or structural heuristics (no LLM dependency).
  3. Retrieve with each anchor as the next query (Hop 1+).
  4. Deduplicate and compose a merged context pool.
  5. Return a structured MultiHopResult with provenance for each hop.

The LLM-guided path (when an inference_fn is provided) generates the next
query by reasoning about what it needs to know next given current evidence.
The heuristic path extracts noun phrases and proper nouns as anchors.

Integration: MultiHopRetrievalService.retrieve() returns a MultiHopResult
that can be injected directly into the cognitive_loop read result.

Usage::

    svc = MultiHopRetrievalService(
        inference_fn=engine.infer_plain,  # or None for heuristic mode
        max_hops=3,
        top_k_per_hop=20,
    )
    result = await svc.retrieve(
        query="Who approved the Q3 budget?",
        retrieve_fn=lambda q: router.read(tenant_id, session_id, q),
    )
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "what", "when",
    "where", "who", "how", "why", "which", "that", "this", "these", "those",
    "did", "does", "do", "will", "would", "could", "should", "may", "might",
    "have", "has", "had", "be", "been", "being", "about", "after", "before",
    "between", "during", "until", "while", "then", "than", "not", "no",
    "person", "people", "someone", "they", "their", "he", "she", "it",
})

_CHAIN_PATTERNS = [
    r"who\s+\w+\s+to\s+the\s+person",
    r"who\s+reported",
    r"who\s+approved",
    r"who\s+was\s+responsible",
    r"what\s+caused.*what\s+caused",
    r"why\s+did.*because\s+of",
    r"led\s+to",
    r"resulted\s+in",
    r"due\s+to.*which\s+caused",
]
_CHAIN_RE = re.compile("|".join(_CHAIN_PATTERNS), re.IGNORECASE)

_NEXT_HOP_PROMPT = """You are a retrieval planner. Given a question and the evidence retrieved so far,
identify what SINGLE piece of information is still missing to answer the question.
Express it as a short search query (5-10 words max). Output ONLY the query, no explanation.

Question: {question}
Evidence so far:
{evidence_summary}

Missing information query:"""


@dataclass
class HopResult:
    hop_index: int
    query: str
    rationale: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    anchors_extracted: list[str] = field(default_factory=list)


@dataclass
class MultiHopResult:
    original_query: str
    hops: list[HopResult] = field(default_factory=list)
    all_chunks: list[dict[str, Any]] = field(default_factory=list)   # deduplicated
    is_chain_question: bool = False
    n_hops_taken: int = 0
    chain_rationale: str = ""    # human-readable trace of why each hop was taken

    @property
    def total_chunks_retrieved(self) -> int:
        return len(self.all_chunks)


# Callable types
RetrieveFn = Callable[[str], Awaitable[Any]]
InferenceFn = Callable[[str], Awaitable[str]]


class MultiHopRetrievalService:
    """Iterative retrieval with optional LLM-guided query generation.

    Parameters
    ----------
    inference_fn:
        Async callable ``(prompt: str) → str``. When provided, used to
        generate next-hop queries from current evidence. When None, uses
        the heuristic anchor extractor.
    max_hops:
        Maximum number of additional hops beyond the initial retrieval.
        Total retrievals = max_hops + 1.
    top_k_per_hop:
        Max chunks to request from the retrieval backend per hop.
    min_new_chunks_to_continue:
        Skip remaining hops when a hop returns fewer than this many
        new (non-duplicate) chunks — the evidence pool is saturated.
    """

    def __init__(
        self,
        *,
        inference_fn: InferenceFn | None = None,
        max_hops: int = 3,
        top_k_per_hop: int = 20,
        min_new_chunks_to_continue: int = 2,
    ) -> None:
        self._inference_fn = inference_fn
        self._max_hops = max(1, int(max_hops))
        self._top_k = int(top_k_per_hop)
        self._min_new_chunks_to_continue = int(min_new_chunks_to_continue)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        retrieve_fn: RetrieveFn,
    ) -> MultiHopResult:
        """Run multi-hop retrieval for the given query.

        Parameters
        ----------
        query:       The original user question.
        retrieve_fn: Async callable (query_str) → ReadResult-like object.
                     Must have a .qdrant_chunks attribute (list of dicts).
        """
        result = MultiHopResult(
            original_query=query,
            is_chain_question=self._is_chain_question(query),
        )

        seen_ids: set[str] = set()
        rationale_lines: list[str] = []

        # Hop 0: initial retrieval with the original query
        hop0 = await self._run_hop(
            hop_index=0,
            query=query,
            rationale="Initial retrieval with original question.",
            retrieve_fn=retrieve_fn,
            seen_ids=seen_ids,
        )
        result.hops.append(hop0)
        result.all_chunks.extend(hop0.chunks)
        rationale_lines.append(f"Hop 0: '{query}' → {len(hop0.chunks)} chunks")

        # Hops 1..N
        for hop_idx in range(1, self._max_hops + 1):
            if not result.all_chunks:
                break

            next_query, rationale = await self._plan_next_hop(
                original_query=query,
                current_chunks=result.all_chunks,
                hop_index=hop_idx,
            )
            if not next_query:
                rationale_lines.append(f"Hop {hop_idx}: no next query generated; stopping.")
                break

            hop = await self._run_hop(
                hop_index=hop_idx,
                query=next_query,
                rationale=rationale,
                retrieve_fn=retrieve_fn,
                seen_ids=seen_ids,
            )
            result.hops.append(hop)
            result.all_chunks.extend(hop.chunks)
            rationale_lines.append(f"Hop {hop_idx}: '{next_query}' → {len(hop.chunks)} new chunks")

            if len(hop.chunks) < self._min_new_chunks_to_continue:
                rationale_lines.append(f"  Stopping: only {len(hop.chunks)} new chunks (below threshold).")
                break

        result.n_hops_taken = len(result.hops)
        result.chain_rationale = "\n".join(rationale_lines)
        return result

    # ------------------------------------------------------------------
    # Hop execution
    # ------------------------------------------------------------------

    async def _run_hop(
        self,
        *,
        hop_index: int,
        query: str,
        rationale: str,
        retrieve_fn: RetrieveFn,
        seen_ids: set[str],
    ) -> HopResult:
        hop = HopResult(hop_index=hop_index, query=query, rationale=rationale)
        try:
            read_result = await retrieve_fn(query)
            raw_chunks = getattr(read_result, "qdrant_chunks", None) or []
            for ch in raw_chunks[: self._top_k]:
                chunk_id = str(ch.get("id", ""))
                if chunk_id and chunk_id in seen_ids:
                    continue
                if chunk_id:
                    seen_ids.add(chunk_id)
                hop.chunks.append(ch)
        except Exception as exc:
            logger.warning("MultiHopRetrieval hop %d failed: %s", hop_index, exc)

        hop.anchors_extracted = self._extract_anchors(query)
        return hop

    # ------------------------------------------------------------------
    # Next-hop query planning
    # ------------------------------------------------------------------

    async def _plan_next_hop(
        self,
        *,
        original_query: str,
        current_chunks: list[dict[str, Any]],
        hop_index: int,
    ) -> tuple[str, str]:
        """Return (next_query, rationale). Empty string = stop."""
        if self._inference_fn is not None:
            return await self._llm_plan(original_query, current_chunks, hop_index)
        return self._heuristic_plan(original_query, current_chunks, hop_index)

    async def _llm_plan(
        self,
        original_query: str,
        chunks: list[dict[str, Any]],
        hop_index: int,
    ) -> tuple[str, str]:
        evidence_lines: list[str] = []
        for ch in chunks[:8]:
            p = ch.get("payload", {}) or {}
            text = str(p.get("text") or p.get("content") or p.get("summary") or "")[:300]
            if text:
                evidence_lines.append(f"- {text}")

        if not evidence_lines:
            return "", "No evidence to plan next hop."

        evidence_summary = "\n".join(evidence_lines)
        prompt = _NEXT_HOP_PROMPT.format(
            question=original_query,
            evidence_summary=evidence_summary,
        )
        try:
            next_query = await self._inference_fn(prompt)
            next_query = str(next_query or "").strip().strip('"').strip("'")
            if not next_query or len(next_query) < 3:
                return "", "LLM returned empty next-hop query."
            return next_query, f"LLM-generated next hop {hop_index}: '{next_query}'"
        except Exception as exc:
            logger.warning("LLM next-hop planning failed: %s; falling back to heuristic", exc)
            return self._heuristic_plan(original_query, chunks, hop_index)

    def _heuristic_plan(
        self,
        original_query: str,
        chunks: list[dict[str, Any]],
        hop_index: int,
    ) -> tuple[str, str]:
        """Extract proper nouns and entity names from retrieved chunks as next queries."""
        all_anchors: list[str] = []
        for ch in chunks[:10]:
            p = ch.get("payload", {}) or {}
            text = str(p.get("text") or p.get("content") or "")
            all_anchors.extend(self._extract_anchors(text))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for a in all_anchors:
            key = a.lower()
            if key not in seen and len(a) >= 3:
                seen.add(key)
                unique.append(a)

        if not unique:
            return "", "No anchors found; stopping."

        # Pick the most novel anchor (not in the original query)
        orig_tokens = set(original_query.lower().split())
        candidates = [a for a in unique if a.lower() not in orig_tokens]
        if not candidates:
            candidates = unique

        next_query = candidates[0]
        rationale = f"Heuristic hop {hop_index}: pivoting to '{next_query}' from {len(unique)} anchors"
        return next_query, rationale

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_chain_question(self, query: str) -> bool:
        """Detect questions that require chained retrieval."""
        return bool(_CHAIN_RE.search(query))

    @staticmethod
    def _extract_anchors(text: str) -> list[str]:
        """Extract capitalized proper nouns and entity-like tokens."""
        words = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
        phrases = [w.strip() for w in words if w.lower().strip() not in _STOPWORDS and len(w) >= 3]
        return phrases[:10]
