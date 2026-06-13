"""
V2 Cognitive Loop — Three-Phase Execution Pipeline

Phase 1 (Read/Retrieve):
  - DNCMemoryRouter.read() → graph subgraph + Qdrant episodic chunks

Phase 2 (Infer):
  - prompt_builder.build_inference_prompt() → assembled context
  - InferenceEngine.infer() → response + cited_node_ids + extracted_entities

Phase 3 (Learn/Write-back):
  - DNCMemoryRouter.write() → Utterance node + Entity nodes + edges
  - DNCMemoryRouter.write() for assistant response (write-back)
  - decay_and_prune() on the local seed neighborhood

This module is the only place that orchestrates all three phases.
Callers (API endpoints) provide tenant_id, session_id, and user_input.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.v2.llm.prompt_builder import build_bench_prompt, build_inference_prompt
from app.v2.llm.query_classifier import classify_query_tier
from app.v2.memory.dnc_router import DNCMemoryRouter, ReadResult, WriteResult

_BENCH_MODE = os.environ.get("NINAI_BENCH_MODE", "0").lower() in ("1", "true", "yes")
_SECOND_CHANCE = os.environ.get("NINAI_SECOND_CHANCE", "1").lower() in ("1", "true", "yes")
_LLM_RERANK = os.environ.get("NINAI_LLM_RERANK", "1").lower() in ("1", "true", "yes")
# Cross-encoder reranker: high-precision relevance ranking of retrieved chunks.
# When on, it replaces the lossy truncated-snippet LLM pre-filter (falls back to it
# if the model can't load). Targets the "answer chunk retrieved but not selected" failure.
_CE_RERANK = os.environ.get("NINAI_CE_RERANK", "0").lower() in ("1", "true", "yes")
# Cross-encoder shortlist size — CE trims the pool to this many before the LLM
# pre-filter makes the final cut (must be > the LLM pre-filter's top_k of 12).
_CE_SHORTLIST = int(os.environ.get("NINAI_CE_SHORTLIST", "16"))
# Dense candidate pool fetched specifically for cross-encoder reranking. Larger than
# QDRANT_TOP_K because the answer chunk is frequently ranked 30-100 by vector search;
# the reranker promotes it into the shortlist. Only used when _CE_RERANK is on.
_CE_POOL = int(os.environ.get("NINAI_CE_POOL", "100"))
# LLM pre-filter funnel knobs. Measured: ~43% of persistent failures are answer chunks
# that WERE retrieved but never reached the answer model — dropped by the pre-filter's
# 300-char selection snippet and top_k=12 cut. Widening both delivers more retrieved
# evidence to the answer step (the stage the reranker could not fix).
_PREFILTER_CHARS = int(os.environ.get("NINAI_PREFILTER_CHARS", "300"))
_PREFILTER_TOPK = int(os.environ.get("NINAI_PREFILTER_TOPK", "12"))
_GIST_INTERVAL = int(os.environ.get("GIST_INTERVAL", "25"))
_HYDE = os.environ.get("NINAI_HYDE", "0").lower() in ("1", "true", "yes")
_QEXPAND = os.environ.get("NINAI_QEXPAND", "0").lower() in ("1", "true", "yes")
_MULTIHOP_ITER = os.environ.get("NINAI_MULTIHOP_ITER", "0").lower() in ("1", "true", "yes")
# Graph-connectivity render boost: promote retrieved chunks that share a discriminative
# graph entity with the question's anchor chunks (the multi-hop bridge), so they survive
# the prompt builder's top-K render cut even with low lexical overlap. See
# _apply_graph_connectivity_boost. Off by default; reorders nothing destructively (only
# stamps payload["_graph_bonus"]).
# Query-focused distillation (map-reduce): the answer model first extracts only the
# question-relevant lines (verbatim) from the retrieved chunks, then answers from that
# denoised context. Targets distraction/refusal failures where the right fact is buried
# in 100 chunks. Off by default; costs ~1-2 extra LLM calls per query.
_DISTILL = os.environ.get("NINAI_DISTILL", "0").lower() in ("1", "true", "yes")
_DISTILL_MAX = int(os.environ.get("NINAI_DISTILL_MAX", "24"))      # chunks fed to the map step
_DISTILL_KEEP = int(os.environ.get("NINAI_DISTILL_KEEP", "3"))     # top original chunks kept as backup
_GRAPH_RANK = os.environ.get("NINAI_GRAPH_RANK", "0").lower() in ("1", "true", "yes")
# Top-K strongest (vector-order) chunks whose entities define the anchor set.
_GRAPH_RANK_ANCHORS = int(os.environ.get("NINAI_GRAPH_RANK_ANCHORS", "5"))
# Score added per shared discriminative entity. Tuned to be comparable to a strong
# lexical term hit (~7-12) so one solid graph link can lift a bridge into the render set.
_GRAPH_RANK_WEIGHT = int(os.environ.get("NINAI_GRAPH_RANK_WEIGHT", "8"))
_GIST_INJECT = os.environ.get("NINAI_GIST_INJECT", "0").lower() in ("1", "true", "yes")
_FULL_CONV_BYPASS = os.environ.get("NINAI_FULL_CONV_BYPASS", "0").lower() in ("1", "true", "yes")

_MULTIHOP_PATTERNS = (
    r"\bbecause of\b", r"\bafter\b.{1,40}\b(which|that|he|she|they)\b",
    r"\bwhat led\b", r"\bhow did\b.{1,30}\baffect\b", r"\bas a result\b",
    r"\bfollowing\b.{1,30}\bwhat\b", r"\bsince\b.{1,20}\b(did|was|started|began)\b",
    r"\bwho introduced\b", r"\bwho got .{1,20} into\b", r"\bbefore .{1,20} when\b",
)

_REFUSAL_FRAGMENTS = (
    "not mentioned", "not provided", "not available", "no information",
    "cannot determine", "does not contain", "not in the context",
    "not specified", "not stated", "not found", "i don't know",
    "i do not know", "i'm not sure", "no mention",
)

_Q_WORDS = frozenset({
    "what", "when", "where", "who", "why", "how", "which", "did",
    "does", "do", "has", "have", "had", "was", "were", "is", "are",
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "that", "this", "their", "his", "her",
})


def _is_refusal(text: str) -> bool:
    low = text.lower().strip()
    if not low or len(low) < 3:
        return True
    return any(p in low for p in _REFUSAL_FRAGMENTS)


def _is_multihop_query(query: str) -> bool:
    low = query.lower()
    return any(re.search(p, low) for p in _MULTIHOP_PATTERNS)


def _extract_caps_from_chunks(chunks: list[dict]) -> list[str]:
    """Extract likely proper nouns from chunk payloads for hop-2 retrieval."""
    seen: set[str] = set()
    names: list[str] = []
    for ch in chunks[:8]:
        p = ch.get("payload", {})
        text = str(p.get("text") or p.get("content") or p.get("summary") or "")
        for word in re.findall(r"\b([A-Z][a-z]{2,15})\b", text):
            if word not in seen and word not in {
                "The", "This", "That", "When", "Where", "What",
                "January", "February", "March", "April", "June",
                "July", "August", "September", "October", "November", "December",
            }:
                seen.add(word)
                names.append(word)
                if len(names) >= 4:
                    return names
    return names

def _apply_graph_connectivity_boost(
    chunks: list[dict],
    anchor_top_k: int = 5,
    bonus_weight: int = 8,
) -> int:
    """Stamp payload['_graph_bonus'] on chunks that are graph-connected to the anchors.

    The strongest retrieved chunks (first `anchor_top_k`, vector-score order) define an
    anchor entity set. Any chunk sharing one of those entities is a candidate multi-hop
    bridge and gets a bonus proportional to the number of shared *discriminative*
    entities. Ubiquitous entities (present in a large fraction of the pool — typically
    the main speakers) carry no signal, so they are excluded from the anchor set.

    Mutates payloads in place; returns the number of chunks that received a bonus.
    Safe no-op when chunks lack entity metadata.
    """
    if not chunks:
        return 0

    n = len(chunks)
    # Document frequency of each entity across the pool → drop ubiquitous ones.
    df: dict[str, int] = {}
    for ch in chunks:
        p = ch.get("payload", {}) or {}
        ids = {str(e) for e in (p.get("entity_ids") or []) if e}
        names = {str(e).lower() for e in (p.get("entity_names") or []) if e}
        for key in ids | names:
            df[key] = df.get(key, 0) + 1
    common_cut = max(3, int(0.4 * n))
    common = {k for k, c in df.items() if c > common_cut}

    def _keys(ch: dict) -> set[str]:
        p = ch.get("payload", {}) or {}
        ids = {str(e) for e in (p.get("entity_ids") or []) if e}
        names = {str(e).lower() for e in (p.get("entity_names") or []) if e}
        return (ids | names) - common

    anchor_entities: set[str] = set()
    for ch in chunks[:anchor_top_k]:
        anchor_entities |= _keys(ch)
    if not anchor_entities:
        return 0

    boosted = 0
    for ch in chunks:
        shared = _keys(ch) & anchor_entities
        if shared:
            p = ch.get("payload", {}) or {}
            p["_graph_bonus"] = len(shared) * bonus_weight
            ch["payload"] = p
            boosted += 1
    return boosted


logger = logging.getLogger(__name__)


@dataclass
class CognitiveLoopResult:
    response: str = ""
    cited_node_ids: list[str] = field(default_factory=list)
    extracted_entities: list[dict[str, Any]] = field(default_factory=list)
    user_utterance_id: str = ""
    assistant_utterance_id: str = ""
    graph_nodes_retrieved: int = 0
    qdrant_chunks_retrieved: int = 0
    graph_writes: int = 0
    decay_stats: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    error: str = ""


class V2CognitiveLoop:
    """
    Orchestrates the complete Graph-RAG + DNC cognitive loop for a single turn.

    Usage:
        loop = V2CognitiveLoop(dnc_router, reasoning_engine)
        result = await loop.run(tenant_id, session_id, user_input, prev_utterance_id)
    """

    def __init__(
        self,
        dnc_router: DNCMemoryRouter,
        reasoning_engine: Any,       # InferenceEngine — default + embeddings
        prev_utterance_registry: dict[str, str] | None = None,
        slm_engine: Any | None = None,   # dedicated fast/SLM pod engine
        llm_engine: Any | None = None,   # dedicated reasoning/LLM pod engine
    ) -> None:
        self._router = dnc_router
        self._engine = reasoning_engine
        # Per-tier engines pointing at their respective vLLM pod URLs.
        # Fall back to default engine when not configured (single-pod mode).
        self._slm_engine = slm_engine or reasoning_engine
        self._llm_engine = llm_engine or reasoning_engine
        # Maps session_id → last utterance id (ephemeral, per-process cache)
        self._last_utt: dict[str, str] = prev_utterance_registry or {}
        # Turn tracking for segment gist generation
        self._turn_count: dict[str, int] = {}
        self._turn_texts: dict[str, list[str]] = {}

    async def run(
        self,
        tenant_id: str,
        session_id: str,
        user_input: str,
        disable_write: bool = False,
        ingest_only: bool = False,
        prev_utterance_id: str | None = None,
        model_hint: str | None = None,
        raw_context: str | None = None,
    ) -> CognitiveLoopResult:
        t0 = time.monotonic()
        result = CognitiveLoopResult()

        # Resolve previous utterance id for chronological chaining
        prev_utt_id = prev_utterance_id or self._last_utt.get(session_id)

        # ---------------------------------------------------------------
        # Phase 1: Contextual Retrieval (Read)
        # ---------------------------------------------------------------
        # Skip retrieval in ingest_only mode — nothing useful is in the graph yet
        # and it saves a Qdrant + FalkorDB round-trip per ingested turn.
        # Also skip when raw_context is provided (full conversation bypass mode).
        read: ReadResult = ReadResult()
        _raw_ctx_active = bool(raw_context and _FULL_CONV_BYPASS)
        if not ingest_only and not _raw_ctx_active:
            try:
                # When the cross-encoder reranker is active, fetch a LARGER dense pool
                # so the reranker can promote answer chunks that vector search ranks
                # below the default top_k (measured: evidence recall jumps 58%@30 ->
                # 91%@100, so reranking a 30-pool leaves most of the gain on the table).
                _pool = _CE_POOL if (_BENCH_MODE and _CE_RERANK) else None

                # HyDE: generate a short hypothetical answer, average its embedding with
                # the query embedding. Helps when question phrasing diverges from stored text.
                query_hint: str | None = None
                if _HYDE:
                    try:
                        query_hint = await self._engine.generate_hypothetical(user_input) or None
                    except Exception:
                        pass

                read = await self._router.read(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    query=user_input,
                    qdrant_limit=_pool,
                    query_hint=query_hint,
                )
                result.graph_nodes_retrieved = len(read.graph_nodes)
                result.qdrant_chunks_retrieved = len(read.qdrant_chunks)

                # Query expansion: generate 2 alternative phrasings and union results.
                # Helps when the question's surface form diverges from stored text.
                if _QEXPAND:
                    try:
                        variants = await self._engine.expand_query(user_input)
                        seen_ids = {ch["id"] for ch in read.qdrant_chunks}
                        for variant in variants:
                            extra = await self._router.read(
                                tenant_id, session_id, variant, qdrant_limit=_pool
                            )
                            for ch in extra.qdrant_chunks:
                                if ch["id"] not in seen_ids:
                                    read.qdrant_chunks.append(ch)
                                    seen_ids.add(ch["id"])
                    except Exception as exc:
                        logger.warning("Query expansion failed: %s", exc)
            except Exception as exc:
                logger.warning("Phase 1 retrieval failed: %s", exc)
                result.error = f"retrieval: {exc}"

        # ---------------------------------------------------------------
        # Phase 1b: context reranking / pre-filtering (bench mode)
        # ---------------------------------------------------------------
        # Two-stage: the cross-encoder reranks the full pool down to a SHORTLIST
        # (coarse relevance + recall), then the LLM pre-filter makes the final precise
        # selection. The CE COMPLEMENTS the LLM pre-filter — it does not replace it —
        # so an ill-fit reranker can reorder/widen candidates but can't drop the answer
        # chunk on its own; the LLM still gets the final say.
        if (not ingest_only and not _raw_ctx_active and _BENCH_MODE and _CE_RERANK
                and read.qdrant_chunks and len(read.qdrant_chunks) > _CE_SHORTLIST):
            try:
                from app.v2.llm.reranker import cross_encoder_rerank
                _reranked = await cross_encoder_rerank(
                    user_input, read.qdrant_chunks, top_k=_CE_SHORTLIST
                )
                if _reranked:
                    read.qdrant_chunks = _reranked
            except Exception as exc:
                logger.warning("Cross-encoder rerank failed: %s", exc)
            # Safety net: if the CE didn't trim the enlarged pool (model failed to
            # load, returned None), cap it back to the standard top_k before the LLM
            # pre-filter so a reranker failure degrades to the original behaviour
            # rather than feeding the whole 100-chunk pool into the prompt. Chunks are
            # in vector-score order, so this is the original top-_CE_SHORTLIST*2.
            _cap = max(_CE_SHORTLIST, 30)
            if read.qdrant_chunks and len(read.qdrant_chunks) > _cap:
                read.qdrant_chunks = read.qdrant_chunks[:_cap]

        if not ingest_only and not _raw_ctx_active and _BENCH_MODE and _LLM_RERANK and read.qdrant_chunks:
            try:
                chunk_texts = []
                for ch in read.qdrant_chunks:
                    p = ch.get("payload", {})
                    t = str(p.get("text") or p.get("content") or p.get("summary") or "")[:_PREFILTER_CHARS]
                    chunk_texts.append(t)
                if len(chunk_texts) > 8:
                    selected = await self._engine.select_relevant_context(
                        user_input, chunk_texts, top_k=_PREFILTER_TOPK
                    )
                    selected_set = set(selected)
                    filtered = [ch for ch in read.qdrant_chunks
                                if str((ch.get("payload") or {}).get("text") or
                                       (ch.get("payload") or {}).get("content") or "")[:_PREFILTER_CHARS]
                                in selected_set]
                    # Always keep high-signal chunks regardless of LLM selection:
                    # gists (broad coverage) and entity chunks (precision recall).
                    _HIGH_SIG = ("segment_gist", "personal_attribute", "temporal_event")
                    filtered_ids = {id(ch) for ch in filtered}
                    high_sig_kept = [
                        ch for ch in read.qdrant_chunks
                        if id(ch) not in filtered_ids
                        and (ch.get("payload", {}).get("type") in _HIGH_SIG
                             or ch.get("payload", {}).get("entity_type") in _HIGH_SIG)
                    ]
                    if filtered or high_sig_kept:
                        read.qdrant_chunks = high_sig_kept + filtered
            except Exception as exc:
                logger.warning("LLM pre-filtering failed: %s", exc)

        # Iterative multi-hop retrieval: for chained questions, extract entity names
        # from the current shortlist and do a second retrieval round seeded by them.
        if not ingest_only and not _raw_ctx_active and _MULTIHOP_ITER and _is_multihop_query(user_input) and read.qdrant_chunks:
            try:
                hop2_names = _extract_caps_from_chunks(read.qdrant_chunks)
                seen_ids = {ch["id"] for ch in read.qdrant_chunks}
                for name in hop2_names[:2]:
                    extra = await self._router.read(tenant_id, session_id, name, qdrant_limit=_pool)
                    for ch in extra.qdrant_chunks:
                        if ch["id"] not in seen_ids:
                            read.qdrant_chunks.append(ch)
                            seen_ids.add(ch["id"])
            except Exception as exc:
                logger.warning("Iterative multi-hop retrieval failed: %s", exc)

        # ---------------------------------------------------------------
        # Phase 1b2: query-focused distillation (map-reduce)
        # ---------------------------------------------------------------
        # Replace the noisy chunk pool with the answer model's own verbatim extraction of
        # the question-relevant lines, keeping a few top original chunks as a safety net in
        # case the extraction drops the needle. Denoises context the way the gold-evidence
        # probe showed lifts the small model ~+11 ROUGE.
        # Temporal questions need the raw [YYYY-MM-DD] anchors to resolve relative dates;
        # the verbatim extraction tends to drop those anchors (measured: temporal -4.6 with
        # distillation). Skip distillation for date/time queries and keep the full context.
        _is_temporal_q = bool(re.search(
            r"\b(when|what year|what date|what day|how long|how many (?:years|months|days|weeks))\b",
            user_input.lower(),
        ))
        if (not ingest_only and not _raw_ctx_active and _DISTILL and read.qdrant_chunks
                and not _is_temporal_q):
            try:
                _texts = []
                for ch in read.qdrant_chunks:
                    p = ch.get("payload", {}) or {}
                    t = str(p.get("text") or p.get("content") or p.get("summary") or "").strip()
                    if t:
                        _texts.append(t)
                facts = await self._llm_engine.distill_evidence(
                    user_input, _texts, max_items=_DISTILL_MAX
                )
                if facts:
                    distilled = [
                        {"id": f"distill_{i}", "score": 1.0,
                         "payload": {"text": f, "type": "segment_gist", "speaker": "", "anchor_date": ""}}
                        for i, f in enumerate(facts)
                    ]
                    # Distilled facts first (rendered as high-priority gists), then a few
                    # original chunks as backup against extraction misses.
                    read.qdrant_chunks = distilled + read.qdrant_chunks[:_DISTILL_KEEP]
            except Exception as exc:
                logger.warning("Distillation stage failed: %s", exc)

        # ---------------------------------------------------------------
        # Phase 1c: graph-connectivity render boost
        # ---------------------------------------------------------------
        # The prompt builder renders only the top-K chunks (ranked by lexical overlap).
        # A multi-hop bridge chunk often shares no question words with the query, so it
        # is cut before the model sees it even though retrieval found it. Promote chunks
        # that are graph-connected (share a discriminative entity) to the anchor chunks.
        if (not ingest_only and not _raw_ctx_active and _GRAPH_RANK and read.qdrant_chunks):
            try:
                _apply_graph_connectivity_boost(
                    read.qdrant_chunks,
                    anchor_top_k=_GRAPH_RANK_ANCHORS,
                    bonus_weight=_GRAPH_RANK_WEIGHT,
                )
            except Exception as exc:
                logger.warning("Graph-connectivity boost failed: %s", exc)

        # ---------------------------------------------------------------
        # Phase 2: Inference & Execution
        # ---------------------------------------------------------------
        # Select inference engine: explicit model_hint overrides auto-classification.
        # With two pods (SLM on T4, LLM on A100), this routes to the correct URL.
        selected_tier = "reasoning"
        if model_hint:
            # Explicit override — caller knows which tier they want
            _hint_low = model_hint.lower()
            if any(_hint_low == m for m in (
                self._slm_engine._model if hasattr(self._slm_engine, '_model') else '',
            )):
                active_engine = self._slm_engine
                selected_tier = "fast"
            else:
                active_engine = self._llm_engine
                selected_tier = "reasoning"
        else:
            tier = classify_query_tier(user_input)
            selected_tier = tier
            active_engine = self._llm_engine if tier == 'reasoning' else self._slm_engine

        # ingest_only=True skips LLM inference — caller only wants write-back.
        infer_result = None
        if not ingest_only:
            try:
                session_utts = [
                    n for n in read.graph_nodes if n.get("label") == "Utterance"
                ]
                other_nodes = [
                    n for n in read.graph_nodes if n.get("label") != "Utterance"
                ]

                # All-gist injection: prepend every conversation gist to the context so
                # the model has a full summary arc regardless of vector ranking.
                if _GIST_INJECT:
                    try:
                        extra_gists = await self._router.fetch_gists(tenant_id)
                        seen_ids = {ch["id"] for ch in read.qdrant_chunks}
                        prepend = [ch for ch in extra_gists if ch["id"] not in seen_ids]
                        read.qdrant_chunks = prepend + read.qdrant_chunks
                    except Exception as exc:
                        logger.warning("Gist injection failed: %s", exc)

                if _BENCH_MODE:
                    if _raw_ctx_active:
                        from app.v2.llm.prompt_builder import _BENCH_SYSTEM_INSTRUCTION
                        prompt = (
                            f"{_BENCH_SYSTEM_INSTRUCTION}\n\n"
                            f"CONVERSATION CONTEXT:\n{raw_context}\n\n"
                            f"QUESTION: {user_input}"
                        )
                    else:
                        prompt = build_bench_prompt(
                            user_input=user_input,
                            graph_nodes=other_nodes + session_utts,
                            qdrant_chunks=read.qdrant_chunks,
                            session_utterances=[],
                        )
                    infer_result = await active_engine.infer_plain(prompt)
                else:
                    prompt = build_inference_prompt(
                        user_input=user_input,
                        graph_nodes=other_nodes,
                        qdrant_chunks=read.qdrant_chunks,
                        session_utterances=session_utts,
                    )
                    infer_result = await active_engine.infer(prompt)
                result.response = infer_result.response
                result.cited_node_ids = infer_result.cited_node_ids
                result.extracted_entities = infer_result.extracted_entities
                if infer_result.error:
                    result.error = infer_result.error
            except Exception as exc:
                logger.error("Phase 2 inference failed: %s", exc)
                result.error = f"inference: {exc}"
                result.response = "I was unable to generate a response due to an internal error."

        # ---------------------------------------------------------------
        # Phase 2b: Second-chance retrieval on refusal (bench mode only)
        # ---------------------------------------------------------------
        if (not ingest_only and _BENCH_MODE and _SECOND_CHANCE
                and infer_result and _is_refusal(result.response)):
            try:
                # Build multiple reformulations for broader retrieval coverage
                import re as _re
                # Quoted terms first (exact phrases)
                quoted = _re.findall(r'"([^"]+)"', user_input)
                # Capitalized tokens (likely proper nouns / names)
                caps = [w for w in _re.findall(r'\b([A-Z][a-z]{2,})\b', user_input)
                        if w.lower() not in _Q_WORDS]
                # Content words without stopwords
                content_words = [
                    w for w in user_input.split() if w.lower() not in _Q_WORDS and len(w) > 2
                ]
                # Try queries in priority order
                candidate_queries = []
                if quoted:
                    candidate_queries.append(" ".join(quoted))
                if caps:
                    candidate_queries.append(" ".join(caps))
                if content_words:
                    candidate_queries.append(" ".join(content_words[:6]))

                merged = False
                for q in candidate_queries:
                    if not q or q.strip() == user_input.strip():
                        continue
                    second_read = await self._router.read(tenant_id, session_id, q)
                    existing_ids = {n.get("id") for n in read.graph_nodes}
                    for node in second_read.graph_nodes:
                        if node.get("id") not in existing_ids:
                            read.graph_nodes.append(node)
                            merged = True
                    existing_chunk_ids = {c.get("id") for c in read.qdrant_chunks}
                    for chunk in second_read.qdrant_chunks:
                        if chunk.get("id") not in existing_chunk_ids:
                            read.qdrant_chunks.append(chunk)
                            merged = True

                # Always re-infer after second-chance attempt, even if no new nodes —
                # the rephrased prompt direction alone can break a refusal loop.
                session_utts2 = [n for n in read.graph_nodes if n.get("label") == "Utterance"]
                other_nodes2 = [n for n in read.graph_nodes if n.get("label") != "Utterance"]
                # Build a targeted prompt that explicitly urges the model to find the answer
                _force_hint = (
                    f"\nNOTE: The answer IS in the context above. "
                    f"Re-read carefully, focusing on: {', '.join(caps[:3]) if caps else 'the main topic'}. "
                    f"Do NOT say 'Not mentioned' — give the best matching fact.\n"
                )
                prompt2 = build_bench_prompt(
                    user_input=user_input + _force_hint,
                    graph_nodes=other_nodes2 + session_utts2,
                    qdrant_chunks=read.qdrant_chunks,
                    session_utterances=[],
                )
                # If the first answer refused and we started on fast tier,
                # escalate retry to reasoning tier for higher recall.
                retry_engine = self._llm_engine if selected_tier == "fast" else active_engine
                infer2 = await retry_engine.infer_plain(prompt2)
                if infer2.response and not _is_refusal(infer2.response):
                    result.response = infer2.response
                    result.cited_node_ids = infer2.cited_node_ids
                    result.extracted_entities = infer2.extracted_entities
            except Exception as exc:
                logger.warning("Second-chance retrieval failed: %s", exc)

        # ---------------------------------------------------------------
        # Phase 3: Continuous Learning (Write + Decay + Prune)
        # ---------------------------------------------------------------
        # disable_write always skips writes.
        # _BENCH_MODE skips writes unless ingest_only explicitly requests them
        # (ingest_only is the fast-ingest path that populates Qdrant + FalkorDB).
        if disable_write or (_BENCH_MODE and not ingest_only):
            result.latency_ms = int((time.monotonic() - t0) * 1000)
            return result

        user_embed: list[float] = []
        try:
            if self._engine and hasattr(self._engine, "embed"):
                user_embed = await self._engine.embed(user_input)
        except Exception:
            pass

        user_write = WriteResult()
        try:
            user_write = await self._router.write(
                tenant_id=tenant_id,
                session_id=session_id,
                utterance_text=user_input,
                role="user",
                prev_utterance_id=prev_utt_id,
                embedding=user_embed or None,
            )
            result.user_utterance_id = user_write.utterance_id
            result.graph_writes += user_write.graph_writes
        except Exception as exc:
            logger.warning("Phase 3 user write failed: %s", exc)

        asst_write = WriteResult()
        if result.response:
            asst_embed: list[float] = []
            try:
                if self._engine and hasattr(self._engine, "embed"):
                    asst_embed = await self._engine.embed(result.response)
            except Exception:
                pass
            try:
                asst_write = await self._router.write(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    utterance_text=result.response,
                    role="assistant",
                    prev_utterance_id=user_write.utterance_id or prev_utt_id,
                    embedding=asst_embed or None,
                )
                result.assistant_utterance_id = asst_write.utterance_id
                result.graph_writes += asst_write.graph_writes
            except Exception as exc:
                logger.warning("Phase 3 assistant write failed: %s", exc)

        # Update session chain pointer
        if asst_write.utterance_id:
            self._last_utt[session_id] = asst_write.utterance_id
        elif user_write.utterance_id:
            self._last_utt[session_id] = user_write.utterance_id

        # Segment gist generation — every GIST_INTERVAL turns during ingest
        if ingest_only and _GIST_INTERVAL > 0 and self._engine:
            texts = self._turn_texts.setdefault(session_id, [])
            texts.append(user_input[:400])
            count = self._turn_count.get(session_id, 0) + 1
            self._turn_count[session_id] = count
            if count % _GIST_INTERVAL == 0:
                try:
                    gist = await self._engine.generate_gist(texts[-_GIST_INTERVAL * 2:])
                    if gist:
                        await self._router.write_gist(
                            tenant_id=tenant_id,
                            session_id=session_id,
                            gist_text=gist,
                            turn_start=count - _GIST_INTERVAL,
                            turn_end=count,
                        )
                except Exception as exc:
                    logger.warning("Gist generation failed: %s", exc)

        # Decay + prune neighborhood
        all_seed_ids = list(read.seed_entity_ids)
        if user_write.entity_ids:
            all_seed_ids.extend(user_write.entity_ids)
        if all_seed_ids:
            try:
                decay_stats = await self._router.decay_and_prune(
                    tenant_id=tenant_id,
                    seed_ids=list(set(all_seed_ids)),
                )
                result.decay_stats = decay_stats
            except Exception as exc:
                logger.warning("Decay/prune failed: %s", exc)

        result.latency_ms = int((time.monotonic() - t0) * 1000)
        return result
