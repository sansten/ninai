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
import time
from dataclasses import dataclass, field
from typing import Any

from app.v2.llm.prompt_builder import build_bench_prompt, build_inference_prompt
from app.v2.llm.query_classifier import classify_query_tier
from app.v2.memory.dnc_router import DNCMemoryRouter, ReadResult, WriteResult

_BENCH_MODE = os.environ.get("NINAI_BENCH_MODE", "0").lower() in ("1", "true", "yes")
_SECOND_CHANCE = os.environ.get("NINAI_SECOND_CHANCE", "1").lower() in ("1", "true", "yes")
_LLM_RERANK = os.environ.get("NINAI_LLM_RERANK", "1").lower() in ("1", "true", "yes")
_GIST_INTERVAL = int(os.environ.get("GIST_INTERVAL", "25"))

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
        read: ReadResult = ReadResult()
        if not ingest_only:
            try:
                read = await self._router.read(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    query=user_input,
                )
                result.graph_nodes_retrieved = len(read.graph_nodes)
                result.qdrant_chunks_retrieved = len(read.qdrant_chunks)
            except Exception as exc:
                logger.warning("Phase 1 retrieval failed: %s", exc)
                result.error = f"retrieval: {exc}"

        # ---------------------------------------------------------------
        # Phase 1b: LLM context pre-filtering (bench mode, 3B model)
        # ---------------------------------------------------------------
        if not ingest_only and _BENCH_MODE and _LLM_RERANK and read.qdrant_chunks:
            try:
                chunk_texts = []
                for ch in read.qdrant_chunks:
                    p = ch.get("payload", {})
                    t = str(p.get("text") or p.get("content") or p.get("summary") or "")[:300]
                    chunk_texts.append(t)
                if len(chunk_texts) > 8:
                    selected = await self._engine.select_relevant_context(
                        user_input, chunk_texts, top_k=12
                    )
                    selected_set = set(selected)
                    filtered = [ch for ch in read.qdrant_chunks
                                if str((ch.get("payload") or {}).get("text") or
                                       (ch.get("payload") or {}).get("content") or "")[:300]
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

        # ---------------------------------------------------------------
        # Phase 2: Inference & Execution
        # ---------------------------------------------------------------
        # Select inference engine: explicit model_hint overrides auto-classification.
        # With two pods (SLM on T4, LLM on A100), this routes to the correct URL.
        if model_hint:
            # Explicit override — caller knows which tier they want
            _hint_low = model_hint.lower()
            if any(_hint_low == m for m in (
                self._slm_engine._model if hasattr(self._slm_engine, '_model') else '',
            )):
                active_engine = self._slm_engine
            else:
                active_engine = self._llm_engine
        else:
            tier = classify_query_tier(user_input)
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
                if _BENCH_MODE:
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
                infer2 = await active_engine.infer_plain(prompt2)
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
