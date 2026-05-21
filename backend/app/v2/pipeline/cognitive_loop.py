"""
V2 Cognitive Loop — Three-Phase Execution Pipeline

Phase 1 (Read/Retrieve):
  - DNCMemoryRouter.read() → graph subgraph + Qdrant episodic chunks

Phase 2 (Infer):
  - prompt_builder.build_inference_prompt() → assembled context
  - OllamaReasoningEngine.infer() → response + cited_node_ids + extracted_entities

Phase 3 (Learn/Write-back):
  - DNCMemoryRouter.write() → Utterance node + Entity nodes + edges
  - DNCMemoryRouter.write() for assistant response (write-back)
  - decay_and_prune() on the local seed neighborhood

This module is the only place that orchestrates all three phases.
Callers (API endpoints) provide tenant_id, session_id, and user_input.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.v2.llm.prompt_builder import build_inference_prompt
from app.v2.memory.dnc_router import DNCMemoryRouter, ReadResult, WriteResult

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
        reasoning_engine: Any,       # OllamaReasoningEngine
        prev_utterance_registry: dict[str, str] | None = None,
    ) -> None:
        self._router = dnc_router
        self._engine = reasoning_engine
        # Maps session_id → last utterance id (ephemeral, per-process cache)
        self._last_utt: dict[str, str] = prev_utterance_registry or {}

    async def run(
        self,
        tenant_id: str,
        session_id: str,
        user_input: str,
        prev_utterance_id: str | None = None,
    ) -> CognitiveLoopResult:
        t0 = time.monotonic()
        result = CognitiveLoopResult()

        # Resolve previous utterance id for chronological chaining
        prev_utt_id = prev_utterance_id or self._last_utt.get(session_id)

        # ---------------------------------------------------------------
        # Phase 1: Contextual Retrieval (Read)
        # ---------------------------------------------------------------
        read: ReadResult = ReadResult()
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
        # Phase 2: Inference & Execution
        # ---------------------------------------------------------------
        infer_result = None
        try:
            # Separate session utterances from other graph nodes for the prompt
            session_utts = [
                n for n in read.graph_nodes if n.get("label") == "Utterance"
            ]
            other_nodes = [
                n for n in read.graph_nodes if n.get("label") != "Utterance"
            ]
            prompt = build_inference_prompt(
                user_input=user_input,
                graph_nodes=other_nodes,
                qdrant_chunks=read.qdrant_chunks,
                session_utterances=session_utts,
            )
            infer_result = await self._engine.infer(prompt)
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
        # Phase 3: Continuous Learning (Write + Decay + Prune)
        # ---------------------------------------------------------------
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
