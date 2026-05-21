"""
V2 DNC-Inspired Memory Router (Component C)

Acts as the bridging layer between the Reasoning Engine (Ollama) and the
Chronological Knowledge Graph (FalkorDB).

Read Weighting:
  1. Embed query with Ollama (nomic-embed-text)
  2. Dense search in Qdrant → top-K chunk ids (episodic memory)
  3. FalkorDB subgraph traversal seeded from those ids → contextual nodes
  4. Rank by (weight * recency_score), return top-M nodes as prompt context

Write Weighting:
  1. Extract entities from new interaction via Ollama (structured JSON)
  2. MERGE each entity into FalkorDB (create-or-reinforce)
  3. Create Utterance node for the interaction
  4. Link utterance → entities via RESPONDED_TO edges
  5. Upsert interaction embedding into Qdrant

Purge:
  1. Decay neighborhood edges after write
  2. Delete edges below WEIGHT_PRUNE_THRESHOLD (called periodically)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReadResult:
    graph_nodes: list[dict[str, Any]] = field(default_factory=list)
    qdrant_chunks: list[dict[str, Any]] = field(default_factory=list)
    seed_entity_ids: list[str] = field(default_factory=list)


@dataclass
class WriteResult:
    utterance_id: str = ""
    entity_ids: list[str] = field(default_factory=list)
    qdrant_point_id: str = ""
    graph_writes: int = 0


class DNCMemoryRouter:
    """
    Stateless DNC memory router.  Instantiated once per request; does not
    hold any cross-request state — all state lives in FalkorDB and Qdrant.
    """

    def __init__(
        self,
        graph_client: Any,           # V2GraphClient
        qdrant_service: Any | None,  # QdrantService (optional — graceful degrade)
        embedding_fn: Any | None,    # async callable(text) -> list[float]
        entity_extractor: Any | None, # async callable(text) -> list[dict]
        top_k_qdrant: int = 10,
        top_m_graph: int = 20,
        graph_hops: int = 2,
    ) -> None:
        self._graph = graph_client
        self._qdrant = qdrant_service
        self._embed = embedding_fn
        self._extract_entities = entity_extractor
        self._top_k = top_k_qdrant
        self._top_m = top_m_graph
        self._hops = graph_hops

    # ------------------------------------------------------------------
    # Read weighting — Phase 1
    # ------------------------------------------------------------------

    async def read(self, tenant_id: str, session_id: str, query: str) -> ReadResult:
        result = ReadResult()

        # 1. Dense retrieval from Qdrant (episodic memory)
        qdrant_chunks: list[dict[str, Any]] = []
        seed_ids: list[str] = []
        if self._embed and self._qdrant:
            try:
                embedding = await self._embed(query)
                raw = await self._qdrant_search(tenant_id, embedding)
                qdrant_chunks = raw
                # Extract entity/memory ids that appear in Qdrant payload
                for chunk in qdrant_chunks:
                    payload = chunk.get("payload", {})
                    for key in ("entity_id", "memory_id", "id"):
                        val = payload.get(key)
                        if val:
                            seed_ids.append(str(val))
            except Exception as exc:
                logger.warning("Qdrant read failed: %s", exc)

        result.qdrant_chunks = qdrant_chunks
        result.seed_entity_ids = seed_ids

        # 2. Graph traversal seeded from Qdrant hits
        if seed_ids and self._graph.is_available():
            try:
                nodes = await self._graph.fetch_subgraph(
                    tenant_id=tenant_id,
                    seed_ids=seed_ids,
                    hops=self._hops,
                    limit=self._top_m,
                )
                result.graph_nodes = nodes
            except Exception as exc:
                logger.warning("Graph read failed: %s", exc)

        # 3. Always include recent session utterances for context continuity
        if self._graph.is_available():
            try:
                recent = await self._graph.fetch_recent_utterances(
                    tenant_id, session_id, limit=6
                )
                # Merge with graph_nodes, dedup by id
                existing_ids = {n.get("id") for n in result.graph_nodes}
                for node in recent:
                    if node.get("id") not in existing_ids:
                        result.graph_nodes.append(node)
            except Exception as exc:
                logger.warning("Recent utterances fetch failed: %s", exc)

        return result

    async def _qdrant_search(
        self, tenant_id: str, embedding: list[float]
    ) -> list[dict[str, Any]]:
        if not self._qdrant:
            return []
        try:
            hits = await self._qdrant.search(
                collection_name="memories",
                query_vector=embedding,
                limit=self._top_k,
                query_filter={"must": [{"key": "tenant_id", "match": {"value": tenant_id}}]},
            )
            return [
                {"id": str(h.id), "score": h.score, "payload": h.payload or {}}
                for h in (hits or [])
            ]
        except Exception as exc:
            logger.warning("Qdrant search error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Write weighting — Phase 3
    # ------------------------------------------------------------------

    async def write(
        self,
        tenant_id: str,
        session_id: str,
        utterance_text: str,
        role: str,
        prev_utterance_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> WriteResult:
        result = WriteResult()

        if not self._graph.is_available():
            return result

        # 1. Extract entities from utterance
        entities: list[dict[str, Any]] = []
        if self._extract_entities:
            try:
                entities = await self._extract_entities(utterance_text) or []
            except Exception as exc:
                logger.warning("Entity extraction failed: %s", exc)

        # 2. Upsert entity nodes into graph
        entity_ids: list[str] = []
        for ent in entities:
            eid = ent.get("id") or str(uuid.uuid4())
            name = ent.get("name", "unknown")
            etype = ent.get("type", "concept")
            try:
                await self._graph.upsert_entity(tenant_id, eid, name, etype)
                entity_ids.append(eid)
                result.graph_writes += 1
            except Exception as exc:
                logger.warning("Entity upsert failed: %s", exc)

        result.entity_ids = entity_ids

        # 3. Create Utterance node
        utt_id = str(uuid.uuid4())
        try:
            await self._graph.create_utterance(
                tenant_id, utt_id, utterance_text, role, session_id
            )
            result.utterance_id = utt_id
            result.graph_writes += 1
        except Exception as exc:
            logger.warning("Utterance node creation failed: %s", exc)
            result.utterance_id = utt_id

        # 4. Link utterance → entities
        if entity_ids:
            try:
                await self._graph.link_utterance_to_entities(
                    tenant_id, utt_id, entity_ids
                )
                result.graph_writes += len(entity_ids)
            except Exception as exc:
                logger.warning("Utterance→entity linking failed: %s", exc)

        # 5. Chain utterances chronologically
        if prev_utterance_id:
            try:
                await self._graph.link_sequential_utterances(
                    tenant_id, prev_utterance_id, utt_id
                )
                result.graph_writes += 1
            except Exception as exc:
                logger.warning("Sequential utterance link failed: %s", exc)

        # 6. Upsert into Qdrant (episodic vector index)
        if embedding and self._qdrant:
            try:
                point_id = str(uuid.uuid4())
                await self._qdrant_upsert(
                    tenant_id=tenant_id,
                    point_id=point_id,
                    embedding=embedding,
                    payload={
                        "utterance_id": utt_id,
                        "session_id": session_id,
                        "role": role,
                        "text": utterance_text[:500],
                        "tenant_id": tenant_id,
                        "entity_ids": entity_ids,
                        "created_at": int(time.time()),
                    },
                )
                result.qdrant_point_id = point_id
            except Exception as exc:
                logger.warning("Qdrant upsert failed: %s", exc)

        return result

    async def _qdrant_upsert(
        self,
        tenant_id: str,
        point_id: str,
        embedding: list[float],
        payload: dict[str, Any],
    ) -> None:
        if not self._qdrant:
            return
        try:
            from qdrant_client.models import PointStruct
            await self._qdrant.upsert(
                collection_name="memories",
                points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
            )
        except Exception as exc:
            logger.warning("Qdrant upsert internal error: %s", exc)

    # ------------------------------------------------------------------
    # Decay / purge — Phase 3 tail
    # ------------------------------------------------------------------

    async def decay_and_prune(
        self,
        tenant_id: str,
        seed_ids: list[str],
    ) -> dict[str, int]:
        decayed = 0
        pruned = 0
        if not self._graph.is_available():
            return {"decayed": decayed, "pruned": pruned}
        if not seed_ids:
            return {"decayed": decayed, "pruned": pruned}
        try:
            decayed = await self._graph.decay_neighborhood(tenant_id, seed_ids)
        except Exception as exc:
            logger.warning("Decay step failed: %s", exc)
        try:
            pruned = await self._graph.prune_weak_edges(tenant_id)
        except Exception as exc:
            logger.warning("Prune step failed: %s", exc)
        return {"decayed": decayed, "pruned": pruned}
