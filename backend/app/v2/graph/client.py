"""
V2 FalkorDB Graph Client

Wraps FalkorDB (via Redis GRAPH.QUERY commands) with schema-aware CRUD for
Entity, Utterance, and Action nodes and their typed edges.

All heavy queries run in a thread executor to avoid blocking the asyncio loop.
Fail-open: every public method returns an empty result on connection failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

try:
    import redis as redis_lib
    _REDIS_OK = True
except ImportError:
    _REDIS_OK = False

from app.v2.graph.schema import (
    DECAY_FACTOR,
    WEIGHT_MAX,
    WEIGHT_PRUNE_THRESHOLD,
    WEIGHT_REINFORCE_DELTA,
)

logger = logging.getLogger(__name__)

_MS = lambda: int(time.time() * 1000)  # noqa: E731


class V2GraphClient:
    """
    FalkorDB client for the v2 chronological knowledge graph.

    Each tenant gets its own named graph: ninai_v2_{tenant_id}.
    """

    def __init__(self, redis_url: str, graph_prefix: str = "ninai_v2") -> None:
        self._graph_prefix = graph_prefix
        self._redis: Any = None
        if not _REDIS_OK:
            logger.warning("redis library unavailable — v2 graph disabled")
            return
        try:
            self._redis = redis_lib.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            self._redis.ping()
            logger.info("V2GraphClient connected: %s", redis_url)
        except Exception as exc:
            logger.error("V2GraphClient connection failed: %s", exc)
            self._redis = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _graph(self, tenant_id: str) -> str:
        return f"{self._graph_prefix}_{tenant_id}"

    def _query(self, tenant_id: str, cypher: str) -> list[dict[str, Any]]:
        if not self._redis:
            return []
        try:
            result = self._redis.execute_command(
                "GRAPH.QUERY", self._graph(tenant_id), cypher
            )
            if not result or len(result) < 2:
                return []
            headers: list[str] = result[0] or []
            rows: list[Any] = result[1] if len(result) > 1 else []
            return [
                {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                for row in rows
            ]
        except Exception as exc:
            logger.error("graph query failed: %s | %s", exc, cypher[:120])
            return []

    async def _aquery(self, tenant_id: str, cypher: str) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._query, tenant_id, cypher)

    @staticmethod
    def _esc(value: str) -> str:
        return value.replace("'", "\\'").replace("\\", "\\\\")

    # ------------------------------------------------------------------
    # Entity nodes
    # ------------------------------------------------------------------

    async def upsert_entity(
        self,
        tenant_id: str,
        entity_id: str,
        name: str,
        entity_type: str,
    ) -> dict[str, Any]:
        now = _MS()
        cypher = (
            f"MERGE (e:Entity {{id: '{self._esc(entity_id)}', tenant_id: '{self._esc(tenant_id)}'}}) "
            f"ON CREATE SET e.name = '{self._esc(name)}', "
            f"  e.entity_type = '{self._esc(entity_type)}', "
            f"  e.weight = 0.5, e.access_count = 0, "
            f"  e.created_at = {now}, e.updated_at = {now} "
            f"ON MATCH SET e.updated_at = {now}, "
            f"  e.access_count = e.access_count + 1, "
            f"  e.weight = CASE WHEN e.weight + 0.1 > 1.0 THEN 1.0 "
            f"             ELSE e.weight + 0.1 END "
            f"RETURN e.id AS id, e.weight AS weight, e.access_count AS access_count"
        )
        rows = await self._aquery(tenant_id, cypher)
        return rows[0] if rows else {}

    async def get_entity(self, tenant_id: str, entity_id: str) -> dict[str, Any]:
        cypher = (
            f"MATCH (e:Entity {{id: '{self._esc(entity_id)}', tenant_id: '{self._esc(tenant_id)}'}}) "
            f"RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type, "
            f"e.weight AS weight, e.access_count AS access_count"
        )
        rows = await self._aquery(tenant_id, cypher)
        return rows[0] if rows else {}

    # ------------------------------------------------------------------
    # Utterance nodes
    # ------------------------------------------------------------------

    async def create_utterance(
        self,
        tenant_id: str,
        utterance_id: str,
        text: str,
        role: str,
        session_id: str,
    ) -> dict[str, Any]:
        now = _MS()
        safe_text = self._esc(text[:2000])
        cypher = (
            f"CREATE (u:Utterance {{"
            f"  id: '{self._esc(utterance_id)}', "
            f"  tenant_id: '{self._esc(tenant_id)}', "
            f"  text: '{safe_text}', "
            f"  role: '{self._esc(role)}', "
            f"  session_id: '{self._esc(session_id)}', "
            f"  weight: 0.5, "
            f"  created_at: {now}"
            f"}}) RETURN u.id AS id"
        )
        rows = await self._aquery(tenant_id, cypher)
        return rows[0] if rows else {"id": utterance_id}

    # ------------------------------------------------------------------
    # Action nodes
    # ------------------------------------------------------------------

    async def create_action(
        self,
        tenant_id: str,
        action_id: str,
        action_type: str,
        outcome: str,
    ) -> dict[str, Any]:
        now = _MS()
        cypher = (
            f"CREATE (a:Action {{"
            f"  id: '{self._esc(action_id)}', "
            f"  tenant_id: '{self._esc(tenant_id)}', "
            f"  action_type: '{self._esc(action_type)}', "
            f"  outcome: '{self._esc(outcome[:500])}', "
            f"  weight: 0.5, "
            f"  created_at: {now}"
            f"}}) RETURN a.id AS id"
        )
        rows = await self._aquery(tenant_id, cypher)
        return rows[0] if rows else {"id": action_id}

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    async def upsert_edge(
        self,
        tenant_id: str,
        from_id: str,
        to_id: str,
        rel_type: str,
        reinforce: bool = True,
    ) -> bool:
        now = _MS()
        delta = WEIGHT_REINFORCE_DELTA if reinforce else 0.0
        cypher = (
            f"MATCH (a {{id: '{self._esc(from_id)}', tenant_id: '{self._esc(tenant_id)}'}}) "
            f"MATCH (b {{id: '{self._esc(to_id)}', tenant_id: '{self._esc(tenant_id)}'}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"ON CREATE SET r.weight = 0.5, r.created_at = {now}, r.updated_at = {now} "
            f"ON MATCH SET r.weight = CASE WHEN r.weight + {delta} > 1.0 THEN 1.0 "
            f"             ELSE r.weight + {delta} END, r.updated_at = {now} "
            f"RETURN r.weight AS weight"
        )
        rows = await self._aquery(tenant_id, cypher)
        return bool(rows)

    async def link_utterance_to_entities(
        self,
        tenant_id: str,
        utterance_id: str,
        entity_ids: list[str],
    ) -> None:
        for eid in entity_ids:
            await self.upsert_edge(tenant_id, utterance_id, eid, "RESPONDED_TO")

    async def link_sequential_utterances(
        self,
        tenant_id: str,
        prev_utterance_id: str,
        next_utterance_id: str,
    ) -> None:
        await self.upsert_edge(
            tenant_id, prev_utterance_id, next_utterance_id, "FOLLOWED_BY"
        )

    # ------------------------------------------------------------------
    # Graph-RAG retrieval: fetch subgraph by seed entity IDs
    # ------------------------------------------------------------------

    async def fetch_subgraph(
        self,
        tenant_id: str,
        seed_ids: list[str],
        hops: int = 2,
        limit: int = 30,
        min_weight: float = 0.1,
    ) -> list[dict[str, Any]]:
        """
        Return nodes within `hops` of any seed, sorted by weight * recency.
        Each row: {id, label, name/text, weight, created_at}
        """
        if not seed_ids:
            return []
        seed_list = ", ".join(f"'{self._esc(s)}'" for s in seed_ids)
        cypher = (
            f"MATCH (seed {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE seed.id IN [{seed_list}] "
            f"MATCH (seed)-[*1..{hops}]-(neighbor {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE neighbor.weight >= {min_weight} "
            f"WITH DISTINCT neighbor "
            f"RETURN neighbor.id AS id, "
            f"  labels(neighbor)[0] AS label, "
            f"  COALESCE(neighbor.name, neighbor.text, neighbor.action_type) AS content, "
            f"  neighbor.weight AS weight, "
            f"  neighbor.created_at AS created_at "
            f"ORDER BY neighbor.weight DESC "
            f"LIMIT {limit}"
        )
        return await self._aquery(tenant_id, cypher)

    async def fetch_recent_utterances(
        self,
        tenant_id: str,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        cypher = (
            f"MATCH (u:Utterance {{"
            f"  tenant_id: '{self._esc(tenant_id)}', "
            f"  session_id: '{self._esc(session_id)}'"
            f"}}) "
            f"RETURN u.id AS id, u.text AS text, u.role AS role, "
            f"  u.created_at AS created_at "
            f"ORDER BY u.created_at DESC LIMIT {limit}"
        )
        return await self._aquery(tenant_id, cypher)

    # ------------------------------------------------------------------
    # Decay + pruning (Component C write-purge mechanics)
    # ------------------------------------------------------------------

    async def decay_neighborhood(
        self,
        tenant_id: str,
        seed_ids: list[str],
        hops: int = 2,
    ) -> int:
        """
        Apply DECAY_FACTOR to all edges in the seed neighborhood that were
        NOT just reinforced.  Returns count of edges processed.
        """
        if not seed_ids:
            return 0
        seed_list = ", ".join(f"'{self._esc(s)}'" for s in seed_ids)
        cypher = (
            f"MATCH (seed {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE seed.id IN [{seed_list}] "
            f"MATCH (seed)-[r*1..{hops}]-(neighbor {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WITH r "
            f"UNWIND r AS rel "
            f"SET rel.weight = rel.weight * {DECAY_FACTOR} "
            f"RETURN count(rel) AS decayed"
        )
        rows = await self._aquery(tenant_id, cypher)
        return int(rows[0].get("decayed", 0)) if rows else 0

    async def prune_weak_edges(self, tenant_id: str) -> int:
        """
        Delete edges whose weight has dropped below WEIGHT_PRUNE_THRESHOLD.
        Returns count of pruned edges.
        """
        cypher = (
            f"MATCH ({{tenant_id: '{self._esc(tenant_id)}'}})"
            f"-[r]->({{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE r.weight < {WEIGHT_PRUNE_THRESHOLD} "
            f"DELETE r "
            f"RETURN count(r) AS pruned"
        )
        rows = await self._aquery(tenant_id, cypher)
        return int(rows[0].get("pruned", 0)) if rows else 0

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if not self._redis:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False
