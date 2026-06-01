"""
V2 FalkorDB Graph Client

Wraps FalkorDB (via Redis GRAPH.QUERY commands) with schema-aware CRUD for
Entity, Utterance, and Action nodes and their typed edges.

All heavy queries run in a thread executor to avoid blocking the asyncio loop.
Fail-open: every public method returns an empty result on connection failure.
"""

from __future__ import annotations

import asyncio
import logging
import re
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
        return value.replace("\\", "\\\\").replace("'", "\\'")

    # ------------------------------------------------------------------
    # Entity nodes
    # ------------------------------------------------------------------

    async def upsert_entity(
        self,
        tenant_id: str,
        entity_id: str,
        name: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _MS()
        attrs = attributes or {}
        safe_content = self._esc(str(attrs.get("content") or name))
        safe_subject = self._esc(str(attrs.get("subject") or ""))
        safe_attribute = self._esc(str(attrs.get("attribute") or ""))
        safe_value = self._esc(str(attrs.get("value") or ""))
        safe_canonical_date = self._esc(str(attrs.get("canonical_date") or ""))
        cypher = (
            f"MERGE (e:Entity {{id: '{self._esc(entity_id)}', tenant_id: '{self._esc(tenant_id)}'}}) "
            f"ON CREATE SET e.name = '{self._esc(name)}', "
            f"  e.entity_type = '{self._esc(entity_type)}', "
            f"  e.content = '{safe_content}', "
            f"  e.subject = '{safe_subject}', "
            f"  e.attribute = '{safe_attribute}', "
            f"  e.value = '{safe_value}', "
            f"  e.canonical_date = '{safe_canonical_date}', "
            f"  e.weight = 0.5, e.access_count = 0, "
            f"  e.created_at = {now}, e.updated_at = {now} "
            f"ON MATCH SET e.updated_at = {now}, "
            f"  e.access_count = e.access_count + 1, "
            f"  e.content = CASE WHEN e.content IS NULL OR e.content = '' THEN '{safe_content}' ELSE e.content END, "
            f"  e.subject = CASE WHEN e.subject IS NULL OR e.subject = '' THEN '{safe_subject}' ELSE e.subject END, "
            f"  e.attribute = CASE WHEN e.attribute IS NULL OR e.attribute = '' THEN '{safe_attribute}' ELSE e.attribute END, "
            f"  e.value = CASE WHEN e.value IS NULL OR e.value = '' THEN '{safe_value}' ELSE e.value END, "
            f"  e.canonical_date = CASE WHEN e.canonical_date IS NULL OR e.canonical_date = '' THEN '{safe_canonical_date}' ELSE e.canonical_date END, "
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
        anchor_date: str | None = None,
        speaker: str | None = None,
    ) -> dict[str, Any]:
        now = _MS()
        safe_text = self._esc(text[:2000])
        safe_anchor = self._esc(str(anchor_date or ""))
        safe_speaker = self._esc(str(speaker or ""))
        cypher = (
            f"CREATE (u:Utterance {{"
            f"  id: '{self._esc(utterance_id)}', "
            f"  tenant_id: '{self._esc(tenant_id)}', "
            f"  text: '{safe_text}', "
            f"  role: '{self._esc(role)}', "
            f"  session_id: '{self._esc(session_id)}', "
            f"  anchor_date: '{safe_anchor}', "
            f"  speaker: '{safe_speaker}', "
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
            f"MATCH (seed:Utterance {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE seed.id IN [{seed_list}] "
            f"MATCH (seed)-[:FOLLOWED_BY*1..{hops}]-(neighbor:Utterance {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE neighbor.weight >= {min_weight} "
            f"WITH DISTINCT neighbor "
            f"RETURN neighbor.id AS id, "
            f"  labels(neighbor)[0] AS label, "
            f"  COALESCE(neighbor.content, neighbor.name, neighbor.text, neighbor.action_type) AS content, "
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
        # Use a named path + relationships(p) to extract individual edges.
        # A variable-length pattern binds the path, not a single relationship, so
        # `UNWIND r AS rel; SET rel.weight` fails with "alias did not resolve to a
        # graph entity". relationships(p) yields the edge list to unwind + decay.
        cypher = (
            f"MATCH (seed {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE seed.id IN [{seed_list}] "
            f"MATCH p = (seed)-[*1..{hops}]-(neighbor {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"UNWIND relationships(p) AS rel "
            f"SET rel.weight = coalesce(rel.weight, 1.0) * {DECAY_FACTOR} "
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
    # Person profile nodes — accumulate facts per person per tenant
    # ------------------------------------------------------------------

    @staticmethod
    def _esc_snake(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower()).strip("_") or "unknown"

    async def upsert_person_profile(
        self,
        tenant_id: str,
        person_name: str,
        facts: list[str],
    ) -> dict[str, Any]:
        """Upsert a per-person profile node, appending new facts each call."""
        if not facts:
            return {}
        profile_id = f"profile_{self._esc_snake(person_name)}"
        now = _MS()
        safe_name = self._esc(person_name)
        facts_block = self._esc(" | ".join(facts[:50]))
        cypher = (
            f"MERGE (e:Entity {{id: '{self._esc(profile_id)}', tenant_id: '{self._esc(tenant_id)}'}}) "
            f"ON CREATE SET "
            f"  e.name = '{safe_name}', "
            f"  e.entity_type = 'person_profile', "
            f"  e.subject = '{safe_name}', "
            f"  e.content = '{facts_block}', "
            f"  e.weight = 0.8, "
            f"  e.access_count = 0, "
            f"  e.created_at = {now}, e.updated_at = {now} "
            f"ON MATCH SET "
            f"  e.content = e.content + ' | ' + '{facts_block}', "
            f"  e.updated_at = {now}, "
            f"  e.weight = CASE WHEN e.weight + 0.05 > 1.0 THEN 1.0 "
            f"             ELSE e.weight + 0.05 END "
            f"RETURN e.id AS id, e.weight AS weight"
        )
        rows = await self._aquery(tenant_id, cypher)
        return rows[0] if rows else {}

    async def fetch_person_profiles(
        self,
        tenant_id: str,
        person_names: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch accumulated profile nodes for the given person names."""
        if not person_names:
            return []
        name_list = ", ".join(f"'{self._esc(n)}'" for n in person_names)
        # Also match profiles where the subject CONTAINS a queried name (handles
        # "Emma" matching "Emma Watson") or the name is contained in subject.
        contains_clauses = " OR ".join(
            f"e.subject CONTAINS '{self._esc(n)}'"
            for n in person_names[:4]
        )
        cypher = (
            f"MATCH (e:Entity {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE e.entity_type = 'person_profile' AND "
            f"  (e.subject IN [{name_list}] OR {contains_clauses}) "
            f"RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type, "
            f"  e.content AS content, e.subject AS subject, "
            f"  e.weight AS weight, e.created_at AS created_at "
            f"LIMIT 10"
        )
        return await self._aquery(tenant_id, cypher)

    async def fetch_all_profiles(
        self,
        tenant_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Fetch top-N person profiles by weight (fallback for generic questions)."""
        cypher = (
            f"MATCH (e:Entity {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE e.entity_type = 'person_profile' "
            f"RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type, "
            f"  e.content AS content, e.subject AS subject, "
            f"  e.weight AS weight, e.created_at AS created_at "
            f"ORDER BY e.weight DESC LIMIT {limit}"
        )
        return await self._aquery(tenant_id, cypher)

    # ------------------------------------------------------------------
    # Temporal event nodes — (person, activity, date) triples
    # ------------------------------------------------------------------

    async def fetch_temporal_events(
        self,
        tenant_id: str,
        date_prefixes: list[str],
        person_names: list[str],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Fetch temporal event nodes matching date prefixes or person names."""
        if not date_prefixes and not person_names:
            return []
        conditions: list[str] = []
        for dp in date_prefixes[:3]:
            conditions.append(f"e.canonical_date STARTS WITH '{self._esc(dp)}'")
        if person_names:
            name_list = ", ".join(f"'{self._esc(n)}'" for n in person_names)
            conditions.append(f"e.subject IN [{name_list}]")
        where = " OR ".join(conditions)
        cypher = (
            f"MATCH (e:Entity {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE e.entity_type = 'temporal_event' AND ({where}) "
            f"RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type, "
            f"  e.content AS content, e.subject AS subject, "
            f"  e.canonical_date AS canonical_date, "
            f"  e.value AS value, e.weight AS weight, e.created_at AS created_at "
            f"ORDER BY e.canonical_date DESC "
            f"LIMIT {limit}"
        )
        return await self._aquery(tenant_id, cypher)

    async def fetch_temporal_events_asc(
        self,
        tenant_id: str,
        date_prefixes: list[str],
        person_names: list[str],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Like fetch_temporal_events but sorted oldest-first (for 'first time' queries)."""
        if not date_prefixes and not person_names:
            return []
        conditions: list[str] = []
        for dp in date_prefixes[:3]:
            conditions.append(f"e.canonical_date STARTS WITH '{self._esc(dp)}'")
        if person_names:
            name_list = ", ".join(f"'{self._esc(n)}'" for n in person_names)
            conditions.append(f"e.subject IN [{name_list}]")
        where = " OR ".join(conditions)
        cypher = (
            f"MATCH (e:Entity {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE e.entity_type = 'temporal_event' AND ({where}) "
            f"RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type, "
            f"  e.content AS content, e.subject AS subject, "
            f"  e.canonical_date AS canonical_date, "
            f"  e.value AS value, e.weight AS weight, e.created_at AS created_at "
            f"ORDER BY e.canonical_date ASC "
            f"LIMIT {limit}"
        )
        return await self._aquery(tenant_id, cypher)

    async def fetch_entities_by_ids(
        self,
        tenant_id: str,
        entity_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch entity nodes directly by their IDs (for Qdrant-seeded entity lookups)."""
        if not entity_ids:
            return []
        id_list = ", ".join(f"'{self._esc(eid)}'" for eid in entity_ids[:30])
        cypher = (
            f"MATCH (e:Entity {{tenant_id: '{self._esc(tenant_id)}'}}) "
            f"WHERE e.id IN [{id_list}] "
            f"RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type, "
            f"  e.content AS content, e.subject AS subject, "
            f"  e.attribute AS attribute, e.value AS value, "
            f"  e.canonical_date AS canonical_date, "
            f"  e.weight AS weight, e.created_at AS created_at "
            f"LIMIT 25"
        )
        return await self._aquery(tenant_id, cypher)

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
