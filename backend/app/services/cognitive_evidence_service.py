from __future__ import annotations

from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory_semantic_node import MemorySemanticNode
from app.models.memory_topic import MemoryTopic
from app.models.navigation_edge import NavigationEdge
from app.services.cognitive_goal_loop_service import CognitiveGoalLoopService
from app.services.fact_service import FactService
from app.services.temporal_reasoning_service import TemporalReasoningService
from app.services.unified_episode_service import UnifiedEpisodeService


class CognitiveEvidenceService:
    """Builds a reusable structured evidence package for downstream reasoning."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        org_id: str,
        episode_service: UnifiedEpisodeService | None = None,
        goal_loop_service: CognitiveGoalLoopService | None = None,
    ) -> None:
        self.session = session
        self.org_id = org_id
        self.episode_service = episode_service or UnifiedEpisodeService(session, org_id=org_id)
        self.goal_loop_service = goal_loop_service or CognitiveGoalLoopService(session, org_id=org_id)

    async def build_package(
        self,
        *,
        query: str,
        memories: list[dict[str, Any]],
        query_intelligence: dict[str, Any] | None = None,
        planner_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        memory_hits = self._normalize_memory_hits(memories)
        memory_ids = [str(item["memory_id"]) for item in memory_hits if item.get("memory_id")]

        episode_index = await self._load_unified_episodes(memory_ids)
        episodes = self._flatten_episode_index(episode_index)
        episode_ids = [str(item["episode_id"]) for item in episodes if item.get("episode_id")]

        semantic_nodes = await self._load_semantic_nodes(memory_ids, episode_ids)
        topics = await self._load_topics(self._collect_topic_ids(episodes, semantic_nodes))
        graph_neighbors = await self._load_graph_neighbors(episodes, semantic_nodes, topics)
        temporal_anchors = self._collect_temporal_anchors(memory_hits)
        fact_layers = await self._load_fact_layers(memory_ids)
        goal_context = await self.goal_loop_service.build_context()
        temporal_reasoning = await self._build_temporal_reasoning(
            query=query,
            memory_hits=memory_hits,
            facts=list(fact_layers.get("facts") or []),
            query_intelligence=dict(query_intelligence or {}),
            planner_context=dict(planner_context or {}),
        )

        return {
            "query": query,
            "query_intelligence": dict(query_intelligence or {}),
            "planner_context": dict(planner_context or {}),
            "memory_hits": memory_hits,
            "episodes": episodes,
            "facts": list(fact_layers.get("facts") or []),
            "contradictions": list(fact_layers.get("contradictions") or []),
            "semantic_nodes": semantic_nodes,
            "topics": topics,
            "graph_neighbors": graph_neighbors,
            "temporal_anchors": temporal_anchors,
            "temporal_reasoning": temporal_reasoning,
            "multi_hop_trace": list((planner_context or {}).get("multi_hop_trace") or []),
            "goal_context": goal_context,
            "evidence_quality": self._build_quality_summary(
                memory_hits=memory_hits,
                episodes=episodes,
                facts=list(fact_layers.get("facts") or []),
                contradictions=list(fact_layers.get("contradictions") or []),
                semantic_nodes=semantic_nodes,
                topics=topics,
                graph_neighbors=graph_neighbors,
                temporal_anchors=temporal_anchors,
                goal_context=goal_context,
            ),
        }

    async def _load_unified_episodes(
        self,
        memory_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        return await self.episode_service.list_for_memory_ids(memory_ids)

    async def _load_fact_layers(self, memory_ids: list[str]) -> dict[str, Any]:
        return await FactService(self.session, self.org_id).get_structured_evidence_for_memories(memory_ids)

    async def _load_semantic_nodes(
        self,
        memory_ids: list[str],
        episode_ids: list[str],
        *,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        filters = []
        for memory_id in memory_ids[:20]:
            filters.append(MemorySemanticNode.source_memory_ids.contains([memory_id]))
        for episode_id in episode_ids[:20]:
            filters.append(MemorySemanticNode.source_episode_ids.contains([episode_id]))

        if not filters:
            return []

        stmt = (
            select(MemorySemanticNode)
            .where(
                MemorySemanticNode.organization_id == self.org_id,
                or_(*filters),
            )
            .order_by(MemorySemanticNode.composite_quality.desc())
            .limit(max(1, int(limit or 1)))
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [
            {
                "semantic_node_id": str(row.id),
                "content": row.content,
                "composite_quality": float(row.composite_quality or 0.0),
                "topic_id": row.topic_id,
                "tags": list(row.tags or []),
                "entities": list(row.entities or []),
                "source_episode_ids": list(row.source_episode_ids or []),
                "source_memory_ids": list(row.source_memory_ids or []),
            }
            for row in rows
        ]

    async def _load_topics(
        self,
        topic_ids: list[str],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        topic_ids = [str(topic_id) for topic_id in topic_ids if topic_id]
        if not topic_ids:
            return []

        stmt = (
            select(MemoryTopic)
            .where(
                MemoryTopic.organization_id == self.org_id,
                MemoryTopic.id.in_(topic_ids),
            )
            .limit(max(1, int(limit or 1)))
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [
            {
                "topic_id": str(row.id),
                "label": row.label,
                "keywords": list(row.keywords or []),
                "scope": row.scope,
                "scope_id": row.scope_id,
            }
            for row in rows
        ]

    async def _load_graph_neighbors(
        self,
        episodes: list[dict[str, Any]],
        semantic_nodes: list[dict[str, Any]],
        topics: list[dict[str, Any]],
        *,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        disjuncts = []

        for episode in episodes:
            if episode.get("source") != "memory_episode":
                continue
            episode_id = str(episode.get("episode_id") or "").strip()
            if episode_id:
                disjuncts.append(
                    (NavigationEdge.source_type == "episode") & (NavigationEdge.source_id == episode_id)
                )

        for node in semantic_nodes:
            node_id = str(node.get("semantic_node_id") or "").strip()
            if node_id:
                disjuncts.append(
                    (NavigationEdge.source_type == "semantic_node") & (NavigationEdge.source_id == node_id)
                )

        for topic in topics:
            topic_id = str(topic.get("topic_id") or "").strip()
            if topic_id:
                disjuncts.append(
                    (NavigationEdge.source_type == "topic") & (NavigationEdge.source_id == topic_id)
                )

        if not disjuncts:
            return []

        stmt = (
            select(NavigationEdge)
            .where(
                NavigationEdge.organization_id == self.org_id,
                or_(*disjuncts),
            )
            .order_by(NavigationEdge.similarity.desc())
            .limit(max(1, int(limit or 1)))
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [
            {
                "source_type": row.source_type,
                "source_id": str(row.source_id),
                "target_type": row.target_type,
                "target_id": str(row.target_id),
                "similarity": float(row.similarity or 0.0),
                "k_rank": int(row.k_rank or 0),
            }
            for row in rows
        ]

    def _normalize_memory_hits(
        self,
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for memory in memories:
            extra_metadata = dict(memory.get("extra_metadata") or {})
            retrieval_learning = dict(extra_metadata.get("retrieval_learning") or {})
            normalized.append(
                {
                    "memory_id": str(memory.get("id") or memory.get("memory_id") or ""),
                    "title": memory.get("title"),
                    "content_preview": memory.get("content_preview") or memory.get("content"),
                    "score": self._float_or_default(memory.get("score"), 0.0),
                    "occurred_at": memory.get("occurred_at") or extra_metadata.get("event_time"),
                    "classification": memory.get("classification"),
                    "source_type": memory.get("source_type"),
                    "tags": list(memory.get("tags") or []),
                    "entities": dict(memory.get("entities") or {}),
                    "provenance": list(memory.get("provenance") or []),
                    "retrieval_learning": retrieval_learning,
                    "retrieval_prior": self._float_or_default(retrieval_learning.get("relevance_score"), 0.0),
                }
            )
        return normalized

    def _flatten_episode_index(
        self,
        episode_index: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        flattened: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        for memory_id, items in episode_index.items():
            for item in items:
                episode_id = str(item.get("episode_id") or "").strip()
                if not episode_id:
                    continue
                enriched = dict(item)
                linked = list(enriched.get("linked_memory_ids") or [])
                if memory_id and memory_id not in linked:
                    linked.append(memory_id)
                enriched["linked_memory_ids"] = linked

                existing = flattened.get(episode_id)
                if existing is None:
                    flattened[episode_id] = enriched
                    order.append(episode_id)
                    continue

                flattened[episode_id] = self._merge_episode(existing, enriched)

        return [flattened[episode_id] for episode_id in order]

    def _merge_episode(
        self,
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        preferred = dict(existing)
        existing_is_memory = existing.get("source") == "memory_episode"
        incoming_is_memory = incoming.get("source") == "memory_episode"
        if incoming_is_memory and not existing_is_memory:
            preferred = dict(incoming)
        else:
            for key, value in incoming.items():
                if value not in (None, [], {}, "") and preferred.get(key) in (None, [], {}, ""):
                    preferred[key] = value

        linked = list(dict.fromkeys(list(existing.get("linked_memory_ids") or []) + list(incoming.get("linked_memory_ids") or [])))
        tags = list(dict.fromkeys(list(existing.get("tags") or []) + list(incoming.get("tags") or [])))
        entities = dict(existing.get("entities") or {})
        entities.update(dict(incoming.get("entities") or {}))
        preferred["linked_memory_ids"] = linked
        preferred["tags"] = tags
        preferred["entities"] = entities
        return preferred

    def _collect_topic_ids(
        self,
        episodes: list[dict[str, Any]],
        semantic_nodes: list[dict[str, Any]],
    ) -> list[str]:
        topic_ids: list[str] = []
        seen: set[str] = set()

        for item in episodes:
            topic_id = str(item.get("topic_id") or "").strip()
            if topic_id and topic_id not in seen:
                seen.add(topic_id)
                topic_ids.append(topic_id)

        for item in semantic_nodes:
            topic_id = str(item.get("topic_id") or "").strip()
            if topic_id and topic_id not in seen:
                seen.add(topic_id)
                topic_ids.append(topic_id)

        return topic_ids

    def _collect_temporal_anchors(
        self,
        memory_hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = []
        for item in memory_hits:
            occurred_at = item.get("occurred_at")
            if not occurred_at:
                continue
            anchors.append(
                {
                    "memory_id": item.get("memory_id"),
                    "occurred_at": occurred_at,
                }
            )
        return anchors

    async def _build_temporal_reasoning(
        self,
        *,
        query: str,
        memory_hits: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        query_intelligence: dict[str, Any],
        planner_context: dict[str, Any],
    ) -> dict[str, Any]:
        planner_temporal = planner_context.get("temporal_reasoning")
        if isinstance(planner_temporal, dict) and planner_temporal:
            return dict(planner_temporal)

        result = await TemporalReasoningService(session=self.session).temporal_query(
            self.org_id,
            "timeline_of_memories",
            query=query,
            candidate_memories=memory_hits,
            candidate_facts=facts,
            extracted_entities=list(query_intelligence.get("extracted_entities") or []),
            max_events=12,
        )
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"timeline": result}
        return {"timeline": []}

    def _build_quality_summary(
        self,
        *,
        memory_hits: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        contradictions: list[dict[str, Any]],
        semantic_nodes: list[dict[str, Any]],
        topics: list[dict[str, Any]],
        graph_neighbors: list[dict[str, Any]],
        temporal_anchors: list[dict[str, Any]],
        goal_context: dict[str, Any],
    ) -> dict[str, Any]:
        avg_memory_score = (
            sum(self._float_or_default(item.get("score"), 0.0) for item in memory_hits) / len(memory_hits)
            if memory_hits
            else 0.0
        )
        avg_semantic_quality = (
            sum(self._float_or_default(item.get("composite_quality"), 0.0) for item in semantic_nodes) / len(semantic_nodes)
            if semantic_nodes
            else 0.0
        )
        avg_feedback_signal = (
            sum(self._float_or_default(item.get("retrieval_prior"), 0.0) for item in memory_hits) / len(memory_hits)
            if memory_hits
            else 0.0
        )

        if semantic_nodes and topics and facts:
            coverage_mode = "semantic_fact_hybrid"
        elif semantic_nodes and topics:
            coverage_mode = "semantic_hybrid"
        elif episodes:
            coverage_mode = "episodic"
        elif memory_hits:
            coverage_mode = "memory_only"
        else:
            coverage_mode = "empty"

        return {
            "memory_count": len(memory_hits),
            "episode_count": len(episodes),
            "fact_count": len(facts),
            "contradiction_count": len(contradictions),
            "semantic_node_count": len(semantic_nodes),
            "topic_count": len(topics),
            "graph_neighbor_count": len(graph_neighbors),
            "temporal_anchor_count": len(temporal_anchors),
            "active_goal_count": len(goal_context.get("active_goals") or []),
            "knowledge_gap_count": len(goal_context.get("knowledge_gaps") or []),
            "avg_memory_score": round(avg_memory_score, 4),
            "avg_semantic_quality": round(avg_semantic_quality, 4),
            "avg_feedback_signal": round(avg_feedback_signal, 4),
            "coverage_mode": coverage_mode,
        }

    @staticmethod
    def _float_or_default(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
