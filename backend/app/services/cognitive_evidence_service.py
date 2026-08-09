from __future__ import annotations

import inspect
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory_semantic_node import MemorySemanticNode
from app.models.memory_state_snapshot import MemoryStateSnapshot
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
        fact_layers = self._merge_inline_fact_layers(memory_hits, fact_layers)
        goal_context = await self.goal_loop_service.build_context()
        temporal_reasoning = await self._build_temporal_reasoning(
            query=query,
            memory_hits=memory_hits,
            facts=list(fact_layers.get("facts") or []),
            query_intelligence=dict(query_intelligence or {}),
            planner_context=dict(planner_context or {}),
        )
        entity_context = await self._build_entity_context(
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
            "entity_context": entity_context,
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
                    "extra_metadata": extra_metadata,
                }
            )
        return normalized

    def _merge_inline_fact_layers(
        self,
        memory_hits: list[dict[str, Any]],
        fact_layers: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = {
            "facts": list((fact_layers or {}).get("facts") or []),
            "contradictions": list((fact_layers or {}).get("contradictions") or []),
        }
        seen: set[tuple[str, str, str, str]] = set()
        for fact in merged["facts"]:
            key = (
                str(fact.get("subject") or "").strip().lower(),
                str(fact.get("predicate") or "").strip().lower(),
                str(fact.get("object") or "").strip().lower(),
                str(fact.get("source_memory_id") or "").strip(),
            )
            seen.add(key)

        for hit in memory_hits:
            extra = dict(hit.get("extra_metadata") or {})
            inline_facts = []
            fact_support = extra.get("fact_support")
            if isinstance(fact_support, dict):
                inline_facts.append(fact_support)
            for item in list(extra.get("fact_supporting_facts") or []):
                if isinstance(item, dict):
                    inline_facts.append(item)

            source_memory_id = str(hit.get("memory_id") or "")
            for index, fact in enumerate(inline_facts):
                subject = str(fact.get("subject") or "").strip()
                predicate = str(fact.get("predicate") or "").strip()
                obj = str(fact.get("object") or "").strip()
                if not (subject and predicate and obj):
                    continue
                key = (subject.lower(), predicate.lower(), obj.lower(), source_memory_id)
                if key in seen:
                    continue
                seen.add(key)
                merged["facts"].append(
                    {
                        "fact_id": str(fact.get("fact_id") or f"inline::{source_memory_id}::{index}"),
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "confidence": self._float_or_default(fact.get("confidence"), 0.85),
                        "status": str(fact.get("status") or "active"),
                        "source_memory_id": source_memory_id,
                        "valid_from": fact.get("valid_from"),
                        "valid_to": fact.get("valid_to"),
                        "contradiction_group_id": fact.get("contradiction_group_id"),
                        "source_type": "state_space" if source_memory_id.startswith("state::") else "inline",
                    }
                )

        merged["facts"].sort(
            key=lambda item: self._float_or_default(item.get("confidence"), 0.0),
            reverse=True,
        )
        return merged

    async def _build_entity_context(
        self,
        *,
        memory_hits: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        query_intelligence: dict[str, Any],
        planner_context: dict[str, Any],
    ) -> dict[str, Any]:
        entity_names = self._candidate_entity_names(
            memory_hits=memory_hits,
            facts=facts,
            query_intelligence=query_intelligence,
            planner_context=planner_context,
        )
        if not entity_names:
            return {"primary_subject": None, "entities": []}

        snapshots: list[MemoryStateSnapshot] = []
        normalized = [self._normalize_entity_name(name) for name in entity_names if self._normalize_entity_name(name)]
        try:
            stmt = (
                select(MemoryStateSnapshot)
                .where(
                    MemoryStateSnapshot.organization_id == self.org_id,
                    MemoryStateSnapshot.scope_type == "entity",
                    MemoryStateSnapshot.scope_key.in_(normalized[:8]),
                )
                .limit(8)
            )
            result = await self.session.execute(stmt)
            scalars = result.scalars() if hasattr(result, "scalars") else None
            if inspect.isawaitable(scalars):
                scalars = await scalars
            rows = scalars.all() if scalars is not None and hasattr(scalars, "all") else []
            if inspect.isawaitable(rows):
                rows = await rows
            snapshots = list(rows or [])
        except Exception:
            snapshots = []
        snapshot_index = {
            self._normalize_entity_name(snapshot.scope_key): snapshot
            for snapshot in snapshots
        }

        entities: list[dict[str, Any]] = []
        primary_subject = str((planner_context.get("question_frame") or {}).get("primary_subject") or "").strip() or None

        for index, entity_name in enumerate(entity_names[:4]):
            scope_key = self._normalize_entity_name(entity_name)
            snapshot = snapshot_index.get(scope_key)
            state = dict(getattr(snapshot, "symbolic_state", {}) or {})
            state_facts = [
                dict(item)
                for item in list(state.get("facts") or [])
                if self._normalize_entity_name(item.get("subject")) == scope_key
                or self._normalize_entity_name(item.get("object")) == scope_key
            ][:5]
            if not state_facts:
                state_facts = [
                    dict(item)
                    for item in facts
                    if self._normalize_entity_name(item.get("subject")) == scope_key
                    or self._normalize_entity_name(item.get("object")) == scope_key
                ][:5]
            aliases = [
                str(item).strip()
                for item in list(state.get("aliases") or [])
                if str(item).strip()
            ]
            recent_memories = []
            for item in list(state.get("recent_memories") or [])[:3]:
                preview = str(item.get("content_preview") or "").strip()
                if not preview:
                    continue
                recent_memories.append(
                    {
                        "memory_id": str(item.get("memory_id") or "").strip(),
                        "title": str(item.get("title") or "").strip(),
                        "content_preview": preview[:160],
                        "occurred_at": item.get("occurred_at"),
                    }
                )

            mention_memories = self._entity_memory_mentions(memory_hits, entity_name)[:3]
            entity_links = self._entity_links(entity_name=entity_name, facts=state_facts or facts)
            canonical_name = self._canonical_entity_name(entity_name=entity_name, aliases=aliases, facts=state_facts)
            entities.append(
                {
                    "canonical_name": canonical_name,
                    "scope_key": scope_key,
                    "state_version": int(getattr(snapshot, "state_version", 0) or 0),
                    "is_primary_subject": bool(primary_subject and self._normalize_entity_name(primary_subject) == scope_key) or (index == 0 and not primary_subject),
                    "aliases": aliases[:6],
                    "facts": state_facts[:5],
                    "recent_memories": recent_memories,
                    "memory_mentions": mention_memories,
                    "entity_links": entity_links[:5],
                }
            )

        return {
            "primary_subject": primary_subject or (entities[0]["canonical_name"] if entities else None),
            "entities": entities,
        }

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

    def _candidate_entity_names(
        self,
        *,
        memory_hits: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        query_intelligence: dict[str, Any],
        planner_context: dict[str, Any],
    ) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()

        def _add(value: Any) -> None:
            text = str(value or "").strip()
            key = self._normalize_entity_name(text)
            if not text or not key or key in seen:
                return
            seen.add(key)
            names.append(text)

        frame = dict(planner_context.get("question_frame") or {})
        _add(frame.get("primary_subject"))
        for item in list(frame.get("secondary_entities") or []):
            _add(item)
        for item in list(query_intelligence.get("extracted_entities") or []):
            _add(item)
        for fact in facts[:12]:
            subject = fact.get("subject")
            obj = fact.get("object")
            if self._looks_named_entity(subject):
                _add(subject)
            if self._looks_named_entity(obj):
                _add(obj)
        for hit in memory_hits[:8]:
            for value in dict(hit.get("entities") or {}).values():
                if isinstance(value, list):
                    for item in value:
                        if self._looks_named_entity(item):
                            _add(item)
                elif self._looks_named_entity(value):
                    _add(value)
        return names[:4]

    def _entity_memory_mentions(
        self,
        memory_hits: list[dict[str, Any]],
        entity_name: str,
    ) -> list[dict[str, Any]]:
        target = self._normalize_entity_name(entity_name)
        mentions: list[dict[str, Any]] = []
        for hit in memory_hits:
            combined = " ".join(
                [
                    str(hit.get("title") or "").strip(),
                    str(hit.get("content_preview") or "").strip(),
                    " ".join(self._flatten_entity_values(dict(hit.get("entities") or {}))),
                ]
            ).lower()
            if target and target not in combined:
                continue
            mentions.append(
                {
                    "memory_id": str(hit.get("memory_id") or "").strip(),
                    "title": str(hit.get("title") or "").strip(),
                    "content_preview": str(hit.get("content_preview") or "").strip()[:160],
                }
            )
            if len(mentions) >= 3:
                break
        return mentions

    def _entity_links(
        self,
        *,
        entity_name: str,
        facts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        target = self._normalize_entity_name(entity_name)
        links: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for fact in facts:
            subject = str(fact.get("subject") or "").strip()
            predicate = str(fact.get("predicate") or "").strip()
            obj = str(fact.get("object") or "").strip()
            if not (subject and predicate and obj):
                continue
            subject_key = self._normalize_entity_name(subject)
            object_key = self._normalize_entity_name(obj)
            if subject_key == target and self._looks_named_entity(obj):
                key = (subject_key, predicate.lower(), object_key)
                if key in seen:
                    continue
                seen.add(key)
                links.append({"direction": "out", "predicate": predicate, "entity": obj})
            elif object_key == target and self._looks_named_entity(subject):
                key = (object_key, predicate.lower(), subject_key)
                if key in seen:
                    continue
                seen.add(key)
                links.append({"direction": "in", "predicate": predicate, "entity": subject})
        return links

    def _canonical_entity_name(
        self,
        *,
        entity_name: str,
        aliases: list[str],
        facts: list[dict[str, Any]],
    ) -> str:
        for fact in facts:
            subject = str(fact.get("subject") or "").strip()
            if subject and self._normalize_entity_name(subject) == self._normalize_entity_name(entity_name):
                return subject
        for alias in aliases:
            if alias and self._normalize_entity_name(alias) == self._normalize_entity_name(entity_name):
                return alias
        return str(entity_name or "").strip()

    @staticmethod
    def _flatten_entity_values(entities: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for value in entities.values():
            if isinstance(value, list):
                values.extend(str(item).strip() for item in value if str(item).strip())
            else:
                text = str(value or "").strip()
                if text:
                    values.append(text)
        return values

    @staticmethod
    def _normalize_entity_name(value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _looks_named_entity(value: Any) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        return bool(
            text[:1].isupper()
            or any(ch.isupper() for ch in text[1:])
        )

    @staticmethod
    def _float_or_default(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
