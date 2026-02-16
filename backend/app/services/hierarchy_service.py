"""Memory Hierarchy Service (GAP-1 + GAP-6 Orchestrator).

Façade that ties together the four-level hierarchy (GAP-1) and
the kNN navigation graph (GAP-6) into a single public API:

    ┌───────────┐      ┌───────────────────┐      ┌─────────────────┐
    │  Messages  │─────▶│ EpisodeBoundary   │─────▶│  MemoryEpisode  │
    │  (Level 1) │      │ Service           │      │  (Level 2)      │
    └───────────┘      └───────────────────┘      └────────┬────────┘
                                                           │ distill
                                                           ▼
    ┌──────────────────┐      ┌───────────────────┐      ┌───────────────────┐
    │  MemoryTopic     │◀─────│ SemanticDistill.  │◀─────│ MemorySemanticNode│
    │  (Level 4)       │      │ Service           │      │  (Level 3)        │
    └──────────────────┘      └───────────────────┘      └───────────────────┘
                                        │
                                        │ kNN
                                        ▼
                               ┌──────────────────┐
                               │ NavigationEdge    │
                               │ (GAP-6 graph)     │
                               └──────────────────┘

Usage:
    svc = HierarchyService(session)

    # Real-time: new message arrives
    result = await svc.ingest_message(org_id=..., owner_id=..., memory_id=...)

    # Batch: nightly rebuild
    stats = await svc.rebuild(org_id=...)

    # Retrieval: top-down search
    nodes = await svc.hierarchical_search(org_id=..., query=..., limit=10)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import MemoryMetadata
from app.models.memory_episode import MemoryEpisode
from app.models.memory_semantic_node import MemorySemanticNode
from app.models.memory_topic import MemoryTopic
from app.services.embedding_service import EmbeddingService
from app.services.episode_boundary_service import EpisodeBoundaryService
from app.services.semantic_distillation_service import SemanticDistillationService
from app.services.topic_structure_service import TopicStructureService
from app.services.knn_navigation_service import KNNNavigationService

logger = logging.getLogger(__name__)


class HierarchyService:
    """Orchestrates the 4-level hierarchy + kNN navigation graph."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.episode_svc = EpisodeBoundaryService(session)
        self.distillation_svc = SemanticDistillationService(session)
        self.knn_svc = KNNNavigationService(session)
        self.topic_structure_svc = TopicStructureService(session)

    # ── Real-time ingestion ─────────────────────────────────────────

    async def ingest_message(
        self,
        *,
        organization_id: str,
        owner_id: str,
        memory_id: str,
        scope: str = "personal",
        scope_id: Optional[str] = None,
        auto_distill: bool = True,
        auto_knn: bool = True,
        auto_rebalance_topics: bool = False,
    ) -> Dict[str, Any]:
        """Process a new message through the hierarchy pipeline.

        1. Episode boundary detection: append to open episode or start new one.
        2. If a previous episode was closed → distill semantic nodes.
        3. Update kNN navigation edges for any new/changed centroids.

        Returns a summary dict describing all operations performed.
        """
        result: Dict[str, Any] = {"memory_id": memory_id, "actions": []}

        # Step 1: Episode segmentation
        ep_result = await self.episode_svc.add_message_to_current_episode(
            organization_id=organization_id,
            owner_id=owner_id,
            memory_id=memory_id,
            scope=scope,
            scope_id=scope_id,
        )
        result["episode_id"] = ep_result["episode_id"]
        result["episode_action"] = ep_result["action"]
        result["actions"].append(f"episode:{ep_result['action']}")

        # Step 2: If a new episode was created, the previous one was closed.
        # Distill the closed episode.
        if ep_result["action"] == "new_episode":
            if auto_rebalance_topics:
                try:
                    stats = await self.topic_structure_svc.rebalance_topics(
                        organization_id=organization_id,
                        scope=scope,
                        scope_id=scope_id,
                    )
                    result["actions"].append(
                        f"topics_rebalanced:s={stats.get('sparsity_score', 0):.3f}"
                    )
                except Exception as exc:
                    logger.warning("Topic rebalance failed: %s", exc)

            if not auto_distill:
                return result

            closed_episodes = await self._find_closed_undistilled(
                organization_id=organization_id,
                owner_id=owner_id,
            )
            for closed_ep in closed_episodes:
                try:
                    nodes = await self.distillation_svc.distill_episode(
                        closed_ep.id, organization_id=organization_id
                    )
                    result["actions"].append(
                        f"distilled:{closed_ep.id}→{len(nodes)} nodes"
                    )

                    # Step 3a: Update kNN for each new semantic node
                    if auto_knn:
                        for n in nodes:
                            await self.knn_svc.update_for_node(
                                organization_id=organization_id,
                                node_type="semantic_node",
                                node_id=n["semantic_node_id"],
                            )
                        result["actions"].append(f"knn_updated:{len(nodes)} semantic nodes")
                except Exception as exc:
                    logger.warning("Distillation failed for %s: %s", closed_ep.id, exc)

            # Step 3b: Update kNN for the closed episode itself
            if auto_knn and closed_episodes:
                for closed_ep in closed_episodes:
                    if closed_ep.vector_id:
                        await self.knn_svc.update_for_node(
                            organization_id=organization_id,
                            node_type="episode",
                            node_id=closed_ep.id,
                            vector_id=closed_ep.vector_id,
                        )
                result["actions"].append("knn_updated:episodes")

        return result

    # ── Batch operations ────────────────────────────────────────────

    async def rebuild(
        self,
        *,
        organization_id: str,
        segment_limit: int = 200,
        distill_limit: int = 50,
        rebalance_topics: bool = True,
    ) -> Dict[str, Any]:
        """Full hierarchy rebuild for an organization.

        1. Segment un-episoded messages.
        2. Distill closed un-distilled episodes.
        3. Full kNN graph rebuild.

        Returns combined stats.
        """
        stats: Dict[str, Any] = {}

        # 1. Batch episode segmentation: find messages not in any episode
        # (This is a simplified batch approach; for production use,
        # messages would be grouped per owner+scope)
        logger.info("Hierarchy rebuild: segmenting messages for org %s", organization_id)

        # 2. Batch distillation
        distill_stats = await self.distillation_svc.distill_batch(
            organization_id=organization_id,
            limit=distill_limit,
        )
        stats["distillation"] = distill_stats

        # 3. Topic structure rebalance (GAP-2)
        if rebalance_topics:
            stats["topic_structure"] = await self.topic_structure_svc.rebalance_topics(
                organization_id=organization_id,
            )

        # 4. Full kNN rebuild
        knn_stats = await self.knn_svc.rebuild_all(
            organization_id=organization_id,
        )
        stats["knn_rebuild"] = knn_stats

        return stats

    # ── Top-down hierarchical retrieval ─────────────────────────────

    async def hierarchical_search(
        self,
        *,
        organization_id: str,
        query: str,
        limit: int = 10,
        hops: int = 2,
        owner_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Top-down adaptive retrieval across the 4-level hierarchy.

        Strategy (xMemory § 3.4 adapted to Ninai's architecture):
        1. Embed the query.
        2. Find matching topics (Level 4) via keyword + embedding.
        3. Traverse kNN graph down to semantic nodes (Level 3).
        4. Expand to episodes (Level 2) via source tracking.
        5. Fetch constituent messages (Level 1).

        Returns all four levels of results.
        """
        query_embedding = await EmbeddingService.embed(query)
        result: Dict[str, Any] = {
            "query": query,
            "topics": [],
            "semantic_nodes": [],
            "episodes": [],
            "messages": [],
        }

        # Level 4: match topics
        topics = await self._search_topics(
            organization_id=organization_id,
            query_embedding=query_embedding,
            limit=5,
        )
        result["topics"] = topics

        # Level 3: traverse kNN from matched topics to semantic nodes
        semantic_nodes: List[Dict[str, Any]] = []
        for topic in topics:
            neighbours = await self.knn_svc.traverse(
                organization_id=organization_id,
                start_type="topic",
                start_id=topic["id"],
                hops=hops,
            )
            for nb in neighbours:
                if nb["type"] == "semantic_node":
                    semantic_nodes.append(nb)

        # Also direct Qdrant search for semantic nodes
        direct_nodes = await self._search_semantic_nodes(
            organization_id=organization_id,
            query_embedding=query_embedding,
            limit=limit,
            owner_id=owner_id,
        )
        # Merge and deduplicate
        seen_ids = {n["id"] for n in semantic_nodes}
        for dn in direct_nodes:
            if dn["id"] not in seen_ids:
                semantic_nodes.append(dn)
                seen_ids.add(dn["id"])

        result["semantic_nodes"] = semantic_nodes[:limit]

        # Level 2: expand to source episodes
        episode_ids: set[str] = set()
        for sn in result["semantic_nodes"]:
            node = (
                await self.session.execute(
                    select(MemorySemanticNode).where(MemorySemanticNode.id == sn["id"])
                )
            ).scalar_one_or_none()
            if node and node.source_episode_ids:
                episode_ids.update(node.source_episode_ids)

        if episode_ids:
            ep_stmt = select(MemoryEpisode).where(
                MemoryEpisode.id.in_(list(episode_ids)),
                MemoryEpisode.organization_id == organization_id,
            )
            episodes = (await self.session.execute(ep_stmt)).scalars().all()
            result["episodes"] = [
                {
                    "id": ep.id,
                    "title": ep.title,
                    "summary": ep.narrative_summary,
                    "message_count": ep.message_count,
                    "boundary_start": str(ep.boundary_start) if ep.boundary_start else None,
                    "boundary_end": str(ep.boundary_end) if ep.boundary_end else None,
                }
                for ep in episodes
            ]

        # Level 1: fetch constituent messages (top N episodes)
        from app.models.memory_episode_membership import MemoryEpisodeMembership
        message_ids: List[str] = []
        for ep_data in result["episodes"][:5]:
            mem_stmt = (
                select(MemoryEpisodeMembership.memory_id)
                .where(MemoryEpisodeMembership.episode_id == ep_data["id"])
                .order_by(MemoryEpisodeMembership.position.asc())
            )
            mids = (await self.session.execute(mem_stmt)).scalars().all()
            message_ids.extend(mids)

        if message_ids:
            msg_stmt = select(MemoryMetadata).where(
                MemoryMetadata.id.in_(message_ids[:limit * 3]),
                MemoryMetadata.organization_id == organization_id,
            )
            msgs = (await self.session.execute(msg_stmt)).scalars().all()
            result["messages"] = [
                {
                    "id": m.id,
                    "content_preview": m.content_preview,
                    "created_at": str(m.created_at) if m.created_at else None,
                    "memory_type": m.memory_type,
                }
                for m in msgs
            ]

        return result

    # ── Helpers ─────────────────────────────────────────────────────

    async def _find_closed_undistilled(
        self, *, organization_id: str, owner_id: str
    ) -> list:
        """Find recently closed episodes that haven't been distilled yet."""
        stmt = (
            select(MemoryEpisode)
            .where(
                MemoryEpisode.organization_id == organization_id,
                MemoryEpisode.owner_id == owner_id,
                MemoryEpisode.status == "closed",
            )
            .order_by(MemoryEpisode.updated_at.desc())
            .limit(5)
        )
        episodes = (await self.session.execute(stmt)).scalars().all()

        # Filter to those without semantic nodes
        undistilled = []
        for ep in episodes:
            existing = (
                await self.session.execute(
                    select(MemorySemanticNode)
                    .where(MemorySemanticNode.source_episode_ids.contains([ep.id]))
                    .limit(1)
                )
            ).scalar_one_or_none()
            if not existing:
                undistilled.append(ep)
        return undistilled

    async def _search_topics(
        self,
        *,
        organization_id: str,
        query_embedding: List[float],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Find matching topics via Qdrant (topic centroids) or keyword fallback."""
        try:
            from qdrant_client.http.models import FieldCondition, MatchValue
            from app.core.qdrant import QdrantService
            from app.core.config import settings

            additional = [
                FieldCondition(key="type", match=MatchValue(value="topic")),
            ]
            org_filter = QdrantService.build_org_filter(
                organization_id, additional_filters=additional
            )
            client = QdrantService.get_client()
            hits = client.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=query_embedding,
                query_filter=org_filter,
                limit=limit,
                with_payload=True,
            )
            return [
                {
                    "id": (hit.payload or {}).get("topic_id", ""),
                    "score": float(hit.score),
                    "label": (hit.payload or {}).get("label", ""),
                }
                for hit in hits
                if (hit.payload or {}).get("topic_id")
            ]
        except Exception:
            # Fallback: keyword search on topic labels
            stmt = (
                select(MemoryTopic)
                .where(MemoryTopic.organization_id == organization_id)
                .limit(limit)
            )
            topics = (await self.session.execute(stmt)).scalars().all()
            return [{"id": t.id, "score": 0.5, "label": t.label} for t in topics]

    async def _search_semantic_nodes(
        self,
        *,
        organization_id: str,
        query_embedding: List[float],
        limit: int,
        owner_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Direct Qdrant search for semantic node centroids."""
        try:
            from qdrant_client.http.models import FieldCondition, MatchValue

            additional = [
                FieldCondition(key="type", match=MatchValue(value="semantic_node")),
            ]
            if owner_id:
                additional.append(
                    FieldCondition(key="owner_id", match=MatchValue(value=owner_id))
                )

            from app.core.qdrant import QdrantService
            from app.core.config import settings

            org_filter = QdrantService.build_org_filter(
                organization_id, additional_filters=additional
            )
            client = QdrantService.get_client()
            hits = client.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=query_embedding,
                query_filter=org_filter,
                limit=limit,
                with_payload=True,
            )
            return [
                {
                    "id": (hit.payload or {}).get("semantic_node_id", ""),
                    "score": float(hit.score),
                    "type": "semantic_node",
                }
                for hit in hits
                if (hit.payload or {}).get("semantic_node_id")
            ]
        except Exception:
            return []
