"""kNN Navigation Graph Service (GAP-6).

Maintains a **materialized k-nearest-neighbour graph** over the three
upper hierarchy levels (episodes, semantic nodes, topics).  Each node
stores its top-k neighbours by centroid cosine similarity.

This enables **cross-cluster traversal** during retrieval:

    Query → match Topics → traverse NavigationEdges → SemanticNodes → Episodes → Messages

Graph maintenance:
    • On node creation/update: compute k-NN for the new node, update reverse edges.
    • Periodic full rebuild (Celery beat): recompute all edges for an org.
    • Stale edges are pruned via a monotonic generation counter.

The service relies on Qdrant's ``recommend`` / ``search`` API for finding
nearest centroids, then materializes the results in PostgreSQL
(``navigation_edges`` table) for fast relational joins during retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.qdrant import QdrantService
from app.models.memory_episode import MemoryEpisode
from app.models.memory_semantic_node import MemorySemanticNode
from app.models.memory_topic import MemoryTopic
from app.models.navigation_edge import NavigationEdge
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────
DEFAULT_K: int = 5                   # neighbours per node
MIN_SIMILARITY: float = 0.20        # ignore edges below this threshold
QDRANT_SEARCH_LIMIT: int = 20       # over-fetch from Qdrant, then trim to k


class KNNNavigationService:
    """Maintains the kNN navigation graph for an organization."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ── Public API: incremental updates ─────────────────────────────

    async def update_for_node(
        self,
        *,
        organization_id: str,
        node_type: str,
        node_id: str,
        vector_id: Optional[str] = None,
        k: int = DEFAULT_K,
    ) -> Dict[str, Any]:
        """Recompute kNN edges for a single node.

        Called when a new episode, semantic node, or topic is created/updated.

        Returns stats: ``{"edges_created": N, "edges_removed": M}``.
        """
        # Resolve the embedding
        embedding = await self._get_node_embedding(
            organization_id=organization_id,
            node_type=node_type,
            node_id=node_id,
            vector_id=vector_id,
        )
        if not embedding or not any(v != 0 for v in embedding):
            logger.debug("No embedding for %s:%s – skipping kNN update", node_type, node_id)
            return {"edges_created": 0, "edges_removed": 0}

        # Search Qdrant for nearest hierarchy centroids
        neighbours = await self._find_neighbours(
            organization_id=organization_id,
            query_vector=embedding,
            exclude_vector_id=vector_id,
            limit=QDRANT_SEARCH_LIMIT,
        )

        # Filter and rank
        ranked = [
            n for n in neighbours
            if n["score"] >= MIN_SIMILARITY
            and not (n["type"] == node_type and n["id"] == node_id)
        ]
        ranked.sort(key=lambda n: n["score"], reverse=True)
        top_k = ranked[:k]

        # Delete old outgoing edges for this node
        del_stmt = delete(NavigationEdge).where(
            NavigationEdge.organization_id == organization_id,
            NavigationEdge.source_type == node_type,
            NavigationEdge.source_id == node_id,
        )
        result = await self.session.execute(del_stmt)
        removed = result.rowcount

        # Insert new edges
        created = 0
        for rank, nb in enumerate(top_k, start=1):
            edge = NavigationEdge(
                id=str(uuid4()),
                organization_id=organization_id,
                source_type=node_type,
                source_id=node_id,
                target_type=nb["type"],
                target_id=nb["id"],
                similarity=nb["score"],
                k_rank=rank,
            )
            self.session.add(edge)
            created += 1

        await self.session.flush()
        return {"edges_created": created, "edges_removed": removed}

    # ── Public API: full rebuild ────────────────────────────────────

    async def rebuild_all(
        self,
        *,
        organization_id: str,
        k: int = DEFAULT_K,
    ) -> Dict[str, Any]:
        """Full kNN graph rebuild for an organization.

        1. Collect all hierarchy nodes (episodes, semantic_nodes, topics).
        2. For each node, compute kNN from Qdrant.
        3. Upsert edges with incremented generation.
        4. Prune stale edges from prior generations.

        Returns stats dict.
        """
        # Get current max generation
        gen_stmt = select(func.max(NavigationEdge.generation)).where(
            NavigationEdge.organization_id == organization_id
        )
        max_gen = (await self.session.execute(gen_stmt)).scalar() or 0
        new_gen = max_gen + 1

        nodes = await self._collect_all_nodes(organization_id)
        logger.info(
            "kNN rebuild for org %s: %d nodes, generation %d",
            organization_id, len(nodes), new_gen,
        )

        total_created = 0
        for node in nodes:
            embedding = await self._get_node_embedding(
                organization_id=organization_id,
                node_type=node["type"],
                node_id=node["id"],
                vector_id=node.get("vector_id"),
            )
            if not embedding or not any(v != 0 for v in embedding):
                continue

            neighbours = await self._find_neighbours(
                organization_id=organization_id,
                query_vector=embedding,
                exclude_vector_id=node.get("vector_id"),
                limit=QDRANT_SEARCH_LIMIT,
            )
            ranked = [
                n for n in neighbours
                if n["score"] >= MIN_SIMILARITY
                and not (n["type"] == node["type"] and n["id"] == node["id"])
            ]
            ranked.sort(key=lambda n: n["score"], reverse=True)
            top_k = ranked[:k]

            for rank, nb in enumerate(top_k, start=1):
                # Upsert: on conflict update similarity, rank, generation
                stmt = insert(NavigationEdge).values(
                    id=str(uuid4()),
                    organization_id=organization_id,
                    source_type=node["type"],
                    source_id=node["id"],
                    target_type=nb["type"],
                    target_id=nb["id"],
                    similarity=nb["score"],
                    k_rank=rank,
                    generation=new_gen,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        "organization_id",
                        "source_type",
                        "source_id",
                        "target_type",
                        "target_id",
                    ],
                    set_={
                        "similarity": nb["score"],
                        "k_rank": rank,
                        "generation": new_gen,
                    },
                )
                await self.session.execute(stmt)
                total_created += 1

        # Prune stale edges (generation < new_gen)
        prune_stmt = delete(NavigationEdge).where(
            NavigationEdge.organization_id == organization_id,
            NavigationEdge.generation < new_gen,
        )
        prune_result = await self.session.execute(prune_stmt)
        pruned = prune_result.rowcount

        await self.session.flush()
        logger.info(
            "kNN rebuild complete: %d edges created, %d stale pruned",
            total_created, pruned,
        )
        return {
            "nodes_processed": len(nodes),
            "edges_created": total_created,
            "edges_pruned": pruned,
            "generation": new_gen,
        }

    # ── Public API: traversal ───────────────────────────────────────

    async def get_neighbours(
        self,
        *,
        organization_id: str,
        node_type: str,
        node_id: str,
        k: int = DEFAULT_K,
    ) -> List[Dict[str, Any]]:
        """Retrieve the k-nearest neighbours of a node from the materialized graph.

        Returns ``[{"type": ..., "id": ..., "similarity": ..., "k_rank": ...}]``.
        """
        stmt = (
            select(NavigationEdge)
            .where(
                NavigationEdge.organization_id == organization_id,
                NavigationEdge.source_type == node_type,
                NavigationEdge.source_id == node_id,
            )
            .order_by(NavigationEdge.k_rank.asc())
            .limit(k)
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [
            {
                "type": r.target_type,
                "id": r.target_id,
                "similarity": r.similarity,
                "k_rank": r.k_rank,
            }
            for r in rows
        ]

    async def traverse(
        self,
        *,
        organization_id: str,
        start_type: str,
        start_id: str,
        hops: int = 2,
        k: int = DEFAULT_K,
    ) -> List[Dict[str, Any]]:
        """Multi-hop BFS traversal on the navigation graph.

        Returns unique nodes reachable within ``hops`` steps.
        """
        visited: set[tuple[str, str]] = set()
        frontier: List[tuple[str, str]] = [(start_type, start_id)]
        result: List[Dict[str, Any]] = []

        for _hop in range(hops):
            next_frontier: List[tuple[str, str]] = []
            for ntype, nid in frontier:
                key = (ntype, nid)
                if key in visited:
                    continue
                visited.add(key)
                neighbours = await self.get_neighbours(
                    organization_id=organization_id,
                    node_type=ntype,
                    node_id=nid,
                    k=k,
                )
                for nb in neighbours:
                    nb_key = (nb["type"], nb["id"])
                    if nb_key not in visited:
                        result.append(nb)
                        next_frontier.append(nb_key)
            frontier = next_frontier

        return result

    # ── Internal helpers ────────────────────────────────────────────

    async def _collect_all_nodes(
        self, organization_id: str
    ) -> List[Dict[str, Any]]:
        """Collect all hierarchy nodes for an org."""
        nodes: List[Dict[str, Any]] = []

        # Episodes with vectors
        ep_stmt = select(MemoryEpisode).where(
            MemoryEpisode.organization_id == organization_id,
            MemoryEpisode.vector_id.isnot(None),
        )
        for ep in (await self.session.execute(ep_stmt)).scalars().all():
            nodes.append({"type": "episode", "id": ep.id, "vector_id": ep.vector_id})

        # Semantic nodes with vectors
        sn_stmt = select(MemorySemanticNode).where(
            MemorySemanticNode.organization_id == organization_id,
            MemorySemanticNode.vector_id.isnot(None),
        )
        for sn in (await self.session.execute(sn_stmt)).scalars().all():
            nodes.append({"type": "semantic_node", "id": sn.id, "vector_id": sn.vector_id})

        # Topics – topics don't have vector_id yet; compute on the fly
        topic_stmt = select(MemoryTopic).where(
            MemoryTopic.organization_id == organization_id,
        )
        for topic in (await self.session.execute(topic_stmt)).scalars().all():
            nodes.append({"type": "topic", "id": topic.id, "vector_id": None})

        return nodes

    async def _get_node_embedding(
        self,
        *,
        organization_id: str,
        node_type: str,
        node_id: str,
        vector_id: Optional[str],
    ) -> Optional[List[float]]:
        """Retrieve or compute the embedding for a hierarchy node."""
        # If we have a Qdrant vector_id, retrieve directly
        if vector_id:
            try:
                client = QdrantService.get_client()
                points = client.retrieve(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    ids=[vector_id],
                    with_vectors=True,
                )
                if points:
                    vec = points[0].vector
                    if isinstance(vec, list):
                        return vec
            except Exception as exc:
                logger.debug("Qdrant retrieve for %s failed: %s", vector_id, exc)

        # Fallback: compute from content
        if node_type == "topic":
            # Embed topic label + keywords
            topic = (
                await self.session.execute(
                    select(MemoryTopic).where(MemoryTopic.id == node_id)
                )
            ).scalar_one_or_none()
            if topic:
                text = f"{topic.label} {' '.join(topic.keywords or [])}"
                return await EmbeddingService.embed(text)

        if node_type == "semantic_node":
            sn = (
                await self.session.execute(
                    select(MemorySemanticNode).where(MemorySemanticNode.id == node_id)
                )
            ).scalar_one_or_none()
            if sn:
                return await EmbeddingService.embed(sn.content)

        if node_type == "episode":
            ep = (
                await self.session.execute(
                    select(MemoryEpisode).where(MemoryEpisode.id == node_id)
                )
            ).scalar_one_or_none()
            if ep and ep.narrative_summary:
                return await EmbeddingService.embed(ep.narrative_summary)

        return None

    async def _find_neighbours(
        self,
        *,
        organization_id: str,
        query_vector: List[float],
        exclude_vector_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Search Qdrant for nearest hierarchy centroids.

        Filters to only return points tagged with type ∈ {episode, semantic_node, topic}.
        """
        try:
            from qdrant_client.http.models import (
                FieldCondition,
                MatchAny,
                Filter,
            )

            additional_filters = [
                FieldCondition(
                    key="type",
                    match=MatchAny(any=["episode", "semantic_node", "topic"]),
                ),
            ]

            org_filter = QdrantService.build_org_filter(
                organization_id, additional_filters=additional_filters
            )

            client = QdrantService.get_client()
            results = client.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=org_filter,
                limit=limit,
                with_payload=True,
            )

            neighbours: List[Dict[str, Any]] = []
            for hit in results:
                payload = hit.payload or {}
                point_id = str(hit.id)
                if point_id == exclude_vector_id:
                    continue
                ntype = payload.get("type", "")
                nid = (
                    payload.get("episode_id")
                    or payload.get("semantic_node_id")
                    or payload.get("topic_id")
                    or ""
                )
                if ntype and nid:
                    neighbours.append({
                        "type": ntype,
                        "id": nid,
                        "score": float(hit.score),
                        "vector_id": point_id,
                    })
            return neighbours
        except Exception as exc:
            logger.warning("Qdrant kNN search failed: %s", exc)
            return []
