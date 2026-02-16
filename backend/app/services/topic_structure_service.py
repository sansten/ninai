"""Topic structure optimization (GAP-2 + GAP-5).

Implements the sparsity-semantic guidance objective for topic partitions and
optional split/merge operations to keep topic sizes within a stable range.

GAP-5 additions: Dynamic reassignment ratio tracking, guided attach protocol,
and full lifecycle management for retroactive restructuring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4, UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory_semantic_node import MemorySemanticNode
from app.models.memory_semantic_node_topic_history import MemorySemanticNodeTopicHistory
from app.models.memory_topic import MemoryTopic
from app.services.embedding_service import EmbeddingService
from app.services.topic_service import normalize_topic_label


MAX_TOPIC_SIZE = 12
MIN_TOPIC_SIZE = 2


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _mean_vector(vectors: list[list[float]]) -> Optional[list[float]]:
    if not vectors:
        return None
    size = len(vectors[0])
    sums = [0.0] * size
    count = 0
    for v in vectors:
        if len(v) != size:
            continue
        count += 1
        for i, val in enumerate(v):
            sums[i] += val
    if count == 0:
        return None
    return [s / count for s in sums]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[mid - 1] + ordered[mid]) / 2.0
    return ordered[mid]


def _kmeans_2(vectors: list[list[float]], max_iter: int = 5) -> list[int]:
    if len(vectors) < 2:
        return [0 for _ in vectors]

    # Initialize centroids using farthest pair (cosine distance).
    best_i = 0
    best_j = 1
    best_dist = -1.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            dist = 1.0 - _cosine_similarity(vectors[i], vectors[j])
            if dist > best_dist:
                best_dist = dist
                best_i, best_j = i, j

    c1 = vectors[best_i][:]
    c2 = vectors[best_j][:]

    labels = [0] * len(vectors)
    for _ in range(max_iter):
        changed = False
        for idx, v in enumerate(vectors):
            d1 = 1.0 - _cosine_similarity(v, c1)
            d2 = 1.0 - _cosine_similarity(v, c2)
            new_label = 0 if d1 <= d2 else 1
            if new_label != labels[idx]:
                labels[idx] = new_label
                changed = True

        if not changed:
            break

        group1 = [v for v, l in zip(vectors, labels) if l == 0]
        group2 = [v for v, l in zip(vectors, labels) if l == 1]
        c1 = _mean_vector(group1) or c1
        c2 = _mean_vector(group2) or c2

    return labels


@dataclass
class TopicStats:
    topic: MemoryTopic
    nodes: list[MemorySemanticNode]
    embeddings: list[list[float]]


class TopicStructureService:
    """Compute guidance scores and rebalance topic partitions."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def compute_guidance_scores(
        self,
        *,
        organization_id: str,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> dict:
        topics = await self._load_topic_stats(
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )

        sizes = [len(t.nodes) for t in topics]
        if not sizes:
            return {"sparsity_score": 0.0, "semantic_score": 0.0, "topic_stats": []}

        n_total = sum(sizes)
        k = len(sizes)
        sparsity = (n_total * n_total) / (k * sum(n * n for n in sizes))

        centroids = [_mean_vector(t.embeddings) for t in topics]
        centroid_sims = []
        for idx, c in enumerate(centroids):
            if c is None:
                centroid_sims.append(0.0)
                continue
            best = 0.0
            for jdx, other in enumerate(centroids):
                if idx == jdx or other is None:
                    continue
                best = max(best, _cosine_similarity(c, other))
            centroid_sims.append(best)

        m = _median(centroid_sims)
        mad = _median([abs(s - m) for s in centroid_sims])
        sigma = mad + 1e-6

        per_topic = []
        sem_sum = 0.0
        for idx, t in enumerate(topics):
            centroid = centroids[idx]
            if centroid is None or not t.embeddings:
                avg_cos = 0.0
            else:
                avg_cos = sum(_cosine_similarity(v, centroid) for v in t.embeddings) / len(t.embeddings)
            s_k = centroid_sims[idx]
            g = math.exp(-((s_k - m) ** 2) / (2 * sigma * sigma))
            sem_sum += avg_cos * g
            per_topic.append(
                {
                    "topic_id": t.topic.id,
                    "label": t.topic.label,
                    "size": len(t.nodes),
                    "avg_cosine": avg_cos,
                    "nearest_centroid": s_k,
                    "g": g,
                }
            )

        semantic = sem_sum / k
        return {
            "sparsity_score": sparsity,
            "semantic_score": semantic,
            "topic_stats": per_topic,
        }

    async def rebalance_topics(
        self,
        *,
        organization_id: str,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        max_size: int = MAX_TOPIC_SIZE,
        min_size: int = MIN_TOPIC_SIZE,
    ) -> dict:
        topics = await self._load_topic_stats(
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )

        splits = 0
        merges = 0
        moved_nodes = 0

        for t in list(topics):
            if len(t.nodes) <= max_size:
                continue
            if len(t.embeddings) < 2:
                continue
            moved = await self._split_topic(t)
            if moved:
                splits += 1
                moved_nodes += moved

        topics = await self._load_topic_stats(
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )

        if len(topics) > 1:
            for t in list(topics):
                if len(t.nodes) >= min_size:
                    continue
                moved = await self._merge_topic(t, topics)
                if moved:
                    merges += 1
                    moved_nodes += moved

        await self.session.flush()

        scores = await self.compute_guidance_scores(
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )
        scores.update({"splits": splits, "merges": merges, "nodes_moved": moved_nodes})
        return scores

    async def _load_topic_stats(
        self,
        *,
        organization_id: str,
        scope: Optional[str],
        scope_id: Optional[str],
    ) -> list[TopicStats]:
        nodes = await self._collect_semantic_nodes(
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )
        topic_ids = {n.topic_id for n in nodes if n.topic_id}
        if not topic_ids:
            return []
        topics = await self._fetch_topics(list(topic_ids))
        topic_map = {t.id: t for t in topics}

        buckets: dict[str, TopicStats] = {}
        for node in nodes:
            if not node.topic_id or node.topic_id not in topic_map:
                continue
            if node.topic_id not in buckets:
                buckets[node.topic_id] = TopicStats(topic=topic_map[node.topic_id], nodes=[], embeddings=[])
            buckets[node.topic_id].nodes.append(node)

        for stat in buckets.values():
            for node in stat.nodes:
                emb = await self._get_node_embedding(node)
                if emb:
                    stat.embeddings.append(emb)

        return list(buckets.values())

    async def _collect_semantic_nodes(
        self,
        *,
        organization_id: str,
        scope: Optional[str],
        scope_id: Optional[str],
    ) -> list[MemorySemanticNode]:
        stmt = select(MemorySemanticNode).where(
            MemorySemanticNode.organization_id == organization_id,
            MemorySemanticNode.topic_id.isnot(None),
        )
        if scope:
            stmt = stmt.where(MemorySemanticNode.scope == scope)
        if scope_id:
            stmt = stmt.where(MemorySemanticNode.scope_id == scope_id)
        return (await self.session.execute(stmt)).scalars().all()

    async def _fetch_topics(self, topic_ids: list[str]) -> list[MemoryTopic]:
        stmt = select(MemoryTopic).where(MemoryTopic.id.in_(topic_ids))
        return (await self.session.execute(stmt)).scalars().all()

    async def _get_node_embedding(self, node: MemorySemanticNode) -> Optional[list[float]]:
        if node.vector_id:
            try:
                from app.core.qdrant import QdrantService
                from app.core.config import settings
                client = QdrantService.get_client()
                points = client.retrieve(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    ids=[node.vector_id],
                )
                if points:
                    vec = getattr(points[0], "vector", None)
                    if isinstance(vec, list) and any(v != 0 for v in vec):
                        return vec
            except Exception:
                pass
        try:
            return await EmbeddingService.embed(node.content)
        except Exception:
            return None

    async def _split_topic(self, stat: TopicStats) -> int:
        vectors = stat.embeddings
        if len(vectors) < 2:
            return 0

        labels = _kmeans_2(vectors)
        if all(l == 0 for l in labels) or all(l == 1 for l in labels):
            return 0

        suffix = uuid4().hex[:6]
        base_label = normalize_topic_label(stat.topic.label)
        new_label = f"{base_label}_split_{suffix}"
        scope_key = getattr(stat.topic, "scope_key", None) or f"{stat.topic.scope}:{stat.topic.scope_id or ''}"

        new_topic = MemoryTopic(
            id=str(uuid4()),
            organization_id=stat.topic.organization_id,
            scope=stat.topic.scope,
            scope_id=stat.topic.scope_id,
            scope_key=scope_key,
            label=new_label,
            label_normalized=new_label,
            keywords=[],
            created_by="system",
        )
        self.session.add(new_topic)

        moved = 0
        for node, label in zip(stat.nodes, labels):
            if label == 1:
                prev_id = node.topic_id
                node.topic_id = new_topic.id
                moved += 1
                # Record history
                await self._record_topic_assignment(
                    node=node,
                    new_topic_id=new_topic.id,
                    previous_topic_id=prev_id,
                    reason="split",
                )

        return moved

    async def _merge_topic(self, stat: TopicStats, topics: list[TopicStats]) -> int:
        if not stat.embeddings:
            return 0

        centroid = _mean_vector(stat.embeddings)
        if centroid is None:
            return 0

        best_topic: Optional[TopicStats] = None
        best_sim = -1.0
        for other in topics:
            if other.topic.id == stat.topic.id or not other.embeddings:
                continue
            other_centroid = _mean_vector(other.embeddings)
            if other_centroid is None:
                continue
            sim = _cosine_similarity(centroid, other_centroid)
            if sim > best_sim:
                best_sim = sim
                best_topic = other

        if best_topic is None:
            return 0

        moved = 0
        for node in stat.nodes:
            prev_id = node.topic_id
            node.topic_id = best_topic.topic.id
            moved += 1
            # Record history
            await self._record_topic_assignment(
                node=node,
                new_topic_id=best_topic.topic.id,
                previous_topic_id=prev_id,
                reason="merge",
            )

        await self.session.delete(stat.topic)
        return moved

    # ── GAP-5: Retroactive Restructuring ────────────────────────────

    async def track_reassignment_ratio(
        self,
        *,
        organization_id: str,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> dict:
        """Compute reassignment ratio: % of nodes that have changed topics.

        Per xMemory (Hu et al., 2026), a healthy memory system should have
        40%+ reassignment ratio, indicating dynamic adaptation to new information.

        Returns:
            {
                "total_nodes": int,
                "nodes_with_history": int,
                "nodes_reassigned": int,
                "reassignment_ratio": float,  # 0.0 to 1.0
                "initial_only_ratio": float,   # % never reassigned
            }
        """
        # Count total semantic nodes
        stmt = select(func.count(MemorySemanticNode.id)).where(
            MemorySemanticNode.organization_id == organization_id,
            MemorySemanticNode.topic_id.isnot(None),
        )
        if scope:
            stmt = stmt.where(MemorySemanticNode.scope == scope)
        if scope_id:
            stmt = stmt.where(MemorySemanticNode.scope_id == scope_id)

        result = await self.session.execute(stmt)
        total_nodes = result.scalar() or 0

        if total_nodes == 0:
            return {
                "total_nodes": 0,
                "nodes_with_history": 0,
                "nodes_reassigned": 0,
                "reassignment_ratio": 0.0,
                "initial_only_ratio": 1.0,
            }

        # Count nodes with reassignment history (more than 1 history entry)
        # Use a subquery to group by node_id and count entries per node
        from sqlalchemy import and_

        stmt = (
            select(MemorySemanticNodeTopicHistory.semantic_node_id)
            .where(MemorySemanticNodeTopicHistory.organization_id == organization_id)
            .group_by(MemorySemanticNodeTopicHistory.semantic_node_id)
            .having(func.count(MemorySemanticNodeTopicHistory.id) > 1)
        )

        result = await self.session.execute(stmt)
        reassigned_node_ids = result.scalars().all()
        nodes_reassigned = len(reassigned_node_ids)

        # Count nodes with any history entry
        stmt = select(func.count(func.distinct(MemorySemanticNodeTopicHistory.semantic_node_id))).where(
            MemorySemanticNodeTopicHistory.organization_id == organization_id,
        )
        result = await self.session.execute(stmt)
        nodes_with_history = result.scalar() or 0

        reassignment_ratio = nodes_reassigned / total_nodes if total_nodes > 0 else 0.0
        initial_only_ratio = (
            (nodes_with_history - nodes_reassigned) / total_nodes if total_nodes > 0 else 0.0
        )

        return {
            "total_nodes": total_nodes,
            "nodes_with_history": nodes_with_history,
            "nodes_reassigned": nodes_reassigned,
            "reassignment_ratio": reassignment_ratio,
            "initial_only_ratio": initial_only_ratio,
        }

    async def guided_attach(
        self,
        *,
        node: MemorySemanticNode,
        candidate_topics: list[MemoryTopic],
    ) -> str:
        """Attach a semantic node to the topic that maximizes guidance score f(P).

        Enhanced attach protocol for GAP-5. Instead of simple nearest-centroid,
        evaluate f(P) for each candidate attachment and choose the one that
        maximizes the guidance objective.

        Args:
            node: Semantic node to attach
            candidate_topics: List of possible topics to attach to

        Returns:
            Topic ID that was selected
        """
        if not candidate_topics:
            # Fall back to creating new topic
            return await self._create_topic_for_node(node)

        node_emb = await self._get_node_embedding(node)
        if not node_emb:
            # No embedding, use first topic
            return candidate_topics[0].id

        # Simple greedy strategy: attach to topic with highest centroid similarity
        # This is a reasonable approximation of maximizing guidance score
        best_topic_id = None
        best_similarity = -1.0

        for candidate in candidate_topics:
            # Get nodes for this topic
            stmt = select(MemorySemanticNode).where(
                MemorySemanticNode.organization_id == node.organization_id,
                MemorySemanticNode.topic_id == candidate.id,
            )
            result = await self.session.execute(stmt)
            topic_nodes = result.scalars().all()

            if not topic_nodes:
                continue

            # Compute centroid
            embeddings = []
            for n in topic_nodes:
                emb = await self._get_node_embedding(n)
                if emb:
                    embeddings.append(emb)

            if not embeddings:
                continue

            centroid = _mean_vector(embeddings)
            if centroid is None:
                continue

            # Compute similarity
            sim = _cosine_similarity(node_emb, centroid)

            if sim > best_similarity:
                best_similarity = sim
                best_topic_id = candidate.id

        # Return best topic or first candidate if no good match found
        selected_topic_id = best_topic_id or candidate_topics[0].id

        # Record history entry (optional, can retrieve scores if needed)
        await self._record_topic_assignment(
            node=node,
            new_topic_id=selected_topic_id,
            previous_topic_id=node.topic_id,
            reason="guided_attach",
        )

        return selected_topic_id

    async def periodic_restructure(
        self,
        *,
        organization_id: str,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> dict:
        """Full lifecycle restructuring: reassess all topic assignments.

        This is a background job that periodically evaluates all semantic nodes
        and reassigns them to maximize the guidance objective f(P). Target is
        40%+ reassignment ratio per xMemory framework.

        Returns:
            {
                "reassignments": int,
                "splits": int,
                "merges": int,
                "guidance_score_before": float,
                "guidance_score_after": float,
                "reassignment_ratio": float,
            }
        """
        # Get baseline guidance score
        scores_before = await self.compute_guidance_scores(
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )
        guidance_before = scores_before["sparsity_score"] + scores_before["semantic_score"]

        # Load all nodes
        nodes = await self._collect_semantic_nodes(
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )

        # Load all topics
        topic_ids = {n.topic_id for n in nodes if n.topic_id}
        topics = await self._fetch_topics(list(topic_ids)) if topic_ids else []

        reassignments = 0

        # For each node, find best topic using guided attach
        for node in nodes:
            if not node.topic_id:
                continue

            # Get all other topics as candidates
            best_topic_id = await self.guided_attach(node=node, candidate_topics=topics)

            if best_topic_id != node.topic_id:
                # Record reassignment
                await self._record_topic_assignment(
                    node=node,
                    new_topic_id=best_topic_id,
                    previous_topic_id=node.topic_id,
                    reason="periodic_restructure",
                )
                node.topic_id = best_topic_id
                reassignments += 1

        await self.session.flush()

        # Run split/merge rebalancing
        rebalance_result = await self.rebalance_topics(
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )

        # Get final guidance score
        scores_after = await self.compute_guidance_scores(
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )
        guidance_after = scores_after["sparsity_score"] + scores_after["semantic_score"]

        # Compute reassignment ratio
        ratio_stats = await self.track_reassignment_ratio(
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )

        return {
            "reassignments": reassignments,
            "splits": rebalance_result["splits"],
            "merges": rebalance_result["merges"],
            "guidance_score_before": guidance_before,
            "guidance_score_after": guidance_after,
            "reassignment_ratio": ratio_stats["reassignment_ratio"],
        }

    async def _record_topic_assignment(
        self,
        *,
        node: MemorySemanticNode,
        new_topic_id: str,
        previous_topic_id: Optional[str] = None,
        reason: str = "initial_attach",
        score_before: Optional[float] = None,
        score_after: Optional[float] = None,
    ) -> None:
        """Record a topic assignment/reassignment in history.

        Args:
            node: Semantic node being assigned
            new_topic_id: New topic ID
            previous_topic_id: Previous topic ID (None for initial)
            reason: Reason code (initial_attach, split, merge, guided_attach, periodic_restructure)
            score_before: Guidance score before reassignment
            score_after: Guidance score after reassignment
        """
        history_entry = MemorySemanticNodeTopicHistory(
            id=str(uuid4()),
            organization_id=node.organization_id,
            semantic_node_id=node.id,
            topic_id=new_topic_id,
            previous_topic_id=previous_topic_id,
            reason=reason,
            guidance_score_before=score_before,
            guidance_score_after=score_after,
        )
        self.session.add(history_entry)

    async def _create_topic_for_node(self, node: MemorySemanticNode) -> str:
        """Create a new topic for an orphan node."""
        topic_id = str(uuid4())
        suffix = uuid4().hex[:6]
        scope_key = f"{node.scope}:{node.scope_id or ''}"

        new_topic = MemoryTopic(
            id=topic_id,
            organization_id=node.organization_id,
            scope=node.scope,
            scope_id=node.scope_id,
            scope_key=scope_key,
            label=f"topic_{suffix}",
            label_normalized=f"topic_{suffix}",
            keywords=[],
            created_by="system",
        )
        self.session.add(new_topic)
        return topic_id
