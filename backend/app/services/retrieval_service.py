"""Retrieval Service (GAP-3: Top-Down Retrieval with Representative Selection).

Implements **submodular greedy selection** to pick diverse, non-redundant
representatives from the memory hierarchy, then expands through the kNN
navigation graph to assemble full context.

Two-stage pipeline:
    Stage I:  Representative Selection
              Use submodular optimization to maximize:
                  f(S) = relevance(S) + λ·coverage(S)
              where coverage counts unique nodes reachable via kNN edges.

    Stage II: Adaptive Expansion
              For each representative, traverse kNN graph up to a depth limit,
              collecting related episodes and messages.

This addresses the "lack of cross-cluster traversal" gap identified in
xMemory framework analysis.  By selecting representatives that cover
diverse regions of memory space, we avoid redundant retrieval and improve
diversity of retrieved context.

Reference: xMemory Equation 7 (submodular greedy)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.qdrant import QdrantService
from app.models.memory_episode import MemoryEpisode
from app.models.memory_episode_membership import MemoryEpisodeMembership
from app.models.memory import MemoryMetadata
from app.models.memory_semantic_node import MemorySemanticNode
from app.models.memory_topic import MemoryTopic
from app.models.navigation_edge import NavigationEdge
from app.services.embedding_service import EmbeddingService
from app.services.uncertainty_gating_service import UncertaintyGatingService

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────
DEFAULT_MAX_REPRESENTATIVES = 5
DEFAULT_COVERAGE_WEIGHT = 0.3       # λ in f(S) = relevance + λ·coverage
DEFAULT_MAX_RESULTS = 20
DEFAULT_EXPANSION_DEPTH = 2         # kNN hops for expansion
MIN_RELEVANCE_THRESHOLD = 0.15      # filter out weak matches


class RetrievalService:
    """Top-down retrieval with representative selection (GAP-3) + uncertainty gating (GAP-4)."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.embedding_svc = EmbeddingService()
        self.qdrant = QdrantService()
        self.uncertainty_svc = UncertaintyGatingService()

    # ── Public API ──────────────────────────────────────────────────

    async def retrieve(
        self,
        *,
        query: str,
        organization_id: UUID,
        scope: str,
        scope_id: Optional[UUID] = None,
        max_representatives: int = DEFAULT_MAX_REPRESENTATIVES,
        coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
        max_results: int = DEFAULT_MAX_RESULTS,
        expansion_depth: int = DEFAULT_EXPANSION_DEPTH,
    ) -> Dict[str, Any]:
        """Two-stage retrieval: representative selection + kNN expansion.

        Args:
            query: Natural language query
            organization_id: Tenant scope
            scope: "personal" | "team" | "organization"
            scope_id: Owner/team ID if scoped
            max_representatives: Budget for Stage I submodular selection
            coverage_weight: λ in objective function f(S) = rel + λ·cov
            max_results: Max memory IDs to return after expansion
            expansion_depth: kNN graph traversal depth

        Returns:
            {
                "memory_ids": [str, ...],           # Ordered by relevance
                "representatives": [                 # Stage I selected nodes
                    {"type": str, "id": str, "score": float},
                    ...
                ],
                "coverage_stats": {
                    "total_nodes_covered": int,
                    "sparsity_score": float,
                },
            }
        """
        logger.info(
            "retrieve: query='%s' org=%s scope=%s max_repr=%d",
            query[:50],
            organization_id,
            scope,
            max_representatives,
        )

        # Stage I: Representative Selection
        representatives = await self._select_representatives(
            query=query,
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
            max_count=max_representatives,
            coverage_weight=coverage_weight,
        )

        # Stage II: Adaptive Expansion
        memory_ids = await self._expand_through_knn(
            representatives=representatives,
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
            depth=expansion_depth,
            max_results=max_results,
        )

        # Compute coverage stats
        coverage_stats = await self._compute_coverage_stats(
            representatives=representatives,
            organization_id=organization_id,
        )

        return {
            "memory_ids": memory_ids,
            "representatives": representatives,
            "coverage_stats": coverage_stats,
        }

    async def retrieve_with_gating(
        self,
        *,
        query: str,
        organization_id: UUID,
        scope: str,
        scope_id: Optional[UUID] = None,
        max_representatives: int = DEFAULT_MAX_REPRESENTATIVES,
        coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
        entropy_threshold: float = 0.1,
        max_expansion_items: int = 10,
    ) -> Dict[str, Any]:
        """Two-stage retrieval with uncertainty-gated expansion (GAP-3 + GAP-4).

        Stage I:  Submodular greedy selection of representatives
        Stage II: Entropy-gated adaptive evidence admission per Eq. (8)

        Args:
            query: Natural language query
            organization_id: Tenant scope
            scope: "personal" | "team" | "organization"
            scope_id: Owner/team ID if scoped
            max_representatives: Budget for Stage I
            coverage_weight: λ in Stage I objective
            entropy_threshold: Minimum ΔH for evidence inclusion (bits)
            max_expansion_items: Maximum items to add in Stage II

        Returns:
            {
                "final_context": str,
                "representatives": [...],
                "included_items": [{"id": str, "type": str, "entropy_reduction": float}, ...],
                "total_entropy_reduction": float,
                "expansion_stopped_reason": str,
                "coverage_stats": {...},
            }
        """
        logger.info(
            "retrieve_with_gating: query='%s' org=%s ΔH_threshold=%.3f",
            query[:50],
            organization_id,
            entropy_threshold,
        )

        # Stage I: Submodular greedy representative selection
        representatives = await self._select_representatives(
            query=query,
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
            max_count=max_representatives,
            coverage_weight=coverage_weight,
        )

        # Build initial context from representatives (summaries)
        initial_context = await self._build_initial_context(
            representatives=representatives,
            organization_id=organization_id,
        )

        # Get candidate evidence for Stage II (episodes + messages)
        candidate_items = await self._get_candidate_evidence(
            representatives=representatives,
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )

        # Stage II: Uncertainty-gated expansion
        gating_result = await self.uncertainty_svc.expand_with_gating(
            query=query,
            initial_context=initial_context,
            candidate_items=candidate_items,
            max_items=max_expansion_items,
            entropy_threshold=entropy_threshold,
        )

        # Compute coverage stats
        coverage_stats = await self._compute_coverage_stats(
            representatives=representatives,
            organization_id=organization_id,
        )

        return {
            "final_context": gating_result["final_context"],
            "representatives": representatives,
            "included_items": gating_result["included_items"],
            "total_entropy_reduction": gating_result["total_entropy_reduction"],
            "expansion_stopped_reason": gating_result["expansion_stopped_reason"],
            "coverage_stats": coverage_stats,
        }

    # ── Stage I: Submodular Greedy Selection ────────────────────────

    async def _select_representatives(
        self,
        *,
        query: str,
        organization_id: UUID,
        scope: str,
        scope_id: Optional[UUID],
        max_count: int,
        coverage_weight: float,
    ) -> List[Dict[str, Any]]:
        """Greedy submodular maximization: f(S) = relevance(S) + λ·coverage(S).

        1. Get candidate topics + semantic nodes from Qdrant
        2. For each candidate, compute marginal gain = Δrelevance + λ·Δcoverage
        3. Greedily select candidate with max marginal gain
        4. Repeat until budget exhausted
        """
        # Embed query
        query_vec = await self.embedding_svc.get_embedding(
            text=query,
            model=settings.LOCAL_EMBEDDING_MODEL,
        )
        if not query_vec or not any(v != 0 for v in query_vec):
            logger.warning("Empty query embedding – returning empty results")
            return []

        # Qdrant search: get topics + semantic nodes
        candidates = await self._get_candidates(
            query_vector=query_vec,
            organization_id=organization_id,
            scope=scope,
            scope_id=scope_id,
        )

        if not candidates:
            logger.info("No candidates found for query")
            return []

        # Precompute kNN coverage for each candidate
        coverage_map = await self._build_coverage_map(
            candidates=candidates,
            organization_id=organization_id,
        )

        # Greedy selection
        selected: List[Dict[str, Any]] = []
        covered_nodes: Set[str] = set()
        total_relevance = 0.0

        for _ in range(min(max_count, len(candidates))):
            best_gain = -float("inf")
            best_idx = -1

            for i, cand in enumerate(candidates):
                if cand["selected"]:
                    continue

                # Marginal gain
                new_coverage = coverage_map[cand["key"]] - covered_nodes
                delta_cov = len(new_coverage)
                delta_rel = cand["score"]
                gain = delta_rel + coverage_weight * delta_cov

                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx == -1:
                break

            # Select best candidate
            cand = candidates[best_idx]
            cand["selected"] = True
            selected.append({
                "type": cand["type"],
                "id": cand["id"],
                "score": cand["score"],
            })
            covered_nodes.update(coverage_map[cand["key"]])
            total_relevance += cand["score"]

        logger.info(
            "Selected %d representatives (coverage=%d nodes, relevance=%.3f)",
            len(selected),
            len(covered_nodes),
            total_relevance,
        )
        return selected

    async def _get_candidates(
        self,
        *,
        query_vector: List[float],
        organization_id: UUID,
        scope: str,
        scope_id: Optional[UUID],
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates from Qdrant: topics + semantic nodes."""
        candidates = []

        # Build filter for scope
        org_id_str = str(organization_id)
        scope_filter = {
            "must": [
                {"key": "organization_id", "match": {"value": org_id_str}},
            ],
        }

        # Search topics
        try:
            topic_results = await self.qdrant.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=scope_filter,
                limit=30,
            )
            for hit in topic_results:
                payload = hit.payload or {}
                if payload.get("memory_type") != "topic":
                    continue
                if hit.score < MIN_RELEVANCE_THRESHOLD:
                    continue
                candidates.append({
                    "type": "topic",
                    "id": payload.get("memory_id", ""),
                    "key": f"topic:{payload.get('memory_id')}",
                    "score": hit.score,
                    "selected": False,
                })
        except Exception as exc:
            logger.warning("Qdrant topic search failed: %s", exc)

        # Search semantic nodes
        try:
            node_results = await self.qdrant.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=scope_filter,
                limit=50,
            )
            for hit in node_results:
                payload = hit.payload or {}
                if payload.get("memory_type") != "semantic_node":
                    continue
                if hit.score < MIN_RELEVANCE_THRESHOLD:
                    continue
                candidates.append({
                    "type": "semantic_node",
                    "id": payload.get("memory_id", ""),
                    "key": f"semantic_node:{payload.get('memory_id')}",
                    "score": hit.score,
                    "selected": False,
                })
        except Exception as exc:
            logger.warning("Qdrant semantic node search failed: %s", exc)

        # Sort by score descending
        candidates.sort(key=lambda c: c["score"], reverse=True)
        logger.debug("Found %d candidates", len(candidates))
        return candidates

    async def _build_coverage_map(
        self,
        *,
        candidates: List[Dict[str, Any]],
        organization_id: UUID,
    ) -> Dict[str, Set[str]]:
        """Build map of candidate -> set of covered node IDs via kNN graph."""
        coverage_map = {}

        for cand in candidates:
            key = cand["key"]
            node_type = cand["type"]
            node_id = cand["id"]

            # Get outgoing kNN edges
            stmt = select(NavigationEdge).where(
                and_(
                    NavigationEdge.organization_id == str(organization_id),
                    NavigationEdge.source_type == node_type,
                    NavigationEdge.source_id == node_id,
                )
            )
            result = await self.session.execute(stmt)
            edges = result.scalars().all()

            covered = {f"{node_type}:{node_id}"}  # Include self
            for edge in edges:
                covered.add(f"{edge.target_type}:{edge.target_id}")

            coverage_map[key] = covered

        return coverage_map

    # ── Stage II: Adaptive Expansion ────────────────────────────────

    async def _expand_through_knn(
        self,
        *,
        representatives: List[Dict[str, Any]],
        organization_id: UUID,
        scope: str,
        scope_id: Optional[UUID],
        depth: int,
        max_results: int,
    ) -> List[str]:
        """Expand each representative through kNN graph to collect memory IDs."""
        memory_ids_set: Set[str] = set()
        visited_nodes: Set[str] = set()

        for rep in representatives:
            node_key = f"{rep['type']}:{rep['id']}"
            frontier = [(node_key, 0)]  # (node_key, current_depth)

            while frontier and len(memory_ids_set) < max_results:
                current_key, current_depth = frontier.pop(0)
                if current_key in visited_nodes:
                    continue
                visited_nodes.add(current_key)

                # Collect memory IDs from this node
                await self._collect_memory_ids(
                    node_key=current_key,
                    organization_id=organization_id,
                    scope=scope,
                    scope_id=scope_id,
                    memory_ids_set=memory_ids_set,
                )

                # Expand neighbors if depth allows
                if current_depth < depth:
                    neighbors = await self._get_neighbors(
                        node_key=current_key,
                        organization_id=organization_id,
                    )
                    for neighbor_key in neighbors:
                        if neighbor_key not in visited_nodes:
                            frontier.append((neighbor_key, current_depth + 1))

        # Convert to list (ordered by insertion for stability)
        memory_ids = list(memory_ids_set)[:max_results]
        logger.info("Expanded to %d memory IDs", len(memory_ids))
        return memory_ids

    async def _collect_memory_ids(
        self,
        *,
        node_key: str,
        organization_id: UUID,
        scope: str,
        scope_id: Optional[UUID],
        memory_ids_set: Set[str],
    ) -> None:
        """Collect memory IDs from a given node (topic/semantic_node/episode)."""
        node_type, node_id_str = node_key.split(":", 1)

        if node_type == "topic":
            # Get all semantic nodes in this topic
            stmt = select(MemorySemanticNode.id).where(
                and_(
                    MemorySemanticNode.organization_id == str(organization_id),
                    MemorySemanticNode.topic_id == node_id_str,
                )
            )
            result = await self.session.execute(stmt)
            node_ids = [str(row[0]) for row in result.all()]
            memory_ids_set.update(node_ids)

        elif node_type == "semantic_node":
            # Add the semantic node itself + source memories
            memory_ids_set.add(node_id_str)
            stmt = select(MemorySemanticNode).where(
                and_(
                    MemorySemanticNode.organization_id == str(organization_id),
                    MemorySemanticNode.id == node_id_str,
                )
            )
            result = await self.session.execute(stmt)
            node = result.scalar_one_or_none()
            if node and node.source_memory_ids:
                memory_ids_set.update(node.source_memory_ids)

        elif node_type == "episode":
            # Get all messages in episode via episode memberships
            stmt = select(MemoryEpisodeMembership.memory_id).where(
                and_(
                    MemoryEpisodeMembership.organization_id == str(organization_id),
                    MemoryEpisodeMembership.episode_id == node_id_str,
                )
            )
            result = await self.session.execute(stmt)
            msg_ids = [str(row[0]) for row in result.all()]
            memory_ids_set.update(msg_ids)

    async def _get_neighbors(
        self,
        *,
        node_key: str,
        organization_id: UUID,
    ) -> List[str]:
        """Get kNN neighbors of a node."""
        node_type, node_id = node_key.split(":", 1)

        stmt = select(NavigationEdge).where(
            and_(
                NavigationEdge.organization_id == str(organization_id),
                NavigationEdge.source_type == node_type,
                NavigationEdge.source_id == node_id,
            )
        )
        result = await self.session.execute(stmt)
        edges = result.scalars().all()

        neighbors = [f"{edge.target_type}:{edge.target_id}" for edge in edges]
        return neighbors

    # ── GAP-4: Uncertainty Gating Helpers ───────────────────────────

    async def _build_initial_context(
        self,
        *,
        representatives: List[Dict[str, Any]],
        organization_id: UUID,
    ) -> str:
        """Build coarse context from Stage I representatives (summaries only)."""
        context_parts = []

        for rep in representatives:
            node_type = rep["type"]
            node_id = rep["id"]

            if node_type == "topic":
                # Fetch topic summary
                stmt = select(MemorySemanticNode).where(
                    and_(
                        MemorySemanticNode.organization_id == str(organization_id),
                        MemorySemanticNode.id == node_id,
                    )
                )
                result = await self.session.execute(stmt)
                topic = result.scalar_one_or_none()
                if topic and topic.summary:
                    context_parts.append(f"Topic: {topic.summary}")

            elif node_type == "semantic_node":
                # Fetch node summary
                stmt = select(MemorySemanticNode).where(
                    and_(
                        MemorySemanticNode.organization_id == str(organization_id),
                        MemorySemanticNode.id == node_id,
                    )
                )
                result = await self.session.execute(stmt)
                node = result.scalar_one_or_none()
                if node and node.summary:
                    context_parts.append(f"Node: {node.summary}")

        return "\n\n".join(context_parts)

    async def _get_candidate_evidence(
        self,
        *,
        representatives: List[Dict[str, Any]],
        organization_id: UUID,
        scope: str,
        scope_id: Optional[UUID],
    ) -> List[Dict[str, Any]]:
        """Fetch candidate evidence items (episodes + messages) for Stage II."""
        from app.models.memory import MemoryMetadata

        candidate_items = []

        # Collect all semantic node IDs from representatives
        semantic_node_ids = []
        for rep in representatives:
            if rep["type"] == "semantic_node":
                semantic_node_ids.append(rep["id"])
            elif rep["type"] == "topic":
                # Get all nodes in topic
                stmt = select(MemorySemanticNode.id).where(
                    and_(
                        MemorySemanticNode.organization_id == str(organization_id),
                        MemorySemanticNode.topic_id == rep["id"],
                    )
                )
                result = await self.session.execute(stmt)
                node_ids = [str(row[0]) for row in result.all()]
                semantic_node_ids.extend(node_ids)

        # Get episodes containing these nodes
        episode_ids_set: Set[str] = set()
        for node_id in semantic_node_ids:
            stmt = select(MemorySemanticNode).where(
                and_(
                    MemorySemanticNode.organization_id == str(organization_id),
                    MemorySemanticNode.id == node_id,
                )
            )
            result = await self.session.execute(stmt)
            node = result.scalar_one_or_none()
            if node and node.episode_id:
                episode_ids_set.add(node.episode_id)

        # Fetch episode messages
        for episode_id in episode_ids_set:
            # Get all messages in episode
            stmt = (
                select(MemoryMetadata)
                .join(
                    MemoryEpisodeMembership,
                    MemoryMetadata.id == MemoryEpisodeMembership.memory_id,
                )
                .where(
                    and_(
                        MemoryEpisodeMembership.organization_id == str(organization_id),
                        MemoryEpisodeMembership.episode_id == episode_id,
                        MemoryMetadata.organization_id == str(organization_id),
                    )
                )
            )

            # Apply scope filter
            if scope == "personal" and scope_id:
                stmt = stmt.where(MemoryMetadata.owner_user_id == str(scope_id))
            elif scope == "team" and scope_id:
                stmt = stmt.where(MemoryMetadata.team_id == str(scope_id))

            result = await self.session.execute(stmt)
            messages = result.scalars().all()

            for msg in messages:
                candidate_items.append({
                    "id": str(msg.id),
                    "type": "message",
                    "content": msg.full_text or "",
                    "metadata": {
                        "episode_id": episode_id,
                        "created_at": msg.created_at.isoformat() if msg.created_at else None,
                    },
                })

        logger.info("Collected %d candidate evidence items", len(candidate_items))
        return candidate_items

    # ── Coverage Statistics ─────────────────────────────────────────

    async def _compute_coverage_stats(
        self,
        *,
        representatives: List[Dict[str, Any]],
        organization_id: UUID,
    ) -> Dict[str, Any]:
        """Compute how many unique nodes are covered by selected representatives."""
        all_covered: Set[str] = set()

        for rep in representatives:
            node_type = rep["type"]
            node_id = rep["id"]

            # Get outgoing edges
            stmt = select(NavigationEdge).where(
                and_(
                    NavigationEdge.organization_id == str(organization_id),
                    NavigationEdge.source_type == node_type,
                    NavigationEdge.source_id == node_id,
                )
            )
            result = await self.session.execute(stmt)
            edges = result.scalars().all()

            all_covered.add(f"{node_type}:{node_id}")
            for edge in edges:
                all_covered.add(f"{edge.target_type}:{edge.target_id}")

        # Sparsity approximation
        num_repr = len(representatives)
        avg_coverage_per_repr = len(all_covered) / max(num_repr, 1)
        sparsity = avg_coverage_per_repr / max(num_repr, 1)

        return {
            "total_nodes_covered": len(all_covered),
            "sparsity_score": sparsity,
        }
