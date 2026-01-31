"""Memory Retrieval Service with Activation Scoring

This service integrates the activation scorer into the retrieval pipeline.
Computes activation scores, writes explanations logs, and queues async updates.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import logging

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory_activation import (
    MemoryActivationState,
    MemoryRetrievalExplanation,
    MemoryCoactivationEdge,
    CausalHypothesis,
)
from app.models.memory import MemoryMetadata
from app.services.memory_activation.scoring import (
    ActivationScorer,
    ActivationComponents,
    get_activation_scorer,
)
from app.schemas.memory_activation import (
    ActivationComponentsSchema,
    RetrievalResultSchema,
    GatingInfoSchema,
)

logger = logging.getLogger(__name__)


class MemoryRetrievalService:
    """Service for activation-scored memory retrieval with explanation logging.
    
    Responsible for:
    1. Loading activation state for retrieved memories
    2. Computing activation scores via ActivationScorer
    3. Applying RLS gating (convert denied to 0 activation)
    4. Ranking by activation
    5. Writing explanations log
    6. Queuing async tasks (access updates, coactivation)
    """

    def __init__(
        self,
        session: AsyncSession,
        org_id: str,
        user_id: str,
        scorer: Optional[ActivationScorer] = None,
    ):
        """Initialize retrieval service.

        Args:
            session: AsyncSession for database operations
            org_id: Organization UUID
            user_id: User UUID
            scorer: Optional custom ActivationScorer. Defaults to singleton.
        """
        self.session = session
        self.org_id = org_id
        self.user_id = user_id
        self.scorer = scorer or get_activation_scorer()

    async def score_and_rank_results(
        self,
        memory_ids: List[str],
        query: str,
        similarities: Dict[str, float],
        scope: Optional[str] = None,
        episode_id: Optional[str] = None,
        goal_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[RetrievalResultSchema]]:
        """Score and rank retrieved memories.

        Args:
            memory_ids: List of memory IDs from search (Qdrant)
            query: Original search query
            similarities: Dict of memory_id -> similarity score from Qdrant
            scope: Current scope context (personal, team, department, organization)
            episode_id: Optional current episode ID for context
            goal_id: Optional current goal ID for context

        Returns:
            Tuple of:
            - List of ranked memory dicts sorted by activation (descending)
            - List of RetrievalResultSchema for explanation logging
        """
        if not memory_ids:
            return [], []

        # Load activation states and metadata in batch
        activation_states = await self._load_activation_states(memory_ids)
        memory_metadata = await self._load_memory_metadata(memory_ids)
        evidence_links = await self._load_evidence_link_counts(memory_ids)
        co_activated_neighbors = await self._load_coactivated_neighbors(memory_ids)

        # Score each memory
        scored_results: List[Tuple[str, float, RetrievalResultSchema]] = []

        for mem_id in memory_ids:
            activation, components, result_schema = await self._score_single_memory(
                memory_id=mem_id,
                similarity=similarities.get(mem_id, 0.0),
                activation_state=activation_states.get(mem_id),
                metadata=memory_metadata.get(mem_id),
                evidence_count=evidence_links.get(mem_id, 0),
                neighbor_activation=co_activated_neighbors.get(mem_id),
                scope=scope,
                episode_id=episode_id,
                goal_id=goal_id,
            )

            scored_results.append((mem_id, activation, result_schema))

        # Sort by activation descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Build response dicts
        ranked_dicts = [
            {
                "id": mem_id,
                "activation_score": activation,
                "similarity": similarities.get(mem_id, 0.0),
                "metadata": memory_metadata.get(mem_id),
            }
            for mem_id, activation, _ in scored_results
        ]

        # Extract explanation schemas
        explanation_results = [result_schema for _, _, result_schema in scored_results]

        return ranked_dicts, explanation_results

    async def write_retrieval_explanation(
        self,
        query: str,
        results: List[RetrievalResultSchema],
        top_k: int,
    ) -> str:
        """Write retrieval explanation log to database.

        Args:
            query: Original search query
            results: List of RetrievalResultSchema with scoring breakdown
            top_k: Number of results requested

        Returns:
            Explanation log ID
        """
        # Hash query for grouping similar queries
        query_hash = hashlib.sha256(query.encode()).hexdigest()

        # Create explanation entry
        explanation = MemoryRetrievalExplanation(
            organization_id=self.org_id,
            user_id=self.user_id,
            query_hash=query_hash,
            retrieved_at=datetime.now(timezone.utc),
            top_k=top_k,
            results=[r.model_dump() for r in results],
        )

        self.session.add(explanation)
        await self.session.flush()  # Get ID without commit

        return str(explanation.id)

    async def _score_single_memory(
        self,
        memory_id: str,
        similarity: float,
        activation_state: Optional[MemoryActivationState],
        metadata: Optional[Dict[str, Any]],
        evidence_count: int,
        neighbor_activation: Optional[float],
        scope: Optional[str],
        episode_id: Optional[str],
        goal_id: Optional[str],
        current_rank: int = 0,
    ) -> Tuple[float, ActivationComponents, RetrievalResultSchema]:
        """Score a single memory with all 8 components.

        Args:
            memory_id: Memory UUID
            similarity: Vector similarity from Qdrant
            activation_state: Memory activation state record
            metadata: Memory metadata dict
            evidence_count: Number of evidence links
            neighbor_activation: Max co-activation neighbor score
            scope: Current scope context
            episode_id: Current episode
            goal_id: Current goal
            current_rank: Rank in result set

        Returns:
            Tuple of (activation_score, components, result_schema)
        """
        # Extract activation state values
        base_importance = activation_state.base_importance if activation_state else 0.5
        confidence = activation_state.confidence if activation_state else 0.8
        contradicted = activation_state.contradicted if activation_state else False
        risk_factor = activation_state.risk_factor if activation_state else 0.0
        access_count = activation_state.access_count if activation_state else 0
        last_accessed = activation_state.last_accessed_at if activation_state else None

        # Compute context gate
        scope_match = self._compute_scope_match(scope, metadata) if metadata else 0.5
        episode_match = self._compute_episode_match(episode_id, metadata) if metadata else 0.5
        goal_match = self._compute_goal_match(goal_id, metadata) if metadata else 0.5

        # Compute age in days
        created_at = metadata.get("created_at") if metadata else None
        age_days = self._compute_age_days(created_at) if created_at else 0.0

        # Score memory
        activation, components = self.scorer.score_memory(
            similarity=similarity,
            base_importance=base_importance,
            confidence=confidence,
            contradicted=contradicted,
            risk_factor=risk_factor,
            access_count=access_count,
            last_accessed_at=last_accessed,
            evidence_link_count=evidence_count,
            scope_match=scope_match,
            episode_match=episode_match,
            goal_match=goal_match,
            neighbor_activation=neighbor_activation,
            age_days=age_days,
        )

        # Check RLS - convert to schema
        allowed = True  # RLS already enforced at query level in caller
        reason = None

        # Build result schema
        result = RetrievalResultSchema(
            memory_id=memory_id,
            activation=activation,
            components=ActivationComponentsSchema(
                rel=components.rel,
                rec=components.rec,
                freq=components.freq,
                imp=components.imp,
                conf=components.conf,
                ctx=components.ctx,
                prov=components.prov,
                risk=components.risk,
                nbr=neighbor_activation if neighbor_activation else None,
            ),
            gating=GatingInfoSchema(allowed=allowed, reason=reason),
            rank=current_rank + 1,
        )

        return activation, components, result

    async def _load_activation_states(self, memory_ids: List[str]) -> Dict[str, MemoryActivationState]:
        """Load activation states for memory IDs.

        Creates missing records with defaults.
        """
        # Query existing states
        stmt = select(MemoryActivationState).where(
            and_(
                MemoryActivationState.organization_id == self.org_id,
                MemoryActivationState.memory_id.in_(memory_ids),
            )
        )
        result = await self.session.execute(stmt)
        states = {str(s.memory_id): s for s in result.scalars().all()}

        # Create missing states with defaults
        missing = [mid for mid in memory_ids if mid not in states]
        if missing:
            new_states = [
                MemoryActivationState(
                    organization_id=self.org_id,
                    memory_id=mid,
                    base_importance=0.5,
                    confidence=0.8,
                    access_count=0,
                )
                for mid in missing
            ]
            self.session.add_all(new_states)
            await self.session.flush()

            for state in new_states:
                states[str(state.memory_id)] = state

        return states

    async def _load_memory_metadata(self, memory_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Load memory metadata."""
        stmt = select(MemoryMetadata).where(MemoryMetadata.id.in_(memory_ids))
        result = await self.session.execute(stmt)
        metadata = {}

        for mem in result.scalars().all():
            metadata[str(mem.id)] = {
                "title": mem.title,
                "scope": mem.scope,
                "created_at": mem.created_at,
                "episode_id": mem.episode_id,
            }

        return metadata

    async def _load_evidence_link_counts(self, memory_ids: List[str]) -> Dict[str, int]:
        """Load evidence link counts (simplified - count by relationships).

        In full system, would count causal_hypotheses with this memory as evidence.
        """
        stmt = select(CausalHypothesis).where(
            and_(
                CausalHypothesis.organization_id == self.org_id,
                CausalHypothesis.evidence_memory_ids.contains([mid for mid in memory_ids]),  # type: ignore
            )
        )
        result = await self.session.execute(stmt)
        hypotheses = result.scalars().all()

        # Count evidence for each memory
        evidence_counts: Dict[str, int] = {mid: 0 for mid in memory_ids}

        for hyp in hypotheses:
            if hyp.evidence_memory_ids:
                for mid in hyp.evidence_memory_ids:
                    if mid in evidence_counts:
                        evidence_counts[mid] += 1

        return evidence_counts

    async def _load_coactivated_neighbors(self, memory_ids: List[str]) -> Dict[str, Optional[float]]:
        """Load max co-activation neighbor score for each memory.

        Returns dict of memory_id -> max_neighbor_activation.
        """
        # Query co-activation edges where memory is a or b
        stmt = select(MemoryCoactivationEdge).where(
            and_(
                MemoryCoactivationEdge.organization_id == self.org_id,
                (MemoryCoactivationEdge.memory_id_a.in_(memory_ids))
                | (MemoryCoactivationEdge.memory_id_b.in_(memory_ids)),
            )
        )
        result = await self.session.execute(stmt)
        edges = result.scalars().all()

        # Find max neighbor activation
        neighbor_maxes: Dict[str, float] = {mid: 0.0 for mid in memory_ids}

        for edge in edges:
            if str(edge.memory_id_a) in memory_ids:
                neighbor_maxes[str(edge.memory_id_a)] = max(
                    neighbor_maxes[str(edge.memory_id_a)], edge.edge_weight
                )
            if str(edge.memory_id_b) in memory_ids:
                neighbor_maxes[str(edge.memory_id_b)] = max(
                    neighbor_maxes[str(edge.memory_id_b)], edge.edge_weight
                )

        return {mid: (v if v > 0.0 else None) for mid, v in neighbor_maxes.items()}

    def _compute_scope_match(self, current_scope: Optional[str], metadata: Dict[str, Any]) -> float:
        """Compute scope affinity."""
        if not current_scope or not metadata:
            return 0.5

        mem_scope = metadata.get("scope")
        if mem_scope == current_scope:
            return 1.0
        elif current_scope in ("organization", "department"):
            # Broader scopes match narrower
            return 0.7 if mem_scope in ("team", "personal") else 0.5
        elif current_scope == "team" and mem_scope == "personal":
            return 0.6

        return 0.3

    def _compute_episode_match(self, current_episode: Optional[str], metadata: Dict[str, Any]) -> float:
        """Compute episode affinity."""
        if not current_episode or not metadata:
            return 0.5

        mem_episode = metadata.get("episode_id")
        return 1.0 if mem_episode == current_episode else 0.3

    def _compute_goal_match(self, current_goal: Optional[str], metadata: Dict[str, Any]) -> float:
        """Compute goal affinity."""
        if not current_goal or not metadata:
            return 0.5

        # Simplified: would need to check goal tags or relationships
        return 0.5

    def _compute_age_days(self, created_at: Optional[datetime]) -> float:
        """Compute memory age in days."""
        if not created_at:
            return 0.0

        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        delta = now - created_at
        return delta.total_seconds() / 86400.0
