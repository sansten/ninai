"""
Causal Reasoning Service (PR-1)
================================

Service for discovering, validating, and reasoning over causal edges.

Key capabilities:
1. discover_causal_edges(): Find cause-effect relationships from episodes
2. validate_edge(): Update edge strength from new observations
3. predict(): Forecast outcomes given causal chains
4. explain(): Trace effects back to root causes
5. counterfactual_query(): "What would happen if...?"
6. do_calculus(): Estimate causal effect of interventions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import and_, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.causal_edge import CausalEdge, CounterfactualScenario
from app.models.memory_fact import MemoryFact, MemoryFactStatus
from app.models.memory import MemoryMetadata
from app.models.memory_episode import MemoryEpisode
from app.models.memory_episode_membership import MemoryEpisodeMembership
from app.services.audit_service import AuditService


logger = logging.getLogger(__name__)


class PredictionResult:
    """Result of a causal prediction."""

    def __init__(
        self,
        outcome_id: str,
        probability: float,
        confidence: float,
        causal_chain: List[str],
    ):
        self.outcome_id = outcome_id
        self.probability = probability  # 0-1 likelihood
        self.confidence = confidence  # 0-1 how sure are we?
        self.causal_chain = causal_chain  # path of edges leading to outcome

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "probability": self.probability,
            "confidence": self.confidence,
            "causal_chain": self.causal_chain,
        }


class CausalExplanation:
    """Explanation of why something happened."""

    def __init__(self, effect_id: str, root_causes: List[Dict[str, Any]]):
        self.effect_id = effect_id
        self.root_causes = root_causes  # [{cause_id, strength, causal_chain}]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "root_causes": self.root_causes,
        }


class CausalReasoningService:
    """
    Discovers and reasons about causal relationships.
    
    Design:
    - RLS-safe: assumes tenant context set on AsyncSession
    - Idempotent: discovering same edge twice updates, doesn't duplicate
    - Fail-safe: LLM failures don't block discovery, just mark confidence lower
    """

    def __init__(self, session: AsyncSession, user_id: str, org_id: str):
        self.session = session
        self.user_id = user_id
        self.org_id = org_id
        self.audit_service = AuditService(session)

    # =========================================================================
    # Discovery: Find causal edges from episodes
    # =========================================================================

    async def discover_causal_edges_from_episode(
        self, episode_id: str, auto_save: bool = True
    ) -> List[CausalEdge]:
        """
        Analyze an episode for causal relationships.
        
        Three signals:
        1. Temporal: fact A appears before fact B → potential causation
        2. Correlational: co-occurrence in same episode
        3. LLM-detected: ask LLM "What causes what in this episode?"
        
        Args:
            episode_id: Episode to analyze
            auto_save: Save discovered edges to database
            
        Returns:
            List of discovered CausalEdges (saved if auto_save=True)
        """
        logger.info(f"Discovering causal edges from episode {episode_id}")

        # Retrieve episode and its member facts
        episode_stmt = select(MemoryEpisode).where(
            MemoryEpisode.id == episode_id,
            MemoryEpisode.organization_id == self.org_id,
        )
        episode_result = await self.session.execute(episode_stmt)
        episode = episode_result.scalar_one_or_none()

        if not episode:
            raise ValueError(f"Episode {episode_id} not found")

        # Get member memories in temporal order
        members_stmt = (
            select(MemoryEpisodeMembership, MemoryMetadata)
            .join(MemoryMetadata, MemoryMetadata.id == MemoryEpisodeMembership.memory_id)
            .where(
                MemoryEpisodeMembership.episode_id == episode_id,
                MemoryEpisodeMembership.organization_id == self.org_id,
            )
            .order_by(MemoryEpisodeMembership.position)
        )
        members_result = await self.session.execute(members_stmt)
        members = members_result.all()

        if not members:
            logger.info(f"Episode {episode_id} has no member memories")
            return []

        discovered_edges = []

        # Signal 1: Temporal causation (A precedes B → A might cause B)
        for i, (membership_i, memory_i) in enumerate(members):
            for membership_j, memory_j in members[i + 1 :]:
                # Potential causal edge: memory_i.id → memory_j.id
                temporal_strength = self._estimate_temporal_strength(
                    i, i + members[i + 1 :].index((membership_j, memory_j)) + 1, len(members)
                )

                edge = await self._create_or_update_edge(
                    cause_entity_id=memory_i.id,
                    cause_entity_type="memory",
                    effect_entity_id=memory_j.id,
                    effect_entity_type="memory",
                    mechanism="temporal",
                    initial_strength=temporal_strength,
                    evidence_memory_ids=[episode_id],
                )
                if edge:
                    discovered_edges.append(edge)

        # Signal 2: Extract facts from memories and build fact-to-fact edges
        facts_stmt = select(MemoryFact).where(
            MemoryFact.organization_id == self.org_id,
            MemoryFact.source_memory_id.in_([m[1].id for m in members]),
            MemoryFact.status == MemoryFactStatus.ACTIVE,
        )
        facts_result = await self.session.execute(facts_stmt)
        facts = list(facts_result.scalars().all())

        # Fact-to-fact causation: subject→object relationships
        for i, fact_i in enumerate(facts):
            for fact_j in facts[i + 1 :]:
                # Heuristic: if fact_i.object == fact_j.subject, there's causation
                if fact_i.object and fact_j.subject and fact_i.object in fact_j.subject:
                    edge = await self._create_or_update_edge(
                        cause_entity_id=fact_i.id,
                        cause_entity_type="fact",
                        effect_entity_id=fact_j.id,
                        effect_entity_type="fact",
                        mechanism="direct",
                        initial_strength=0.6,
                        evidence_memory_ids=[fact_i.source_memory_id, fact_j.source_memory_id],
                    )
                    if edge:
                        discovered_edges.append(edge)

        # Signal 3: LLM causal detection (if configured)
        if settings.ENABLE_LLM_CAUSAL_DISCOVERY:
            llm_edges = await self._discover_via_llm(episode, members)
            discovered_edges.extend(llm_edges)

        # Audit log
        await self.audit_service.log_event(
            event_type="causal_discovery",
            actor_id=self.user_id,
            organization_id=self.org_id,
            resource_type="episode",
            resource_id=episode_id,
            details={"edges_discovered": len(discovered_edges)},
        )

        return discovered_edges

    def _estimate_temporal_strength(self, position_i: int, position_j: int, total: int) -> float:
        """
        Estimate causal strength based on temporal proximity and ordering.
        
        Closer events are more likely to be causally related.
        Strength decreases with distance.
        """
        distance = position_j - position_i
        max_distance = total - 1
        # Inverse distance: closer = stronger
        strength = max(0.1, 1.0 - (distance / max_distance) * 0.7)
        return strength

    async def _create_or_update_edge(
        self,
        cause_entity_id: str,
        cause_entity_type: str,
        effect_entity_id: str,
        effect_entity_type: str,
        mechanism: str,
        initial_strength: float,
        evidence_memory_ids: List[str],
    ) -> Optional[CausalEdge]:
        """Create or update a causal edge."""
        # Check if edge already exists
        existing_stmt = select(CausalEdge).where(
            and_(
                CausalEdge.organization_id == self.org_id,
                CausalEdge.cause_entity_id == cause_entity_id,
                CausalEdge.effect_entity_id == effect_entity_id,
            )
        )
        existing_result = await self.session.execute(existing_stmt)
        existing_edge = existing_result.scalar_one_or_none()

        if existing_edge:
            # Update: strengthen existing edge with new evidence
            existing_edge.validation_count += 1
            existing_edge.evidence_memory_ids = list(
                set(existing_edge.evidence_memory_ids + evidence_memory_ids)
            )
            # Update strength via Bayesian average
            existing_edge.strength = (
                existing_edge.strength * existing_edge.validation_count + initial_strength
            ) / (existing_edge.validation_count + 1)
            existing_edge.last_validated_at = datetime.now(timezone.utc)
            self.session.add(existing_edge)
            await self.session.flush()
            return existing_edge

        # Create new edge
        new_edge = CausalEdge(
            id=str(uuid4()),
            organization_id=self.org_id,
            cause_entity_id=cause_entity_id,
            cause_entity_type=cause_entity_type,
            effect_entity_id=effect_entity_id,
            effect_entity_type=effect_entity_type,
            mechanism=mechanism,
            strength=initial_strength,
            evidence_memory_ids=evidence_memory_ids,
        )
        self.session.add(new_edge)
        await self.session.flush()
        return new_edge

    async def _discover_via_llm(
        self, episode: MemoryEpisode, members: List[tuple]
    ) -> List[CausalEdge]:
        """
        Ask LLM to identify causal relationships in episode.
        
        Prompt: "In this conversation, what causes what?"
        """
        # Placeholder for LLM integration
        # In production, call LLM with episode text, parse causal edges
        logger.info("LLM causal discovery not yet implemented")
        return []

    # =========================================================================
    # Prediction: What will happen if...?
    # =========================================================================

    async def predict(
        self, cause_fact_ids: List[str], target_fact_id: str, max_depth: int = 3
    ) -> Optional[PredictionResult]:
        """
        Forecast outcome probability given cause facts.
        
        Use causal chains: cause_fact → ... → target_fact
        
        Args:
            cause_fact_ids: Starting facts
            target_fact_id: What we want to predict
            max_depth: Max hops in causal chain
            
        Returns:
            PredictionResult with probability and confidence
        """
        # BFS through causal graph to find path from cause to target
        paths = await self._find_causal_paths(cause_fact_ids, target_fact_id, max_depth)

        if not paths:
            return None

        # Probability based on strongest path
        best_path = max(paths, key=lambda p: p["strength"])
        strength = best_path["strength"]
        confidence = sum(e.strength for e in best_path["edges"]) / len(best_path["edges"])

        return PredictionResult(
            outcome_id=target_fact_id,
            probability=strength,
            confidence=confidence,
            causal_chain=best_path["path"],
        )

    async def _find_causal_paths(
        self, source_ids: List[str], target_id: str, max_depth: int
    ) -> List[Dict[str, Any]]:
        """BFS to find causal paths from sources to target."""
        # Simplified: assume single-hop for now
        # In production, implement proper BFS with memoization
        paths = []

        edges_stmt = select(CausalEdge).where(
            and_(
                CausalEdge.organization_id == self.org_id,
                CausalEdge.cause_entity_id.in_(source_ids),
                CausalEdge.effect_entity_id == target_id,
            )
        )
        edges_result = await self.session.execute(edges_stmt)
        edges = list(edges_result.scalars().all())

        for edge in edges:
            paths.append(
                {
                    "path": [edge.cause_entity_id, edge.effect_entity_id],
                    "strength": edge.strength,
                    "edges": [edge],
                }
            )

        return paths

    # =========================================================================
    # Explanation: Why did something happen?
    # =========================================================================

    async def explain(self, effect_fact_id: str) -> CausalExplanation:
        """
        Trace an effect back to root causes.
        
        Returns all causal edges that could have led to effect_fact_id.
        """
        # Query: all edges where effect_entity_id == effect_fact_id
        edges_stmt = select(CausalEdge).where(
            and_(
                CausalEdge.organization_id == self.org_id,
                CausalEdge.effect_entity_id == effect_fact_id,
            )
        ).order_by(CausalEdge.strength.desc())

        edges_result = await self.session.execute(edges_stmt)
        edges = list(edges_result.scalars().all())

        root_causes = [
            {
                "cause_id": edge.cause_entity_id,
                "strength": edge.strength,
                "mechanism": edge.mechanism,
                "causal_chain": [edge.cause_entity_id, edge.effect_entity_id],
            }
            for edge in edges
        ]

        return CausalExplanation(effect_fact_id, root_causes)

    # =========================================================================
    # Counterfactual queries: What would happen if...?
    # =========================================================================

    async def counterfactual_query(
        self, condition: str, query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a counterfactual query.
        
        Example:
        condition: "if memory_id X increases by 20%"
        query: {outcome: "customer_satisfaction"}
        
        Returns: predicted outcome under this condition
        """
        # Create scenario
        scenario = CounterfactualScenario(
            id=str(uuid4()),
            organization_id=self.org_id,
            scenario_type="prediction",
            condition=condition,
            predicted_outcomes={"placeholder": 0.5},  # stub
        )
        self.session.add(scenario)
        await self.session.flush()

        return {
            "scenario_id": scenario.id,
            "condition": condition,
            "status": "created",
        }

    # =========================================================================
    # Do-calculus: Estimate causal effects of interventions
    # =========================================================================

    async def do_calculus(
        self, treatment: str, outcome_var: str
    ) -> Dict[str, Any]:
        """
        Estimate the causal effect of an intervention.
        
        E[Y | do(X=x)] - the expected value of Y if we intervene to set X=x.
        
        Simplified implementation: use edge strength as proxy for effect size.
        """
        # Find path from treatment to outcome
        edges_stmt = select(CausalEdge).where(
            and_(
                CausalEdge.organization_id == self.org_id,
                CausalEdge.cause_entity_id == treatment,
                CausalEdge.effect_entity_id == outcome_var,
            )
        )
        edges_result = await self.session.execute(edges_stmt)
        edges = list(edges_result.scalars().all())

        if not edges:
            return {"treatment": treatment, "outcome": outcome_var, "effect": 0.0}

        # Average effect across discovered edges
        avg_effect = sum(e.strength for e in edges) / len(edges)

        return {
            "treatment": treatment,
            "outcome": outcome_var,
            "causal_effect": avg_effect,
            "num_paths": len(edges),
        }

    # =========================================================================
    # Validation: Update edge strength from new observations
    # =========================================================================

    async def validate_edge(
        self, edge_id: str, new_evidence: Dict[str, Any]
    ) -> Optional[CausalEdge]:
        """
        Update edge strength based on new observed evidence.
        
        Args:
            edge_id: Edge to validate
            new_evidence: {success: bool, strength_adjustment: float}
            
        Returns:
            Updated CausalEdge
        """
        edge_stmt = select(CausalEdge).where(
            and_(
                CausalEdge.id == edge_id,
                CausalEdge.organization_id == self.org_id,
            )
        )
        edge_result = await self.session.execute(edge_stmt)
        edge = edge_result.scalar_one_or_none()

        if not edge:
            return None

        success = new_evidence.get("success", False)
        adjustment = new_evidence.get("strength_adjustment", 0.0)

        if success:
            edge.validation_count += 1
            edge.strength = min(1.0, edge.strength + adjustment)
        else:
            edge.invalidation_count += 1
            edge.strength = max(0.0, edge.strength - adjustment)

        edge.last_validated_at = datetime.now(timezone.utc)
        self.session.add(edge)
        await self.session.flush()

        return edge
