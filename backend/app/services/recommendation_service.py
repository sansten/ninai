"""
Memory Recommendation Engine - Rank and recommend related memories

Recommendation Algorithm:
- 50% Semantic Similarity (from graph relationships)
- 20% Recency (recently accessed/created memories ranked higher)
- 20% Interaction History (viewed, edited, referenced often)
- 10% User Feedback (upvotes/downvotes on past recommendations)

Caches recommendations for performance and tracks feedback for ML improvements.
"""

from __future__ import annotations

import uuid
import logging
import json
import inspect
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
import math
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

from app.models.memory import MemoryMetadata
from app.models.graph_relationship import GraphRelationship
from app.models.recommendation_feedback import RecommendationFeedback

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Memory recommendation engine using multi-factor ranking.
    
    Ranks memories by composite score:
    score = 0.5*similarity + 0.2*recency + 0.2*interaction + 0.1*feedback
    
    Results are cached for performance and limited to top N results.
    """

    def __init__(self, db: AsyncSession, redis_client: Any = None):
        self.db = db
        self.redis = redis_client
        self.cache_ttl = 86400  # 24 hours
        self.weights = {
            "similarity": 0.5,
            "recency": 0.2,
            "interaction": 0.2,
            "feedback": 0.1
        }

    async def _cache_get(self, key: str) -> Optional[str]:
        if not self.redis:
            return None
        getter = getattr(self.redis, "get", None)
        if getter is None:
            return None
        try:
            value = getter(key)
            if inspect.isawaitable(value):
                value = await value
            return value
        except Exception:
            return None

    async def _cache_setex(self, key: str, ttl_seconds: int, value: str) -> None:
        if not self.redis:
            return
        setter = getattr(self.redis, "setex", None)
        if setter is None:
            return
        try:
            result = setter(key, ttl_seconds, value)
            if inspect.isawaitable(result):
                await result
        except Exception:
            return

    async def get_recommendations(
        self,
        memory_id: str,
        org_id: str,
        limit: int = 10,
        use_cache: bool = True,
        min_similarity: float = 0.0,
        max_age_days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get personalized memory recommendations for a given memory.
        
        Args:
            memory_id: Base memory ID
            org_id: Organization ID
            limit: Max recommendations (1-50)
            use_cache: Use Redis cache if available
            min_similarity: Minimum relationship similarity
            max_age_days: Exclude memories older than N days
            
        Returns:
            Ranked list of {"memory_id", "score", "factors", "title", ...}
        """
        
        limit = min(limit, 50)  # Cap at 50
        cache_key = f"recommendations:{org_id}:{memory_id}:{limit}"
        
        # Try cache
        if use_cache:
            cached = await self._cache_get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return json.loads(cached)
        
        logger.info(f"Computing recommendations for memory {memory_id}")
        
        try:
            # Get related memories via graph
            related = await self._get_related_memories(
                memory_id,
                org_id,
                min_similarity=min_similarity
            )
            
            if not related:
                logger.warning(f"No related memories found for {memory_id}")
                return []
            
            # Calculate ranking factors for each memory
            scored_memories = []
            
            for rel in related:
                related_id = rel["target_memory_id"]
                similarity_score = rel["similarity_score"]
                
                # Get other factors
                recency = await self._calculate_recency_score(related_id)
                interaction = await self._calculate_interaction_score(related_id)
                feedback = await self._calculate_feedback_score(
                    memory_id,
                    related_id,
                    org_id
                )
                
                # Composite score
                composite_score = (
                    self.weights["similarity"] * similarity_score +
                    self.weights["recency"] * recency +
                    self.weights["interaction"] * interaction +
                    self.weights["feedback"] * feedback
                )
                
                scored_memories.append({
                    "related_memory_id": related_id,
                    "score": composite_score,
                    "factors": {
                        "similarity": similarity_score,
                        "recency": recency,
                        "interaction": interaction,
                        "feedback": feedback
                    },
                    "relationship_type": rel.get("relationship_type", "RELATES_TO")
                })
            
            # Sort by score and take top N
            scored_memories.sort(key=lambda x: x["score"], reverse=True)
            top_recommendations = scored_memories[:limit]
            
            # Enrich with memory details
            recommendations = await self._enrich_with_memory_details(
                top_recommendations,
                org_id,
                max_age_days
            )
            
            # Cache results
            await self._cache_setex(cache_key, self.cache_ttl, json.dumps(recommendations))
            
            logger.info(f"Generated {len(recommendations)} recommendations for {memory_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)
            return []

    async def _get_related_memories(
        self,
        memory_id: str,
        org_id: str,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get all memories related to this one via graph relationships.
        
        Returns relationships both incoming and outgoing.
        """
        
        # Get outgoing relationships
        stmt_out = select(GraphRelationship).where(
            and_(
                GraphRelationship.organization_id == uuid.UUID(org_id),
                GraphRelationship.from_memory_id == memory_id,
                GraphRelationship.similarity_score.isnot(None),
                GraphRelationship.similarity_score >= min_similarity
            )
        ).order_by(GraphRelationship.similarity_score.desc())
        
        result_out = await self.db.execute(stmt_out)
        outgoing = result_out.scalars().all()
        
        # Get incoming relationships
        stmt_in = select(GraphRelationship).where(
            and_(
                GraphRelationship.organization_id == uuid.UUID(org_id),
                GraphRelationship.to_memory_id == memory_id,
                GraphRelationship.similarity_score.isnot(None),
                GraphRelationship.similarity_score >= min_similarity
            )
        ).order_by(GraphRelationship.similarity_score.desc())
        
        result_in = await self.db.execute(stmt_in)
        incoming = result_in.scalars().all()
        
        # Combine and deduplicate
        related_dict = {}
        
        for rel in outgoing:
            target_id = rel.to_memory_id
            if target_id not in related_dict:
                related_dict[target_id] = {
                    "target_memory_id": target_id,
                    "similarity_score": rel.similarity_score,
                    "relationship_type": rel.relationship_type
                }
        
        for rel in incoming:
            source_id = rel.from_memory_id
            if source_id not in related_dict:
                related_dict[source_id] = {
                    "target_memory_id": source_id,
                    "similarity_score": rel.similarity_score,
                    "relationship_type": rel.relationship_type
                }
            else:
                # Take max similarity if both incoming and outgoing exist
                related_dict[source_id]["similarity_score"] = max(
                    related_dict[source_id]["similarity_score"],
                    rel.similarity_score
                )
        
        return list(related_dict.values())

    async def _calculate_recency_score(self, memory_id: str) -> float:
        """
        Calculate recency score: 1.0 for recent, 0.0 for old.
        
        Exponential decay: 1.0 at 0 days, 0.5 at 30 days, 0.1 at 90 days.
        """
        
        stmt = select(MemoryMetadata.created_at, MemoryMetadata.updated_at).where(
            MemoryMetadata.id == memory_id
        )
        
        result = await self.db.execute(stmt)
        row = result.first()
        
        if not row:
            return 0.0
        
        now = datetime.now(timezone.utc)

        created_at = row.created_at
        updated_at = row.updated_at
        if created_at is not None and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if updated_at is not None and updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        most_recent = max(created_at or datetime.min.replace(tzinfo=timezone.utc), updated_at or datetime.min.replace(tzinfo=timezone.utc))
        days_old = max(0, (now - most_recent).days)
        
        # Exponential decay: e^(-days/30)
        recency = math.exp(-days_old / 30.0)
        return float(min(1.0, max(0.0, recency)))

    async def _calculate_interaction_score(self, memory_id: str) -> float:
        """
        Calculate interaction score based on memory access patterns.
        
        Factors:
        - View count (each view)
        - Edit count (each edit weighted higher)
        - Share count (each share)
        - Time in memory (total seconds)
        
        Normalized to 0.0-1.0 range.
        """
        
        stmt = select(MemoryMetadata.access_count, MemoryMetadata.extra_metadata).where(MemoryMetadata.id == memory_id)
        
        result = await self.db.execute(stmt)
        row = result.first()
        
        if not row:
            return 0.0

        access_count = int(row[0] or 0)
        metadata = row[1] or {}
        
        # Extract interaction counts
        view_count = int(metadata.get("view_count", access_count) or access_count)
        edit_count = metadata.get("edit_count", 0)
        share_count = metadata.get("share_count", 0)
        time_seconds = metadata.get("time_spent_seconds", 0)
        
        # Weighted score
        score = (
            0.3 * math.log1p(view_count) +
            0.4 * math.log1p((edit_count or 0) * 2) +
            0.2 * math.log1p((share_count or 0) * 3) +
            0.1 * math.log1p((time_seconds or 0) / 60.0)
        )
        
        # Normalize to 0.0-1.0
        try:
            normalized = 1.0 / (1.0 + math.exp(-score))
        except OverflowError:
            normalized = 1.0

        return float(min(1.0, max(0.0, normalized)))

    async def _calculate_feedback_score(
        self,
        base_memory_id: str,
        target_memory_id: str,
        org_id: str
    ) -> float:
        """
        Calculate feedback score from user reactions to past recommendations.
        
        Weighted average of upvotes/downvotes on similar recommendations.
        Score: 1.0 for all upvotes, 0.0 for neutral, -1.0 for all downvotes.
        Mapped to 0.0-1.0 range.
        """
        
        stmt = select(RecommendationFeedback).where(
            and_(
                RecommendationFeedback.organization_id == uuid.UUID(org_id),
                RecommendationFeedback.base_memory_id == base_memory_id,
                RecommendationFeedback.recommended_memory_id == target_memory_id,
                RecommendationFeedback.created_at >= datetime.utcnow() - timedelta(days=90)
            )
        )
        
        result = await self.db.execute(stmt)
        feedback_list = result.scalars().all()
        
        if not feedback_list:
            return 0.5  # Neutral if no feedback
        
        # Calculate weighted average
        weights = []
        values = []
        
        for feedback in feedback_list:
            # Weight by recency
            age_days = (datetime.utcnow() - feedback.created_at).days
            recency_weight = math.exp(-age_days / 30.0)
            weights.append(recency_weight)
            
            # Map helpful/not_helpful to score
            if feedback.helpful:
                values.append(1.0)
            else:
                values.append(0.0)
        
        if not weights:
            return 0.5
        
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        weight_total = sum(weights) or 1.0
        weighted_avg = weighted_sum / weight_total
        return float(min(1.0, max(0.0, weighted_avg)))

    async def _enrich_with_memory_details(
        self,
        recommendations: List[Dict[str, Any]],
        org_id: str,
        max_age_days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch full memory details for recommendations.
        
        Args:
            recommendations: List with at least 'related_memory_id'
            org_id: Organization ID
            max_age_days: Exclude memories created more than N days ago
            
        Returns:
            Enriched recommendations with title, summary, created_at, etc.
        """
        
        if not recommendations:
            return []
        
        memory_ids = [r["related_memory_id"] for r in recommendations]
        
        stmt = select(MemoryMetadata).where(
            and_(
                MemoryMetadata.id.in_(memory_ids),
                MemoryMetadata.organization_id == org_id,
                MemoryMetadata.is_active.is_(True),
            )
        )
        
        if max_age_days:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            stmt = stmt.where(MemoryMetadata.created_at >= cutoff_date)
        
        result = await self.db.execute(stmt)
        memories = {str(m.id): m for m in result.scalars().all()}
        
        # Merge memory details into recommendations
        enriched = []
        
        for rec in recommendations:
            memory_id = rec["related_memory_id"]
            
            if memory_id not in memories:
                continue
            
            memory = memories[memory_id]
            
            enriched.append({
                "memory_id": memory_id,
                "title": memory.title,
                "summary": (memory.content_preview or "")[:200],
                "score": rec["score"],
                "factors": rec["factors"],
                "relationship_type": rec["relationship_type"],
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            })
        
        return enriched

    async def submit_feedback(
        self,
        base_memory_id: str,
        recommended_memory_id: str,
        org_id: str,
        user_id: str,
        helpful: bool,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit user feedback on a recommendation.
        
        Stores feedback for future ML improvements.
        """
        
        feedback = RecommendationFeedback(
            id=uuid.uuid4(),
            organization_id=uuid.UUID(org_id),
            base_memory_id=base_memory_id,
            recommended_memory_id=recommended_memory_id,
            user_id=uuid.UUID(user_id),
            helpful=helpful,
            reason=reason,
            created_at=datetime.utcnow()
        )
        
        self.db.add(feedback)
        await self.db.commit()
        
        # Invalidate cache
        cache_key = f"recommendations:{org_id}:{base_memory_id}:*"
        # Note: Redis doesn't have wildcard delete, would need to track all variations
        
        logger.info(f"Stored feedback: {base_memory_id} -> {recommended_memory_id} (helpful={helpful})")
        
        return {
            "status": "feedback_submitted",
            "helpful": helpful,
            "base_memory_id": base_memory_id,
            "recommended_memory_id": recommended_memory_id
        }

    async def get_recommendation_metrics(self, org_id: str) -> Dict[str, Any]:
        """
        Get metrics about recommendations for organization.
        
        Returns:
            - total_recommendations_given
            - feedback_count
            - helpful_ratio
            - avg_score_given
            - most_recommended
        """
        
        # Get all feedback
        stmt = select(RecommendationFeedback).where(
            RecommendationFeedback.organization_id == uuid.UUID(org_id)
        )
        
        result = await self.db.execute(stmt)
        feedback_list = result.scalars().all()
        
        if not feedback_list:
            return {
                "total_feedback": 0,
                "helpful_ratio": 0.0,
                "total_recommendations_given": 0
            }
        
        helpful_count = sum(1 for f in feedback_list if f.helpful)
        
        return {
            "total_feedback": len(feedback_list),
            "helpful_ratio": helpful_count / len(feedback_list),
            "helpful_count": helpful_count,
            "not_helpful_count": len(feedback_list) - helpful_count,
            "total_recommendations_given": len(feedback_list)  # Approximation
        }

    async def get_weights(self) -> Dict[str, float]:
        """Get current ranking weights."""
        return self.weights.copy()

    async def update_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Update ranking weights.
        
        Weights must sum to 1.0.
        """
        
        # Validate
        valid_keys = {"similarity", "recency", "interaction", "feedback"}
        if not all(k in valid_keys for k in weights.keys()):
            raise ValueError(f"Invalid weight keys. Must be: {valid_keys}")
        
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.weights = weights
        logger.info(f"Updated recommendation weights: {weights}")
        
        return self.weights


async def get_recommendation_service(
    db: AsyncSession,
    redis_client: Any
) -> RecommendationService:
    """Dependency to get service instance."""
    return RecommendationService(db, redis_client)
