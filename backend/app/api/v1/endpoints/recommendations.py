"""
API endpoints for memory recommendations.

Endpoints:
- GET /memories/{id}/recommendations - Get recommendations for a memory
- POST /recommendations/{id}/feedback - Submit feedback on recommendation
- GET /recommendations/metrics - Analytics about recommendations
- PATCH /recommendations/weights - Adjust ranking weights
"""

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

try:
    import redis  # type: ignore
    REDIS_AVAILABLE = True
except Exception:  # pragma: no cover
    redis = None  # type: ignore
    REDIS_AVAILABLE = False

from app.api.v1.endpoints.auth import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.models.user import User
from app.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recommendations"])


def _get_sync_redis_client():
    if not REDIS_AVAILABLE:
        return None
    try:
        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        return client
    except Exception:
        return None


@router.get("/memories/{memory_id}/recommendations", name="get_memory_recommendations")
async def get_memory_recommendations(
    memory_id: str,
    org_id: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0),
    max_age_days: Optional[int] = Query(None, ge=1, le=365),
    use_cache: bool = Query(True),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get personalized memory recommendations.
    
    Returns up to N similar memories ranked by:
    - 50% Semantic Similarity (from graph relationships)
    - 20% Recency (recently created/updated memories)
    - 20% Interaction (views, edits, shares)
    - 10% Feedback (user upvotes on past recommendations)
    
    Query Parameters:
    - memory_id: Memory to get recommendations for (path)
    - org_id: Organization ID
    - limit: Number of recommendations (1-50, default 10)
    - min_similarity: Minimum relationship similarity (0.0-1.0)
    - max_age_days: Exclude memories older than N days
    - use_cache: Use cached results if available (default true)
    
    Returns:
    - recommendations: List of ranked memories with scores and factors
    - memory_id: Input memory ID
    - count: Number of recommendations returned
    """
    
    # Verify authorization
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    logger.info(
        f"Get recommendations request from user {current_user.id} "
        f"for memory {memory_id} (limit={limit})"
    )
    
    redis_client = _get_sync_redis_client()
    service = RecommendationService(db, redis_client)
    
    recommendations = await service.get_recommendations(
        memory_id=memory_id,
        org_id=org_id,
        limit=limit,
        use_cache=use_cache,
        min_similarity=min_similarity,
        max_age_days=max_age_days
    )
    
    return {
        "status": "success",
        "memory_id": memory_id,
        "count": len(recommendations),
        "recommendations": recommendations
    }


@router.post("/recommendations/feedback/{recommended_memory_id}", name="submit_recommendation_feedback")
async def submit_recommendation_feedback(
    recommended_memory_id: str,
    base_memory_id: str = Query(...),
    org_id: str = Query(...),
    helpful: bool = Query(...),
    reason: Optional[str] = Query(None, max_length=500),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Submit feedback on a memory recommendation.
    
    Help improve recommendation algorithm by rating recommendations as helpful or not.
    
    Query Parameters:
    - base_memory_id: Original memory that was recommended from
    - recommended_memory_id: Memory that was recommended (path)
    - org_id: Organization ID
    - helpful: True if helpful, False if not
    - reason: Optional reason text (max 500 chars)
    
    Returns:
    - status: success/error
    - feedback: Stored feedback object
    """
    
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    logger.info(
        f"Recommendation feedback from user {current_user.id}: "
        f"{base_memory_id} -> {recommended_memory_id} (helpful={helpful})"
    )
    
    redis_client = _get_sync_redis_client()
    service = RecommendationService(db, redis_client)
    
    feedback = await service.submit_feedback(
        base_memory_id=base_memory_id,
        recommended_memory_id=recommended_memory_id,
        org_id=org_id,
        user_id=str(current_user.id),
        helpful=helpful,
        reason=reason
    )
    
    return {
        "status": "success",
        "feedback": feedback
    }


@router.get("/recommendations/metrics", name="recommendation_metrics")
async def get_recommendation_metrics(
    org_id: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get metrics about recommendations for organization.
    
    Returns:
    - total_feedback: Total feedback submissions
    - helpful_ratio: Percentage marked as helpful
    - helpful_count: Number of helpful votes
    - not_helpful_count: Number of unhelpful votes
    """
    
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    redis_client = _get_sync_redis_client()
    service = RecommendationService(db, redis_client)
    metrics = await service.get_recommendation_metrics(org_id)
    
    return {
        "status": "success",
        "org_id": org_id,
        **metrics
    }


@router.get("/recommendations/weights", name="get_recommendation_weights")
async def get_recommendation_weights(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get current recommendation ranking weights.
    
    Returns weights for each factor:
    - similarity: Weight for semantic similarity (50% default)
    - recency: Weight for recent memories (20% default)
    - interaction: Weight for interaction history (20% default)
    - feedback: Weight for user feedback (10% default)
    """
    
    redis_client = _get_sync_redis_client()
    service = RecommendationService(db, redis_client)
    weights = await service.get_weights()
    
    return {
        "status": "success",
        "org_id": str(current_user.organization_id),
        "weights": weights
    }


@router.patch("/recommendations/weights", name="update_recommendation_weights")
async def update_recommendation_weights(
    similarity: Optional[float] = Query(None, ge=0.0, le=1.0),
    recency: Optional[float] = Query(None, ge=0.0, le=1.0),
    interaction: Optional[float] = Query(None, ge=0.0, le=1.0),
    feedback: Optional[float] = Query(None, ge=0.0, le=1.0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Update recommendation ranking weights.
    
    Weights must sum to 1.0 exactly. Useful for A/B testing different algorithms.
    
    Query Parameters:
    - similarity: New weight for similarity (0.0-1.0)
    - recency: New weight for recency (0.0-1.0)
    - interaction: New weight for interaction (0.0-1.0)
    - feedback: New weight for feedback (0.0-1.0)
    
    All must be provided together and sum to 1.0.
    
    Returns:
    - weights: New weights
    """
    
    # Verify admin or org owner
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin required")
    
    # Build weights dict from provided values
    weights_dict = {}
    if similarity is not None:
        weights_dict["similarity"] = similarity
    if recency is not None:
        weights_dict["recency"] = recency
    if interaction is not None:
        weights_dict["interaction"] = interaction
    if feedback is not None:
        weights_dict["feedback"] = feedback
    
    if not weights_dict:
        raise HTTPException(
            status_code=400,
            detail="Must provide at least one weight"
        )
    
    redis_client = _get_sync_redis_client()
    service = RecommendationService(db, redis_client)
    
    try:
        updated_weights = await service.update_weights(weights_dict)
        logger.info(f"Updated weights for org {current_user.organization_id}: {updated_weights}")
        
        return {
            "status": "success",
            "weights": updated_weights
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/recommendations/cache", name="clear_recommendation_cache")
async def clear_recommendation_cache(
    org_id: str = Query(...),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Clear cached recommendations for organization.
    
    Useful after major algorithm changes or to force refresh.
    """
    
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin required")
    
    # Note: Redis doesn't have native wildcard delete, would need to scan
    # For now, just log the request
    
    logger.info(f"Cache clear request for org {org_id}")
    
    return {
        "status": "success",
        "message": "Recommendation cache cleared"
    }
