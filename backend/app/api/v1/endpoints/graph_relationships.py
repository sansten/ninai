"""
API endpoints for graph relationship management.

Endpoints:
- POST /api/v1/graph/populate - Trigger relationship population
- GET /api/v1/graph/relationships - List relationships
- GET /api/v1/graph/stats - Get relationship statistics
- GET /api/v1/graph/config - Get relationship generation config
- PATCH /api/v1/graph/config - Update relationship generation config
"""

import logging
from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

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
from app.models.graph_relationship import GraphRelationship
from app.services.graph_relationship_service import GraphRelationshipService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph/relationships", tags=["graph"])


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


@router.post("/populate", name="populate_relationships")
async def populate_relationships(
    org_id: str = Query(..., description="Organization ID"),
    similarity_threshold: float = Query(0.75, ge=0.0, le=1.0),
    max_relationships: int = Query(5, ge=1, le=50),
    async_: bool = Query(True, description="Run async via Celery"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Populate graph relationships based on embedding similarity.
    
    Finds similar memories and creates RELATES_TO relationships.
    
    Query Parameters:
    - org_id: Organization ID (required)
    - similarity_threshold: Min similarity 0.0-1.0 (default 0.75)
    - max_relationships: Max relationships per memory (default 5)
    - async_: Run async via Celery (default true)
    
    Returns:
    - If async: {task_id, status, org_id}
    - If sync: {relationships_created, similarities, ...}
    """
    
    # Check authorization
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    logger.info(
        f"Populate graph relationships request from user {current_user.id} "
        f"for org {org_id} "
        f"(threshold={similarity_threshold}, async={async_})"
    )
    
    if async_:
        # Queue Celery task (lazy import so test/startup doesn't require Celery wiring)
        try:
            from app.tasks.graph_population import populate_graph_relationships
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Celery task unavailable: {e}")

        task = populate_graph_relationships.delay(
            org_id=org_id,
            similarity_threshold=similarity_threshold,
            batch_size=100,
        )
        
        return {
            "status": "queued",
            "task_id": task.id,
            "org_id": org_id,
            "similarity_threshold": similarity_threshold,
            "max_relationships": max_relationships
        }
    else:
        # Run synchronously
        redis_client = _get_sync_redis_client()
        if redis_client is None:
            raise HTTPException(status_code=503, detail="Redis not available - graph features disabled")

        service = GraphRelationshipService(db, redis_client)
        
        result = await service.populate_relationships(
            org_id=org_id,
            similarity_threshold=similarity_threshold,
            max_relationships_per_memory=max_relationships
        )
        
        return {
            "status": "completed",
            "org_id": org_id,
            **result
        }


@router.get("", name="list_relationships")
async def list_relationships(
    org_id: str = Query(...),
    memory_id: Optional[str] = Query(None, description="Filter by memory ID"),
    relationship_type: Optional[str] = Query(None),
    auto_created_only: bool = Query(False),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    List graph relationships for organization.
    
    Query Parameters:
    - org_id: Organization ID
    - memory_id: Filter by specific memory (both incoming and outgoing)
    - relationship_type: Filter by type (RELATES_TO, DEPENDS_ON, etc)
    - auto_created_only: Show only auto-generated relationships
    - min_similarity: Filter by minimum similarity score
    - limit: Results per page
    - offset: Pagination offset
    """
    
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Build query
    try:
        org_uuid = UUID(org_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid org_id")

    filters = [GraphRelationship.organization_id == org_uuid]
    
    if memory_id:
        filters.append(
            or_(
                GraphRelationship.from_memory_id == memory_id,
                GraphRelationship.to_memory_id == memory_id,
            )
        )
    
    if relationship_type:
        filters.append(GraphRelationship.relationship_type == relationship_type)
    
    if auto_created_only:
        filters.append(GraphRelationship.auto_created.is_(True))
    
    if min_similarity > 0.0:
        filters.append(GraphRelationship.similarity_score >= min_similarity)
    
    stmt = select(GraphRelationship).where(and_(*filters))
    
    # Get total count
    count_stmt = select(func.count()).select_from(GraphRelationship).where(and_(*filters))
    count_result = await db.execute(count_stmt)
    total = int(count_result.scalar() or 0)
    
    # Get paginated results
    stmt = stmt.order_by(GraphRelationship.similarity_score.desc()).offset(offset).limit(limit)
    result = await db.execute(stmt)
    relationships = result.scalars().all()
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "relationships": [r.to_dict() for r in relationships]
    }


@router.get("/stats", name="relationship_stats")
async def get_relationship_stats(
    org_id: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get statistics about graph relationships for organization.
    
    Returns:
    - total: Total relationships
    - auto_created: Count of auto-generated
    - manually_created: Count of manual
    - avg_similarity: Average similarity score
    - min/max_similarity: Range of scores
    - by_type: Breakdown by relationship type
    """
    
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    try:
        org_uuid = UUID(org_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid org_id")
    
    redis_client = _get_sync_redis_client()
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not available - graph features disabled")

    service = GraphRelationshipService(db, redis_client)
    stats = await service.get_relationship_stats(org_id)
    
    # Add breakdown by type
    type_result = await db.execute(
        select(
            GraphRelationship.relationship_type,
            GraphRelationship.auto_created
        ).where(GraphRelationship.organization_id == org_uuid)
    )
    
    by_type = {}
    for rel_type, auto in type_result:
        if rel_type not in by_type:
            by_type[rel_type] = {"auto": 0, "manual": 0}
        if auto:
            by_type[rel_type]["auto"] += 1
        else:
            by_type[rel_type]["manual"] += 1
    
    stats["by_type"] = by_type
    
    return stats


@router.get("/config", name="get_graph_config")
async def get_graph_config(
    org_id: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get graph relationship generation config for organization."""
    
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    redis_client = _get_sync_redis_client()
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not available - graph features disabled")

    service = GraphRelationshipService(db, redis_client)
    config = await service.get_config(org_id)
    
    return {
        "org_id": org_id,
        **config
    }


@router.patch("/config", name="update_graph_config")
async def update_graph_config(
    org_id: str = Query(...),
    similarity_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_relationships: Optional[int] = Query(None, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Update graph relationship generation config.
    
    These settings will be used for next population run.
    """
    
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    redis_client = _get_sync_redis_client()
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not available - graph features disabled")

    service = GraphRelationshipService(db, redis_client)
    config = await service.update_config(
        org_id,
        similarity_threshold=similarity_threshold,
        max_relationships_per_memory=max_relationships
    )
    
    logger.info(f"Updated graph config for org {org_id}")
    
    return {
        "status": "updated",
        "org_id": org_id,
        "config": config
    }


@router.get("/memory/{memory_id}/related", name="get_related_memories")
async def get_related_memories(
    memory_id: str,
    org_id: str = Query(...),
    relationship_type: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get memories related to a specific memory via graph relationships.
    
    Returns both incoming and outgoing relationships.
    """
    
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Get relationships where this memory is the source
    try:
        org_uuid = UUID(org_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid org_id")

    from_stmt = (
        select(GraphRelationship)
        .where(
            and_(
                GraphRelationship.organization_id == org_uuid,
                GraphRelationship.from_memory_id == memory_id,
            )
        )
        .order_by(GraphRelationship.similarity_score.desc())
        .limit(limit)
    )
    
    if relationship_type:
        from_stmt = from_stmt.where(GraphRelationship.relationship_type == relationship_type)
    
    from_result = await db.execute(from_stmt)
    from_rels = from_result.scalars().all()
    
    # Get relationships where this memory is the target
    to_stmt = (
        select(GraphRelationship)
        .where(
            and_(
                GraphRelationship.organization_id == org_uuid,
                GraphRelationship.to_memory_id == memory_id,
            )
        )
        .order_by(GraphRelationship.similarity_score.desc())
        .limit(limit)
    )
    
    if relationship_type:
        to_stmt = to_stmt.where(GraphRelationship.relationship_type == relationship_type)
    
    to_result = await db.execute(to_stmt)
    to_rels = to_result.scalars().all()
    
    return {
        "memory_id": memory_id,
        "outgoing": [r.to_dict() for r in from_rels],
        "incoming": [r.to_dict() for r in to_rels],
    }
