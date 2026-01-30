"""Dead Letter Queue API endpoints."""

from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.core.rbac import require_capability, Capabilities
from app.models.user import User
from app.models.dead_letter_queue import DeadLetterTask
from app.services.dead_letter_queue_service import DeadLetterQueueService
from app.schemas.dead_letter_queue import (
    DeadLetterTaskResponse,
    DeadLetterTaskRequeue,
    DeadLetterTaskDiscard,
    DeadLetterQueueStats,
)

router = APIRouter()


@router.get("/stats", response_model=DeadLetterQueueStats)
@require_capability(Capabilities.DLQ_VIEW)
async def get_dlq_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get Dead Letter Queue statistics.
    
    Requires: dlq.view capability
    """
    org_id = current_user.organization_id

    # Total unresolved tasks
    total_stmt = select(func.count()).where(
        and_(
            DeadLetterTask.organization_id == org_id,
            DeadLetterTask.is_resolved == False,
        )
    )
    total_result = await db.execute(total_stmt)
    total_unresolved = total_result.scalar_one()

    # By failure reason
    reason_stmt = (
        select(
            DeadLetterTask.failure_reason,
            func.count().label("count"),
        )
        .where(
            and_(
                DeadLetterTask.organization_id == org_id,
                DeadLetterTask.is_resolved == False,
            )
        )
        .group_by(DeadLetterTask.failure_reason)
    )
    reason_result = await db.execute(reason_stmt)
    by_reason = {row[0]: row[1] for row in reason_result.all()}

    # By task type
    type_stmt = (
        select(
            DeadLetterTask.task_type,
            func.count().label("count"),
        )
        .where(
            and_(
                DeadLetterTask.organization_id == org_id,
                DeadLetterTask.is_resolved == False,
            )
        )
        .group_by(DeadLetterTask.task_type)
    )
    type_result = await db.execute(type_stmt)
    by_task_type = {row[0]: row[1] for row in type_result.all()}

    # High priority count (priority >= 8)
    high_priority_stmt = select(func.count()).where(
        and_(
            DeadLetterTask.organization_id == org_id,
            DeadLetterTask.is_resolved == False,
            DeadLetterTask.review_priority >= 8,
        )
    )
    high_priority_result = await db.execute(high_priority_stmt)
    high_priority_count = high_priority_result.scalar_one()

    return DeadLetterQueueStats(
        total_unresolved=total_unresolved,
        by_failure_reason=by_reason,
        by_task_type=by_task_type,
        high_priority_count=high_priority_count,
    )


@router.get("", response_model=List[DeadLetterTaskResponse])
@require_capability(Capabilities.DLQ_VIEW)
async def list_dlq_tasks(
    is_resolved: Optional[bool] = Query(None),
    task_type: Optional[str] = Query(None),
    failure_reason: Optional[str] = Query(None),
    min_priority: Optional[int] = Query(None, ge=1, le=10),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List Dead Letter Queue tasks with optional filtering.
    
    Requires: dlq.view capability
    """
    org_id = current_user.organization_id

    # Build query
    conditions = [DeadLetterTask.organization_id == org_id]
    
    if is_resolved is not None:
        conditions.append(DeadLetterTask.is_resolved == is_resolved)
    
    if task_type:
        conditions.append(DeadLetterTask.task_type == task_type)
    
    if failure_reason:
        conditions.append(DeadLetterTask.failure_reason == failure_reason)
    
    if min_priority is not None:
        conditions.append(DeadLetterTask.review_priority >= min_priority)

    stmt = (
        select(DeadLetterTask)
        .where(and_(*conditions))
        .order_by(
            desc(DeadLetterTask.review_priority),
            DeadLetterTask.quarantined_at.asc(),
        )
        .limit(limit)
        .offset(offset)
    )

    result = await db.execute(stmt)
    tasks = result.scalars().all()

    return [
        DeadLetterTaskResponse(
            id=t.id,
            organization_id=t.organization_id,
            original_task_id=t.original_task_id,
            task_type=t.task_type,
            failure_reason=t.failure_reason,
            total_attempts=t.total_attempts,
            last_error=t.last_error,
            error_pattern=t.error_pattern,
            task_payload=t.task_payload,
            metadata=t.task_metadata,
            quarantined_at=t.quarantined_at,
            reviewed_at=t.reviewed_at,
            reviewed_by=t.reviewed_by,
            resolution=t.resolution,
            resolution_notes=t.resolution_notes,
            is_resolved=t.is_resolved,
            review_priority=t.review_priority,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in tasks
    ]


@router.get("/{dlq_id}", response_model=DeadLetterTaskResponse)
@require_capability(Capabilities.DLQ_VIEW)
async def get_dlq_task(
    dlq_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific Dead Letter Queue task.
    
    Requires: dlq.view capability
    """
    stmt = select(DeadLetterTask).where(
        and_(
            DeadLetterTask.id == dlq_id,
            DeadLetterTask.organization_id == current_user.organization_id,
        )
    )
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail="DLQ task not found")

    return DeadLetterTaskResponse(
        id=task.id,
        organization_id=task.organization_id,
        original_task_id=task.original_task_id,
        task_type=task.task_type,
        failure_reason=task.failure_reason,
        total_attempts=task.total_attempts,
        last_error=task.last_error,
        error_pattern=task.error_pattern,
        task_payload=task.task_payload,
        metadata=task.task_metadata,
        quarantined_at=task.quarantined_at,
        reviewed_at=task.reviewed_at,
        reviewed_by=task.reviewed_by,
        resolution=task.resolution,
        resolution_notes=task.resolution_notes,
        is_resolved=task.is_resolved,
        review_priority=task.review_priority,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@router.post("/{dlq_id}/requeue", response_model=DeadLetterTaskResponse)
@require_capability(Capabilities.DLQ_MANAGE)
async def requeue_dlq_task(
    dlq_id: UUID,
    payload: DeadLetterTaskRequeue,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Requeue a DLQ task back to the pipeline.
    
    Requires: dlq.manage capability
    """
    dlq_service = DeadLetterQueueService(db)

    try:
        await dlq_service.requeue_task(
            dlq_id=dlq_id,
            user_id=current_user.id,
            notes=payload.notes,
        )
        await db.commit()

        # Return updated DLQ task
        stmt = select(DeadLetterTask).where(DeadLetterTask.id == dlq_id)
        result = await db.execute(stmt)
        task = result.scalar_one()

        return DeadLetterTaskResponse(
            id=task.id,
            organization_id=task.organization_id,
            original_task_id=task.original_task_id,
            task_type=task.task_type,
            failure_reason=task.failure_reason,
            total_attempts=task.total_attempts,
            last_error=task.last_error,
            error_pattern=task.error_pattern,
            task_payload=task.task_payload,
            metadata=task.task_metadata,
            quarantined_at=task.quarantined_at,
            reviewed_at=task.reviewed_at,
            reviewed_by=task.reviewed_by,
            resolution=task.resolution,
            resolution_notes=task.resolution_notes,
            is_resolved=task.is_resolved,
            review_priority=task.review_priority,
            created_at=task.created_at,
            updated_at=task.updated_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dlq_id}/discard", response_model=DeadLetterTaskResponse)
@require_capability(Capabilities.DLQ_MANAGE)
async def discard_dlq_task(
    dlq_id: UUID,
    payload: DeadLetterTaskDiscard,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Permanently discard a DLQ task.
    
    Requires: dlq.manage capability
    """
    dlq_service = DeadLetterQueueService(db)

    try:
        task = await dlq_service.discard_task(
            dlq_id=dlq_id,
            user_id=current_user.id,
            notes=payload.notes,
        )
        await db.commit()

        return DeadLetterTaskResponse(
            id=task.id,
            organization_id=task.organization_id,
            original_task_id=task.original_task_id,
            task_type=task.task_type,
            failure_reason=task.failure_reason,
            total_attempts=task.total_attempts,
            last_error=task.last_error,
            error_pattern=task.error_pattern,
            task_payload=task.task_payload,
            metadata=task.task_metadata,
            quarantined_at=task.quarantined_at,
            reviewed_at=task.reviewed_at,
            reviewed_by=task.reviewed_by,
            resolution=task.resolution,
            resolution_notes=task.resolution_notes,
            is_resolved=task.is_resolved,
            review_priority=task.review_priority,
            created_at=task.created_at,
            updated_at=task.updated_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/detect-poison-messages")
@require_capability(Capabilities.DLQ_MANAGE)
async def detect_poison_messages(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Scan for and quarantine poison messages.
    
    Requires: dlq.manage capability
    """
    dlq_service = DeadLetterQueueService(db)

    quarantined = await dlq_service.detect_poison_messages(
        organization_id=current_user.organization_id
    )
    await db.commit()

    return {
        "quarantined_count": len(quarantined),
        "task_ids": [str(t.id) for t in quarantined],
    }
