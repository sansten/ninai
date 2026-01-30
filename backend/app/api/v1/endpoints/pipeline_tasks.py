"""Pipeline task API endpoints for monitoring and management."""

from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.core.rbac import require_capability, Capabilities
from app.models.pipeline_task import PipelineTask, PipelineTaskType, PipelineTaskStatus
from app.models.user import User
from app.schemas.pipeline_tasks import (
    PipelineTaskResponse,
    PipelineTaskCreate,
    PipelineTaskUpdatePriority,
    PipelineStatsResponse,
)
from app.services.sla_scheduler_service import SLASchedulerService
from app.services.audit_service import AuditService


router = APIRouter()


def _task_to_response(task: PipelineTask) -> PipelineTaskResponse:
    """Convert PipelineTask model to response schema."""
    return PipelineTaskResponse(
        id=str(task.id),
        organization_id=str(task.organization_id),
        task_type=task.task_type,
        status=task.status,
        priority=task.priority,
        sla_deadline=task.sla_deadline,
        sla_category=task.sla_category,
        sla_remaining_ms=task.sla_remaining_ms,
        sla_breached=task.sla_breached,
        estimated_tokens=task.estimated_tokens,
        actual_tokens=task.actual_tokens,
        estimated_latency_ms=task.estimated_latency_ms,
        duration_ms=task.duration_ms,
        blocks_on_task_id=str(task.blocks_on_task_id) if task.blocks_on_task_id else None,
        blocked_by_quota=task.blocked_by_quota,
        attempts=task.attempts,
        max_attempts=task.max_attempts,
        last_error=task.last_error,
        metadata=task.metadata,
        trace_id=task.trace_id,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        updated_at=task.updated_at,
    )


@router.get("/pipelines/stats", response_model=PipelineStatsResponse)
@require_capability(Capabilities.PIPELINES_VIEW)
async def get_pipeline_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get pipeline queue statistics and metrics.
    
    Requires: pipelines.view capability
    Returns stats for the user's organization.
    """
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    org_id = current_user.organization_id
    
    # Total tasks
    total_result = await db.execute(
        select(func.count()).select_from(PipelineTask).where(
            PipelineTask.organization_id == org_id
        )
    )
    total_tasks = total_result.scalar_one()
    
    # Status counts
    queued_result = await db.execute(
        select(func.count()).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.status == PipelineTaskStatus.QUEUED.value,
            )
        )
    )
    queued_tasks = queued_result.scalar_one()
    
    running_result = await db.execute(
        select(func.count()).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.status == PipelineTaskStatus.RUNNING.value,
            )
        )
    )
    running_tasks = running_result.scalar_one()
    
    blocked_result = await db.execute(
        select(func.count()).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.status == PipelineTaskStatus.BLOCKED.value,
            )
        )
    )
    blocked_tasks = blocked_result.scalar_one()
    
    # Last hour succeeded/failed
    succeeded_result = await db.execute(
        select(func.count()).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.status == PipelineTaskStatus.SUCCEEDED.value,
                PipelineTask.completed_at >= one_hour_ago,
            )
        )
    )
    succeeded_tasks_last_hour = succeeded_result.scalar_one()
    
    failed_result = await db.execute(
        select(func.count()).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.status == PipelineTaskStatus.FAILED.value,
                PipelineTask.completed_at >= one_hour_ago,
            )
        )
    )
    failed_tasks_last_hour = failed_result.scalar_one()
    
    # SLA breached count
    sla_breached_result = await db.execute(
        select(func.count()).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.sla_deadline.isnot(None),
                PipelineTask.sla_deadline < now,
                PipelineTask.status.in_([
                    PipelineTaskStatus.QUEUED.value,
                    PipelineTaskStatus.RUNNING.value,
                ])
            )
        )
    )
    sla_breached_count = sla_breached_result.scalar_one()
    
    # SLA compliance rate (completed tasks in last hour)
    completed_with_sla_result = await db.execute(
        select(func.count()).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.sla_deadline.isnot(None),
                PipelineTask.completed_at >= one_hour_ago,
            )
        )
    )
    completed_with_sla = completed_with_sla_result.scalar_one()
    
    sla_met_result = await db.execute(
        select(func.count()).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.sla_deadline.isnot(None),
                PipelineTask.completed_at >= one_hour_ago,
                PipelineTask.completed_at <= PipelineTask.sla_deadline,
            )
        )
    )
    sla_met = sla_met_result.scalar_one()
    
    sla_compliance_rate = (sla_met / completed_with_sla * 100) if completed_with_sla > 0 else 100.0
    
    # Average times
    avg_queue_time_result = await db.execute(
        select(func.avg(
            func.extract('epoch', PipelineTask.started_at - PipelineTask.created_at) * 1000
        )).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.started_at.isnot(None),
                PipelineTask.started_at >= one_hour_ago,
            )
        )
    )
    avg_queue_time_ms = avg_queue_time_result.scalar_one()
    
    avg_exec_time_result = await db.execute(
        select(func.avg(PipelineTask.duration_ms)).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.duration_ms.isnot(None),
                PipelineTask.completed_at >= one_hour_ago,
            )
        )
    )
    avg_execution_time_ms = avg_exec_time_result.scalar_one()
    
    # Token consumption
    tokens_result = await db.execute(
        select(func.sum(PipelineTask.actual_tokens)).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.actual_tokens.isnot(None),
                PipelineTask.completed_at >= one_hour_ago,
            )
        )
    )
    total_tokens_consumed_last_hour = tokens_result.scalar_one() or 0
    
    avg_tokens_result = await db.execute(
        select(func.avg(PipelineTask.actual_tokens)).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.actual_tokens.isnot(None),
                PipelineTask.completed_at >= one_hour_ago,
            )
        )
    )
    avg_tokens_per_task = avg_tokens_result.scalar_one()
    
    # Queue depth by type
    queue_depth_result = await db.execute(
        select(
            PipelineTask.task_type,
            func.count()
        ).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.status == PipelineTaskStatus.QUEUED.value,
            )
        ).group_by(PipelineTask.task_type)
    )
    queue_depth_by_type = {row[0]: row[1] for row in queue_depth_result.all()}
    
    # SLA breach by category
    sla_breach_result = await db.execute(
        select(
            PipelineTask.sla_category,
            func.count()
        ).select_from(PipelineTask).where(
            and_(
                PipelineTask.organization_id == org_id,
                PipelineTask.sla_deadline.isnot(None),
                PipelineTask.sla_deadline < now,
                PipelineTask.status.in_([
                    PipelineTaskStatus.QUEUED.value,
                    PipelineTaskStatus.RUNNING.value,
                ])
            )
        ).group_by(PipelineTask.sla_category)
    )
    sla_breach_by_category = {row[0] or "unknown": row[1] for row in sla_breach_result.all()}
    
    return PipelineStatsResponse(
        total_tasks=total_tasks,
        queued_tasks=queued_tasks,
        running_tasks=running_tasks,
        blocked_tasks=blocked_tasks,
        succeeded_tasks_last_hour=succeeded_tasks_last_hour,
        failed_tasks_last_hour=failed_tasks_last_hour,
        sla_breached_count=sla_breached_count,
        sla_compliance_rate=round(sla_compliance_rate, 2),
        avg_queue_time_ms=round(avg_queue_time_ms, 2) if avg_queue_time_ms else None,
        avg_execution_time_ms=round(avg_execution_time_ms, 2) if avg_execution_time_ms else None,
        total_tokens_consumed_last_hour=total_tokens_consumed_last_hour,
        avg_tokens_per_task=round(avg_tokens_per_task, 2) if avg_tokens_per_task else None,
        queue_depth_by_type=queue_depth_by_type,
        sla_breach_by_category=sla_breach_by_category,
    )


@router.get("/pipelines", response_model=list[PipelineTaskResponse])
@require_capability(Capabilities.PIPELINES_VIEW)
async def list_pipeline_tasks(
    status_filter: Optional[str] = Query(None, description="Filter by status: queued, running, blocked, succeeded, failed"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    sla_breached_only: bool = Query(False, description="Show only SLA-breached tasks"),
    limit: int = Query(100, ge=1, le=500, description="Max tasks to return"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List pipeline tasks with optional filtering.
    
    Requires: pipelines.view capability
    Returns tasks for the user's organization.
    """
    now = datetime.now(timezone.utc)
    
    # Build query
    stmt = select(PipelineTask).where(
        PipelineTask.organization_id == current_user.organization_id
    )
    
    # Apply filters
    if status_filter:
        stmt = stmt.where(PipelineTask.status == status_filter.upper())
    
    if task_type:
        stmt = stmt.where(PipelineTask.task_type == task_type.upper())
    
    if sla_breached_only:
        stmt = stmt.where(
            and_(
                PipelineTask.sla_deadline.isnot(None),
                PipelineTask.sla_deadline < now,
                PipelineTask.status.in_([
                    PipelineTaskStatus.QUEUED.value,
                    PipelineTaskStatus.RUNNING.value,
                ])
            )
        )
    
    # Order by: breached first, then by deadline, then priority
    stmt = stmt.order_by(
        (PipelineTask.sla_deadline < now).desc(),
        PipelineTask.sla_deadline.asc(),
        PipelineTask.priority.desc(),
        PipelineTask.created_at.asc(),
    ).limit(limit)
    
    result = await db.execute(stmt)
    tasks = result.scalars().all()
    
    return [_task_to_response(task) for task in tasks]


@router.get("/pipelines/{task_id}", response_model=PipelineTaskResponse)
@require_capability(Capabilities.PIPELINES_VIEW)
async def get_pipeline_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific pipeline task by ID.
    
    Requires: pipelines.view capability
    """
    stmt = select(PipelineTask).where(
        and_(
            PipelineTask.id == task_id,
            PipelineTask.organization_id == current_user.organization_id,
        )
    )
    
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline task {task_id} not found"
        )
    
    return _task_to_response(task)


@router.post("/pipelines", response_model=PipelineTaskResponse, status_code=status.HTTP_201_CREATED)
@require_capability(Capabilities.PIPELINES_MANAGE)
async def create_pipeline_task(
    body: PipelineTaskCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Manually enqueue a pipeline task.
    
    Requires: pipelines.manage capability
    Uses SLASchedulerService to ensure proper scheduling.
    """
    # Calculate SLA deadline
    sla_deadline = None
    if body.sla_deadline_minutes:
        sla_deadline = datetime.now(timezone.utc) + timedelta(minutes=body.sla_deadline_minutes)
    
    # Use SLASchedulerService to enqueue
    scheduler = SLASchedulerService(db)
    
    task = await scheduler.enqueue_pipeline_task(
        organization_id=str(current_user.organization_id),
        task_type=body.task_type.upper(),
        priority=body.priority,
        sla_deadline=sla_deadline,
        sla_category=body.sla_category,
        estimated_tokens=body.estimated_tokens,
        estimated_latency_ms=body.estimated_latency_ms,
        metadata=body.metadata,
        blocks_on_task_id=body.blocks_on_task_id,
        scopes={"pipeline.enqueue"},
    )
    
    await db.commit()
    await db.refresh(task)
    
    return _task_to_response(task)


@router.put("/pipelines/{task_id}/priority", response_model=PipelineTaskResponse)
@require_capability(Capabilities.PIPELINES_ADMIN)
async def update_task_priority(
    task_id: str,
    body: PipelineTaskUpdatePriority,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update the priority of a queued or blocked pipeline task.
    
    Requires: pipelines.admin capability
    Cannot change priority of running/completed tasks.
    """
    stmt = select(PipelineTask).where(
        and_(
            PipelineTask.id == task_id,
            PipelineTask.organization_id == current_user.organization_id,
        )
    )
    
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline task {task_id} not found"
        )
    
    if task.status not in [PipelineTaskStatus.QUEUED.value, PipelineTaskStatus.BLOCKED.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot change priority of {task.status} task (only queued/blocked tasks)"
        )
    
    old_priority = task.priority
    task.priority = body.priority
    task.updated_at = datetime.now(timezone.utc)
    
    # Audit the change
    audit = AuditService(db)
    await audit.log_event(
        event_type="pipeline.priority_updated",
        organization_id=str(current_user.organization_id),
        actor_id=str(current_user.id),
        resource_type="pipeline_task",
        resource_id=task_id,
        success=True,
        details={
            "old_priority": old_priority,
            "new_priority": body.priority,
            "reason": body.reason,
        },
    )
    
    await db.commit()
    await db.refresh(task)
    
    return _task_to_response(task)


@router.post("/pipelines/{task_id}/cancel", response_model=PipelineTaskResponse)
@require_capability(Capabilities.PIPELINES_ADMIN)
async def cancel_pipeline_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel a queued or running pipeline task.
    
    Requires: pipelines.admin capability
    Sets status to FAILED with cancellation reason.
    """
    stmt = select(PipelineTask).where(
        and_(
            PipelineTask.id == task_id,
            PipelineTask.organization_id == current_user.organization_id,
        )
    )
    
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline task {task_id} not found"
        )
    
    if task.status not in [PipelineTaskStatus.QUEUED.value, PipelineTaskStatus.RUNNING.value, PipelineTaskStatus.BLOCKED.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel {task.status} task (only queued/running/blocked)"
        )
    
    old_status = task.status
    task.status = PipelineTaskStatus.FAILED.value
    task.last_error = f"Cancelled by {current_user.email}"
    task.completed_at = datetime.now(timezone.utc)
    task.updated_at = datetime.now(timezone.utc)
    
    # Audit the cancellation
    audit = AuditService(db)
    await audit.log_event(
        event_type="pipeline.task_cancelled",
        organization_id=str(current_user.organization_id),
        actor_id=str(current_user.id),
        resource_type="pipeline_task",
        resource_id=task_id,
        success=True,
        details={
            "old_status": old_status,
            "task_type": task.task_type,
        },
    )
    
    await db.commit()
    await db.refresh(task)
    
    return _task_to_response(task)


@router.post("/pipelines/{task_id}/retry", response_model=PipelineTaskResponse)
@require_capability(Capabilities.PIPELINES_ADMIN)
async def retry_pipeline_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retry a failed or cancelled pipeline task.
    
    Requires: pipelines.admin capability
    Resets status to QUEUED and clears error.
    """
    stmt = select(PipelineTask).where(
        and_(
            PipelineTask.id == task_id,
            PipelineTask.organization_id == current_user.organization_id,
        )
    )
    
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline task {task_id} not found"
        )
    
    if task.status != PipelineTaskStatus.FAILED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot retry {task.status} task (only failed tasks)"
        )
    
    if task.attempts >= task.max_attempts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task has reached max attempts ({task.max_attempts})"
        )
    
    task.status = PipelineTaskStatus.QUEUED.value
    task.last_error = None
    task.completed_at = None
    task.started_at = None
    task.duration_ms = None
    task.updated_at = datetime.now(timezone.utc)
    
    # Audit the retry
    audit = AuditService(db)
    await audit.log_event(
        event_type="pipeline.task_retried",
        organization_id=str(current_user.organization_id),
        actor_id=str(current_user.id),
        resource_type="pipeline_task",
        resource_id=task_id,
        success=True,
        details={
            "attempts": task.attempts,
            "max_attempts": task.max_attempts,
        },
    )
    
    await db.commit()
    await db.refresh(task)
    
    return _task_to_response(task)


@router.get("/pipelines/stats/history", response_model=list[dict])
@require_capability(Capabilities.PIPELINES_VIEW)
async def get_pipeline_stats_history(
    hours: int = Query(24, ge=1, le=168, description="Hours of history to return"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get historical pipeline statistics for trending.
    
    Requires: pipelines.view capability
    Returns hourly snapshots of queue stats for the specified time range.
    """
    now = datetime.now(timezone.utc)
    org_id = current_user.organization_id
    history = []
    
    # Generate hourly snapshots
    for hour_offset in range(hours, 0, -1):
        snapshot_time = now - timedelta(hours=hour_offset)
        hour_start = snapshot_time.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)
        
        # Completed tasks in this hour
        completed_result = await db.execute(
            select(func.count()).select_from(PipelineTask).where(
                and_(
                    PipelineTask.organization_id == org_id,
                    PipelineTask.completed_at >= hour_start,
                    PipelineTask.completed_at < hour_end,
                )
            )
        )
        completed = completed_result.scalar_one()
        
        # SLA compliance for this hour
        with_sla_result = await db.execute(
            select(func.count()).select_from(PipelineTask).where(
                and_(
                    PipelineTask.organization_id == org_id,
                    PipelineTask.sla_deadline.isnot(None),
                    PipelineTask.completed_at >= hour_start,
                    PipelineTask.completed_at < hour_end,
                )
            )
        )
        with_sla = with_sla_result.scalar_one()
        
        sla_met_result = await db.execute(
            select(func.count()).select_from(PipelineTask).where(
                and_(
                    PipelineTask.organization_id == org_id,
                    PipelineTask.sla_deadline.isnot(None),
                    PipelineTask.completed_at >= hour_start,
                    PipelineTask.completed_at < hour_end,
                    PipelineTask.completed_at <= PipelineTask.sla_deadline,
                )
            )
        )
        sla_met = sla_met_result.scalar_one()
        
        sla_compliance = (sla_met / with_sla * 100) if with_sla > 0 else 100.0
        
        # Avg execution time
        avg_duration_result = await db.execute(
            select(func.avg(PipelineTask.duration_ms)).select_from(PipelineTask).where(
                and_(
                    PipelineTask.organization_id == org_id,
                    PipelineTask.duration_ms.isnot(None),
                    PipelineTask.completed_at >= hour_start,
                    PipelineTask.completed_at < hour_end,
                )
            )
        )
        avg_duration = avg_duration_result.scalar_one()
        
        history.append({
            "timestamp": hour_start.isoformat(),
            "completed_tasks": completed,
            "sla_compliance_rate": round(sla_compliance, 2),
            "avg_duration_ms": round(avg_duration, 2) if avg_duration else None,
        })
    
    return history


@router.get("/pipelines/export")
@require_capability(Capabilities.PIPELINES_VIEW)
async def export_pipeline_tasks(
    format: str = Query("csv", regex="^(csv|json)$", description="Export format"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    hours: int = Query(24, ge=1, le=720, description="Hours of history"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Export pipeline tasks as CSV or JSON.
    
    Requires: pipelines.view capability
    Downloads task history for analysis.
    """
    from fastapi.responses import StreamingResponse
    import csv
    import json
    from io import StringIO
    
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)
    
    stmt = select(PipelineTask).where(
        and_(
            PipelineTask.organization_id == current_user.organization_id,
            PipelineTask.created_at >= since,
        )
    )
    
    if status_filter:
        stmt = stmt.where(PipelineTask.status == status_filter.upper())
    
    stmt = stmt.order_by(PipelineTask.created_at.desc()).limit(10000)
    
    result = await db.execute(stmt)
    tasks = result.scalars().all()
    
    if format == "csv":
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "id", "task_type", "status", "priority", "sla_category", "sla_breached",
            "estimated_tokens", "actual_tokens", "duration_ms", "attempts",
            "blocked_by_quota", "created_at", "started_at", "completed_at", "last_error"
        ])
        writer.writeheader()
        
        for task in tasks:
            writer.writerow({
                "id": str(task.id),
                "task_type": task.task_type,
                "status": task.status,
                "priority": task.priority,
                "sla_category": task.sla_category or "",
                "sla_breached": task.sla_breached,
                "estimated_tokens": task.estimated_tokens or "",
                "actual_tokens": task.actual_tokens or "",
                "duration_ms": task.duration_ms or "",
                "attempts": task.attempts,
                "blocked_by_quota": task.blocked_by_quota,
                "created_at": task.created_at.isoformat() if task.created_at else "",
                "started_at": task.started_at.isoformat() if task.started_at else "",
                "completed_at": task.completed_at.isoformat() if task.completed_at else "",
                "last_error": task.last_error or "",
            })
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=pipeline_tasks_{now.strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    
    else:  # json
        data = [_task_to_response(task).model_dump(mode='json') for task in tasks]
        return StreamingResponse(
            iter([json.dumps(data, indent=2)]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=pipeline_tasks_{now.strftime('%Y%m%d_%H%M%S')}.json"}
        )


@router.get("/pipelines/{task_id}/dependencies", response_model=dict)
@require_capability(Capabilities.PIPELINES_VIEW)
async def get_task_dependencies(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get dependency tree for a pipeline task.
    
    Requires: pipelines.view capability
    Returns tasks that this task blocks on (dependencies) and tasks blocked by this task (dependents).
    """
    # Get the target task
    stmt = select(PipelineTask).where(
        and_(
            PipelineTask.id == task_id,
            PipelineTask.organization_id == current_user.organization_id,
        )
    )
    result = await db.execute(stmt)
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline task {task_id} not found"
        )
    
    # Find dependencies (tasks this blocks on)
    dependencies = []
    if task.blocks_on_task_id:
        dep_stmt = select(PipelineTask).where(
            and_(
                PipelineTask.id == task.blocks_on_task_id,
                PipelineTask.organization_id == current_user.organization_id,
            )
        )
        dep_result = await db.execute(dep_stmt)
        dep_task = dep_result.scalar_one_or_none()
        if dep_task:
            dependencies.append(_task_to_response(dep_task))
    
    # Find dependents (tasks that block on this)
    dependents_stmt = select(PipelineTask).where(
        and_(
            PipelineTask.blocks_on_task_id == task_id,
            PipelineTask.organization_id == current_user.organization_id,
        )
    )
    dependents_result = await db.execute(dependents_stmt)
    dependents_tasks = dependents_result.scalars().all()
    dependents = [_task_to_response(t) for t in dependents_tasks]
    
    return {
        "task": _task_to_response(task),
        "dependencies": dependencies,
        "dependents": dependents,
    }
