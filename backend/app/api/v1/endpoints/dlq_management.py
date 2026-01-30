"""
DLQ Management Endpoints

REST API for managing dead letter queue (failed Celery tasks).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from pydantic import BaseModel
import uuid

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.middleware.tenant_context import get_tenant_context, TenantContext
from app.services.dlq_service import DLQService
import logging

router = APIRouter(prefix="/dlq", tags=["DLQ Management"])
logger = logging.getLogger(__name__)


# Schemas
class DLQTaskResponse(BaseModel):
    """DLQ task response."""
    task_id: str
    task_name: str
    retry_count: int
    max_retries: int
    exception: str
    created_at: str
    next_retry_at: Optional[str] = None

    class Config:
        from_attributes = True


class DLQStatsResponse(BaseModel):
    """DLQ statistics response."""
    total_failed_tasks: int
    total_retrying: int
    by_task: dict  # {task_name: {count, avg_retries}}
    oldest_task_age_hours: Optional[float] = None


class RetryTaskRequest(BaseModel):
    """Request to retry a task."""
    task_id: str


class RetryAllRequest(BaseModel):
    """Request to retry all failed tasks."""
    task_filter: Optional[str] = None  # task name filter


class PurgeTaskRequest(BaseModel):
    """Request to purge a task."""
    task_id: str


class PurgeOldRequest(BaseModel):
    """Request to purge old tasks."""
    days: int = 30


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Dependency to require admin role."""
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user


@router.get("/tasks", response_model=List[DLQTaskResponse])
async def list_dlq_tasks(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    task_name: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List all failed tasks in DLQ.
    
    Admin only.
    """
    try:
        service = DLQService(db)
        tasks = await service.get_failed_tasks(
            task_name_filter=task_name,
            limit=limit,
            offset=offset
        )
        return tasks
    except Exception as e:
        logger.error(f"Error listing DLQ tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to list DLQ tasks")


@router.post("/retry-task")
async def retry_dlq_task(
    request: RetryTaskRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin)
):
    """
    Retry a single failed task with exponential backoff.
    
    Admin only. Task will be retried after 60/300/900 seconds (attempt 1/2/3).
    """
    try:
        service = DLQService(db)
        success = await service.retry_task(request.task_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Task {request.task_id} not found")
        
        return {
            "task_id": request.task_id,
            "status": "queued_for_retry",
            "message": "Task will be retried with exponential backoff"
        }
    except Exception as e:
        logger.error(f"Error retrying task: {e}")
        raise HTTPException(status_code=500, detail="Failed to retry task")


@router.post("/retry-all")
async def retry_all_dlq_tasks(
    request: RetryAllRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin)
):
    """
    Retry all failed tasks, optionally filtered by task name.
    
    Admin only. All matching tasks will be queued for retry.
    """
    try:
        service = DLQService(db)
        count = await service.retry_all_failed(
            task_name_filter=request.task_filter
        )
        
        return {
            "tasks_retried": count,
            "message": f"Queued {count} tasks for retry",
            "filter": request.task_filter
        }
    except Exception as e:
        logger.error(f"Error retrying all tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retry tasks")


@router.delete("/purge-task")
async def purge_dlq_task(
    request: PurgeTaskRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin)
):
    """
    Permanently delete a task from DLQ.
    
    Admin only. This cannot be undone.
    """
    try:
        service = DLQService(db)
        success = await service.purge_task(request.task_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Task {request.task_id} not found")
        
        return {
            "task_id": request.task_id,
            "status": "purged",
            "message": "Task permanently deleted from DLQ"
        }
    except Exception as e:
        logger.error(f"Error purging task: {e}")
        raise HTTPException(status_code=500, detail="Failed to purge task")


@router.delete("/purge-old")
async def purge_old_dlq_tasks(
    request: PurgeOldRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin)
):
    """
    Delete tasks older than N days from DLQ.
    
    Admin only. Default: 30 days.
    """
    try:
        service = DLQService(db)
        count = await service.purge_old_tasks(days=request.days)
        
        return {
            "tasks_purged": count,
            "days": request.days,
            "message": f"Deleted {count} tasks older than {request.days} days"
        }
    except Exception as e:
        logger.error(f"Error purging old tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to purge old tasks")


@router.get("/stats", response_model=DLQStatsResponse)
async def get_dlq_stats(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin)
):
    """
    Get DLQ statistics and metrics.
    
    Admin only. Returns task counts, retry statistics, and age information.
    """
    try:
        service = DLQService(db)
        stats = await service.get_dlq_stats()
        
        return {
            "total_failed_tasks": stats.get("total_failed", 0),
            "total_retrying": stats.get("total_retrying", 0),
            "by_task": stats.get("by_task_name", {}),
            "oldest_task_age_hours": stats.get("oldest_age_hours")
        }
    except Exception as e:
        logger.error(f"Error getting DLQ stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get DLQ stats")
