"""
Dead Letter Queue (DLQ) Service

Handles failed Celery tasks with retry logic, error tracking, and admin management.
"""

import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from app.models.pipeline_task import PipelineTask, PipelineTaskStatus, PipelineTaskType
from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)


class DLQService:
    """Service for managing dead letter queue (failed tasks)."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.max_retries = 3
        self.retry_delays = [60, 300, 900]  # 1min, 5min, 15min
    
    async def enqueue_failed_task(
        self,
        task_id: str,
        task_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        exception: str,
        traceback: str,
        org_id: uuid.UUID
    ) -> PipelineTask:
        """
        Add a failed task to the DLQ.
        
        Args:
            task_id: Celery task ID
            task_name: Task name (e.g., 'memory_pipeline')
            args: Task arguments
            kwargs: Task keyword arguments
            exception: Exception message
            traceback: Full traceback
            org_id: Organization ID
            
        Returns:
            Created DLQ task record
        """
        try:
            # Check if task already in DLQ
            stmt = select(PipelineTask).where(
                PipelineTask.task_id == task_id
            )
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing record
                existing.status = PipelineTaskStatus.FAILED
                existing.error_message = exception
                existing.retry_count = (existing.retry_count or 0) + 1
                existing.updated_at = datetime.utcnow()
                
                if existing.metadata:
                    existing.metadata["traceback"] = traceback
                    existing.metadata["last_failure"] = datetime.utcnow().isoformat()
                
                dlq_task = existing
            else:
                # Create new DLQ record
                dlq_task = PipelineTask(
                    id=uuid.uuid4(),
                    task_id=task_id,
                    task_name=task_name,
                    status=PipelineTaskStatus.FAILED,
                    organization_id=org_id,
                    args=json.dumps(args),
                    kwargs=json.dumps(kwargs),
                    error_message=exception,
                    retry_count=1,
                    metadata={
                        "traceback": traceback,
                        "first_failure": datetime.utcnow().isoformat(),
                        "last_failure": datetime.utcnow().isoformat()
                    }
                )
                self.db.add(dlq_task)
            
            await self.db.commit()
            
            logger.info(f"Added task {task_id} to DLQ (retry {dlq_task.retry_count})")
            return dlq_task
        
        except Exception as e:
            logger.error(f"Error adding task to DLQ: {e}")
            await self.db.rollback()
            raise
    
    async def get_failed_tasks(
        self,
        org_id: Optional[uuid.UUID] = None,
        task_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[PipelineTask]:
        """Get failed tasks from DLQ."""
        stmt = select(PipelineTask).where(
            PipelineTask.status == PipelineTaskStatus.FAILED
        )
        
        if org_id:
            stmt = stmt.where(PipelineTask.organization_id == org_id)
        if task_name:
            stmt = stmt.where(PipelineTask.task_name == task_name)
        
        stmt = stmt.order_by(PipelineTask.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def retry_task(self, task_id: str) -> bool:
        """
        Retry a failed task.
        
        Returns True if task was retried, False if max retries exceeded.
        """
        try:
            stmt = select(PipelineTask).where(
                PipelineTask.task_id == task_id
            )
            result = await self.db.execute(stmt)
            task = result.scalar_one_or_none()
            
            if not task:
                logger.error(f"Task {task_id} not found in DLQ")
                return False
            
            # Check retry limit
            retry_count = task.retry_count or 0
            if retry_count >= self.max_retries:
                logger.warning(f"Task {task_id} exceeded max retries ({self.max_retries})")
                task.status = PipelineTaskStatus.FAILED
                task.metadata = task.metadata or {}
                task.metadata["max_retries_exceeded"] = True
                await self.db.commit()
                return False
            
            # Calculate retry delay
            delay = self.retry_delays[min(retry_count, len(self.retry_delays) - 1)]
            
            # Resubmit task
            args = json.loads(task.args) if task.args else []
            kwargs = json.loads(task.kwargs) if task.kwargs else {}
            
            # Get task function from Celery
            celery_task = celery_app.tasks.get(task.task_name)
            if not celery_task:
                logger.error(f"Task {task.task_name} not found in Celery registry")
                return False
            
            # Apply async with countdown
            new_task = celery_task.apply_async(
                args=args,
                kwargs=kwargs,
                countdown=delay
            )
            
            # Update DLQ record
            task.status = PipelineTaskStatus.RETRYING
            task.retry_count = retry_count + 1
            task.metadata = task.metadata or {}
            task.metadata["retry_task_id"] = new_task.id
            task.metadata["retry_scheduled"] = datetime.utcnow().isoformat()
            task.metadata["retry_delay_seconds"] = delay
            task.updated_at = datetime.utcnow()
            
            await self.db.commit()
            
            logger.info(f"Retrying task {task_id} (attempt {retry_count + 1}) with {delay}s delay")
            return True
        
        except Exception as e:
            logger.error(f"Error retrying task: {e}")
            await self.db.rollback()
            return False
    
    async def retry_all_failed(self, org_id: Optional[uuid.UUID] = None) -> Dict[str, int]:
        """
        Retry all failed tasks.
        
        Returns dict with retry stats.
        """
        failed_tasks = await self.get_failed_tasks(org_id=org_id)
        
        retried = 0
        skipped = 0
        
        for task in failed_tasks:
            if await self.retry_task(task.task_id):
                retried += 1
            else:
                skipped += 1
        
        return {
            "total": len(failed_tasks),
            "retried": retried,
            "skipped": skipped
        }
    
    async def purge_task(self, task_id: str) -> bool:
        """
        Permanently delete a task from DLQ.
        
        Returns True if deleted, False if not found.
        """
        try:
            stmt = select(PipelineTask).where(
                PipelineTask.task_id == task_id
            )
            result = await self.db.execute(stmt)
            task = result.scalar_one_or_none()
            
            if not task:
                return False
            
            await self.db.delete(task)
            await self.db.commit()
            
            logger.info(f"Purged task {task_id} from DLQ")
            return True
        
        except Exception as e:
            logger.error(f"Error purging task: {e}")
            await self.db.rollback()
            return False
    
    async def purge_old_tasks(self, days: int = 30) -> int:
        """
        Delete tasks older than specified days.
        
        Returns count of deleted tasks.
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            stmt = select(PipelineTask).where(
                and_(
                    PipelineTask.status == PipelineTaskStatus.FAILED,
                    PipelineTask.created_at < cutoff
                )
            )
            result = await self.db.execute(stmt)
            tasks = result.scalars().all()
            
            count = 0
            for task in tasks:
                await self.db.delete(task)
                count += 1
            
            await self.db.commit()
            
            logger.info(f"Purged {count} old tasks from DLQ")
            return count
        
        except Exception as e:
            logger.error(f"Error purging old tasks: {e}")
            await self.db.rollback()
            return 0
    
    async def get_dlq_stats(self, org_id: Optional[uuid.UUID] = None) -> Dict[str, Any]:
        """Get DLQ statistics."""
        try:
            # Base query
            base_stmt = select(PipelineTask).where(
                PipelineTask.status == PipelineTaskStatus.FAILED
            )
            
            if org_id:
                base_stmt = base_stmt.where(PipelineTask.organization_id == org_id)
            
            # Total failed
            result = await self.db.execute(base_stmt)
            all_failed = result.scalars().all()
            
            # Group by task name
            by_task_name = {}
            by_retry_count = {0: 0, 1: 0, 2: 0, 3: 0, "3+": 0}
            
            for task in all_failed:
                # By task name
                name = task.task_name or "unknown"
                by_task_name[name] = by_task_name.get(name, 0) + 1
                
                # By retry count
                retries = task.retry_count or 0
                if retries >= 3:
                    by_retry_count["3+"] += 1
                else:
                    by_retry_count[retries] += 1
            
            # Recent failures (last 24h)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_stmt = base_stmt.where(PipelineTask.created_at >= recent_cutoff)
            result = await self.db.execute(recent_stmt)
            recent_failed = result.scalars().all()
            
            return {
                "total_failed": len(all_failed),
                "recent_24h": len(recent_failed),
                "by_task_name": by_task_name,
                "by_retry_count": by_retry_count,
                "avg_retries": sum(t.retry_count or 0 for t in all_failed) / len(all_failed) if all_failed else 0
            }
        
        except Exception as e:
            logger.error(f"Error getting DLQ stats: {e}")
            return {}
