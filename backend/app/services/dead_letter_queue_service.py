"""Dead Letter Queue service for managing failed pipeline tasks."""

from datetime import datetime, timezone
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dead_letter_queue import DeadLetterTask
from app.models.pipeline_task import PipelineTask, PipelineTaskStatus
from app.services.audit_service import AuditService


class DeadLetterQueueService:
    """Service for managing dead letter queue operations."""

    # Poison message detection thresholds
    POISON_MESSAGE_FAILURE_THRESHOLD = 3  # Same error 3+ times
    POISON_MESSAGE_TIME_WINDOW_HOURS = 24

    def __init__(self, db: AsyncSession):
        self.db = db
        self.audit = AuditService(db)

    async def check_and_quarantine(
        self,
        task: PipelineTask,
        reason: str = "max_retries_exceeded",
    ) -> Optional[DeadLetterTask]:
        """
        Check if a failed task should be quarantined to DLQ.
        
        Args:
            task: Failed pipeline task
            reason: Reason for quarantine
            
        Returns:
            DeadLetterTask if quarantined, None otherwise
        """
        # Only quarantine failed tasks
        if task.status != PipelineTaskStatus.FAILED.value:
            return None

        # Check if already in DLQ
        existing_stmt = select(DeadLetterTask).where(
            DeadLetterTask.original_task_id == task.id
        )
        existing_result = await self.db.execute(existing_stmt)
        if existing_result.scalar_one_or_none():
            return None  # Already quarantined

        # Detect error pattern for poison messages
        error_pattern = None
        if task.last_error:
            # Simple pattern: first 100 chars of error
            error_pattern = task.last_error[:100]

        # Calculate review priority based on SLA and task type
        review_priority = 5
        if task.sla_breached:
            review_priority += 3
        if task.task_type in ["CONSOLIDATION", "CRITIQUE"]:
            review_priority += 2
        review_priority = min(review_priority, 10)

        # Create DLQ entry
        dlq_task = DeadLetterTask(
            organization_id=task.organization_id,
            original_task_id=task.id,
            task_type=task.task_type,
            failure_reason=reason,
            total_attempts=task.attempts,
            last_error=task.last_error,
            error_pattern=error_pattern,
            task_payload={
                "task_type": task.task_type,
                "priority": task.priority,
                "sla_category": task.sla_category,
                "metadata": task.task_metadata,
            },
            metadata={
                "sla_breached": task.sla_breached,
                "blocked_by_quota": task.blocked_by_quota,
                "estimated_tokens": task.estimated_tokens,
            },
            quarantined_at=datetime.now(timezone.utc),
            review_priority=review_priority,
        )

        self.db.add(dlq_task)
        
        # Flush to generate the ID
        await self.db.flush()

        # Audit the quarantine
        await self.audit.log_event(
            event_type="dlq.task_quarantined",
            organization_id=str(task.organization_id),
            resource_type="dead_letter_task",
            resource_id=str(dlq_task.id),
            success=True,
            details={
                "original_task_id": str(task.id),
                "task_type": task.task_type,
                "reason": reason,
                "attempts": task.attempts,
                "review_priority": review_priority,
            },
        )

        await self.db.flush()
        return dlq_task

    async def detect_poison_messages(
        self,
        organization_id: UUID,
    ) -> List[DeadLetterTask]:
        """
        Detect poison messages: tasks with same error pattern failing repeatedly.
        
        Returns list of newly quarantined poison messages.
        """
        # Find failed tasks with similar errors
        # This is a simplified version - in production, use better pattern matching
        
        # Get recent failed tasks grouped by error pattern
        stmt = (
            select(
                func.substring(PipelineTask.last_error, 1, 100).label("error_pattern"),
                func.count().label("failure_count"),
                func.array_agg(PipelineTask.id).label("task_ids"),
            )
            .where(
                and_(
                    PipelineTask.organization_id == organization_id,
                    PipelineTask.status == PipelineTaskStatus.FAILED.value,
                    PipelineTask.last_error.isnot(None),
                )
            )
            .group_by("error_pattern")
            .having(func.count() >= self.POISON_MESSAGE_FAILURE_THRESHOLD)
        )

        result = await self.db.execute(stmt)
        patterns = result.all()

        quarantined = []
        for pattern, count, task_ids in patterns:
            # Quarantine these tasks as poison messages
            for task_id in task_ids:
                task_stmt = select(PipelineTask).where(PipelineTask.id == task_id)
                task_result = await self.db.execute(task_stmt)
                task = task_result.scalar_one_or_none()
                
                if task:
                    dlq_task = await self.check_and_quarantine(
                        task,
                        reason="poison_message"
                    )
                    if dlq_task:
                        quarantined.append(dlq_task)

        return quarantined

    async def requeue_task(
        self,
        dlq_id: UUID,
        user_id: UUID,
        notes: Optional[str] = None,
    ) -> PipelineTask:
        """
        Requeue a task from DLQ back to pipeline.
        
        Creates a new pipeline task with same payload and marks DLQ entry as resolved.
        """
        from app.services.sla_scheduler_service import SLASchedulerService

        # Get DLQ task
        stmt = select(DeadLetterTask).where(DeadLetterTask.id == dlq_id)
        result = await self.db.execute(stmt)
        dlq_task = result.scalar_one_or_none()

        if not dlq_task:
            raise ValueError(f"DLQ task {dlq_id} not found")

        if dlq_task.is_resolved:
            raise ValueError("Task already resolved")

        # Create new pipeline task
        scheduler = SLASchedulerService(self.db)
        payload = dlq_task.task_payload or {}
        
        new_task = await scheduler.enqueue_pipeline_task(
            organization_id=str(dlq_task.organization_id),
            task_type=payload.get("task_type", dlq_task.task_type),
            priority=payload.get("priority", 5),
            sla_category=payload.get("sla_category"),
            metadata={
                **payload.get("metadata", {}),
                "requeued_from_dlq": str(dlq_id),
                "original_task_id": str(dlq_task.original_task_id),
            },
            scopes={"pipeline.enqueue"},
        )

        # Mark DLQ task as resolved
        dlq_task.is_resolved = True
        dlq_task.resolution = "requeued"
        dlq_task.resolution_notes = notes
        dlq_task.reviewed_at = datetime.now(timezone.utc)
        dlq_task.reviewed_by = user_id
        dlq_task.updated_at = datetime.now(timezone.utc)

        # Audit
        await self.audit.log_event(
            event_type="dlq.task_requeued",
            organization_id=str(dlq_task.organization_id),
            actor_id=str(user_id),
            resource_type="dead_letter_task",
            resource_id=str(dlq_id),
            success=True,
            details={
                "new_task_id": str(new_task.id),
                "original_task_id": str(dlq_task.original_task_id),
                "notes": notes,
            },
        )

        await self.db.flush()
        return new_task

    async def discard_task(
        self,
        dlq_id: UUID,
        user_id: UUID,
        notes: Optional[str] = None,
    ) -> DeadLetterTask:
        """
        Permanently discard a DLQ task without requeuing.
        """
        stmt = select(DeadLetterTask).where(DeadLetterTask.id == dlq_id)
        result = await self.db.execute(stmt)
        dlq_task = result.scalar_one_or_none()

        if not dlq_task:
            raise ValueError(f"DLQ task {dlq_id} not found")

        dlq_task.is_resolved = True
        dlq_task.resolution = "discarded"
        dlq_task.resolution_notes = notes
        dlq_task.reviewed_at = datetime.now(timezone.utc)
        dlq_task.reviewed_by = user_id
        dlq_task.updated_at = datetime.now(timezone.utc)

        # Audit
        await self.audit.log_event(
            event_type="dlq.task_discarded",
            organization_id=str(dlq_task.organization_id),
            actor_id=str(user_id),
            resource_type="dead_letter_task",
            resource_id=str(dlq_id),
            success=True,
            details={
                "original_task_id": str(dlq_task.original_task_id),
                "reason": dlq_task.failure_reason,
                "notes": notes,
            },
        )

        await self.db.flush()
        return dlq_task

    async def get_pending_review(
        self,
        organization_id: UUID,
        limit: int = 100,
    ) -> List[DeadLetterTask]:
        """Get unresolved DLQ tasks ordered by review priority."""
        stmt = (
            select(DeadLetterTask)
            .where(
                and_(
                    DeadLetterTask.organization_id == organization_id,
                    DeadLetterTask.is_resolved == False,
                )
            )
            .order_by(
                desc(DeadLetterTask.review_priority),
                DeadLetterTask.quarantined_at.asc(),
            )
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return list(result.scalars().all())
