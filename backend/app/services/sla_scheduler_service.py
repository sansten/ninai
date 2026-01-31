"""SLA-based pipeline scheduler service.

Manages consolidation/critique/eval pipelines with SLA ordering, backpressure,
and fair resource allocation across tenants.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import and_, or_, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.pipeline_task import (
    PipelineTask,
    PipelineTaskStatus,
    PipelineTaskType,
)
from app.services.audit_service import AuditService
from app.core.task_execution import TaskExecutionContext


class SLASchedulerService:
    """Pipeline scheduler with SLA-based ordering and backpressure handling."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.audit = AuditService(db)

    async def enqueue_pipeline_task(
        self,
        *,
        organization_id: str,
        task_type: str,
        input_session_id: str,
        target_resource_id: str,
        sla_deadline: datetime,
        sla_category: str = "normal",
        priority: int = 0,
        estimated_tokens: int = 0,
        estimated_latency_ms: int = 0,
        task_metadata: dict | None = None,
        actor_user_id: str,
        scopes: set[str] | None = None,
    ) -> PipelineTask:
        """Enqueue a new pipeline task.

        Args:
            organization_id: Organization ID
            task_type: Type of pipeline task (consolidation, critique, evaluation, etc.)
            input_session_id: Source cognitive session ID
            target_resource_id: Target resource (memory/run/session ID)
            sla_deadline: When this task must complete
            sla_category: SLA category (critical, high, normal, low)
            priority: Priority for scheduling (higher = sooner)
            estimated_tokens: Estimated token cost
            estimated_latency_ms: Estimated latency
            task_metadata: Task-specific data
            actor_user_id: User enqueueing the task
            scopes: Capability scopes (must include "pipeline.enqueue")

        Returns:
            Created PipelineTask

        Raises:
            PermissionError: If scopes don't include "pipeline.enqueue"
        """
        if scopes and "pipeline.enqueue" not in scopes:
            raise PermissionError("Missing scope: pipeline.enqueue")

        task = PipelineTask(
            id=str(uuid4()),
            organization_id=organization_id,
            task_type=task_type,
            status=PipelineTaskStatus.QUEUED.value,
            input_session_id=input_session_id,
            target_resource_id=target_resource_id,
            sla_deadline=sla_deadline,
            sla_category=sla_category,
            priority=priority,
            estimated_tokens=estimated_tokens,
            estimated_latency_ms=estimated_latency_ms,
            task_metadata=task_metadata or {},
            attempts=0,
            max_attempts=3,
        )

        self.db.add(task)
        await self.audit.log_event(
            event_type="pipeline.task.enqueued",
            actor_id=actor_user_id,
            organization_id=organization_id,
            resource_type="pipeline_task",
            resource_id=task.id,
            success=True,
            details={
                "task_type": task_type,
                "sla_deadline": sla_deadline.isoformat(),
                "priority": priority,
                "estimated_tokens": estimated_tokens,
            },
        )

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

        return task

    async def dequeue_next_by_sla(
        self,
        *,
        organization_id: str,
        scopes: set[str] | None = None,
    ) -> PipelineTask | None:
        """Dequeue next task using SLA-based ordering.

        Ordering: SLA-breached tasks first, then by SLA remaining time (ascending),
        then by priority (descending).

        Args:
            organization_id: Organization ID
            scopes: Capability scopes (must include "pipeline.dequeue")

        Returns:
            Next PipelineTask or None if queue is empty

        Raises:
            PermissionError: If scopes don't include "pipeline.dequeue"
        """
        if scopes and "pipeline.dequeue" not in scopes:
            raise PermissionError("Missing scope: pipeline.dequeue")

        now = datetime.now(timezone.utc)

        # Fetch tasks: breached first, then by remaining time, then by priority
        stmt = (
            select(PipelineTask)
            .where(
                and_(
                    PipelineTask.organization_id == organization_id,
                    PipelineTask.status == PipelineTaskStatus.QUEUED.value,
                    PipelineTask.attempts < PipelineTask.max_attempts,
                    PipelineTask.blocked_by_quota == False,
                )
            )
            .order_by(
                # Breached SLAs first (sla_deadline < now = breached)
                (PipelineTask.sla_deadline < now).desc(),
                # Then by remaining time (ascending = sooner deadline first)
                PipelineTask.sla_deadline.asc(),
                # Then by priority (descending = higher priority first)
                PipelineTask.priority.desc(),
                # Tiebreaker: creation time
                PipelineTask.created_at.asc(),
            )
            .limit(1)
            .with_for_update(skip_locked=True)
        )

        result = await self.db.execute(stmt)
        task = result.scalar_one_or_none()

        if task:
            task.status = PipelineTaskStatus.RUNNING.value
            task.started_at = now
            task.attempts += 1
            await self.audit.log_event(
                event_type="pipeline.task.started",
                organization_id=organization_id,
                resource_type="pipeline_task",
                resource_id=task.id,
                success=True,
                details={
                    "attempt": task.attempts,
                    "sla_remaining_ms": task.sla_remaining_ms,
                },
            )
            if not self.db.info.get("auto_commit", True):
                await self.db.flush()

        return task

    async def mark_succeeded(
        self,
        *,
        task_id: str,
        actual_tokens: int | None = None,
        actual_latency_ms: int | None = None,
        scopes: set[str] | None = None,
    ) -> PipelineTask:
        """Mark task as succeeded.

        Args:
            task_id: Pipeline task ID
            actual_tokens: Actual token cost incurred
            actual_latency_ms: Actual latency in milliseconds
            scopes: Capability scopes (must include "pipeline.update")

        Returns:
            Updated PipelineTask

        Raises:
            PermissionError: If scopes don't include "pipeline.update"
        """
        if scopes and "pipeline.update" not in scopes:
            raise PermissionError("Missing scope: pipeline.update")

        task = await self.db.get(PipelineTask, task_id)
        if not task:
            raise ValueError(f"Pipeline task {task_id} not found")

        task.status = PipelineTaskStatus.SUCCEEDED.value
        task.finished_at = datetime.now(timezone.utc)
        task.actual_tokens = actual_tokens
        task.actual_latency_ms = actual_latency_ms
        if task.started_at and task.finished_at:
            task.duration_ms = int(
                (task.finished_at - task.started_at).total_seconds() * 1000
            )

        await self.audit.log_event(
            event_type="pipeline.task.succeeded",
            organization_id=task.organization_id,
            resource_type="pipeline_task",
            resource_id=task.id,
            success=True,
            details={
                "attempts": task.attempts,
                "duration_ms": task.duration_ms,
                "actual_tokens": actual_tokens,
            },
        )

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

        return task

    async def record_task_metrics(
        self,
        *,
        task_id: str,
        actual_tokens: int,
        model_latency_ms: float = 0.0,
        preprocessing_ms: float = 0.0,
        postprocessing_ms: float = 0.0,
        peak_memory_mb: float = 0.0,
        avg_memory_mb: float = 0.0,
        scopes: set[str] | None = None,
    ) -> None:
        """Record resource metrics for completed task.

        Args:
            task_id: Pipeline task ID
            actual_tokens: Actual token usage
            model_latency_ms: Time spent in model inference
            preprocessing_ms: Time spent in preprocessing
            postprocessing_ms: Time spent in postprocessing
            peak_memory_mb: Peak memory usage
            avg_memory_mb: Average memory usage
            scopes: Capability scopes (must include "pipeline.update")

        Raises:
            PermissionError: If scopes don't include "pipeline.update"
        """
        from app.core.resource_profiler import resource_profiler

        if scopes and "pipeline.update" not in scopes:
            raise PermissionError("Missing scope: pipeline.update")

        task = await self.db.get(PipelineTask, task_id)
        if not task:
            return  # Task not found, skip profiling

        # Record to resource profiler
        try:
            cost_per_1k = 0.0001  # Ollama local model cost estimate
            estimated_cost_usd = (actual_tokens / 1000.0) * cost_per_1k
            queued_ms = 0.0
            if task.created_at and task.started_at:
                queued_ms = (task.started_at - task.created_at).total_seconds() * 1000

            resource_profiler.record_metrics(
                task_id=task_id,
                organization_id=task.organization_id,
                task_type=task.task_type,
                queued_duration_ms=queued_ms,
                execution_duration_ms=float(task.duration_ms or 0),
                estimated_tokens=task.estimated_tokens,
                actual_tokens=actual_tokens,
                model_latency_ms=model_latency_ms,
                preprocessing_ms=preprocessing_ms,
                postprocessing_ms=postprocessing_ms,
                peak_memory_mb=peak_memory_mb,
                avg_memory_mb=avg_memory_mb,
                estimated_cost_usd=estimated_cost_usd,
                succeeded=task.status == PipelineTaskStatus.SUCCEEDED.value,
            )
        except Exception as e:
            # Don't fail task if profiling fails
            import logging
            logger = logging.getLogger(__name__)
            logger.exception(f"Failed to record metrics for task {task_id}: {e}")

    async def mark_failed(
        self,
        *,
        task_id: str,
        error: str,
        scopes: set[str] | None = None,
    ) -> PipelineTask:
        """Mark task as failed.

        Args:
            task_id: Pipeline task ID
            error: Error message/reason for failure
            scopes: Capability scopes (must include "pipeline.update")

        Returns:
            Updated PipelineTask

        Raises:
            PermissionError: If scopes don't include "pipeline.update"
        """
        from app.services.dead_letter_queue_service import DeadLetterQueueService
        
        if scopes and "pipeline.update" not in scopes:
            raise PermissionError("Missing scope: pipeline.update")

        task = await self.db.get(PipelineTask, task_id)
        if not task:
            raise ValueError(f"Pipeline task {task_id} not found")

        should_quarantine = task.attempts >= task.max_attempts
        
        if should_quarantine:
            task.status = PipelineTaskStatus.FAILED.value
            task.finished_at = datetime.now(timezone.utc)
        else:
            task.status = PipelineTaskStatus.QUEUED.value

        task.last_error = error
        if task.started_at:
            task.finished_at = datetime.now(timezone.utc)
            task.duration_ms = int(
                (task.finished_at - task.started_at).total_seconds() * 1000
            )

        await self.audit.log_event(
            event_type="pipeline.task.failed",
            organization_id=task.organization_id,
            resource_type="pipeline_task",
            resource_id=task.id,
            success=False,
            error_message=error,
            details={
                "attempts": task.attempts,
                "max_attempts": task.max_attempts,
                "will_retry": task.status == PipelineTaskStatus.QUEUED.value,
                "will_quarantine": should_quarantine,
            },
        )
        
        # Auto-quarantine to DLQ if max attempts exceeded
        if should_quarantine:
            dlq_service = DeadLetterQueueService(self.db)
            await dlq_service.check_and_quarantine(
                task=task,
                reason="max_retries_exceeded"
            )

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

        return task

    async def mark_blocked(
        self,
        *,
        task_id: str,
        reason: str = "backpressure",
        blocks_on_task_id: str | None = None,
        scopes: set[str] | None = None,
    ) -> PipelineTask:
        """Mark task as blocked (backpressure or dependency).

        Args:
            task_id: Pipeline task ID
            reason: Reason for blocking ("backpressure", "quota", "dependency", etc.)
            blocks_on_task_id: If blocked on another task, provide its ID
            scopes: Capability scopes (must include "pipeline.update")

        Returns:
            Updated PipelineTask

        Raises:
            PermissionError: If scopes don't include "pipeline.update"
        """
        if scopes and "pipeline.update" not in scopes:
            raise PermissionError("Missing scope: pipeline.update")

        task = await self.db.get(PipelineTask, task_id)
        if not task:
            raise ValueError(f"Pipeline task {task_id} not found")

        task.status = PipelineTaskStatus.BLOCKED.value
        task.blocks_on_task_id = blocks_on_task_id
        task.blocked_by_quota = reason == "quota"

        await self.audit.log_event(
            event_type="pipeline.task.blocked",
            organization_id=task.organization_id,
            resource_type="pipeline_task",
            resource_id=task.id,
            success=True,
            details={
                "reason": reason,
                "blocks_on_task_id": blocks_on_task_id,
            },
        )

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

        return task

    async def get_queue_stats(
        self,
        *,
        organization_id: str,
    ) -> dict:
        """Get queue statistics for an organization.

        Returns:
            Stats dict with counts per status and SLA breach info
        """
        now = datetime.now(timezone.utc)

        # Count tasks per status
        status_counts = {}
        for status in PipelineTaskStatus:
            stmt = select(func.count()).where(
                and_(
                    PipelineTask.organization_id == organization_id,
                    PipelineTask.status == status.value,
                )
            )
            count = await self.db.scalar(stmt)
            status_counts[status.value] = count or 0

        # Count breached SLAs
        breached_stmt = select(func.count()).where(
            and_(
                PipelineTask.organization_id == organization_id,
                PipelineTask.status == PipelineTaskStatus.QUEUED.value,
                PipelineTask.sla_deadline < now,
            )
        )
        breached_count = await self.db.scalar(breached_stmt)

        # Avg latency for completed tasks
        latency_stmt = select(func.avg(PipelineTask.actual_latency_ms)).where(
            and_(
                PipelineTask.organization_id == organization_id,
                PipelineTask.status == PipelineTaskStatus.SUCCEEDED.value,
            )
        )
        avg_latency = await self.db.scalar(latency_stmt)

        return {
            "status_counts": status_counts,
            "breached_sla_count": breached_count or 0,
            "avg_latency_ms": int(avg_latency) if avg_latency else None,
            "timestamp": now.isoformat(),
        }
