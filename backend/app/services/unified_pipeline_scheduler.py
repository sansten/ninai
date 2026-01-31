"""
Unified Pipeline Scheduler Service - Phase 3

Central queue for consolidation, critique, and evaluation tasks with:
- SLA enforcement (latency, throughput)
- Rate limiting per organization
- Per-tenant concurrency caps
- Backpressure management
- Starvation prevention (fairness)
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, func
import logging

from app.models.agent_process import AgentProcess, ProcessStatus
from app.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class PipelineTaskType(str, Enum):
    """Types of pipeline tasks."""
    CONSOLIDATION = "consolidation"  # Merge duplicate knowledge
    CRITIQUE = "critique"  # Review and validate
    EVALUATION = "evaluation"  # Score and rank


class PipelineSLA:
    """SLA configuration for pipeline tasks."""

    def __init__(
        self,
        task_type: PipelineTaskType,
        max_latency_seconds: int = 300,  # 5 minutes
        target_throughput_per_minute: int = 10,
        max_retry_attempts: int = 3,
        priority: int = 5  # 1-10, lower is higher priority
    ):
        self.task_type = task_type
        self.max_latency_seconds = max_latency_seconds
        self.target_throughput_per_minute = target_throughput_per_minute
        self.max_retry_attempts = max_retry_attempts
        self.priority = priority

    def is_sla_violated(self, task: AgentProcess) -> bool:
        """Check if task's SLA is violated."""
        if task.started_at is None:
            return False

        elapsed = (datetime.utcnow() - task.started_at).total_seconds()
        return elapsed > self.max_latency_seconds


class UnifiedPipelineScheduler:
    """
    Central scheduler for all pipeline tasks (consolidation, critique, eval).
    
    Features:
    - SLA enforcement: Track and alert on violations
    - Rate limiting: Per-org request limit
    - Concurrency caps: Max concurrent tasks per org
    - Backpressure: Queue overflow handling
    - Fairness: Prevent org starvation
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        # Default SLAs per task type
        self.slas = {
            PipelineTaskType.CONSOLIDATION: PipelineSLA(
                PipelineTaskType.CONSOLIDATION,
                max_latency_seconds=600,  # 10 minutes
                target_throughput_per_minute=5
            ),
            PipelineTaskType.CRITIQUE: PipelineSLA(
                PipelineTaskType.CRITIQUE,
                max_latency_seconds=1800,  # 30 minutes (human review)
                target_throughput_per_minute=2
            ),
            PipelineTaskType.EVALUATION: PipelineSLA(
                PipelineTaskType.EVALUATION,
                max_latency_seconds=300,  # 5 minutes
                target_throughput_per_minute=10
            )
        }

    async def enqueue_pipeline_task(
        self,
        organization_id: uuid.UUID,
        task_type: PipelineTaskType,
        task_name: str,
        input_data: Dict[str, Any],
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        priority: Optional[int] = None
    ) -> AgentProcess:
        """
        Enqueue a pipeline task.
        
        Raises:
            ValueError: If queue is full (backpressure)
        """
        # Check rate limit
        if not await self._check_rate_limit(organization_id, task_type):
            raise ValueError(f"Rate limit exceeded for {task_type.value}")

        # Check queue depth (backpressure)
        queue_depth = await self._get_queue_depth(organization_id)
        if queue_depth > 1000:  # Hardcoded limit
            raise ValueError("Pipeline queue overflow - backpressure applied")

        # Use SLA priority if not specified
        if priority is None:
            priority = self.slas[task_type].priority

        # Create process
        process = AgentProcess(
            id=uuid.uuid4(),
            organization_id=organization_id,
            agent_name=f"pipeline_{task_type.value}",
            status=ProcessStatus.queued,
            priority=priority,
            created_by_user_id=user_id,
            session_id=session_id,
            metadata={
                "task_type": task_type.value,
                "task_name": task_name,
                "input_data": input_data,
                "sla_deadline": (datetime.utcnow() + timedelta(
                    seconds=self.slas[task_type].max_latency_seconds
                )).isoformat()
            }
        )

        self.db.add(process)
        await self.db.flush()

        # Audit
        audit_svc = AuditService(self.db)
        await audit_svc.log_event(
            event_type=f"pipeline.{task_type.value}.enqueued",
            actor_id=str(user_id) if user_id else None,
            organization_id=str(organization_id),
            resource_type="pipeline_task",
            resource_id=str(process.id),
            success=True,
            details={"task_name": task_name, "priority": priority}
        )

        logger.info(
            f"Enqueued pipeline task: org={organization_id} "
            f"type={task_type.value} priority={priority} task_id={process.id}"
        )

        return process

    async def dequeue_next_task(self) -> Optional[AgentProcess]:
        """
        Dequeue next task respecting fairness and concurrency limits.
        
        Algorithm:
        1. Find org with lowest current load (fairness)
        2. Check concurrency cap for that org
        3. Return highest priority queued task from that org
        """
        # Find org with lowest running task count
        org_loads = await self._get_org_loads()

        for org_id in sorted(org_loads.keys(), key=lambda x: org_loads[x]):
            # Check if org has hit concurrency cap
            running_count = org_loads[org_id]
            if running_count >= await self._get_concurrency_cap(org_id):
                continue

            # Get highest priority queued task from this org
            stmt = select(AgentProcess).where(
                and_(
                    AgentProcess.organization_id == org_id,
                    AgentProcess.status == ProcessStatus.queued
                )
            ).order_by(
                AgentProcess.priority.asc(),  # Lower priority value = higher priority
                AgentProcess.created_at.asc()  # FIFO for same priority
            ).limit(1)

            result = await self.db.execute(stmt)
            task = result.scalar_one_or_none()

            if task:
                task.status = ProcessStatus.running
                task.started_at = datetime.utcnow()
                await self.db.flush()

                logger.info(f"Dequeued pipeline task: {task.id} (org={task.organization_id})")
                return task

        return None

    async def mark_task_succeeded(
        self,
        task_id: uuid.UUID,
        output_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark pipeline task as succeeded."""
        stmt = select(AgentProcess).where(AgentProcess.id == task_id)
        result = await self.db.execute(stmt)
        task = result.scalar_one_or_none()

        if not task:
            raise ValueError("Task not found")

        task.status = ProcessStatus.succeeded
        task.completed_at = datetime.utcnow()
        if output_data:
            task.metadata["output_data"] = output_data

        # Check SLA
        sla = self.slas.get(PipelineTaskType(task.metadata["task_type"]))
        sla_violated = sla and sla.is_sla_violated(task)

        if sla_violated:
            logger.warning(
                f"SLA VIOLATED: task={task.id} "
                f"elapsed={(datetime.utcnow() - task.started_at).total_seconds()}s "
                f"limit={sla.max_latency_seconds}s"
            )

        await self.db.flush()

    async def mark_task_failed(
        self,
        task_id: uuid.UUID,
        error_message: str
    ) -> None:
        """Mark pipeline task as failed."""
        stmt = select(AgentProcess).where(AgentProcess.id == task_id)
        result = await self.db.execute(stmt)
        task = result.scalar_one_or_none()

        if not task:
            raise ValueError("Task not found")

        # Check retry count
        max_retries = self.slas[
            PipelineTaskType(task.metadata["task_type"])
        ].max_retry_attempts

        if (task.attempt_count or 0) < max_retries:
            # Retry
            task.status = ProcessStatus.queued
            task.attempt_count = (task.attempt_count or 0) + 1
            task.metadata["last_error"] = error_message
            logger.info(
                f"Task failed, retrying: {task.id} "
                f"(attempt {task.attempt_count}/{max_retries})"
            )
        else:
            # Give up
            task.status = ProcessStatus.failed
            task.completed_at = datetime.utcnow()
            task.metadata["final_error"] = error_message
            logger.error(f"Task failed permanently: {task.id} - {error_message}")

        await self.db.flush()

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    async def _check_rate_limit(
        self,
        organization_id: uuid.UUID,
        task_type: PipelineTaskType
    ) -> bool:
        """Check if org is under rate limit for this task type."""
        # Count requests in last minute
        one_minute_ago = datetime.utcnow() - timedelta(minutes=1)

        stmt = select(func.count(AgentProcess.id)).where(
            and_(
                AgentProcess.organization_id == organization_id,
                AgentProcess.created_at >= one_minute_ago
            )
        )
        result = await self.db.execute(stmt)
        count = result.scalar() or 0

        sla = self.slas[task_type]
        return count < sla.target_throughput_per_minute

    async def _get_queue_depth(self, organization_id: uuid.UUID) -> int:
        """Get number of queued tasks for org."""
        stmt = select(func.count(AgentProcess.id)).where(
            and_(
                AgentProcess.organization_id == organization_id,
                AgentProcess.status == ProcessStatus.queued
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar() or 0

    async def _get_org_loads(self) -> Dict[uuid.UUID, int]:
        """Get current running task count per org."""
        stmt = select(
            AgentProcess.organization_id,
            func.count(AgentProcess.id).label("count")
        ).where(
            AgentProcess.status == ProcessStatus.running
        ).group_by(
            AgentProcess.organization_id
        )
        result = await self.db.execute(stmt)
        rows = result.all()

        return {row[0]: row[1] for row in rows}

    async def _get_concurrency_cap(self, organization_id: uuid.UUID) -> int:
        """Get max concurrent tasks allowed for org."""
        # In production, look up org settings
        # For now, default to 5
        return 5
