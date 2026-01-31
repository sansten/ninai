"""Tests for SLA-based pipeline scheduler service.

Tests cover:
- SLA ordering (breached deadlines first, then by remaining time)
- Fair resource allocation (backpressure handling)
- Starvation prevention (priority ordering)
- Quota enforcement and blocking
- Retry logic with attempt limits
- Queue statistics and monitoring
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pipeline_task import (
    PipelineTask,
    PipelineTaskStatus,
    PipelineTaskType,
)
from app.services.sla_scheduler_service import SLASchedulerService


@pytest.fixture
def test_org_id():
    return str(uuid4())


@pytest.fixture
def test_user_id():
    return str(uuid4())


@pytest.mark.asyncio
async def test_enqueue_pipeline_task(db_session: AsyncSession, test_org_id, test_user_id):
    """Test enqueueing a new pipeline task."""
    scheduler = SLASchedulerService(db_session)
    db_session.info["auto_commit"] = False

    now = datetime.now(timezone.utc)
    deadline = now + timedelta(minutes=30)

    task = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="consolidation",
        input_session_id="session123",
        target_resource_id="mem456",
        sla_deadline=deadline,
        sla_category="normal",
        priority=1,
        estimated_tokens=100,
        estimated_latency_ms=5000,
        task_metadata={"source": "user_feedback"},
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    assert task.id is not None
    assert task.organization_id == test_org_id
    assert task.task_type == "consolidation"
    assert task.status == PipelineTaskStatus.QUEUED.value
    assert task.priority == 1
    assert task.sla_deadline == deadline
    assert task.attempts == 0
    assert task.estimated_tokens == 100


@pytest.mark.asyncio
async def test_enqueue_permission_denied(db_session: AsyncSession, test_org_id, test_user_id):
    """Test that enqueue requires proper scope."""
    scheduler = SLASchedulerService(db_session)

    deadline = datetime.now(timezone.utc) + timedelta(minutes=30)

    with pytest.raises(PermissionError, match="pipeline.enqueue"):
        await scheduler.enqueue_pipeline_task(
            organization_id=test_org_id,
            task_type="consolidation",
            input_session_id="session123",
            target_resource_id="mem456",
            sla_deadline=deadline,
            actor_user_id=test_user_id,
            scopes={"other.scope"},
        )


@pytest.mark.asyncio
async def test_sla_ordering_breached_first(db_session: AsyncSession, test_org_id, test_user_id):
    """Test that SLA-breached tasks are dequeued first."""
    scheduler = SLASchedulerService(db_session)
    db_session.info["auto_commit"] = False

    now = datetime.now(timezone.utc)

    # Create three tasks: one breached, one high priority but not breached, one low priority
    breached_deadline = now - timedelta(minutes=5)  # Already past deadline
    normal_deadline = now + timedelta(minutes=30)

    # Enqueue normal task first
    normal_task = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="critique",
        input_session_id="session1",
        target_resource_id="mem1",
        sla_deadline=normal_deadline,
        priority=0,
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    # Enqueue breached task
    breached_task = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="consolidation",
        input_session_id="session2",
        target_resource_id="mem2",
        sla_deadline=breached_deadline,
        priority=0,
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    # Dequeue - should get breached task first
    dequeued = await scheduler.dequeue_next_by_sla(
        organization_id=test_org_id,
        scopes={"pipeline.dequeue"},
    )

    assert dequeued.id == breached_task.id
    assert dequeued.status == PipelineTaskStatus.RUNNING.value
    assert dequeued.attempts == 1


@pytest.mark.asyncio
async def test_sla_ordering_by_remaining_time(db_session: AsyncSession, test_org_id, test_user_id):
    """Test ordering by remaining SLA time when none are breached."""
    scheduler = SLASchedulerService(db_session)
    db_session.info["auto_commit"] = False

    now = datetime.now(timezone.utc)

    # Create tasks with different deadlines (all future)
    deadline_soon = now + timedelta(minutes=5)
    deadline_later = now + timedelta(minutes=60)

    # Enqueue later deadline first
    later_task = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="evaluation",
        input_session_id="session1",
        target_resource_id="mem1",
        sla_deadline=deadline_later,
        priority=10,
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    # Enqueue sooner deadline
    soon_task = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="evaluation",
        input_session_id="session2",
        target_resource_id="mem2",
        sla_deadline=deadline_soon,
        priority=1,  # Lower priority, but closer deadline
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    # Should dequeue the one with sooner deadline
    dequeued = await scheduler.dequeue_next_by_sla(
        organization_id=test_org_id,
        scopes={"pipeline.dequeue"},
    )

    assert dequeued.id == soon_task.id


@pytest.mark.asyncio
async def test_sla_ordering_by_priority(db_session: AsyncSession, test_org_id, test_user_id):
    """Test ordering by priority when SLAs and deadlines are equal."""
    scheduler = SLASchedulerService(db_session)
    db_session.info["auto_commit"] = False

    now = datetime.now(timezone.utc)
    same_deadline = now + timedelta(minutes=30)

    # Enqueue low priority task first
    low_prio = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="feedback_loop",
        input_session_id="session1",
        target_resource_id="mem1",
        sla_deadline=same_deadline,
        priority=1,
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    # Enqueue high priority task
    high_prio = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="feedback_loop",
        input_session_id="session2",
        target_resource_id="mem2",
        sla_deadline=same_deadline,
        priority=10,
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    # Should dequeue high priority first
    dequeued = await scheduler.dequeue_next_by_sla(
        organization_id=test_org_id,
        scopes={"pipeline.dequeue"},
    )

    assert dequeued.id == high_prio.id


@pytest.mark.asyncio
async def test_mark_succeeded(db_session: AsyncSession, test_org_id, test_user_id):
    """Test marking a task as succeeded."""
    scheduler = SLASchedulerService(db_session)
    db_session.info["auto_commit"] = False

    now = datetime.now(timezone.utc)
    task = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="consolidation",
        input_session_id="session123",
        target_resource_id="mem456",
        sla_deadline=now + timedelta(minutes=30),
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    # Mark as running first
    await scheduler.dequeue_next_by_sla(organization_id=test_org_id, scopes={"pipeline.dequeue"})

    # Mark as succeeded
    succeeded = await scheduler.mark_succeeded(
        task_id=task.id,
        actual_tokens=87,
        actual_latency_ms=4200,
        scopes={"pipeline.update"},
    )

    assert succeeded.status == PipelineTaskStatus.SUCCEEDED.value
    assert succeeded.actual_tokens == 87
    assert succeeded.actual_latency_ms == 4200
    assert succeeded.duration_ms is not None


@pytest.mark.asyncio
async def test_mark_failed_with_retries(db_session: AsyncSession, test_org_id, test_user_id):
    """Test that failed tasks are requeued if attempts < max_attempts."""
    scheduler = SLASchedulerService(db_session)
    db_session.info["auto_commit"] = False

    now = datetime.now(timezone.utc)
    task = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="consolidation",
        input_session_id="session123",
        target_resource_id="mem456",
        sla_deadline=now + timedelta(minutes=30),
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    # Dequeue (attempt 1)
    await scheduler.dequeue_next_by_sla(organization_id=test_org_id, scopes={"pipeline.dequeue"})

    # Mark as failed
    failed = await scheduler.mark_failed(
        task_id=task.id,
        error="Network timeout",
        scopes={"pipeline.update"},
    )

    # Should be requeued since attempts (1) < max_attempts (3)
    assert failed.status == PipelineTaskStatus.QUEUED.value
    assert failed.last_error == "Network timeout"


@pytest.mark.asyncio
async def test_mark_failed_exhausted_retries(db_session: AsyncSession, test_org_id, test_user_id):
    """Test that tasks are marked FAILED when max_attempts exceeded."""
    scheduler = SLASchedulerService(db_session)
    db_session.info["auto_commit"] = False

    now = datetime.now(timezone.utc)
    task = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="consolidation",
        input_session_id="session123",
        target_resource_id="mem456",
        sla_deadline=now + timedelta(minutes=30),
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    # Simulate 3 failed attempts
    for i in range(3):
        dequeued = await scheduler.dequeue_next_by_sla(organization_id=test_org_id, scopes={"pipeline.dequeue"})
        if not dequeued:
            break
        task = await db_session.get(PipelineTask, task.id)
        # Mark as failed - after 3 attempts, it should be marked FAILED, not requeued
        await scheduler.mark_failed(
            task_id=task.id,
            error=f"Attempt {i+1} failed",
            scopes={"pipeline.update"},
        )

    # Fetch updated task
    task = await db_session.get(PipelineTask, task.id)

    # Should be FAILED since attempts >= max_attempts
    assert task.status == PipelineTaskStatus.FAILED.value
    assert task.attempts == 3


@pytest.mark.asyncio
async def test_mark_blocked(db_session: AsyncSession, test_org_id, test_user_id):
    """Test blocking a task due to backpressure or quota."""
    scheduler = SLASchedulerService(db_session)
    db_session.info["auto_commit"] = False

    now = datetime.now(timezone.utc)
    task = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="consolidation",
        input_session_id="session123",
        target_resource_id="mem456",
        sla_deadline=now + timedelta(minutes=30),
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    blocked = await scheduler.mark_blocked(
        task_id=task.id,
        reason="quota",
        scopes={"pipeline.update"},
    )

    assert blocked.status == PipelineTaskStatus.BLOCKED.value
    assert blocked.blocked_by_quota is True


@pytest.mark.asyncio
async def test_blocked_tasks_skipped_in_dequeue(db_session: AsyncSession, test_org_id, test_user_id):
    """Test that blocked tasks are not dequeued."""
    scheduler = SLASchedulerService(db_session)
    db_session.info["auto_commit"] = False

    now = datetime.now(timezone.utc)

    # Create two tasks
    task1 = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="consolidation",
        input_session_id="session1",
        target_resource_id="mem1",
        sla_deadline=now + timedelta(minutes=30),
        priority=10,  # Higher priority
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    task2 = await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="consolidation",
        input_session_id="session2",
        target_resource_id="mem2",
        sla_deadline=now + timedelta(minutes=30),
        priority=1,
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    # Block task1
    await scheduler.mark_blocked(task_id=task1.id, reason="quota", scopes={"pipeline.update"})

    # Dequeue should get task2 (not blocked)
    dequeued = await scheduler.dequeue_next_by_sla(
        organization_id=test_org_id,
        scopes={"pipeline.dequeue"},
    )

    assert dequeued.id == task2.id


@pytest.mark.asyncio
async def test_queue_stats(db_session: AsyncSession, test_org_id, test_user_id):
    """Test getting queue statistics."""
    scheduler = SLASchedulerService(db_session)
    db_session.info["auto_commit"] = False

    now = datetime.now(timezone.utc)

    # Create tasks with different statuses
    await scheduler.enqueue_pipeline_task(
        organization_id=test_org_id,
        task_type="consolidation",
        input_session_id="session1",
        target_resource_id="mem1",
        sla_deadline=now + timedelta(minutes=30),
        actor_user_id=test_user_id,
        scopes={"pipeline.enqueue"},
    )

    stats = await scheduler.get_queue_stats(organization_id=test_org_id)

    assert stats["status_counts"]["queued"] == 1
    assert stats["status_counts"]["running"] == 0
    assert "timestamp" in stats
