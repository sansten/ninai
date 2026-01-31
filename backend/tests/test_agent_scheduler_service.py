import pytest

from app.services.agent_scheduler_service import AgentSchedulerService


@pytest.mark.asyncio
async def test_scheduler_orders_by_priority(db_session, test_org_id):
    svc = AgentSchedulerService(db_session, max_running_per_org=1)

    low = await svc.enqueue(
        organization_id=test_org_id,
        agent_name="agent-low",
        priority=1,
        scopes={"scheduler.enqueue"},
    )
    high = await svc.enqueue(
        organization_id=test_org_id,
        agent_name="agent-high",
        priority=5,
        scopes={"scheduler.enqueue"},
    )

    picked = await svc.dequeue_next(org_id=test_org_id, scopes={"scheduler.dequeue"})

    assert picked is not None
    assert picked.id == high.id
    assert picked.status == "running"


@pytest.mark.asyncio
async def test_scheduler_respects_running_cap(db_session, test_org_id):
    svc = AgentSchedulerService(db_session, max_running_per_org=1)

    p1 = await svc.enqueue(
        organization_id=test_org_id,
        agent_name="agent-a",
        priority=1,
        scopes={"scheduler.enqueue"},
    )
    p2 = await svc.enqueue(
        organization_id=test_org_id,
        agent_name="agent-b",
        priority=1,
        scopes={"scheduler.enqueue"},
    )

    first = await svc.dequeue_next(org_id=test_org_id, scopes={"scheduler.dequeue"})
    assert first is not None

    second = await svc.dequeue_next(org_id=test_org_id, scopes={"scheduler.dequeue"})
    assert second is None

    await svc.mark_succeeded(process_id=p1.id, scopes={"scheduler.update"})

    third = await svc.dequeue_next(org_id=test_org_id, scopes={"scheduler.dequeue"})
    assert third is not None
    assert third.id == p2.id


@pytest.mark.asyncio
async def test_scheduler_attempt_limits_and_requeue(db_session, test_org_id):
    svc = AgentSchedulerService(db_session, max_running_per_org=2)

    proc = await svc.enqueue(
        organization_id=test_org_id,
        agent_name="agent-attempts",
        priority=1,
        max_attempts=2,
        scopes={"scheduler.enqueue"},
    )

    first = await svc.dequeue_next(org_id=test_org_id, scopes={"scheduler.dequeue"})
    assert first is not None
    assert first.id == proc.id

    # Simulate backpressure and requeue the same process.
    await svc.reset_to_queue(process_id=proc.id, reason="backpressure", scopes={"scheduler.update"})

    second = await svc.dequeue_next(org_id=test_org_id, scopes={"scheduler.dequeue"})
    assert second is not None
    assert second.id == proc.id

    # Exceed attempt budget; further dequeues should skip it.
    await svc.mark_failed(process_id=proc.id, reason="exceeded", scopes={"scheduler.update"})

    none_left = await svc.dequeue_next(org_id=test_org_id, scopes={"scheduler.dequeue"})
    assert none_left is None
