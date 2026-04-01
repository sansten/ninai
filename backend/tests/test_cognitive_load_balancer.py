from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.load_snapshot import LoadSnapshot
from app.models.organization import Organization
from app.services.cognitive_load_balancer import CognitiveLoadBalancer


def _balancer() -> CognitiveLoadBalancer:
    return CognitiveLoadBalancer()


def test_classify_empty_dict_low():
    assert _balancer().classify_load(queue_depths={}) == "low"


def test_classify_zero_total_low():
    assert _balancer().classify_load(queue_depths={"q.agent_topics": 0}) == "low"


def test_classify_low_boundary_50_is_low():
    assert _balancer().classify_load(queue_depths={"q.agent_topics": 50}) == "low"


def test_classify_51_is_medium():
    assert _balancer().classify_load(queue_depths={"q.agent_topics": 51}) == "medium"


def test_classify_200_is_medium():
    assert _balancer().classify_load(queue_depths={"q.agent_topics": 200}) == "medium"


def test_classify_201_is_high():
    assert _balancer().classify_load(queue_depths={"q.agent_topics": 201}) == "high"


def test_classify_500_is_high():
    assert _balancer().classify_load(queue_depths={"q.agent_topics": 500}) == "high"


def test_classify_501_is_critical():
    assert _balancer().classify_load(queue_depths={"q.agent_topics": 501}) == "critical"


def test_classify_sums_multiple_queues():
    assert _balancer().classify_load(queue_depths={"a": 120, "b": 90}) == "high"


def test_classify_negative_depths_clamped_to_zero():
    assert _balancer().classify_load(queue_depths={"a": -100, "b": 20}) == "low"


def test_should_skip_critical_topics_true():
    assert _balancer().should_skip(task_name="q.agent_topics", load_level="critical") is True


def test_should_skip_critical_patterns_true():
    assert _balancer().should_skip(task_name="q.agent_patterns", load_level="critical") is True


def test_should_skip_critical_graph_true():
    assert _balancer().should_skip(task_name="q.agent_graph", load_level="critical") is True


def test_should_skip_critical_memory_ingest_false():
    assert _balancer().should_skip(task_name="q.memory_ingest", load_level="critical") is False


def test_should_skip_critical_cognitive_loop_false():
    assert _balancer().should_skip(task_name="q.cognitive_loop", load_level="critical") is False


def test_should_skip_high_topics_true():
    assert _balancer().should_skip(task_name="q.agent_topics", load_level="high") is True


def test_should_skip_high_patterns_true():
    assert _balancer().should_skip(task_name="q.agent_patterns", load_level="high") is True


def test_should_skip_high_graph_true():
    assert _balancer().should_skip(task_name="q.agent_graph", load_level="high") is True


def test_should_skip_high_memory_ingest_false():
    assert _balancer().should_skip(task_name="q.memory_ingest", load_level="high") is False


def test_should_skip_medium_topics_true():
    assert _balancer().should_skip(task_name="q.agent_topics", load_level="medium") is True


def test_should_skip_medium_patterns_false():
    assert _balancer().should_skip(task_name="q.agent_patterns", load_level="medium") is False


def test_should_skip_low_topics_false():
    assert _balancer().should_skip(task_name="q.agent_topics", load_level="low") is False


def test_should_skip_unknown_level_defaults_false():
    assert _balancer().should_skip(task_name="q.agent_topics", load_level="weird") is False


def test_throttle_factor_low():
    assert _balancer().throttle_factor(load_level="low") == 0.0


def test_throttle_factor_medium():
    assert _balancer().throttle_factor(load_level="medium") == 0.5


def test_throttle_factor_high():
    assert _balancer().throttle_factor(load_level="high") == 2.0


def test_throttle_factor_critical():
    assert _balancer().throttle_factor(load_level="critical") == 5.0


def test_throttle_factor_unknown_defaults_zero():
    assert _balancer().throttle_factor(load_level="mystery") == 0.0


@pytest.mark.asyncio
async def test_record_snapshot_writes_correct_load_level(db_session: AsyncSession, test_org_id: str):
    balancer = _balancer()
    snapshot = await balancer.record_snapshot(
        db=db_session,
        org_id=test_org_id,
        queue_depths={"q.agent_topics": 501},
        active_workers=7,
    )

    assert snapshot.load_level == "critical"


@pytest.mark.asyncio
async def test_record_snapshot_persists_fields(db_session: AsyncSession, test_org_id: str):
    balancer = _balancer()
    snapshot = await balancer.record_snapshot(
        db=db_session,
        org_id=test_org_id,
        queue_depths={"q.agent_graph": 42},
        active_workers=3,
    )

    row = await db_session.get(LoadSnapshot, snapshot.id)
    assert row is not None
    assert row.org_id == test_org_id
    assert row.queue_depths == {"q.agent_graph": 42}
    assert row.active_workers == 3
    assert row.sampled_at is not None


@pytest.mark.asyncio
async def test_record_snapshot_clamps_negative_values(db_session: AsyncSession, test_org_id: str):
    balancer = _balancer()
    snapshot = await balancer.record_snapshot(
        db=db_session,
        org_id=test_org_id,
        queue_depths={"q.agent_topics": -5},
        active_workers=-2,
    )

    assert snapshot.queue_depths == {"q.agent_topics": 0}
    assert snapshot.active_workers == 0
    assert snapshot.load_level == "low"


@pytest.mark.asyncio
async def test_get_recent_load_returns_desc_order(db_session: AsyncSession, test_org_id: str):
    now = datetime.now(timezone.utc)
    older = LoadSnapshot(
        id=str(uuid4()),
        org_id=test_org_id,
        queue_depths={"q.agent_topics": 10},
        active_workers=1,
        load_level="low",
        sampled_at=now - timedelta(minutes=5),
    )
    newer = LoadSnapshot(
        id=str(uuid4()),
        org_id=test_org_id,
        queue_depths={"q.agent_topics": 300},
        active_workers=2,
        load_level="high",
        sampled_at=now,
    )
    db_session.add_all([older, newer])
    await db_session.commit()

    rows = await _balancer().get_recent_load(db=db_session, org_id=test_org_id, limit=10)
    assert len(rows) >= 2
    assert rows[0].sampled_at >= rows[1].sampled_at


@pytest.mark.asyncio
async def test_get_recent_load_respects_limit(db_session: AsyncSession, test_org_id: str):
    now = datetime.now(timezone.utc)
    for i in range(3):
        db_session.add(
            LoadSnapshot(
                id=str(uuid4()),
                org_id=test_org_id,
                queue_depths={"q.agent_topics": i},
                active_workers=1,
                load_level="low",
                sampled_at=now + timedelta(seconds=i),
            )
        )
    await db_session.commit()

    rows = await _balancer().get_recent_load(db=db_session, org_id=test_org_id, limit=2)
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_get_recent_load_filters_by_org(db_session: AsyncSession, test_org_id: str):
    other_org_id = str(uuid4())

    # Insert a second organization because LoadSnapshot has org FK.
    db_session.add(
        Organization(
            id=other_org_id,
            name="Other Org",
            slug="other-org-load-balancer",
            is_active=True,
        )
    )
    await db_session.commit()

    db_session.add(
        LoadSnapshot(
            id=str(uuid4()),
            org_id=test_org_id,
            queue_depths={"q.agent_topics": 10},
            active_workers=1,
            load_level="low",
        )
    )
    db_session.add(
        LoadSnapshot(
            id=str(uuid4()),
            org_id=other_org_id,
            queue_depths={"q.agent_topics": 600},
            active_workers=10,
            load_level="critical",
        )
    )
    await db_session.commit()

    rows = await _balancer().get_recent_load(db=db_session, org_id=test_org_id, limit=10)
    assert rows
    assert all(row.org_id == test_org_id for row in rows)


@pytest.mark.asyncio
async def test_get_recent_load_limit_minimum_one(db_session: AsyncSession, test_org_id: str):
    balancer = _balancer()
    for i in range(2):
        await balancer.record_snapshot(
            db=db_session,
            org_id=test_org_id,
            queue_depths={"q.agent_topics": i + 1},
            active_workers=1,
        )

    rows = await balancer.get_recent_load(db=db_session, org_id=test_org_id, limit=0)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_record_snapshot_visible_in_recent_query(db_session: AsyncSession, test_org_id: str):
    balancer = _balancer()
    created = await balancer.record_snapshot(
        db=db_session,
        org_id=test_org_id,
        queue_depths={"q.agent_topics": 205},
        active_workers=5,
    )

    rows = await balancer.get_recent_load(db=db_session, org_id=test_org_id, limit=5)
    assert rows[0].id == created.id
    assert rows[0].load_level == "high"


@pytest.mark.asyncio
async def test_record_snapshot_row_count_increases(db_session: AsyncSession, test_org_id: str):
    before = (
        await db_session.execute(
            select(LoadSnapshot).where(LoadSnapshot.org_id == test_org_id)
        )
    ).scalars().all()
    before_count = len(before)

    await _balancer().record_snapshot(
        db=db_session,
        org_id=test_org_id,
        queue_depths={"q.agent_topics": 20},
        active_workers=2,
    )

    after = (
        await db_session.execute(
            select(LoadSnapshot).where(LoadSnapshot.org_id == test_org_id)
        )
    ).scalars().all()
    assert len(after) == before_count + 1
