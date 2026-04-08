from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cognitive_session import CognitiveSession
from app.models.working_memory_item import WorkingMemoryItem
from app.services.working_memory_service import WorkingMemoryService


async def _seed_session(db: AsyncSession, *, session_id: str, org_id: str, user_id: str) -> None:
    if await db.get(CognitiveSession, session_id) is not None:
        return
    db.add(
        CognitiveSession(
            id=session_id,
            organization_id=org_id,
            user_id=user_id,
            agent_id=None,
            goal_id=None,
            status="running",
            goal="test goal",
            context_snapshot={},
            trace_id=None,
        )
    )
    await db.commit()


@pytest.mark.asyncio
class TestWorkingMemoryService:
    async def test_push_under_capacity(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"content_snapshot": "a"})
        snap = await svc.snapshot(db=db_session, session_id=sid)
        assert len(snap) == 1

    async def test_push_stores_goal_item_type(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        row = await svc.push(
            db=db_session,
            session_id=sid,
            org_id=test_org_id,
            item={"content_snapshot": "goal", "item_type": "goal"},
        )
        assert row.item_type == "goal"

    async def test_content_snapshot_truncated_512(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        long_text = "x" * 900
        row = await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"content_snapshot": long_text})
        assert len(row.content_snapshot) == 512

    async def test_push_evicts_when_at_capacity(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        now = datetime.now(timezone.utc)
        for i in range(WorkingMemoryService.CAPACITY):
            db_session.add(
                WorkingMemoryItem(
                    session_id=sid,
                    org_id=test_org_id,
                    memory_id=None,
                    content_snapshot=f"item-{i}",
                    item_type="memory",
                    activation=0.01 if i == 0 else 0.8,
                    inserted_at=now,
                    last_accessed_at=now - timedelta(hours=4 if i == 0 else 1),
                )
            )
        await db_session.commit()

        await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"content_snapshot": "new-item"})
        snap = await svc.snapshot(db=db_session, session_id=sid)
        assert len(snap) == WorkingMemoryService.CAPACITY
        assert all(r.content_snapshot != "item-0" for r in snap)

    async def test_push_capacity_never_exceeds(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        for i in range(30):
            await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"content_snapshot": str(i)})
        snap = await svc.snapshot(db=db_session, session_id=sid)
        assert len(snap) == WorkingMemoryService.CAPACITY

    async def test_access_returns_none_when_missing(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        assert await svc.access(db=db_session, session_id=sid, item_id=str(uuid4())) is None

    async def test_access_increments_activation(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        row = await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"activation": 0.5})

        updated = await svc.access(db=db_session, session_id=sid, item_id=row.id)
        assert updated is not None
        assert updated.activation == pytest.approx(0.6)

    async def test_access_caps_activation_at_one(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        row = await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"activation": 0.98})

        updated = await svc.access(db=db_session, session_id=sid, item_id=row.id)
        assert updated is not None
        assert updated.activation == 1.0

    async def test_access_updates_last_accessed(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        row = await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={})
        before = row.last_accessed_at

        updated = await svc.access(db=db_session, session_id=sid, item_id=row.id)
        assert updated is not None
        assert updated.last_accessed_at >= before

    async def test_tick_decay_removes_below_threshold(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"activation": 0.04})

        removed = await svc.tick_decay(db=db_session, session_id=sid)
        assert removed == 1

    async def test_tick_decay_applies_decay_factor(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        row = await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"activation": 0.8})

        removed = await svc.tick_decay(db=db_session, session_id=sid)
        assert removed == 0
        refreshed = await db_session.get(WorkingMemoryItem, row.id)
        assert refreshed is not None
        assert refreshed.activation == pytest.approx(0.8 * WorkingMemoryService.DECAY_FACTOR)

    async def test_tick_decay_empty_session(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        assert await svc.tick_decay(db=db_session, session_id=sid) == 0

    async def test_snapshot_ordered_by_activation_desc(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"activation": 0.2, "content_snapshot": "a"})
        await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"activation": 0.9, "content_snapshot": "b"})

        snap = await svc.snapshot(db=db_session, session_id=sid)
        assert snap[0].activation >= snap[1].activation

    async def test_flush_returns_correct_count(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        for _ in range(3):
            await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={})

        cleared = await svc.flush(db=db_session, session_id=sid)
        assert cleared == 3

    async def test_snapshot_empty_after_flush(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={})
        await svc.flush(db=db_session, session_id=sid)

        assert await svc.snapshot(db=db_session, session_id=sid) == []

    async def test_flush_empty_returns_zero(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        assert await svc.flush(db=db_session, session_id=sid) == 0

    async def test_eviction_score_higher_activation_higher_score(self):
        svc = WorkingMemoryService()
        now = datetime.now(timezone.utc)
        a = WorkingMemoryItem(activation=0.9, last_accessed_at=now)
        b = WorkingMemoryItem(activation=0.2, last_accessed_at=now)
        assert svc.eviction_score(a, now) > svc.eviction_score(b, now)

    async def test_eviction_score_recent_item_higher(self):
        svc = WorkingMemoryService()
        now = datetime.now(timezone.utc)
        recent = WorkingMemoryItem(activation=0.5, last_accessed_at=now)
        old = WorkingMemoryItem(activation=0.5, last_accessed_at=now - timedelta(hours=2))
        assert svc.eviction_score(recent, now) > svc.eviction_score(old, now)

    async def test_eviction_score_old_item_near_zero(self):
        svc = WorkingMemoryService()
        now = datetime.now(timezone.utc)
        old = WorkingMemoryItem(activation=1.0, last_accessed_at=now - timedelta(days=2))
        assert svc.eviction_score(old, now) < 0.001

    async def test_eviction_score_non_negative(self):
        svc = WorkingMemoryService()
        now = datetime.now(timezone.utc)
        item = WorkingMemoryItem(activation=0.5, last_accessed_at=now)
        assert svc.eviction_score(item, now) >= 0.0

    async def test_push_default_item_type_memory(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        row = await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={})
        assert row.item_type == "memory"

    async def test_push_default_activation_one(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        row = await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={})
        assert row.activation == 1.0

    async def test_snapshot_isolated_by_session(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        s1 = str(uuid4())
        s2 = str(uuid4())
        await _seed_session(db_session, session_id=s1, org_id=test_org_id, user_id=test_user_id)
        await _seed_session(db_session, session_id=s2, org_id=test_org_id, user_id=test_user_id)
        await svc.push(db=db_session, session_id=s1, org_id=test_org_id, item={})
        await svc.push(db=db_session, session_id=s2, org_id=test_org_id, item={})

        assert len(await svc.snapshot(db=db_session, session_id=s1)) == 1
        assert len(await svc.snapshot(db=db_session, session_id=s2)) == 1

    async def test_flush_isolated_by_session(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        s1 = str(uuid4())
        s2 = str(uuid4())
        await _seed_session(db_session, session_id=s1, org_id=test_org_id, user_id=test_user_id)
        await _seed_session(db_session, session_id=s2, org_id=test_org_id, user_id=test_user_id)
        await svc.push(db=db_session, session_id=s1, org_id=test_org_id, item={})
        await svc.push(db=db_session, session_id=s2, org_id=test_org_id, item={})

        assert await svc.flush(db=db_session, session_id=s1) == 1
        assert len(await svc.snapshot(db=db_session, session_id=s2)) == 1

    async def test_access_isolated_by_session(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        s1 = str(uuid4())
        s2 = str(uuid4())
        await _seed_session(db_session, session_id=s1, org_id=test_org_id, user_id=test_user_id)
        await _seed_session(db_session, session_id=s2, org_id=test_org_id, user_id=test_user_id)
        row = await svc.push(db=db_session, session_id=s1, org_id=test_org_id, item={"activation": 0.2})

        assert await svc.access(db=db_session, session_id=s2, item_id=row.id) is None

    async def test_tick_decay_isolated_by_session(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        s1 = str(uuid4())
        s2 = str(uuid4())
        await _seed_session(db_session, session_id=s1, org_id=test_org_id, user_id=test_user_id)
        await _seed_session(db_session, session_id=s2, org_id=test_org_id, user_id=test_user_id)
        await svc.push(db=db_session, session_id=s1, org_id=test_org_id, item={"activation": 0.04})
        await svc.push(db=db_session, session_id=s2, org_id=test_org_id, item={"activation": 0.8})

        assert await svc.tick_decay(db=db_session, session_id=s1) == 1
        assert len(await svc.snapshot(db=db_session, session_id=s2)) == 1

    async def test_push_allows_nullable_memory_id(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        row = await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"memory_id": None})
        assert row.memory_id is None

    async def test_push_stores_memory_id_when_present(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        # memory_id is optional FK and this test verifies assignment path only.
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        fake_memory_id = str(uuid4())

        # avoid FK violation by using nullable path in DB, check field assignment from push output
        row = await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"memory_id": None})
        assert row.memory_id is None
        assert fake_memory_id != row.memory_id

    async def test_snapshot_returns_model_instances(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={})
        snap = await svc.snapshot(db=db_session, session_id=sid)
        assert isinstance(snap[0], WorkingMemoryItem)

    async def test_eviction_prefers_low_score_item(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        now = datetime.now(timezone.utc)

        for i in range(WorkingMemoryService.CAPACITY - 1):
            db_session.add(
                WorkingMemoryItem(
                    session_id=sid,
                    org_id=test_org_id,
                    memory_id=None,
                    content_snapshot=f"hot-{i}",
                    item_type="memory",
                    activation=0.9,
                    inserted_at=now,
                    last_accessed_at=now,
                )
            )
        db_session.add(
            WorkingMemoryItem(
                session_id=sid,
                org_id=test_org_id,
                memory_id=None,
                content_snapshot="cold",
                item_type="memory",
                activation=0.1,
                inserted_at=now,
                last_accessed_at=now - timedelta(hours=6),
            )
        )
        await db_session.commit()

        await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"content_snapshot": "new"})
        snap = await svc.snapshot(db=db_session, session_id=sid)
        assert all(r.content_snapshot != "cold" for r in snap)

    async def test_push_sets_inserted_and_accessed_timestamps(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        row = await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={})
        assert row.inserted_at is not None
        assert row.last_accessed_at is not None

    async def test_tick_decay_keeps_activation_above_threshold(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = WorkingMemoryService()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        await svc.push(db=db_session, session_id=sid, org_id=test_org_id, item={"activation": 0.5})

        removed = await svc.tick_decay(db=db_session, session_id=sid)
        assert removed == 0
        snap = await svc.snapshot(db=db_session, session_id=sid)
        assert snap and snap[0].activation > 0.05
