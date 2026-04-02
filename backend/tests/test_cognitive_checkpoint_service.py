from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cognitive_session import CognitiveSession
from app.models.cognitive_state_checkpoint import CognitiveStateCheckpoint
from app.models.organization import Organization
from app.services.cognitive_checkpoint_service import CognitiveCheckpointService


def _service() -> CognitiveCheckpointService:
    return CognitiveCheckpointService()


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
            goal="checkpoint test",
            context_snapshot={},
            trace_id=None,
        )
    )
    await db.flush()


@pytest.mark.asyncio
class TestSave:
    async def test_save_creates_seq_one_for_new_session(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        row = await svc.save(
            db=db_session,
            session_id=sid,
            org_id=test_org_id,
            loop_iteration=1,
            active_goal="goal",
            completed_steps=[0],
            pending_steps=[1, 2],
            working_memory=[{"id": "m1"}],
            last_output="out",
        )
        assert row.checkpoint_seq == 1

    async def test_save_increments_sequence(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        first = await svc.save(
            db=db_session,
            session_id=sid,
            org_id=test_org_id,
            loop_iteration=1,
            active_goal="g",
            completed_steps=[],
            pending_steps=[0],
            working_memory=[],
            last_output="a",
        )
        second = await svc.save(
            db=db_session,
            session_id=sid,
            org_id=test_org_id,
            loop_iteration=2,
            active_goal="g",
            completed_steps=[0],
            pending_steps=[1],
            working_memory=[],
            last_output="b",
        )
        assert first.checkpoint_seq == 1
        assert second.checkpoint_seq == 2

    async def test_save_persists_working_memory_list(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        row = await svc.save(
            db=db_session,
            session_id=sid,
            org_id=test_org_id,
            loop_iteration=1,
            active_goal="g",
            completed_steps=[],
            pending_steps=[],
            working_memory=[{"item": "x"}, {"item": "y"}],
            last_output="ok",
        )
        assert isinstance(row.working_memory_snapshot, list)
        assert len(row.working_memory_snapshot) == 2

    async def test_save_status_defaults_active(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        row = await svc.save(
            db=db_session,
            session_id=sid,
            org_id=test_org_id,
            loop_iteration=0,
            active_goal="g",
            completed_steps=[],
            pending_steps=[],
            working_memory=[],
            last_output="",
        )
        assert row.status == "active"

    async def test_save_stringifies_goal_and_output(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        row = await svc.save(
            db=db_session,
            session_id=sid,
            org_id=test_org_id,
            loop_iteration=0,
            active_goal=123,
            completed_steps=[],
            pending_steps=[],
            working_memory=[],
            last_output=456,
        )
        assert row.active_goal == "123"
        assert row.last_output == "456"

    async def test_save_copies_input_lists(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        completed = [0]
        pending = [1]
        memory = [{"m": 1}]

        row = await svc.save(
            db=db_session,
            session_id=sid,
            org_id=test_org_id,
            loop_iteration=1,
            active_goal="g",
            completed_steps=completed,
            pending_steps=pending,
            working_memory=memory,
            last_output="x",
        )

        completed.append(99)
        pending.append(88)
        memory.append({"m": 2})

        assert row.completed_step_indices == [0]
        assert row.pending_step_indices == [1]
        assert row.working_memory_snapshot == [{"m": 1}]


@pytest.mark.asyncio
class TestLatest:
    async def test_latest_returns_highest_sequence(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=1, active_goal="g", completed_steps=[], pending_steps=[1], working_memory=[], last_output="a")
        expected = await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=2, active_goal="g", completed_steps=[1], pending_steps=[2], working_memory=[], last_output="b")

        latest = await svc.latest(db=db_session, session_id=sid)
        assert latest is not None
        assert latest.id == expected.id

    async def test_latest_returns_none_when_missing(self, db_session: AsyncSession):
        svc = _service()
        assert await svc.latest(db=db_session, session_id=str(uuid4())) is None

    async def test_latest_is_session_scoped(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid1 = str(uuid4())
        sid2 = str(uuid4())
        await _seed_session(db_session, session_id=sid1, org_id=test_org_id, user_id=test_user_id)
        await _seed_session(db_session, session_id=sid2, org_id=test_org_id, user_id=test_user_id)

        await svc.save(db=db_session, session_id=sid1, org_id=test_org_id, loop_iteration=1, active_goal="a", completed_steps=[], pending_steps=[], working_memory=[], last_output="a")
        row2 = await svc.save(db=db_session, session_id=sid2, org_id=test_org_id, loop_iteration=1, active_goal="b", completed_steps=[], pending_steps=[], working_memory=[], last_output="b")

        latest_sid2 = await svc.latest(db=db_session, session_id=sid2)
        assert latest_sid2 is not None
        assert latest_sid2.id == row2.id


@pytest.mark.asyncio
class TestRestore:
    async def test_restore_returns_correct_state_dict(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        await svc.save(
            db=db_session,
            session_id=sid,
            org_id=test_org_id,
            loop_iteration=7,
            active_goal="ship release",
            completed_steps=[0, 1],
            pending_steps=[2, 3],
            working_memory=[{"key": "v"}],
            last_output="ok",
        )
        state = await svc.restore(db=db_session, session_id=sid)

        assert state is not None
        assert state["loop_iteration"] == 7
        assert state["active_goal"] == "ship release"
        assert state["completed_steps"] == [0, 1]
        assert state["pending_steps"] == [2, 3]
        assert state["working_memory"] == [{"key": "v"}]
        assert state["last_output"] == "ok"

    async def test_restore_marks_status_restored(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        row = await svc.save(
            db=db_session,
            session_id=sid,
            org_id=test_org_id,
            loop_iteration=1,
            active_goal="g",
            completed_steps=[],
            pending_steps=[1],
            working_memory=[],
            last_output="x",
        )

        await svc.restore(db=db_session, session_id=sid)
        refreshed = await db_session.get(CognitiveStateCheckpoint, row.id)
        assert refreshed is not None
        assert refreshed.status == "restored"

    async def test_restore_returns_none_when_no_checkpoint(self, db_session: AsyncSession):
        svc = _service()
        assert await svc.restore(db=db_session, session_id=str(uuid4())) is None

    async def test_restore_uses_latest_checkpoint(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=1, active_goal="a", completed_steps=[], pending_steps=[1], working_memory=[], last_output="a")
        await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=2, active_goal="b", completed_steps=[1], pending_steps=[2], working_memory=[], last_output="b")

        state = await svc.restore(db=db_session, session_id=sid)
        assert state is not None
        assert state["loop_iteration"] == 2
        assert state["active_goal"] == "b"

    async def test_restore_required_keys_present(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=1, active_goal="g", completed_steps=[], pending_steps=[], working_memory=[], last_output="")

        state = await svc.restore(db=db_session, session_id=sid)
        assert state is not None
        assert set(state) == {
            "loop_iteration",
            "active_goal",
            "completed_steps",
            "pending_steps",
            "working_memory",
            "last_output",
        }


@pytest.mark.asyncio
class TestMarkCompleted:
    async def test_mark_completed_sets_status_completed(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=1, active_goal="g", completed_steps=[], pending_steps=[1], working_memory=[], last_output="a")
        changed = await svc.mark_completed(db=db_session, session_id=sid)

        rows = list((await db_session.execute(select(CognitiveStateCheckpoint).where(CognitiveStateCheckpoint.session_id == sid))).scalars().all())
        assert changed is True
        assert rows
        assert all(r.status == "completed" for r in rows)

    async def test_mark_completed_returns_false_when_missing(self, db_session: AsyncSession):
        svc = _service()
        assert await svc.mark_completed(db=db_session, session_id=str(uuid4())) is False

    async def test_mark_completed_updates_multiple_rows(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=1, active_goal="g", completed_steps=[], pending_steps=[1], working_memory=[], last_output="a")
        await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=2, active_goal="g", completed_steps=[1], pending_steps=[2], working_memory=[], last_output="b")
        assert await svc.mark_completed(db=db_session, session_id=sid) is True

        rows = list((await db_session.execute(select(CognitiveStateCheckpoint).where(CognitiveStateCheckpoint.session_id == sid))).scalars().all())
        assert len(rows) == 2
        assert all(r.status == "completed" for r in rows)

    async def test_mark_completed_does_not_touch_other_sessions(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid1 = str(uuid4())
        sid2 = str(uuid4())
        await _seed_session(db_session, session_id=sid1, org_id=test_org_id, user_id=test_user_id)
        await _seed_session(db_session, session_id=sid2, org_id=test_org_id, user_id=test_user_id)

        row1 = await svc.save(db=db_session, session_id=sid1, org_id=test_org_id, loop_iteration=1, active_goal="a", completed_steps=[], pending_steps=[], working_memory=[], last_output="")
        row2 = await svc.save(db=db_session, session_id=sid2, org_id=test_org_id, loop_iteration=1, active_goal="b", completed_steps=[], pending_steps=[], working_memory=[], last_output="")

        assert await svc.mark_completed(db=db_session, session_id=sid1) is True
        keep_active = await db_session.get(CognitiveStateCheckpoint, row2.id)
        updated = await db_session.get(CognitiveStateCheckpoint, row1.id)
        assert updated is not None and updated.status == "completed"
        assert keep_active is not None and keep_active.status == "active"


@pytest.mark.asyncio
class TestCleanupOld:
    async def test_cleanup_old_deletes_beyond_keep_sessions(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sessions = [str(uuid4()) for _ in range(3)]
        for sid in sessions:
            await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        rows: list[CognitiveStateCheckpoint] = []
        for i, sid in enumerate(sessions):
            row = await svc.save(
                db=db_session,
                session_id=sid,
                org_id=test_org_id,
                loop_iteration=i,
                active_goal=f"g{i}",
                completed_steps=[],
                pending_steps=[i],
                working_memory=[],
                last_output="x",
            )
            rows.append(row)

        rows[0].created_at = datetime.now(timezone.utc) - timedelta(days=3)
        rows[1].created_at = datetime.now(timezone.utc) - timedelta(days=2)
        rows[2].created_at = datetime.now(timezone.utc) - timedelta(days=1)
        await db_session.flush()

        deleted = await svc.cleanup_old(db=db_session, org_id=test_org_id, keep_sessions=1)
        assert deleted == 2

        remaining = list((await db_session.execute(select(CognitiveStateCheckpoint).where(CognitiveStateCheckpoint.org_id == test_org_id))).scalars().all())
        assert len(remaining) == 1
        assert remaining[0].session_id == sessions[2]

    async def test_cleanup_old_keep_zero_deletes_all(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=1, active_goal="g", completed_steps=[], pending_steps=[1], working_memory=[], last_output="a")

        deleted = await svc.cleanup_old(db=db_session, org_id=test_org_id, keep_sessions=0)
        assert deleted == 1

    async def test_cleanup_old_returns_zero_when_under_limit(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=1, active_goal="g", completed_steps=[], pending_steps=[1], working_memory=[], last_output="a")

        deleted = await svc.cleanup_old(db=db_session, org_id=test_org_id, keep_sessions=10)
        assert deleted == 0

    async def test_cleanup_old_scoped_by_org(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        other_org = str(uuid4())
        db_session.add(Organization(id=other_org, name="Other Org", slug=f"other-{other_org[:8]}", is_active=True))
        await db_session.flush()

        sid1 = str(uuid4())
        sid2 = str(uuid4())
        await _seed_session(db_session, session_id=sid1, org_id=test_org_id, user_id=test_user_id)
        await _seed_session(db_session, session_id=sid2, org_id=other_org, user_id=test_user_id)

        row1 = await svc.save(db=db_session, session_id=sid1, org_id=test_org_id, loop_iteration=1, active_goal="a", completed_steps=[], pending_steps=[], working_memory=[], last_output="")
        row2 = await svc.save(db=db_session, session_id=sid2, org_id=other_org, loop_iteration=1, active_goal="b", completed_steps=[], pending_steps=[], working_memory=[], last_output="")
        row1.created_at = datetime.now(timezone.utc) - timedelta(days=2)
        row2.created_at = datetime.now(timezone.utc)
        await db_session.flush()

        deleted = await svc.cleanup_old(db=db_session, org_id=test_org_id, keep_sessions=0)
        assert deleted == 1
        other = await db_session.get(CognitiveStateCheckpoint, row2.id)
        assert other is not None

    async def test_cleanup_old_empty_returns_zero(self, db_session: AsyncSession, test_org_id: str):
        svc = _service()
        deleted = await svc.cleanup_old(db=db_session, org_id=test_org_id, keep_sessions=1)
        assert deleted == 0

    async def test_cleanup_old_negative_keep_treated_as_zero(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)
        await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=1, active_goal="g", completed_steps=[], pending_steps=[], working_memory=[], last_output="")

        deleted = await svc.cleanup_old(db=db_session, org_id=test_org_id, keep_sessions=-5)
        assert deleted == 1


class TestDiffSteps:
    def test_diff_steps_returns_remaining(self):
        svc = _service()
        assert svc.diff_steps(completed=[0, 1], full_plan=[0, 1, 2]) == [2]

    def test_diff_steps_all_completed_empty(self):
        svc = _service()
        assert svc.diff_steps(completed=[0, 1, 2], full_plan=[0, 1, 2]) == []

    def test_diff_steps_empty_completed_returns_full(self):
        svc = _service()
        assert svc.diff_steps(completed=[], full_plan=[3, 4]) == [3, 4]

    def test_diff_steps_preserves_plan_order(self):
        svc = _service()
        assert svc.diff_steps(completed=[2], full_plan=[4, 2, 1, 3]) == [4, 1, 3]

    def test_diff_steps_handles_duplicates(self):
        svc = _service()
        assert svc.diff_steps(completed=[1, 1], full_plan=[1, 2, 2, 3]) == [2, 2, 3]

    def test_diff_steps_casts_ints(self):
        svc = _service()
        assert svc.diff_steps(completed=["1"], full_plan=["1", "2"]) == [2]

    def test_diff_steps_empty_plan(self):
        svc = _service()
        assert svc.diff_steps(completed=[1], full_plan=[]) == []


@pytest.mark.asyncio
class TestServiceSanity:
    async def test_save_row_count_increases(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        before = list((await db_session.execute(select(CognitiveStateCheckpoint).where(CognitiveStateCheckpoint.org_id == test_org_id))).scalars().all())
        await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=1, active_goal="g", completed_steps=[], pending_steps=[1], working_memory=[], last_output="a")
        after = list((await db_session.execute(select(CognitiveStateCheckpoint).where(CognitiveStateCheckpoint.org_id == test_org_id))).scalars().all())
        assert len(after) == len(before) + 1

    async def test_latest_tie_break_by_sequence(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        svc = _service()
        sid = str(uuid4())
        await _seed_session(db_session, session_id=sid, org_id=test_org_id, user_id=test_user_id)

        a = await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=1, active_goal="a", completed_steps=[], pending_steps=[1], working_memory=[], last_output="a")
        b = await svc.save(db=db_session, session_id=sid, org_id=test_org_id, loop_iteration=2, active_goal="b", completed_steps=[1], pending_steps=[2], working_memory=[], last_output="b")
        a.created_at = datetime.now(timezone.utc)
        b.created_at = datetime.now(timezone.utc)
        await db_session.flush()

        latest = await svc.latest(db=db_session, session_id=sid)
        assert latest is not None
        assert latest.checkpoint_seq == 2
