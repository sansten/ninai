"""Checkpoint/restore service for cognitive loop resilience (Phase 80)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cognitive_state_checkpoint import CognitiveStateCheckpoint


class CognitiveCheckpointService:
    async def save(
        self,
        *,
        db: AsyncSession,
        session_id: str,
        org_id: str,
        loop_iteration: int,
        active_goal: str,
        completed_steps: list[int],
        pending_steps: list[int],
        working_memory: list[dict],
        last_output: str,
    ) -> CognitiveStateCheckpoint:
        next_seq = (
            (
                await db.execute(
                    select(func.max(CognitiveStateCheckpoint.checkpoint_seq)).where(
                        CognitiveStateCheckpoint.session_id == str(session_id)
                    )
                )
            ).scalar_one_or_none()
            or 0
        ) + 1

        row = CognitiveStateCheckpoint(
            session_id=str(session_id),
            org_id=str(org_id),
            checkpoint_seq=int(next_seq),
            loop_iteration=int(loop_iteration),
            active_goal=str(active_goal),
            completed_step_indices=list(completed_steps or []),
            pending_step_indices=list(pending_steps or []),
            working_memory_snapshot=list(working_memory or []),
            last_output=str(last_output),
            status="active",
            created_at=datetime.now(timezone.utc),
        )
        db.add(row)
        await db.flush()
        return row

    async def latest(self, *, db: AsyncSession, session_id: str) -> CognitiveStateCheckpoint | None:
        stmt = (
            select(CognitiveStateCheckpoint)
            .where(CognitiveStateCheckpoint.session_id == str(session_id))
            .order_by(CognitiveStateCheckpoint.checkpoint_seq.desc(), CognitiveStateCheckpoint.created_at.desc())
            .limit(1)
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    async def restore(self, *, db: AsyncSession, session_id: str) -> dict | None:
        row = await self.latest(db=db, session_id=session_id)
        if row is None:
            return None

        row.status = "restored"
        await db.flush()

        return {
            "loop_iteration": int(row.loop_iteration),
            "active_goal": str(row.active_goal),
            "completed_steps": list(row.completed_step_indices or []),
            "pending_steps": list(row.pending_step_indices or []),
            "working_memory": list(row.working_memory_snapshot or []),
            "last_output": str(row.last_output),
        }

    async def mark_completed(self, *, db: AsyncSession, session_id: str) -> bool:
        result = await db.execute(
            update(CognitiveStateCheckpoint)
            .where(CognitiveStateCheckpoint.session_id == str(session_id))
            .values(status="completed")
        )
        await db.flush()
        return bool(int(getattr(result, "rowcount", 0) or 0))

    async def cleanup_old(self, *, db: AsyncSession, org_id: str, keep_sessions: int = 100) -> int:
        keep = max(0, int(keep_sessions if keep_sessions is not None else 100))

        rows = list(
            (
                await db.execute(
                    select(CognitiveStateCheckpoint)
                    .where(CognitiveStateCheckpoint.org_id == str(org_id))
                    .order_by(CognitiveStateCheckpoint.created_at.desc(), CognitiveStateCheckpoint.checkpoint_seq.desc())
                )
            )
            .scalars()
            .all()
        )

        if not rows:
            return 0

        ordered_sessions: list[str] = []
        seen: set[str] = set()
        for row in rows:
            sid = str(row.session_id)
            if sid not in seen:
                seen.add(sid)
                ordered_sessions.append(sid)

        keep_set = set(ordered_sessions[:keep]) if keep > 0 else set()
        stale_sessions = {sid for sid in ordered_sessions if sid not in keep_set}
        if not stale_sessions:
            return 0

        result = await db.execute(
            delete(CognitiveStateCheckpoint).where(
                CognitiveStateCheckpoint.org_id == str(org_id),
                CognitiveStateCheckpoint.session_id.in_(stale_sessions),
            )
        )
        await db.flush()
        return int(getattr(result, "rowcount", 0) or 0)

    def diff_steps(self, *, completed: list[int], full_plan: list[int]) -> list[int]:
        done = {int(step) for step in (completed or [])}
        return [int(step) for step in (full_plan or []) if int(step) not in done]
