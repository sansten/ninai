"""DB persistence helpers for CognitiveLoop.

Separates SQLAlchemy persistence from orchestration logic to enable unit testing
without a live database.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cognitive_iteration import CognitiveIteration
from app.models.cognitive_session import CognitiveSession


class CognitiveLoopRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_session(self, session_id: str) -> CognitiveSession | None:
        return await self.session.get(CognitiveSession, session_id)

    async def save_session_status(self, sess: CognitiveSession, status: str) -> None:
        sess.status = status
        await self.session.flush()

    async def get_iteration(self, *, session_id: str, iteration_num: int) -> CognitiveIteration | None:
        stmt = select(CognitiveIteration).where(
            CognitiveIteration.session_id == session_id,
            CognitiveIteration.iteration_num == iteration_num,
        )
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def create_iteration(self, *, session_id: str, iteration_num: int) -> CognitiveIteration:
        now = datetime.now(timezone.utc)
        row = CognitiveIteration(
            session_id=session_id,
            iteration_num=iteration_num,
            plan_json={},
            execution_json={},
            critique_json={},
            evaluation="retry",
            started_at=now,
            finished_at=now,
            metrics={},
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def finalize_iteration(
        self,
        *,
        iteration: CognitiveIteration,
        plan_json: dict,
        execution_json: dict,
        critique_json: dict,
        evaluation: str,
        metrics: dict,
        finished_at: datetime | None = None,
    ) -> None:
        iteration.plan_json = plan_json or {}
        iteration.execution_json = execution_json or {}
        iteration.critique_json = critique_json or {}
        iteration.evaluation = evaluation
        iteration.finished_at = finished_at or datetime.now(timezone.utc)
        iteration.metrics = metrics or {}
        await self.session.flush()
