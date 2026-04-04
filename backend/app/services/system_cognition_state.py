"""SystemCognitionState service — upsert + read per-org cognitive state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.system_cognition_state import SystemCognitionState


@dataclass
class CognitionStateUpdate:
    current_focus: str
    active_session_ids: list[str]
    unresolved_anomalies_count: int
    stale_goals_count: int
    cognitive_load: float
    next_scheduled_action: str | None


class SystemCognitionStateService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def upsert(self, org_id: str, update: CognitionStateUpdate) -> SystemCognitionState:
        """INSERT ... ON CONFLICT (organization_id) DO UPDATE SET ...

        Creates a new row on first heartbeat for an org; updates in-place
        on every subsequent heartbeat.
        """
        from app.models.base import generate_uuid, utc_now

        now = datetime.now(timezone.utc)
        values = {
            "id": generate_uuid(),
            "organization_id": org_id,
            "current_focus": update.current_focus,
            "active_session_ids": update.active_session_ids,
            "unresolved_anomalies_count": update.unresolved_anomalies_count,
            "stale_goals_count": update.stale_goals_count,
            "cognitive_load": update.cognitive_load,
            "last_heartbeat_at": now,
            "next_scheduled_action": update.next_scheduled_action,
            "created_at": now,
            "updated_at": now,
        }
        stmt = (
            pg_insert(SystemCognitionState)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["organization_id"],
                set_={
                    "current_focus": update.current_focus,
                    "active_session_ids": update.active_session_ids,
                    "unresolved_anomalies_count": update.unresolved_anomalies_count,
                    "stale_goals_count": update.stale_goals_count,
                    "cognitive_load": update.cognitive_load,
                    "last_heartbeat_at": now,
                    "next_scheduled_action": update.next_scheduled_action,
                    "updated_at": now,
                },
            )
            .returning(SystemCognitionState)
        )
        result = await self._db.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            # Fallback: fetch after upsert (driver may not return on conflict)
            row = await self.get(org_id)
        return row  # type: ignore[return-value]

    async def get(self, org_id: str) -> SystemCognitionState | None:
        result = await self._db.execute(
            select(SystemCognitionState).where(
                SystemCognitionState.organization_id == org_id
            )
        )
        return result.scalar_one_or_none()
