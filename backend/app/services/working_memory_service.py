"""Working memory manager service (Phase 61)."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.working_memory_item import WorkingMemoryItem


class WorkingMemoryService:
    CAPACITY = 20
    DECAY_FACTOR = 0.85

    def eviction_score(self, item: WorkingMemoryItem, now: datetime) -> float:
        """Return activation-weighted recency score for eviction ranking."""
        last = item.last_accessed_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        seconds_since = max(0.0, (current - last).total_seconds())
        recency_score = math.exp(-seconds_since / 300.0)
        return float(item.activation) * recency_score

    async def push(
        self,
        *,
        db: AsyncSession,
        session_id: str,
        org_id: str,
        item: dict[str, Any],
    ) -> WorkingMemoryItem:
        now = datetime.now(timezone.utc)

        existing = list(
            (
                await db.execute(
                    select(WorkingMemoryItem).where(
                        WorkingMemoryItem.session_id == session_id
                    )
                )
            )
            .scalars()
            .all()
        )

        if len(existing) >= self.CAPACITY:
            evict = min(existing, key=lambda row: self.eviction_score(row, now))
            await db.delete(evict)
            await db.flush()

        content_snapshot = str(item.get("content_snapshot") or item.get("content") or "")[:512]
        row = WorkingMemoryItem(
            session_id=session_id,
            org_id=org_id,
            memory_id=item.get("memory_id"),
            content_snapshot=content_snapshot,
            item_type=str(item.get("item_type") or "memory"),
            activation=float(item.get("activation") or 1.0),
            inserted_at=now,
            last_accessed_at=now,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row

    async def access(
        self,
        *,
        db: AsyncSession,
        session_id: str,
        item_id: str,
    ) -> WorkingMemoryItem | None:
        stmt = select(WorkingMemoryItem).where(
            WorkingMemoryItem.session_id == session_id,
            WorkingMemoryItem.id == item_id,
        )
        row = (await db.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None

        row.last_accessed_at = datetime.now(timezone.utc)
        row.activation = min(1.0, float(row.activation) + 0.1)
        await db.commit()
        await db.refresh(row)
        return row

    async def tick_decay(
        self,
        *,
        db: AsyncSession,
        session_id: str,
    ) -> int:
        rows = list(
            (
                await db.execute(
                    select(WorkingMemoryItem).where(
                        WorkingMemoryItem.session_id == session_id
                    )
                )
            )
            .scalars()
            .all()
        )

        removed = 0
        for row in rows:
            row.activation = float(row.activation) * self.DECAY_FACTOR
            if row.activation < 0.05:
                await db.delete(row)
                removed += 1

        await db.commit()
        return removed

    async def snapshot(
        self,
        *,
        db: AsyncSession,
        session_id: str,
    ) -> list[WorkingMemoryItem]:
        stmt = (
            select(WorkingMemoryItem)
            .where(WorkingMemoryItem.session_id == session_id)
            .order_by(WorkingMemoryItem.activation.desc())
        )
        return list((await db.execute(stmt)).scalars().all())

    async def flush(
        self,
        *,
        db: AsyncSession,
        session_id: str,
    ) -> int:
        rows = await self.snapshot(db=db, session_id=session_id)
        count = len(rows)
        if count == 0:
            return 0
        await db.execute(
            delete(WorkingMemoryItem).where(WorkingMemoryItem.session_id == session_id)
        )
        await db.commit()
        return count
