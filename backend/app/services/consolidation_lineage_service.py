"""Consolidation Lineage Service — Gap C.

MemoryConsolidationAgent recommends merges, and SemanticDistillationService
creates MemoryConsolidation records for distillation flows.  But when the
enrichment pipeline's merge recommendation is acted on (e.g. by the sleep
cycle or a manual merge), no MemoryConsolidation record is written — so you
cannot trace which source memories produced a given consolidated memory.

This service closes that gap by writing a MemoryConsolidation record whenever
a merge is committed, with the full list of source_memory_ids.

Design rules:
- Pure service: wraps DB insert; no business logic beyond record creation.
- Returns a LineageRecord dataclass so callers can log/test without touching DB.
- Does NOT raise on duplicate consolidation records (uses on_conflict_do_nothing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class LineageRecord:
    """Lightweight view of a created MemoryConsolidation row."""
    id: str
    consolidated_memory_id: str | None
    source_memory_ids: list[str]
    org_id: str
    created_by: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConsolidationLineageService:
    """Persist consolidation provenance for merged memories.

    Typical usage (after a merge is committed):
        svc = ConsolidationLineageService(session=db)
        record = await svc.record_merge(
            org_id="org-1",
            user_id="user-1",
            consolidated_memory_id="mem-new",
            source_memory_ids=["mem-a", "mem-b", "mem-c"],
            title="Weekly incident summary",
            created_by="memory_sleep_pipeline",
        )
    """

    def __init__(self, *, session: Any) -> None:
        self._session = session

    async def record_merge(
        self,
        *,
        org_id: str,
        user_id: str,
        consolidated_memory_id: str | None,
        source_memory_ids: list[str],
        title: str | None = None,
        summary: str | None = None,
        created_by: str = "agent",
    ) -> LineageRecord:
        """Write a MemoryConsolidation record for a completed merge.

        Parameters
        ----------
        org_id:                   Organisation that owns the memories.
        user_id:                  User context for the consolidation.
        consolidated_memory_id:   ID of the resulting merged memory (may be None
                                  if the merge produced no new memory, e.g. a
                                  deduplicate-into-primary operation).
        source_memory_ids:        IDs of the memories that were merged.
        title:                    Optional title for the consolidated memory.
        summary:                  Optional summary of what was merged.
        created_by:               Identifies which pipeline performed the merge
                                  (e.g. "memory_sleep_pipeline", "manual").
        """
        from app.models.memory_consolidation import MemoryConsolidation

        record = MemoryConsolidation(
            organization_id=org_id,
            user_id=user_id,
            consolidated_memory_id=consolidated_memory_id,
            source_memory_ids=list(source_memory_ids),
            title=title,
            summary=summary,
            created_by=created_by,
            status="completed",
        )
        self._session.add(record)
        await self._session.flush()

        return LineageRecord(
            id=str(record.id),
            consolidated_memory_id=consolidated_memory_id,
            source_memory_ids=list(source_memory_ids),
            org_id=org_id,
            created_by=created_by,
        )

    async def get_lineage(
        self,
        *,
        org_id: str,
        consolidated_memory_id: str,
    ) -> list[str]:
        """Return the source_memory_ids for a consolidated memory.

        Returns an empty list if no consolidation record exists.
        """
        from sqlalchemy import select
        from app.models.memory_consolidation import MemoryConsolidation

        stmt = select(MemoryConsolidation.source_memory_ids).where(
            MemoryConsolidation.organization_id == org_id,
            MemoryConsolidation.consolidated_memory_id == consolidated_memory_id,
        ).order_by(MemoryConsolidation.created_at.desc()).limit(1)

        res = await self._session.execute(stmt)
        row = res.scalar_one_or_none()
        return list(row or [])

    async def get_all_lineage_for_org(
        self,
        *,
        org_id: str,
        limit: int = 100,
    ) -> list[LineageRecord]:
        """Return recent consolidation records for an org (newest first)."""
        from sqlalchemy import select
        from app.models.memory_consolidation import MemoryConsolidation

        stmt = (
            select(MemoryConsolidation)
            .where(MemoryConsolidation.organization_id == org_id)
            .order_by(MemoryConsolidation.created_at.desc())
            .limit(limit)
        )
        res = await self._session.execute(stmt)
        rows = res.scalars().all()
        return [
            LineageRecord(
                id=str(r.id),
                consolidated_memory_id=r.consolidated_memory_id,
                source_memory_ids=list(r.source_memory_ids or []),
                org_id=org_id,
                created_by=str(r.created_by),
                created_at=r.created_at,
            )
            for r in rows
        ]
