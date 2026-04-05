"""Temporal Knowledge Graph service (Feature 24.6).

Implements temporal validity and conflict handling for graph edges:
- edge validity windows (valid_from / valid_until)
- as-of filtering for time-travel queries
- temporal overlap conflict detection
- proactive invalidation when contradicting evidence appears
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.graph_relationship import GraphRelationship


def _parse_iso(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class TemporalKnowledgeGraphService:
    def __init__(self, db: AsyncSession):
        self.db = db

    def edge_valid_as_of(self, relationship: GraphRelationship, as_of: datetime) -> bool:
        meta = relationship.metadata_ or {}
        valid_from = _parse_iso(meta.get("valid_from"))
        valid_until = _parse_iso(meta.get("valid_until"))
        point = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)

        if valid_from and point < valid_from:
            return False
        if valid_until and point >= valid_until:
            return False
        return True

    def filter_edges_as_of(
        self,
        relationships: list[GraphRelationship],
        as_of: datetime,
    ) -> list[GraphRelationship]:
        return [rel for rel in relationships if self.edge_valid_as_of(rel, as_of)]

    def detect_temporal_conflict(
        self,
        *,
        candidate_type: str,
        candidate_valid_from: datetime,
        candidate_valid_until: datetime | None,
        existing_type: str,
        existing_valid_from: datetime,
        existing_valid_until: datetime | None,
    ) -> bool:
        """Detect contradictory overlap in temporal windows.

        Conservative rule:
        - CONTRADICTS conflicts with any non-CONTRADICTS overlapping edge.
        - Also treat RELATES_TO vs DEPENDS_ON overlap as a soft conflict.
        """
        contradictory_pair = (
            candidate_type == "CONTRADICTS" and existing_type != "CONTRADICTS"
        ) or (
            existing_type == "CONTRADICTS" and candidate_type != "CONTRADICTS"
        )

        soft_pair = {candidate_type, existing_type} == {"RELATES_TO", "DEPENDS_ON"}
        if not (contradictory_pair or soft_pair):
            return False

        cand_end = candidate_valid_until or datetime.max.replace(tzinfo=timezone.utc)
        ex_end = existing_valid_until or datetime.max.replace(tzinfo=timezone.utc)

        latest_start = max(candidate_valid_from, existing_valid_from)
        earliest_end = min(cand_end, ex_end)
        return latest_start < earliest_end

    async def invalidate_contradicted_edges(
        self,
        *,
        organization_id: str,
        from_memory_id: str,
        to_memory_id: str,
        contradiction_at: datetime | None = None,
    ) -> int:
        """Close validity windows on active non-CONTRADICTS edges for a memory pair."""
        invalidated_at = contradiction_at or _now_utc()

        stmt = select(GraphRelationship).where(
            and_(
                GraphRelationship.organization_id == organization_id,
                GraphRelationship.from_memory_id == from_memory_id,
                GraphRelationship.to_memory_id == to_memory_id,
                GraphRelationship.relationship_type != "CONTRADICTS",
            )
        )
        rows = (await self.db.execute(stmt)).scalars().all()

        count = 0
        for rel in rows:
            meta = dict(rel.metadata_ or {})
            if meta.get("valid_until"):
                continue
            meta["valid_until"] = invalidated_at.isoformat()
            meta["invalidated_reason"] = "contradiction"
            rel.metadata_ = meta
            count += 1

        if count:
            await self.db.commit()
        return count
