"""Fact enrichment service (PR3).

Provides enrichment payload for memory search responses.
"""

from __future__ import annotations

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory_fact import MemoryFact, MemoryFactStatus


class FactService:
    def __init__(self, session: AsyncSession, org_id: str):
        self.session = session
        self.org_id = org_id

    async def get_enrichment_for_memories(self, memory_ids: list[str]) -> dict:
        if not memory_ids:
            return {"facts_used": [], "disputed_facts": []}

        active_stmt = (
            select(MemoryFact)
            .where(
                and_(
                    MemoryFact.organization_id == self.org_id,
                    MemoryFact.source_memory_id.in_(memory_ids),
                    MemoryFact.status == MemoryFactStatus.ACTIVE,
                )
            )
            .order_by(MemoryFact.confidence.desc())
        )
        active = list((await self.session.execute(active_stmt)).scalars().all())

        disputed_stmt = (
            select(MemoryFact)
            .where(
                and_(
                    MemoryFact.organization_id == self.org_id,
                    MemoryFact.source_memory_id.in_(memory_ids),
                    MemoryFact.status == MemoryFactStatus.DISPUTED,
                )
            )
            .order_by(MemoryFact.updated_at.desc())
        )
        disputed = list((await self.session.execute(disputed_stmt)).scalars().all())

        def _serialize(fact: MemoryFact) -> dict:
            return {
                "id": fact.id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "confidence": fact.confidence,
                "status": fact.status.value if hasattr(fact.status, "value") else str(fact.status),
                "source_memory_id": fact.source_memory_id,
                "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
                "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
                "contradiction_group_id": fact.contradiction_group_id,
            }

        return {
            "facts_used": [_serialize(f) for f in active],
            "disputed_facts": [_serialize(f) for f in disputed],
        }
