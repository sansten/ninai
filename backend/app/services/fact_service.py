"""Fact enrichment service (PR3).

Provides enrichment payload for memory search responses.
"""

from __future__ import annotations

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.contradiction import Contradiction
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

    async def get_structured_evidence_for_memories(self, memory_ids: list[str]) -> dict:
        """Return fact and contradiction evidence for a set of memories."""
        if not memory_ids:
            return {"facts": [], "contradictions": []}

        stmt = (
            select(MemoryFact)
            .where(
                and_(
                    MemoryFact.organization_id == self.org_id,
                    MemoryFact.source_memory_id.in_(memory_ids),
                )
            )
            .order_by(MemoryFact.confidence.desc(), MemoryFact.updated_at.desc())
        )
        facts = list((await self.session.execute(stmt)).scalars().all())
        if not facts:
            return {"facts": [], "contradictions": []}

        fact_ids = [fact.id for fact in facts]
        contradiction_stmt = (
            select(Contradiction)
            .where(
                and_(
                    Contradiction.organization_id == self.org_id,
                    Contradiction.resolved_at.is_(None),
                    (Contradiction.fact_a.in_(fact_ids) | Contradiction.fact_b.in_(fact_ids)),
                )
            )
            .order_by(Contradiction.created_at.desc())
        )
        contradictions = list((await self.session.execute(contradiction_stmt)).scalars().all())

        fact_index = {str(fact.id): fact for fact in facts}

        def _serialize_fact(fact: MemoryFact) -> dict:
            return {
                "fact_id": str(fact.id),
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "confidence": float(fact.confidence or 0.0),
                "status": fact.status.value if hasattr(fact.status, "value") else str(fact.status),
                "source_memory_id": str(fact.source_memory_id),
                "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
                "valid_to": fact.valid_to.isoformat() if fact.valid_to else None,
                "contradiction_group_id": fact.contradiction_group_id,
            }

        def _serialize_contradiction(contradiction: Contradiction) -> dict:
            fact_a = fact_index.get(str(contradiction.fact_a))
            fact_b = fact_index.get(str(contradiction.fact_b))
            return {
                "contradiction_id": str(contradiction.id),
                "fact_a": str(contradiction.fact_a),
                "fact_b": str(contradiction.fact_b),
                "fact_a_object": getattr(fact_a, "object", None),
                "fact_b_object": getattr(fact_b, "object", None),
                "reason": contradiction.reason,
                "severity": contradiction.severity.value if hasattr(contradiction.severity, "value") else str(contradiction.severity),
                "created_at": contradiction.created_at.isoformat() if contradiction.created_at else None,
            }

        return {
            "facts": [_serialize_fact(fact) for fact in facts],
            "contradictions": [_serialize_contradiction(item) for item in contradictions],
        }
