"""Concept registry persistence service (Phase 59)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.learned_concept import LearnedConcept


class ConceptRegistryService:
    async def upsert_concepts(
        self,
        *,
        db: AsyncSession,
        org_id: str,
        new_concepts: list[dict],
        updated_concepts: list[dict],
    ) -> int:
        affected = 0
        now = datetime.now(timezone.utc)

        for concept in new_concepts or []:
            name = str(concept.get("concept_name") or "").strip()
            if not name:
                continue
            stmt = select(LearnedConcept).where(
                LearnedConcept.org_id == org_id,
                LearnedConcept.concept_name == name,
            )
            existing = (await db.execute(stmt)).scalar_one_or_none()
            member_ids = [str(m) for m in (concept.get("member_ids") or concept.get("member_memory_ids") or []) if str(m)]
            canonical_terms = [str(t) for t in (concept.get("canonical_terms") or []) if str(t)]
            conf = float(concept.get("confidence") or 0.5)

            if existing is None:
                row = LearnedConcept(
                    org_id=org_id,
                    concept_name=name,
                    member_memory_ids=member_ids,
                    canonical_terms=canonical_terms,
                    occurrence_count=len(member_ids),
                    first_seen=now,
                    last_seen=now,
                    confidence=conf,
                )
                db.add(row)
                affected += 1
            else:
                merged_ids = list(dict.fromkeys((existing.member_memory_ids or []) + member_ids))
                existing.member_memory_ids = merged_ids
                existing.canonical_terms = canonical_terms or existing.canonical_terms
                existing.occurrence_count = len(merged_ids)
                existing.last_seen = now
                existing.confidence = conf
                affected += 1

        for concept in updated_concepts or []:
            name = str(concept.get("concept_name") or "").strip()
            if not name:
                continue
            stmt = select(LearnedConcept).where(
                LearnedConcept.org_id == org_id,
                LearnedConcept.concept_name == name,
            )
            existing = (await db.execute(stmt)).scalar_one_or_none()
            member_ids = [
                str(m)
                for m in (
                    concept.get("member_ids")
                    or concept.get("member_memory_ids")
                    or concept.get("new_member_ids")
                    or []
                )
                if str(m)
            ]
            canonical_terms = [str(t) for t in (concept.get("canonical_terms") or []) if str(t)]
            conf = float(concept.get("confidence") or 0.5)

            if existing is None:
                row = LearnedConcept(
                    org_id=org_id,
                    concept_name=name,
                    member_memory_ids=member_ids,
                    canonical_terms=canonical_terms,
                    occurrence_count=len(member_ids),
                    first_seen=now,
                    last_seen=now,
                    confidence=conf,
                )
                db.add(row)
                affected += 1
            else:
                merged_ids = list(dict.fromkeys((existing.member_memory_ids or []) + member_ids))
                existing.member_memory_ids = merged_ids
                if canonical_terms:
                    existing.canonical_terms = canonical_terms
                existing.occurrence_count = len(merged_ids)
                existing.last_seen = now
                existing.confidence = conf
                affected += 1

        await db.commit()
        return affected

    async def get_concepts_for_org(
        self,
        *,
        db: AsyncSession,
        org_id: str,
        limit: int = 50,
    ) -> list[LearnedConcept]:
        safe_limit = max(1, int(limit))
        stmt = (
            select(LearnedConcept)
            .where(LearnedConcept.org_id == org_id)
            .order_by(LearnedConcept.last_seen.desc())
            .limit(safe_limit)
        )
        return list((await db.execute(stmt)).scalars().all())

    async def find_concept_for_memory(
        self,
        *,
        db: AsyncSession,
        org_id: str,
        memory_id: str,
    ) -> LearnedConcept | None:
        stmt = select(LearnedConcept).where(LearnedConcept.org_id == org_id)
        concepts = list((await db.execute(stmt)).scalars().all())
        for concept in concepts:
            if memory_id in (concept.member_memory_ids or []):
                return concept
        return None
