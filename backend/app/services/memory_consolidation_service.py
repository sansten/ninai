"""PR-2 Memory Consolidation ("sleep cycle") service."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import MemoryMetadata
from app.models.memory_arc import MemoryArc
from app.models.memory_consolidation_session import ConsolidationSession


class MemoryConsolidationService:
    """Orchestrates offline memory consolidation operations."""

    FORGETTING_HALF_LIFE_DAYS = 30.0
    MIN_STABILITY_TO_KEEP = 0.1
    MAX_MEASUREMENTS_PER_ARC = 20

    def __init__(self, session: AsyncSession, user_id: str, org_id: str):
        self.session = session
        self.user_id = user_id
        self.org_id = org_id

    @staticmethod
    def retention_score(days_since_access: float, stability_days: float) -> float:
        """Ebbinghaus-style retention score R(t) = exp(-t / S)."""
        if stability_days <= 0:
            stability_days = 1.0
        return math.exp(-days_since_access / stability_days)

    @staticmethod
    def infer_trend(measurements: List[Dict[str, Any]]) -> str:
        if len(measurements) < 2:
            return "stable"
        first = float(measurements[0].get("strength", 0.0))
        last = float(measurements[-1].get("strength", 0.0))
        delta = last - first
        if delta > 0.1:
            return "strengthening"
        if delta < -0.1:
            return "weakening"
        if last > 0.75 and first < 0.4:
            return "rediscovered"
        return "stable"

    async def start_consolidation_session(self, session_type: str = "triggered") -> ConsolidationSession:
        quality_before = await self.compute_memory_quality()
        session = ConsolidationSession(
            organization_id=self.org_id,
            session_type=session_type,
            started_at=datetime.now(timezone.utc),
            status="in_progress",
            operations={},
            memory_quality_before=quality_before,
            memory_quality_after=None,
        )
        self.session.add(session)
        await self.session.flush()
        return session

    async def list_sessions(self, limit: int = 20) -> List[ConsolidationSession]:
        stmt = (
            select(ConsolidationSession)
            .where(ConsolidationSession.organization_id == self.org_id)
            .order_by(ConsolidationSession.started_at.desc())
            .limit(limit)
        )
        rows = await self.session.execute(stmt)
        return list(rows.scalars().all())

    async def discover_connections(self) -> List[Dict[str, Any]]:
        stmt = (
            select(MemoryMetadata)
            .where(
                and_(
                    MemoryMetadata.organization_id == self.org_id,
                    MemoryMetadata.is_active.is_(True),
                )
            )
            .limit(120)
        )
        rows = await self.session.execute(stmt)
        memories = list(rows.scalars().all())

        discovered: List[Dict[str, Any]] = []
        for i in range(len(memories)):
            m1 = memories[i]
            tags1 = set(m1.tags or [])
            ent1 = set((m1.entities or {}).keys())
            for j in range(i + 1, len(memories)):
                m2 = memories[j]
                tags2 = set(m2.tags or [])
                ent2 = set((m2.entities or {}).keys())

                shared_tags = sorted(tags1.intersection(tags2))
                shared_entities = sorted(ent1.intersection(ent2))
                if not shared_tags and not shared_entities:
                    continue

                novelty = 1.0
                if shared_tags:
                    novelty -= min(0.6, len(shared_tags) * 0.15)
                if shared_entities:
                    novelty -= min(0.3, len(shared_entities) * 0.1)

                discovered.append(
                    {
                        "from_memory_id": str(m1.id),
                        "to_memory_id": str(m2.id),
                        "shared_tags": shared_tags,
                        "shared_entity_types": shared_entities,
                        "novelty_score": max(0.1, round(novelty, 3)),
                    }
                )

                if len(discovered) >= 20:
                    return discovered
        return discovered

    async def merge_redundant_facts(self) -> Dict[str, int]:
        stmt = select(MemoryMetadata).where(
            and_(
                MemoryMetadata.organization_id == self.org_id,
                MemoryMetadata.is_active.is_(True),
            )
        )
        rows = await self.session.execute(stmt)
        memories = list(rows.scalars().all())

        by_hash: Dict[str, List[MemoryMetadata]] = {}
        for memory in memories:
            key = (memory.content_hash or "").strip()
            if not key:
                continue
            by_hash.setdefault(key, []).append(memory)

        merged_groups = 0
        merged_memories = 0
        for group in by_hash.values():
            if len(group) < 2:
                continue
            primary = group[0]
            duplicates = group[1:]
            for duplicate in duplicates:
                duplicate.is_active = False
                md = dict(duplicate.extra_metadata or {})
                md["merged_into"] = str(primary.id)
                duplicate.extra_metadata = md
                self.session.add(duplicate)
                merged_memories += 1
            merged_groups += 1

        return {
            "merged_fact_groups": merged_groups,
            "merged_facts": merged_memories,
        }

    async def apply_forgetting_curve(self) -> Dict[str, int]:
        stmt = select(MemoryMetadata).where(
            and_(
                MemoryMetadata.organization_id == self.org_id,
                MemoryMetadata.is_active.is_(True),
            )
        )
        rows = await self.session.execute(stmt)
        memories = list(rows.scalars().all())

        now = datetime.now(timezone.utc)
        pruned = 0
        for memory in memories:
            if memory.legal_hold:
                continue

            md = dict(memory.extra_metadata or {})
            if bool(md.get("consolidation_pinned", False)):
                continue

            last_touch = memory.last_accessed_at or memory.created_at
            if not last_touch:
                continue

            last_touch_utc = last_touch.astimezone(timezone.utc) if last_touch.tzinfo else last_touch.replace(tzinfo=timezone.utc)
            days_since_access = max(0.0, (now - last_touch_utc).total_seconds() / 86400.0)
            stability_days = self.FORGETTING_HALF_LIFE_DAYS + float(memory.access_count or 0) * 2.0
            score = self.retention_score(days_since_access, stability_days)

            md["retention_score"] = round(score, 4)
            memory.extra_metadata = md

            if score < self.MIN_STABILITY_TO_KEEP:
                memory.is_active = False
                pruned += 1
            self.session.add(memory)

        return {"pruned_memories": pruned}

    async def compute_memory_trajectories(self) -> Dict[str, int]:
        stmt = select(MemoryMetadata).where(MemoryMetadata.organization_id == self.org_id).limit(200)
        rows = await self.session.execute(stmt)
        memories = list(rows.scalars().all())

        updated = 0
        for memory in memories:
            access_count = float(memory.access_count or 0)
            strength = min(1.0, 0.2 + access_count / 12.0)
            relevance = min(1.0, 0.3 + access_count / 15.0)
            measurement = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strength": round(strength, 4),
                "access_count": int(access_count),
                "relevance_score": round(relevance, 4),
            }

            existing_stmt = select(MemoryArc).where(
                and_(
                    MemoryArc.organization_id == self.org_id,
                    MemoryArc.memory_id == str(memory.id),
                )
            )
            existing = (await self.session.execute(existing_stmt)).scalar_one_or_none()

            if existing is None:
                arc = MemoryArc(
                    organization_id=self.org_id,
                    memory_id=str(memory.id),
                    measurements=[measurement],
                    trend="stable",
                    trajectory_type="linear_decay",
                    prediction_next_access=None,
                    last_computed_at=datetime.now(timezone.utc),
                )
            else:
                measurements = list(existing.measurements or [])
                measurements.append(measurement)
                measurements = measurements[-self.MAX_MEASUREMENTS_PER_ARC :]
                existing.measurements = measurements
                existing.trend = self.infer_trend(measurements)
                existing.trajectory_type = (
                    "recently_boosted" if access_count >= 5 else "linear_decay"
                )
                # naive next-access prediction: inverse proportional to activity
                horizon_days = max(1, int(14 - min(10, access_count)))
                existing.prediction_next_access = datetime.now(timezone.utc).replace(microsecond=0)
                existing.last_computed_at = datetime.now(timezone.utc)
                arc = existing

            self.session.add(arc)
            updated += 1

        return {"trajectory_updates": updated}

    async def dream_like_association(self) -> List[Dict[str, Any]]:
        stmt = (
            select(MemoryMetadata)
            .where(
                and_(
                    MemoryMetadata.organization_id == self.org_id,
                    MemoryMetadata.is_active.is_(True),
                )
            )
            .limit(80)
        )
        rows = await self.session.execute(stmt)
        memories = list(rows.scalars().all())

        associations: List[Dict[str, Any]] = []
        for i in range(len(memories)):
            a = memories[i]
            tags_a = set(a.tags or [])
            for j in range(i + 1, len(memories)):
                b = memories[j]
                tags_b = set(b.tags or [])
                if tags_a.intersection(tags_b):
                    continue
                associations.append(
                    {
                        "memory_a": str(a.id),
                        "memory_b": str(b.id),
                        "hypothesis": "Potential cross-domain transfer opportunity",
                    }
                )
                if len(associations) >= 10:
                    return associations
        return associations

    async def finalize_session(
        self,
        session_id: str,
        operations: Optional[Dict[str, Any]] = None,
        failed: bool = False,
    ) -> ConsolidationSession:
        stmt = select(ConsolidationSession).where(
            and_(
                ConsolidationSession.organization_id == self.org_id,
                ConsolidationSession.id == session_id,
            )
        )
        session = (await self.session.execute(stmt)).scalar_one_or_none()
        if session is None:
            raise ValueError(f"Consolidation session not found: {session_id}")

        completed = datetime.now(timezone.utc)
        started = session.started_at
        started_utc = started.astimezone(timezone.utc) if started.tzinfo else started.replace(tzinfo=timezone.utc)

        session.completed_at = completed
        session.duration_seconds = int((completed - started_utc).total_seconds())
        session.status = "failed" if failed else "completed"
        session.operations = operations or session.operations or {}
        session.memory_quality_after = await self.compute_memory_quality()
        self.session.add(session)
        return session

    async def run_full_consolidation_cycle(self, session_type: str = "triggered") -> ConsolidationSession:
        session = await self.start_consolidation_session(session_type=session_type)
        operations: Dict[str, Any] = {}

        connections = await self.discover_connections()
        operations["discovered_connections"] = len(connections)

        merge_stats = await self.merge_redundant_facts()
        operations.update(merge_stats)

        prune_stats = await self.apply_forgetting_curve()
        operations.update(prune_stats)

        trajectory_stats = await self.compute_memory_trajectories()
        operations.update(trajectory_stats)

        dream_links = await self.dream_like_association()
        operations["dream_associations"] = len(dream_links)

        session = await self.finalize_session(session.id, operations=operations, failed=False)
        await self.session.commit()
        await self.session.refresh(session)
        return session

    async def get_session_report(self, session_id: str) -> Optional[ConsolidationSession]:
        stmt = select(ConsolidationSession).where(
            and_(
                ConsolidationSession.organization_id == self.org_id,
                ConsolidationSession.id == session_id,
            )
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_memory_arc(self, memory_id: str) -> Optional[MemoryArc]:
        stmt = select(MemoryArc).where(
            and_(
                MemoryArc.organization_id == self.org_id,
                MemoryArc.memory_id == memory_id,
            )
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def pin_memory(self, memory_id: str) -> bool:
        memory = await self._get_memory(memory_id)
        if memory is None:
            return False
        md = dict(memory.extra_metadata or {})
        md["consolidation_pinned"] = True
        memory.extra_metadata = md
        self.session.add(memory)
        await self.session.commit()
        return True

    async def unpin_memory(self, memory_id: str) -> bool:
        memory = await self._get_memory(memory_id)
        if memory is None:
            return False
        md = dict(memory.extra_metadata or {})
        md["consolidation_pinned"] = False
        memory.extra_metadata = md
        self.session.add(memory)
        await self.session.commit()
        return True

    async def compute_memory_quality(self) -> float:
        stmt = select(MemoryMetadata).where(MemoryMetadata.organization_id == self.org_id)
        rows = await self.session.execute(stmt)
        memories = list(rows.scalars().all())
        if not memories:
            return 0.0

        total = len(memories)
        active = sum(1 for m in memories if bool(m.is_active))
        avg_access = sum(int(m.access_count or 0) for m in memories) / total

        active_ratio = active / total
        normalized_access = min(1.0, avg_access / 10.0)
        quality = (0.65 * active_ratio) + (0.35 * normalized_access)
        return round(float(quality), 4)

    async def _get_memory(self, memory_id: str) -> Optional[MemoryMetadata]:
        stmt = select(MemoryMetadata).where(
            and_(
                MemoryMetadata.organization_id == self.org_id,
                MemoryMetadata.id == memory_id,
            )
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()
