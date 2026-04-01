"""Cognitive load balancer service (Phase 55)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.load_snapshot import LoadSnapshot


class CognitiveLoadBalancer:
    """Classify system load and decide whether lower-priority tasks should run."""

    THRESHOLDS: dict[str, int] = {
        "low": 50,
        "medium": 200,
        "high": 500,
    }

    _CRITICAL_ALLOWLIST = {"q.memory_ingest", "q.cognitive_loop"}
    _HIGH_SKIP = {"q.agent_topics", "q.agent_patterns", "q.agent_graph"}
    _MEDIUM_SKIP = {"q.agent_topics"}

    _THROTTLE_FACTORS: dict[str, float] = {
        "low": 0.0,
        "medium": 0.5,
        "high": 2.0,
        "critical": 5.0,
    }

    def classify_load(self, *, queue_depths: dict[str, int]) -> str:
        """Classify load level from total queue depth."""
        total = sum(max(0, int(depth)) for depth in (queue_depths or {}).values())

        if total > self.THRESHOLDS["high"]:
            return "critical"
        if total > self.THRESHOLDS["medium"]:
            return "high"
        if total > self.THRESHOLDS["low"]:
            return "medium"
        return "low"

    def should_skip(self, *, task_name: str, load_level: str) -> bool:
        """Return True when a task should be skipped at the given load level."""
        level = (load_level or "low").lower()

        if level == "critical":
            return task_name not in self._CRITICAL_ALLOWLIST
        if level == "high":
            return task_name in self._HIGH_SKIP
        if level == "medium":
            return task_name in self._MEDIUM_SKIP
        return False

    def throttle_factor(self, *, load_level: str) -> float:
        """Return multiplier for delay/sleep behavior by load level."""
        return self._THROTTLE_FACTORS.get((load_level or "low").lower(), 0.0)

    async def record_snapshot(
        self,
        *,
        db: AsyncSession,
        org_id: str,
        queue_depths: dict[str, int],
        active_workers: int,
    ) -> LoadSnapshot:
        """Persist a load snapshot and return the saved row."""
        normalized_queue_depths = {
            key: max(0, int(value)) for key, value in (queue_depths or {}).items()
        }
        snapshot = LoadSnapshot(
            org_id=org_id,
            queue_depths=normalized_queue_depths,
            active_workers=max(0, int(active_workers)),
            load_level=self.classify_load(queue_depths=normalized_queue_depths),
        )
        db.add(snapshot)
        await db.commit()
        await db.refresh(snapshot)
        return snapshot

    async def get_recent_load(
        self,
        *,
        db: AsyncSession,
        org_id: str,
        limit: int = 10,
    ) -> list[LoadSnapshot]:
        """Return latest load snapshots for an org ordered by newest first."""
        safe_limit = max(1, int(limit))
        stmt = (
            select(LoadSnapshot)
            .where(LoadSnapshot.org_id == org_id)
            .order_by(LoadSnapshot.sampled_at.desc())
            .limit(safe_limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())
