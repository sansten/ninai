"""Intelligence Digest Service — P46.

Builds per-organization daily/weekly intelligence digests by querying
existing memory, goal, anomaly, and review queue data.
Follows the same aggregation pattern as memory_insights.py.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.digest_report import DigestReport

logger = logging.getLogger(__name__)

_DIGEST_WINDOW_DAYS = {"daily": 1, "weekly": 7}


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _build_narrative(
    *,
    org_id: str,
    digest_type: str,
    memories_total: int,
    memories_flagged: int,
    anomalies_count: int,
    goals_active: int,
    goals_completed: int,
    quality_trend: dict,
) -> str:
    """Build a heuristic narrative paragraph for the digest."""
    window = "today" if digest_type == "daily" else "this week"
    direction = quality_trend.get("direction", "stable")
    delta = quality_trend.get("delta", 0.0)
    avg_7d = quality_trend.get("avg_7d", 0.0)

    parts: list[str] = [
        f"Intelligence digest for {window}.",
        f"Memory store contains {memories_total} active records",
        f"({memories_flagged} pending human review).",
    ]

    if anomalies_count > 0:
        parts.append(f"{anomalies_count} anomal{'y' if anomalies_count == 1 else 'ies'} detected.")
    else:
        parts.append("No anomalies detected.")

    if goals_completed > 0:
        parts.append(f"{goals_completed} goal{'s' if goals_completed > 1 else ''} completed.")
    if goals_active > 0:
        parts.append(f"{goals_active} goal{'s' if goals_active > 1 else ''} active.")

    trend_str = f"Memory quality is {direction}"
    if abs(delta) > 0.01:
        sign = "+" if delta >= 0 else ""
        trend_str += f" ({sign}{delta:.2f} vs prior period, 7d avg {avg_7d:.2f})"
    else:
        trend_str += f" (7d avg {avg_7d:.2f})"
    parts.append(trend_str + ".")

    return " ".join(parts)


class DigestService:
    """Builds and persists intelligence digest reports."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _count_active_memories(self, org_id: str) -> int:
        try:
            from app.models.memory import MemoryMetadata
            stmt = select(func.count(MemoryMetadata.id)).where(
                MemoryMetadata.organization_id == org_id,
                MemoryMetadata.memory_type != "short_term",
            )
            result = await self.db.execute(stmt)
            return int(result.scalar() or 0)
        except Exception:
            logger.exception("Failed to count memories", extra={"org_id": org_id})
            return 0

    async def _count_flagged_memories(self, org_id: str) -> int:
        """Count memories queued for human review (HumanReviewQueueAgent outputs)."""
        try:
            from app.models.agent_run import AgentRun
            stmt = select(func.count(func.distinct(AgentRun.memory_id))).where(
                AgentRun.organization_id == org_id,
                AgentRun.agent_name == "HumanReviewQueueAgent",
                AgentRun.status == "success",
            )
            result = await self.db.execute(stmt)
            return int(result.scalar() or 0)
        except Exception:
            return 0

    async def _count_anomalies(self, org_id: str, window_days: int) -> int:
        """Count anomaly detection results within the digest window."""
        try:
            from app.models.agent_result_cache import AgentResultCache
            since = datetime.now(timezone.utc) - timedelta(days=window_days)
            stmt = select(func.count(AgentResultCache.id)).where(
                AgentResultCache.organization_id == org_id,
                AgentResultCache.agent_name == "AnomalyDetectionAgent",
                AgentResultCache.created_at >= since,
            )
            result = await self.db.execute(stmt)
            return int(result.scalar() or 0)
        except Exception:
            return 0

    async def _count_goals(self, org_id: str, window_days: int) -> tuple[int, int]:
        """Return (active_goals, completed_in_window)."""
        try:
            from app.models.goal import Goal
            since = datetime.now(timezone.utc) - timedelta(days=window_days)
            active_stmt = select(func.count(Goal.id)).where(
                Goal.organization_id == org_id,
                Goal.status == "active",
            )
            completed_stmt = select(func.count(Goal.id)).where(
                Goal.organization_id == org_id,
                Goal.status == "completed",
                Goal.updated_at >= since,
            )
            active = int((await self.db.execute(active_stmt)).scalar() or 0)
            completed = int((await self.db.execute(completed_stmt)).scalar() or 0)
            return active, completed
        except Exception:
            return 0, 0

    async def _compute_quality_trend(self, org_id: str) -> dict:
        """Compute 7-day average credibility score and delta vs prior 7 days."""
        try:
            from app.models.agent_result_cache import AgentResultCache
            now = datetime.now(timezone.utc)
            week_ago = now - timedelta(days=7)
            two_weeks_ago = now - timedelta(days=14)

            def _avg_stmt(start, end):
                return (
                    select(func.avg(AgentResultCache.confidence))
                    .where(
                        AgentResultCache.organization_id == org_id,
                        AgentResultCache.agent_name == "CredibilityAgent",
                        AgentResultCache.created_at >= start,
                        AgentResultCache.created_at < end,
                    )
                )

            current_avg = float((await self.db.execute(_avg_stmt(week_ago, now))).scalar() or 0.5)
            prior_avg = float((await self.db.execute(_avg_stmt(two_weeks_ago, week_ago))).scalar() or 0.5)
            delta = round(current_avg - prior_avg, 4)
            direction = "improving" if delta > 0.01 else "declining" if delta < -0.01 else "stable"
            return {"avg_7d": round(current_avg, 4), "delta": delta, "direction": direction}
        except Exception:
            return {"avg_7d": 0.5, "delta": 0.0, "direction": "stable"}

    async def _fetch_top_insights(self, org_id: str, window_days: int) -> list[dict]:
        """Return top 3 agent results by confidence in the window."""
        try:
            from app.models.agent_result_cache import AgentResultCache
            since = datetime.now(timezone.utc) - timedelta(days=window_days)
            stmt = (
                select(AgentResultCache)
                .where(
                    AgentResultCache.organization_id == org_id,
                    AgentResultCache.created_at >= since,
                )
                .order_by(AgentResultCache.confidence.desc())
                .limit(3)
            )
            rows = (await self.db.execute(stmt)).scalars().all()
            return [
                {
                    "title": str(row.agent_name),
                    "summary": f"Confidence {round(float(row.confidence or 0), 3)}",
                    "agent": str(row.agent_name),
                    "confidence": float(row.confidence or 0),
                }
                for row in rows
            ]
        except Exception:
            return []

    async def build_digest(self, org_id: str, digest_type: str = "daily") -> DigestReport:
        """Build a DigestReport for an org without saving it."""
        window_days = _DIGEST_WINDOW_DAYS.get(digest_type, 1)
        today = datetime.now(timezone.utc).date()

        memories_total = await self._count_active_memories(org_id)
        memories_flagged = await self._count_flagged_memories(org_id)
        anomalies_count = await self._count_anomalies(org_id, window_days)
        goals_active, goals_completed = await self._count_goals(org_id, window_days)
        quality_trend = await self._compute_quality_trend(org_id)
        top_insights = await self._fetch_top_insights(org_id, window_days)

        narrative = _build_narrative(
            org_id=org_id,
            digest_type=digest_type,
            memories_total=memories_total,
            memories_flagged=memories_flagged,
            anomalies_count=anomalies_count,
            goals_active=goals_active,
            goals_completed=goals_completed,
            quality_trend=quality_trend,
        )

        report = DigestReport(
            organization_id=org_id,
            run_date=today,
            digest_type=digest_type,
            memories_total=memories_total,
            memories_flagged=memories_flagged,
            anomalies_count=anomalies_count,
            goals_active=goals_active,
            goals_completed=goals_completed,
            top_insights=top_insights,
            quality_trend=quality_trend,
            narrative=narrative,
            completed_at=datetime.now(timezone.utc),
        )
        return report

    async def save_report(self, report: DigestReport) -> DigestReport:
        """Upsert a DigestReport (one per org per run_date per digest_type)."""
        stmt = select(DigestReport).where(
            DigestReport.organization_id == report.organization_id,
            DigestReport.run_date == report.run_date,
            DigestReport.digest_type == report.digest_type,
        )
        existing = (await self.db.execute(stmt)).scalar_one_or_none()

        if existing is not None:
            existing.memories_total = report.memories_total
            existing.memories_flagged = report.memories_flagged
            existing.anomalies_count = report.anomalies_count
            existing.goals_active = report.goals_active
            existing.goals_completed = report.goals_completed
            existing.top_insights = report.top_insights
            existing.quality_trend = report.quality_trend
            existing.narrative = report.narrative
            existing.completed_at = report.completed_at
            await self.db.commit()
            await self.db.refresh(existing)
            return existing

        self.db.add(report)
        await self.db.commit()
        await self.db.refresh(report)
        return report

    async def get_latest(self, org_id: str, digest_type: str = "daily") -> Optional[DigestReport]:
        """Return the most recent digest for an org."""
        stmt = (
            select(DigestReport)
            .where(
                DigestReport.organization_id == org_id,
                DigestReport.digest_type == digest_type,
            )
            .order_by(DigestReport.run_date.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_history(
        self, org_id: str, digest_type: str = "daily", limit: int = 30
    ) -> list[DigestReport]:
        """Return recent digest history for an org."""
        stmt = (
            select(DigestReport)
            .where(
                DigestReport.organization_id == org_id,
                DigestReport.digest_type == digest_type,
            )
            .order_by(DigestReport.run_date.desc())
            .limit(min(limit, 90))
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
