"""UsageService for per-org usage counters and dashboard queries."""

from __future__ import annotations

from datetime import date, timedelta
from uuid import uuid4

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.usage_event import UsageEvent


class UsageService:
    def __init__(self, session: AsyncSession, org_id: str):
        self.session = session
        self.org_id = org_id

    async def increment(self, *, metric: str, value: int = 1, on_date: date | None = None) -> None:
        d = on_date or date.today()
        await self.session.execute(
            text(
                """
                INSERT INTO usage_events (id, organization_id, date, metric, value, created_at, updated_at)
                VALUES (:id, :org, :date, :metric, :val, now(), now())
                ON CONFLICT (organization_id, date, metric)
                DO UPDATE SET value = usage_events.value + :val, updated_at = now()
                """
            ),
            {
                "id": str(uuid4()),
                "org": self.org_id,
                "date": d,
                "metric": metric,
                "val": value,
            },
        )

    async def get_daily(self, *, metric: str, days: int = 30) -> list[dict]:
        since = date.today() - timedelta(days=days)
        res = await self.session.execute(
            select(UsageEvent.date, UsageEvent.value)
            .where(
                UsageEvent.organization_id == self.org_id,
                UsageEvent.metric == metric,
                UsageEvent.date >= since,
            )
            .order_by(UsageEvent.date)
        )
        return [{"date": str(r.date), "value": r.value} for r in res.all()]

    async def get_summary(self, *, days: int = 30) -> dict:
        since = date.today() - timedelta(days=days)
        res = await self.session.execute(
            select(UsageEvent.metric, func.sum(UsageEvent.value).label("total"))
            .where(UsageEvent.organization_id == self.org_id, UsageEvent.date >= since)
            .group_by(UsageEvent.metric)
        )
        return {r.metric: int(r.total) for r in res.all()}
