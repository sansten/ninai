"""Cognitive metrics aggregation and Prometheus instruments for Feature 21."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from prometheus_client import Counter, Gauge, Histogram
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.middleware.prometheus import metrics_registry
from app.models.agent_run import AgentRun
from app.models.audit import AuditEvent
from app.models.memory import MemoryMetadata
from app.models.meta_agent import MetaConflictRegistry


MEMORY_WRITES = Counter(
    "ninai_memory_writes_total",
    "Total memories written",
    ["domain", "org"],
    registry=metrics_registry,
)
AGENT_RUNS = Counter(
    "ninai_agent_runs_total",
    "Total agent runs",
    ["agent_name", "strategy", "status"],
    registry=metrics_registry,
)
COGNITIVE_LATENCY = Histogram(
    "ninai_cognitive_latency_seconds",
    "Cognitive operation latency in seconds",
    ["operation"],
    registry=metrics_registry,
)
ACTIVE_CONFLICTS = Gauge(
    "ninai_active_conflicts",
    "Currently active conflicts",
    ["domain"],
    registry=metrics_registry,
)
HEARTBEAT_FRESHNESS = Gauge(
    "ninai_heartbeat_age_seconds",
    "Seconds since last heartbeat event",
    registry=metrics_registry,
)
LLM_SUCCESS_RATE = Gauge(
    "ninai_llm_success_rate",
    "LLM call success rate estimate",
    registry=metrics_registry,
)


class CognitiveMetricsService:
    async def get_summary(self, *, db: AsyncSession, org_id: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)

        total_memories_res = await db.execute(
            select(func.count()).select_from(MemoryMetadata).where(MemoryMetadata.organization_id == org_id)
        )
        total_memories = int(total_memories_res.scalar_one() or 0)

        recent_memories_res = await db.execute(
            select(func.count())
            .select_from(MemoryMetadata)
            .where(
                MemoryMetadata.organization_id == org_id,
                MemoryMetadata.created_at >= one_hour_ago,
            )
        )
        memories_last_hour = int(recent_memories_res.scalar_one() or 0)

        active_conflicts_res = await db.execute(
            select(func.count())
            .select_from(MetaConflictRegistry)
            .where(
                MetaConflictRegistry.organization_id == org_id,
                MetaConflictRegistry.status == "open",
            )
        )
        active_conflicts = int(active_conflicts_res.scalar_one() or 0)

        heartbeat_res = await db.execute(
            select(func.max(AuditEvent.timestamp)).where(
                AuditEvent.organization_id == org_id,
                AuditEvent.event_type.ilike("%heartbeat%"),
            )
        )
        latest_heartbeat = heartbeat_res.scalar_one_or_none()
        heartbeat_age_seconds = None
        if latest_heartbeat is not None:
            if getattr(latest_heartbeat, "tzinfo", None) is None:
                latest_heartbeat = latest_heartbeat.replace(tzinfo=timezone.utc)
            heartbeat_age_seconds = max(0.0, (now - latest_heartbeat).total_seconds())

        llm_success_res = await db.execute(
            select(
                func.count().label("total"),
                func.sum(
                    case(
                        (AuditEvent.success.is_(True), 1),
                        else_=0,
                    )
                ).label("ok"),
            ).where(
                AuditEvent.organization_id == org_id,
                AuditEvent.event_type.ilike("llm.%"),
                AuditEvent.timestamp >= one_hour_ago,
            )
        )
        llm_total, llm_ok = llm_success_res.one_or_none() or (0, 0)
        llm_total = int(llm_total or 0)
        llm_ok = int(llm_ok or 0)
        llm_success_rate = (llm_ok / llm_total) if llm_total else None

        ACTIVE_CONFLICTS.labels(domain="all").set(float(active_conflicts))
        if heartbeat_age_seconds is not None:
            HEARTBEAT_FRESHNESS.set(float(heartbeat_age_seconds))
        if llm_success_rate is not None:
            LLM_SUCCESS_RATE.set(float(llm_success_rate))

        return {
            "timestamp": now.isoformat(),
            "organization_id": org_id,
            "memory_writes_per_second": round(memories_last_hour / 3600.0, 6),
            "total_memories": total_memories,
            "active_conflicts": active_conflicts,
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "llm_success_rate": llm_success_rate,
        }

    async def get_agent_metrics(self, *, db: AsyncSession, org_id: str) -> dict[str, Any]:
        latency_expr = func.extract("epoch", AgentRun.finished_at - AgentRun.started_at)
        stmt = (
            select(
                AgentRun.agent_name,
                func.count().label("runs"),
                func.avg(latency_expr).label("avg_latency_seconds"),
                func.sum(case((AgentRun.status == "failed", 1), else_=0)).label("failed_runs"),
            )
            .where(AgentRun.organization_id == org_id)
            .group_by(AgentRun.agent_name)
            .order_by(AgentRun.agent_name.asc())
        )
        res = await db.execute(stmt)
        rows = res.all()

        items: list[dict[str, Any]] = []
        for agent_name, runs, avg_latency_seconds, failed_runs in rows:
            run_count = int(runs or 0)
            failures = int(failed_runs or 0)
            error_rate = (failures / run_count) if run_count else 0.0
            latency = float(avg_latency_seconds or 0.0)

            AGENT_RUNS.labels(agent_name=str(agent_name), strategy="unknown", status="all").inc(run_count)
            COGNITIVE_LATENCY.labels(operation=f"agent:{agent_name}").observe(latency)

            items.append(
                {
                    "agent_name": str(agent_name),
                    "run_count": run_count,
                    "avg_latency_seconds": latency,
                    "error_rate": round(error_rate, 6),
                }
            )

        return {"organization_id": org_id, "agents": items, "total_agents": len(items)}

    async def get_memory_metrics(self, *, db: AsyncSession, org_id: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        one_day_ago = now - timedelta(days=1)

        total_res = await db.execute(
            select(func.count()).select_from(MemoryMetadata).where(MemoryMetadata.organization_id == org_id)
        )
        total = int(total_res.scalar_one() or 0)

        recent_growth_res = await db.execute(
            select(func.count())
            .select_from(MemoryMetadata)
            .where(
                MemoryMetadata.organization_id == org_id,
                MemoryMetadata.created_at >= one_day_ago,
            )
        )
        growth_last_24h = int(recent_growth_res.scalar_one() or 0)

        inactive_res = await db.execute(
            select(func.count())
            .select_from(MemoryMetadata)
            .where(
                MemoryMetadata.organization_id == org_id,
                MemoryMetadata.is_active.is_(False),
            )
        )
        inactive = int(inactive_res.scalar_one() or 0)
        decay_rate = (inactive / total) if total else 0.0

        domain_stmt = (
            select(func.coalesce(MemoryMetadata.business_domain, "unknown"), func.count())
            .where(MemoryMetadata.organization_id == org_id)
            .group_by(func.coalesce(MemoryMetadata.business_domain, "unknown"))
            .order_by(func.count().desc())
        )
        domain_res = await db.execute(domain_stmt)
        domain_distribution = {str(domain): int(count) for domain, count in domain_res.all()}

        for domain, count in domain_distribution.items():
            MEMORY_WRITES.labels(domain=domain, org=org_id).inc(int(count))

        return {
            "organization_id": org_id,
            "total_memories": total,
            "growth_last_24h": growth_last_24h,
            "decay_rate": round(decay_rate, 6),
            "domain_distribution": domain_distribution,
        }

    async def get_event_metrics(self, *, db: AsyncSession, org_id: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)

        stmt = (
            select(AuditEvent.event_type, func.count())
            .where(
                AuditEvent.organization_id == org_id,
                AuditEvent.timestamp >= one_hour_ago,
            )
            .group_by(AuditEvent.event_type)
            .order_by(func.count().desc())
        )
        res = await db.execute(stmt)
        rows = res.all()

        events: list[dict[str, Any]] = []
        for event_type, count in rows:
            event_count = int(count or 0)
            events.append(
                {
                    "event_type": str(event_type),
                    "count_last_hour": event_count,
                    "rate_per_minute": round(event_count / 60.0, 6),
                }
            )

        return {"organization_id": org_id, "events": events, "window": "1h"}
