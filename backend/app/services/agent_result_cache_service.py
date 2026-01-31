"""Agent result cache service.

Provides a Postgres-backed cache for agent outputs, keyed by a stable
content/enrichment hash (intentionally excluding memory_id).

Tenant safety:
- Rows are isolated by organization_id and protected by RLS.
- Callers must run inside a tenant-context transaction.

Notes:
- This cache is best-effort. Failures should not break the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_result_cache import AgentResultCache


@dataclass(frozen=True)
class CachedAgentResult:
    outputs: dict[str, Any]
    confidence: float


class AgentResultCacheService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(
        self,
        *,
        organization_id: str,
        agent_name: str,
        agent_version: str,
        strategy: str,
        model: str,
        cache_key: str,
        now: Optional[datetime] = None,
    ) -> Optional[CachedAgentResult]:
        now = now or datetime.now(timezone.utc)

        stmt = (
            select(AgentResultCache)
            .where(
                AgentResultCache.organization_id == organization_id,
                AgentResultCache.agent_name == agent_name,
                AgentResultCache.agent_version == agent_version,
                AgentResultCache.strategy == strategy,
                AgentResultCache.model == model,
                AgentResultCache.cache_key == cache_key,
            )
            .limit(1)
        )

        res = await self.session.execute(stmt)
        row = res.scalar_one_or_none()
        if row is None:
            return None

        if row.expires_at is not None and row.expires_at <= now:
            return None

        row.last_accessed_at = now
        await self.session.flush()
        return CachedAgentResult(outputs=row.outputs or {}, confidence=float(row.confidence or 0.0))

    async def upsert(
        self,
        *,
        organization_id: str,
        agent_name: str,
        agent_version: str,
        strategy: str,
        model: str,
        cache_key: str,
        outputs: dict[str, Any],
        confidence: float,
        ttl_seconds: int | None = None,
        now: Optional[datetime] = None,
    ) -> None:
        now = now or datetime.now(timezone.utc)

        expires_at = None
        if ttl_seconds is not None:
            try:
                ttl_int = int(ttl_seconds)
            except Exception:
                ttl_int = 0
            if ttl_int > 0:
                expires_at = now + timedelta(seconds=ttl_int)

        stmt = insert(AgentResultCache).values(
            organization_id=organization_id,
            agent_name=agent_name,
            agent_version=agent_version,
            strategy=strategy,
            model=model,
            cache_key=cache_key,
            outputs=outputs or {},
            confidence=float(confidence or 0.0),
            last_accessed_at=now,
            expires_at=expires_at,
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=[
                AgentResultCache.organization_id,
                AgentResultCache.agent_name,
                AgentResultCache.agent_version,
                AgentResultCache.strategy,
                AgentResultCache.model,
                AgentResultCache.cache_key,
            ],
            set_={
                "outputs": outputs or {},
                "confidence": float(confidence or 0.0),
                "last_accessed_at": now,
                "expires_at": expires_at,
                "updated_at": now,
            },
        )

        await self.session.execute(stmt)
        await self.session.flush()
