"""Memory GraphQL resolvers."""

from __future__ import annotations

import strawberry
from sqlalchemy import or_, select

from app.middleware.tenant_context import TenantContext
from app.graphql.resolvers.conflicts import GqlConflict, build_conflicts_from_metadata
from app.models.memory import MemoryMetadata


@strawberry.type
class GqlMemory:
    id: str
    content: str
    domain: str
    credibility_score: float
    decay_score: float
    tags: list[str]
    conflicts: list[GqlConflict]


async def resolve_search_memory(
    info: strawberry.Info,
    query: str,
    limit: int = 5,
) -> list[GqlMemory]:
    ctx = info.context
    db = ctx["db"]
    tenant = ctx["tenant"]
    if not isinstance(tenant, TenantContext):
        return []

    q = query.strip()
    if not q:
        return []

    like = f"%{q}%"
    stmt = (
        select(MemoryMetadata)
        .where(
            MemoryMetadata.organization_id == tenant.org_id,
            MemoryMetadata.is_active.is_(True),
            or_(
                MemoryMetadata.title.ilike(like),
                MemoryMetadata.content_preview.ilike(like),
            ),
        )
        .order_by(MemoryMetadata.created_at.desc())
        .limit(max(1, min(limit, 50)))
    )

    rows = (await db.execute(stmt)).scalars().all()
    output: list[GqlMemory] = []

    for row in rows:
        meta = row.extra_metadata or {}
        output.append(
            GqlMemory(
                id=str(row.id),
                content=row.content_preview or "",
                domain=str(row.business_domain or meta.get("domain") or "general"),
                credibility_score=float(meta.get("credibility_score") or 0.0),
                decay_score=float(
                    meta.get("decay_score")
                    or meta.get("freshness_score")
                    or 0.0
                ),
                tags=list(row.tags or []),
                conflicts=build_conflicts_from_metadata(meta),
            )
        )

    return output
