"""Goal GraphQL resolvers."""

from __future__ import annotations

import strawberry
from sqlalchemy import select

from app.middleware.tenant_context import TenantContext
from app.models.goal import Goal


@strawberry.type
class GqlGoal:
    id: str
    title: str
    status: str
    priority: int


async def resolve_goals(
    info: strawberry.Info,
    limit: int = 10,
    status: str | None = None,
) -> list[GqlGoal]:
    ctx = info.context
    tenant = ctx["tenant"]
    if not isinstance(tenant, TenantContext):
        return []

    stmt = (
        select(Goal)
        .where(Goal.organization_id == tenant.org_id)
        .order_by(Goal.updated_at.desc())
        .limit(max(1, min(limit, 50)))
    )
    if status:
        stmt = stmt.where(Goal.status == status)

    rows = (await ctx["db"].execute(stmt)).scalars().all()

    return [
        GqlGoal(
            id=str(goal.id),
            title=goal.title,
            status=goal.status,
            priority=int(goal.priority or 0),
        )
        for goal in rows
    ]
