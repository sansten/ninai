"""GraphQL auth and context wiring."""

from __future__ import annotations

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context, security


async def get_graphql_context(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    """Build an authenticated context and apply tenant RLS settings."""
    credentials = await security(request)
    x_api_key = request.headers.get("X-API-Key")

    tenant = await get_tenant_context(
        credentials=credentials,
        x_api_key=x_api_key,
        db=db,
    )

    await set_tenant_context(
        db,
        tenant.user_id,
        tenant.org_id,
        tenant.roles_string,
        tenant.clearance_level,
    )

    return {
        "tenant": tenant,
        "db": db,
        "request": request,
    }
