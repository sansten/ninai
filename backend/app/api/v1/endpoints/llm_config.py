"""Admin endpoints for LLM provider and model configuration."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.models.org_llm_config import OrgLlmConfig, VALID_PROVIDERS
from app.services.llm_router_service import LlmRouterService

router = APIRouter(prefix="/admin/llm-config", tags=["llm-config"])


class LlmConfigRequest(BaseModel):
    provider: str
    model: str
    api_key_ref: str | None = None
    base_url: str | None = None


@router.get("")
async def get_llm_config(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    res = await db.execute(
        select(OrgLlmConfig).where(OrgLlmConfig.organization_id == tenant.org_id)
    )
    row = res.scalar_one_or_none()
    if not row:
        return {
            "provider": "local",
            "model": "qwen2.5:7b",
            "is_active": True,
        }
    return {
        "provider": row.provider,
        "model": row.model,
        "api_key_ref": row.api_key_ref,
        "base_url": row.base_url,
        "is_active": row.is_active,
    }


@router.put("")
async def set_llm_config(
    body: LlmConfigRequest,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    normalized_provider = (body.provider or "").strip().lower()
    if normalized_provider == "vllm":
        normalized_provider = "local"
    if normalized_provider not in VALID_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider. Must be one of {sorted(VALID_PROVIDERS)}",
        )
    if not body.model or len(body.model) == 0:
        raise HTTPException(status_code=400, detail="Model name required and non-empty")

    res = await db.execute(
        select(OrgLlmConfig).where(OrgLlmConfig.organization_id == tenant.org_id)
    )
    row = res.scalar_one_or_none()

    if not row:
        row = OrgLlmConfig(
            id=str(uuid4()),
            organization_id=tenant.org_id,
        )
        db.add(row)

    row.provider = normalized_provider
    row.model = body.model
    row.api_key_ref = body.api_key_ref
    row.base_url = body.base_url
    row.is_active = True

    await db.commit()
    return {
        "provider": row.provider,
        "model": row.model,
        "api_key_ref": row.api_key_ref,
        "base_url": row.base_url,
        "is_active": row.is_active,
    }
