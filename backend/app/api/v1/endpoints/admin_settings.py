"""Admin settings endpoints.

Exposes runtime-editable configuration (DB-backed) to system administrators.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.schemas.admin_settings import (
    AuthConfig,
    AuthConfigResponse,
    AuthConfigUpdate,
    EnvSetting,
    EnvSettingsResponse,
)
from app.services.app_settings_service import get_effective_auth_config, update_auth_config_overrides


router = APIRouter()


@router.get("/settings/auth", response_model=AuthConfigResponse)
async def get_auth_settings(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    effective, overrides = await get_effective_auth_config(db)
    return AuthConfigResponse(effective=AuthConfig(**effective), overrides=overrides)


@router.put("/settings/auth", response_model=AuthConfigResponse)
async def put_auth_settings(
    body: AuthConfigUpdate,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    patch: dict[str, Any] = body.model_dump(exclude_unset=True)
    overrides = await update_auth_config_overrides(
        db,
        patch=patch,
        updated_by_user_id=tenant.user_id,
    )
    await db.commit()

    effective, _ = await get_effective_auth_config(db)
    return AuthConfigResponse(effective=AuthConfig(**effective), overrides=overrides)


_SENSITIVE_SUBSTRINGS = (
    "PASSWORD",
    "SECRET",
    "TOKEN",
    "PRIVATE",
    "KEY",
)

_ALLOWED_EXCEPTIONS = {
    # Not secrets; keep visible.
    "OIDC_CLIENT_ID",
    "JWT_ALGORITHM",
    "API_KEY",  # not present today, but common; still consider masking in prod
}


def _is_sensitive(key: str) -> bool:
    if key in _ALLOWED_EXCEPTIONS:
        return False
    upper = key.upper()
    return any(s in upper for s in _SENSITIVE_SUBSTRINGS)


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


@router.get("/settings/env", response_model=EnvSettingsResponse)
async def get_env_settings(
    tenant: TenantContext = Depends(require_org_admin()),
):
    # NOTE: This is read-only; changing .env values still requires redeploy/restart.
    raw = settings.model_dump()

    items: list[EnvSetting] = []
    for k in sorted(raw.keys()):
        sensitive = _is_sensitive(k)
        value = None if sensitive else _stringify(raw.get(k))
        if sensitive and raw.get(k) is not None:
            value = "***"
        items.append(
            EnvSetting(
                key=k,
                value=value,
                is_sensitive=sensitive,
                requires_restart=True,
            )
        )

    return EnvSettingsResponse(items=items)
