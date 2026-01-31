"""App settings service.

Provides a small abstraction over the DB-backed app_settings table.
Currently used for runtime-editable authentication/OIDC configuration.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.app_setting import AppSetting


_AUTH_CONFIG_KEY = "auth_config"

# Small in-process cache to avoid hitting DB on every request.
# NOTE: Safe for single-process dev; in production this should be backed by Redis
# or a proper config service if you need immediate cross-worker consistency.
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _normalize_mode(mode: Optional[str]) -> str:
    value = (mode or "password").strip().lower()
    if value not in {"password", "oidc", "both"}:
        return "password"
    return value


def _env_auth_config() -> dict[str, Any]:
    return {
        "auth_mode": _normalize_mode(settings.AUTH_MODE),
        "oidc_issuer": settings.OIDC_ISSUER,
        "oidc_client_id": settings.OIDC_CLIENT_ID,
        "oidc_audience": settings.OIDC_AUDIENCE,
        "oidc_allowed_email_domains": settings.OIDC_ALLOWED_EMAIL_DOMAINS,
        "oidc_default_org_slug": settings.OIDC_DEFAULT_ORG_SLUG,
        "oidc_default_org_id": settings.OIDC_DEFAULT_ORG_ID,
        "oidc_default_role": settings.OIDC_DEFAULT_ROLE,
        "oidc_groups_claim": settings.OIDC_GROUPS_CLAIM,
        "oidc_group_to_role_json": settings.OIDC_GROUP_TO_ROLE_JSON,
    }


def _merge_effective(env_cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    effective = dict(env_cfg)
    for k, v in overrides.items():
        # Allow explicit null to clear an env default.
        effective[k] = v
    # Always normalize auth_mode.
    effective["auth_mode"] = _normalize_mode(effective.get("auth_mode"))
    # Normalize allowed domains: strip @ and whitespace.
    domains = effective.get("oidc_allowed_email_domains")
    if isinstance(domains, list):
        cleaned = [str(d).strip().lstrip("@").lower() for d in domains if str(d).strip()]
        effective["oidc_allowed_email_domains"] = cleaned or None
    return effective


async def get_auth_config_overrides(db: AsyncSession) -> dict[str, Any]:
    result = await db.execute(select(AppSetting).where(AppSetting.key == _AUTH_CONFIG_KEY))
    setting = result.scalar_one_or_none()
    if setting and isinstance(setting.value, dict):
        return dict(setting.value)
    return {}


async def get_effective_auth_config(
    db: AsyncSession,
    *,
    cache_ttl_seconds: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (effective, overrides).

    effective = env defaults merged with DB overrides.
    overrides = raw DB overrides.
    """
    now = time.time()
    cached = _CACHE.get(_AUTH_CONFIG_KEY)
    if cached and (now - cached[0]) < cache_ttl_seconds:
        effective = cached[1]
        # Overrides are only used for UI; fetch them when needed.
        overrides = await get_auth_config_overrides(db)
        return effective, overrides

    overrides = await get_auth_config_overrides(db)
    env_cfg = _env_auth_config()
    effective = _merge_effective(env_cfg, overrides)
    _CACHE[_AUTH_CONFIG_KEY] = (now, effective)
    return effective, overrides


async def update_auth_config_overrides(
    db: AsyncSession,
    *,
    patch: dict[str, Any],
    updated_by_user_id: Optional[str],
) -> dict[str, Any]:
    current = await get_auth_config_overrides(db)

    # Only update keys provided in patch.
    for k, v in patch.items():
        # Convention: null means "inherit .env/default".
        if v is None:
            current.pop(k, None)
        else:
            current[k] = v

    result = await db.execute(select(AppSetting).where(AppSetting.key == _AUTH_CONFIG_KEY))
    setting = result.scalar_one_or_none()

    if setting is None:
        setting = AppSetting(key=_AUTH_CONFIG_KEY, value=current, updated_by_user_id=updated_by_user_id)
        db.add(setting)
    else:
        setting.value = current
        setting.updated_by_user_id = updated_by_user_id

    await db.flush()
    # Bust cache immediately.
    _CACHE.pop(_AUTH_CONFIG_KEY, None)
    return current
