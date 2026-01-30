"""Admin settings schemas.

These schemas support runtime-editable application configuration exposed via an
admin-only API (backed by DB storage).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from app.schemas.base import BaseSchema


class AuthConfig(BaseSchema):
    auth_mode: Literal["password", "oidc", "both"] = "password"

    oidc_issuer: Optional[str] = None
    oidc_client_id: Optional[str] = None
    oidc_audience: Optional[str] = None
    oidc_allowed_email_domains: Optional[list[str]] = None

    oidc_default_org_slug: Optional[str] = None
    oidc_default_org_id: Optional[str] = None
    oidc_default_role: Optional[str] = None

    oidc_groups_claim: Optional[str] = None
    oidc_group_to_role_json: Optional[str] = None


class AuthConfigUpdate(BaseSchema):
    """Patch-like update.

    Any provided field overwrites the saved override.
    To revert a field back to its .env/default value, send it as null.
    """

    auth_mode: Optional[Literal["password", "oidc", "both"]] = None

    oidc_issuer: Optional[str] = None
    oidc_client_id: Optional[str] = None
    oidc_audience: Optional[str] = None
    oidc_allowed_email_domains: Optional[list[str]] = None

    oidc_default_org_slug: Optional[str] = None
    oidc_default_org_id: Optional[str] = None
    oidc_default_role: Optional[str] = None

    oidc_groups_claim: Optional[str] = None
    oidc_group_to_role_json: Optional[str] = None


class AuthConfigResponse(BaseSchema):
    effective: AuthConfig
    overrides: dict[str, Any]


class EnvSetting(BaseSchema):
    key: str
    value: Optional[str] = None
    is_sensitive: bool = False
    requires_restart: bool = True


class EnvSettingsResponse(BaseSchema):
    items: list[EnvSetting]
