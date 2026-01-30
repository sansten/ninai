"""Feature gating for Community vs Enterprise.

Core (OSS) provides the interface and default behavior.
Enterprise (paid) plugs in at runtime by registering an alternate FeatureGate
implementation (typically backed by a signed license token).

Design goals:
- Server-side enforcement (FastAPI dependency).
- Org-scoped entitlements (multi-tenant).
- Safe defaults: if enterprise package is not installed, enterprise features are disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fastapi import Depends, HTTPException, Request, status

from app.middleware.tenant_context import TenantContext, get_tenant_context


class FeatureGate(Protocol):
    def is_enabled(self, *, org_id: str, feature: str) -> bool: ...


@dataclass(frozen=True)
class CommunityFeatureGate:
    """Default gate for OSS builds.

    Enterprise features are disabled unless an Enterprise gate is installed.
    """

    def is_enabled(self, *, org_id: str, feature: str) -> bool:
        # Convention: enterprise-only features are prefixed to keep separation obvious.
        if feature.startswith("enterprise."):
            return False
        return True


def set_feature_gate(request_or_app: Request | object, gate: FeatureGate) -> None:
    """Attach the active FeatureGate to the FastAPI app state."""

    app = getattr(request_or_app, "app", request_or_app)
    setattr(app.state, "feature_gate", gate)


def get_feature_gate(request: Request) -> FeatureGate:
    gate = getattr(request.app.state, "feature_gate", None)
    if gate is None:
        # Extremely defensive default.
        return CommunityFeatureGate()
    return gate


def require_feature(feature: str):
    """FastAPI dependency that enforces org-scoped feature entitlement.

    Intended usage:
      @router.get(...)
      async def endpoint(
          tenant: TenantContext = Depends(require_org_admin()),
          _: None = Depends(require_feature("enterprise.admin_ops")),
      ):
          ...

    Returns None on success; raises 403 if not entitled.
    """

    async def checker(
        request: Request,
        tenant: TenantContext = Depends(get_tenant_context),
    ) -> None:
        gate = get_feature_gate(request)
        if not gate.is_enabled(org_id=tenant.org_id, feature=feature):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "feature_not_enabled",
                    "feature": feature,
                    "message": "This feature requires an Enterprise license.",
                },
            )

    return checker


class EnterpriseFeatures:
    """Canonical feature names for gating.

    These are intentionally strings (not enums) to keep them stable across
    editions and easy to reference from routes/tasks.
    """

    # Licensing / entitlements
    LICENSE_MANAGEMENT = "enterprise.license_management"

    # Operations / observability
    ADMIN_OPS = "enterprise.admin_ops"
    ADVANCED_OBSERVABILITY = "enterprise.observability"

    # Evaluation / drift
    AUTOEVALBENCH = "enterprise.autoevalbench"
    DRIFT_DETECTION = "enterprise.drift_detection"

    # Identity
    SCIM = "enterprise.scim"
    SSO_ADVANCED = "enterprise.sso_advanced"
