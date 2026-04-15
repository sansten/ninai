"""
API v1 Dependency Factories
============================

Factory functions for injectable services used across v1 endpoints.
All functions return cached or request-scoped instances.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from app.services.identity_resolver_service import IdentityResolverService
from app.services.identity_policy_service import IdentityPolicyService


@lru_cache(maxsize=1)
def _build_ad_client():
    """Build ADSCIMClient if AD_SCIM_ENDPOINT is configured; return None otherwise."""
    from app.core.config import settings
    from app.integrations.ad_scim_client import ADSCIMClient

    if not getattr(settings, "AD_SCIM_ENDPOINT", None):
        return None
    return ADSCIMClient(
        endpoint=settings.AD_SCIM_ENDPOINT,
        token=settings.AD_SCIM_TOKEN or "",
    )


async def get_identity_resolver() -> IdentityResolverService:
    """Dependency: return a configured IdentityResolverService."""
    from app.core.redis import RedisClient

    redis = await RedisClient.get_client()
    return IdentityResolverService(redis_client=redis, ad_client=_build_ad_client())


async def get_identity_policy_service() -> IdentityPolicyService:
    """Dependency: return an IdentityPolicyService (stateless, cheap to construct)."""
    return IdentityPolicyService()
