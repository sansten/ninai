"""
Tests for IdentityResolverService
===================================

Covers:
- Anonymous passthrough
- Redis cache hit/miss
- AD/SCIM enrichment path
- JWT fallback path
- Error resilience (bad Redis, bad AD)
- Bot detection via roles_string and AD servicePrincipalType
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.identity_resolver_service import IdentityResolverService, ResolvedIdentity


def _make_redis(cached_value=None, fail=False):
    redis = AsyncMock()
    if fail:
        redis.get.side_effect = Exception("redis down")
        redis.setex.side_effect = Exception("redis down")
    else:
        redis.get.return_value = cached_value
        redis.setex.return_value = True
    return redis


def _make_ad(record=None, fail=False):
    ad = AsyncMock()
    if fail:
        ad.get_user.side_effect = Exception("AD timeout")
    else:
        ad.get_user.return_value = record
    return ad


@pytest.mark.asyncio
async def test_anonymous_user_returns_anonymous_identity():
    svc = IdentityResolverService(redis_client=_make_redis())
    identity = await svc.resolve(user_id="anonymous", org_id="org1", roles_string="")
    assert identity.actor_id == "anonymous"
    assert identity.actor_type == "anonymous"
    assert identity.confidence == 0.0


@pytest.mark.asyncio
async def test_empty_user_id_returns_anonymous_identity():
    svc = IdentityResolverService(redis_client=_make_redis())
    identity = await svc.resolve(user_id="", org_id="org1", roles_string="user")
    assert identity.actor_id == "anonymous"


@pytest.mark.asyncio
async def test_redis_cache_hit_returns_cached_identity():
    cached = {
        "actor_id": "user-123",
        "actor_type": "employee",
        "display_name": "Alice",
        "role": "employee_engineering",
        "department": "Engineering",
        "location": "London",
        "manager_id": "mgr-456",
    }
    redis = _make_redis(cached_value=json.dumps(cached))
    svc = IdentityResolverService(redis_client=redis)

    identity = await svc.resolve(user_id="user-123", org_id="org1", roles_string="user")

    assert identity.actor_id == "user-123"
    assert identity.department == "Engineering"
    assert identity.confidence == 0.8
    # AD should not be called
    redis.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_ad_enrichment_on_cache_miss():
    ad_record = {
        "displayName": "Bob Smith",
        "jobTitle": "employee_leadership",
        "department": "Finance",
        "officeLocation": "NYC",
        "managerId": "mgr-789",
    }
    redis = _make_redis(cached_value=None)
    ad = _make_ad(record=ad_record)
    svc = IdentityResolverService(redis_client=redis, ad_client=ad)

    identity = await svc.resolve(user_id="user-abc", org_id="org1", roles_string="user")

    assert identity.actor_id == "user-abc"
    assert identity.display_name == "Bob Smith"
    assert identity.department == "Finance"
    assert identity.confidence == 1.0
    # Should write to cache
    redis.setex.assert_awaited_once()


@pytest.mark.asyncio
async def test_jwt_fallback_when_no_ad_and_cache_miss():
    redis = _make_redis(cached_value=None)
    svc = IdentityResolverService(redis_client=redis, ad_client=None)

    identity = await svc.resolve(user_id="user-xyz", org_id="org1", roles_string="org_admin")

    assert identity.actor_id == "user-xyz"
    assert identity.department is None
    assert identity.confidence == 0.5
    assert identity.role == "employee_org_admin"


@pytest.mark.asyncio
async def test_redis_failure_falls_through_to_jwt():
    redis = _make_redis(fail=True)
    svc = IdentityResolverService(redis_client=redis, ad_client=None)

    identity = await svc.resolve(user_id="user-xyz", org_id="org1", roles_string="user")

    assert identity.actor_id == "user-xyz"
    assert identity.confidence == 0.5


@pytest.mark.asyncio
async def test_ad_failure_falls_through_to_jwt():
    redis = _make_redis(cached_value=None)
    ad = _make_ad(fail=True)
    svc = IdentityResolverService(redis_client=redis, ad_client=ad)

    identity = await svc.resolve(user_id="user-xyz", org_id="org1", roles_string="user")

    assert identity.actor_id == "user-xyz"
    assert identity.confidence == 0.5


@pytest.mark.asyncio
async def test_bot_detected_via_roles_string():
    redis = _make_redis(cached_value=None)
    svc = IdentityResolverService(redis_client=redis, ad_client=None)

    identity = await svc.resolve(user_id="svc-bot", org_id="org1", roles_string="service,bot")

    assert identity.actor_type == "bot"


@pytest.mark.asyncio
async def test_bot_detected_via_ad_service_principal():
    redis = _make_redis(cached_value=None)
    ad = _make_ad(record={"servicePrincipalType": "Application", "displayName": "AutoBot"})
    svc = IdentityResolverService(redis_client=redis, ad_client=ad)

    identity = await svc.resolve(user_id="bot-app", org_id="org1", roles_string="user")

    assert identity.actor_type == "bot"


@pytest.mark.asyncio
async def test_role_from_jwt_priority_order():
    redis = _make_redis(cached_value=None)
    svc = IdentityResolverService(redis_client=redis, ad_client=None)

    # system_admin should win over org_admin
    identity = await svc.resolve(
        user_id="u1", org_id="o1", roles_string="org_admin,system_admin"
    )
    assert identity.role == "employee_system_admin"
