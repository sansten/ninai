"""
Tests for IdentityPolicyService
==================================

Covers:
- Org mandate forces FULL regardless of user preference
- call_anonymous=True applies ANONYMOUS when org permits
- User preference ROLE_ONLY is respected
- Default is FULL when no policy or preference
- get_org_policy returns safe default when no row
- get_user_pref returns None when no row
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.identity_policy_service import (
    IdentityPolicyService,
    ResolvedActorContext,
    _DEFAULT_ORG_POLICY,
    get_org_policy,
    get_user_pref,
)
from app.services.identity_resolver_service import ResolvedIdentity
from app.models.user_identity_preference import UserIdentityPreference, IdentityMode


def _make_resolved_identity(
    actor_id: str = "user-1",
    actor_type: str = "employee",
    role: str = "employee_engineering",
    department: str = "Engineering",
    confidence: float = 1.0,
) -> ResolvedIdentity:
    return ResolvedIdentity(
        actor_id=actor_id,
        actor_type=actor_type,
        display_name=actor_id,
        role=role,
        department=department,
        location=None,
        manager_id=None,
        confidence=confidence,
    )


def _make_org_policy(
    mandate: bool = False,
    allowed_modes: list = None,
    audit_trail: bool = True,
):
    from types import SimpleNamespace
    return SimpleNamespace(
        mandate_actor_identity=mandate,
        allowed_modes=allowed_modes or ["full", "role_only", "anonymous"],
        enrich_from_directory=True,
        audit_trail_always=audit_trail,
    )


def _make_user_pref(preference: str = "full"):
    from types import SimpleNamespace
    return SimpleNamespace(preference=IdentityMode(preference))


@pytest.mark.asyncio
async def test_org_mandate_forces_full():
    svc = IdentityPolicyService()
    identity = _make_resolved_identity()
    org_policy = _make_org_policy(mandate=True)
    user_pref = _make_user_pref("anonymous")  # user wants anonymous

    ctx = await svc.resolve(
        resolved_identity=identity,
        org_policy=org_policy,
        user_pref=user_pref,
        call_anonymous=True,
    )

    assert ctx.mode_applied == "full"
    assert ctx.actor_id == "user-1"
    assert ctx.mandate_was_active is True


@pytest.mark.asyncio
async def test_call_anonymous_applied_when_org_permits():
    svc = IdentityPolicyService()
    identity = _make_resolved_identity()
    org_policy = _make_org_policy(mandate=False)
    user_pref = _make_user_pref("full")

    ctx = await svc.resolve(
        resolved_identity=identity,
        org_policy=org_policy,
        user_pref=user_pref,
        call_anonymous=True,
    )

    assert ctx.mode_applied == "anonymous"
    assert ctx.actor_id is None
    assert ctx.mandate_was_active is False


@pytest.mark.asyncio
async def test_user_preference_role_only():
    svc = IdentityPolicyService()
    identity = _make_resolved_identity()
    org_policy = _make_org_policy(mandate=False)
    user_pref = _make_user_pref("role_only")

    ctx = await svc.resolve(
        resolved_identity=identity,
        org_policy=org_policy,
        user_pref=user_pref,
        call_anonymous=False,
    )

    assert ctx.mode_applied == "role_only"
    assert ctx.actor_id is None  # stripped in role_only
    assert ctx.role == "employee_engineering"


@pytest.mark.asyncio
async def test_default_full_when_no_preference():
    svc = IdentityPolicyService()
    identity = _make_resolved_identity()
    org_policy = _make_org_policy(mandate=False)

    ctx = await svc.resolve(
        resolved_identity=identity,
        org_policy=org_policy,
        user_pref=None,
        call_anonymous=False,
    )

    assert ctx.mode_applied == "full"
    assert ctx.actor_id == "user-1"
    assert ctx.department == "Engineering"


@pytest.mark.asyncio
async def test_full_mode_includes_all_fields():
    svc = IdentityPolicyService()
    identity = _make_resolved_identity(confidence=0.8)
    org_policy = _make_org_policy(mandate=False)

    ctx = await svc.resolve(
        resolved_identity=identity,
        org_policy=org_policy,
        user_pref=None,
        call_anonymous=False,
    )

    assert ctx.actor_type == "employee"
    assert ctx.role == "employee_engineering"
    assert ctx.department == "Engineering"
    assert ctx.identity_confidence == 0.8


@pytest.mark.asyncio
async def test_anonymous_mode_strips_all_fields():
    svc = IdentityPolicyService()
    identity = _make_resolved_identity()
    org_policy = _make_org_policy(mandate=False)
    user_pref = _make_user_pref("anonymous")

    ctx = await svc.resolve(
        resolved_identity=identity,
        org_policy=org_policy,
        user_pref=user_pref,
        call_anonymous=False,
    )

    assert ctx.mode_applied == "anonymous"
    assert ctx.actor_id is None
    assert ctx.actor_type is None
    assert ctx.role is None
    assert ctx.department is None
    assert ctx.identity_confidence == 0.0


@pytest.mark.asyncio
async def test_get_org_policy_returns_default_on_db_error():
    db = AsyncMock()
    db.scalar.side_effect = Exception("DB conn failed")

    policy = await get_org_policy(db, "org-999")

    assert policy is _DEFAULT_ORG_POLICY
    assert policy.mandate_actor_identity is False


@pytest.mark.asyncio
async def test_get_org_policy_returns_default_when_no_row():
    db = AsyncMock()
    db.scalar.return_value = None

    policy = await get_org_policy(db, "org-new")

    assert policy is _DEFAULT_ORG_POLICY


@pytest.mark.asyncio
async def test_get_user_pref_returns_none_on_db_error():
    db = AsyncMock()
    db.scalar.side_effect = Exception("DB conn failed")

    pref = await get_user_pref(db, "user-abc")

    assert pref is None


@pytest.mark.asyncio
async def test_get_user_pref_returns_none_when_no_row():
    db = AsyncMock()
    db.scalar.return_value = None

    pref = await get_user_pref(db, "user-new")

    assert pref is None
