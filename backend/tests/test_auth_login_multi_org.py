"""Regression tests: /auth/login used to always pick the user's oldest
UserRole row as "the" organization, with no way for a multi-org user to
choose. login now accepts an optional org_slug and rejects it if the user
has no role there, instead of silently logging into the wrong tenant.

Postgres-backed (uses pg_client/pg_db_session) — auto-skips when Postgres
isn't reachable, same as tests/test_admin.py.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_password_hash
from app.models.organization import Organization
from app.models.user import Role, User, UserRole

pytestmark = pytest.mark.asyncio


async def _make_org(db_session: AsyncSession, slug: str) -> Organization:
    org = Organization(id=str(uuid4()), name=slug, slug=slug)
    db_session.add(org)
    await db_session.flush()
    return org


async def _make_role(db_session: AsyncSession, name: str = "member") -> Role:
    role = Role(id=str(uuid4()), name=name, display_name=name, permissions=[])
    db_session.add(role)
    await db_session.flush()
    return role


async def _make_multi_org_user(db_session: AsyncSession) -> tuple[User, Organization, Organization]:
    user = User(
        id=str(uuid4()),
        email="multiorg@test.com",
        full_name="Multi Org User",
        hashed_password=get_password_hash("password123"),
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    org_old = await _make_org(db_session, "old-org")
    org_new = await _make_org(db_session, "new-org")
    role = await _make_role(db_session)

    # Older row created first — this is what the old code silently defaulted to.
    db_session.add(UserRole(id=str(uuid4()), user_id=user.id, role_id=role.id, organization_id=org_old.id))
    await db_session.flush()
    db_session.add(UserRole(id=str(uuid4()), user_id=user.id, role_id=role.id, organization_id=org_new.id))
    await db_session.commit()

    return user, org_old, org_new


async def test_login_without_org_slug_defaults_to_first_org(
    pg_client: AsyncClient, pg_db_session: AsyncSession
):
    """Backward compat: omitting org_slug keeps the existing default behavior."""
    user, org_old, org_new = await _make_multi_org_user(pg_db_session)

    resp = await pg_client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "password123"},
    )
    assert resp.status_code == 200
    assert resp.json()["organization"]["id"] == org_old.id


async def test_login_with_org_slug_selects_requested_org(
    pg_client: AsyncClient, pg_db_session: AsyncSession
):
    """A multi-org user can now choose which org to log into."""
    user, org_old, org_new = await _make_multi_org_user(pg_db_session)

    resp = await pg_client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "password123", "org_slug": "new-org"},
    )
    assert resp.status_code == 200
    assert resp.json()["organization"]["id"] == org_new.id


async def test_login_with_org_slug_user_has_no_role_in_rejected(
    pg_client: AsyncClient, pg_db_session: AsyncSession
):
    """Requesting an org the user has no role in must fail, not silently
    fall back to a different org."""
    user, org_old, org_new = await _make_multi_org_user(pg_db_session)
    other_org = await _make_org(pg_db_session, "unrelated-org")
    await pg_db_session.commit()

    resp = await pg_client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "password123", "org_slug": "unrelated-org"},
    )
    assert resp.status_code == 401
