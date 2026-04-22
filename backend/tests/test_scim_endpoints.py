from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.feature_gate import CommunityFeatureGate
from app.core.security import create_access_token
from app.main import app


class _AllowAllGate:
    def is_enabled(self, *, org_id: str, feature: str) -> bool:
        return True


def _enable_scim(monkeypatch) -> None:
    monkeypatch.setattr(app.state, "feature_gate", _AllowAllGate(), raising=False)


@dataclass
class _Scalars:
    items: list

    def all(self):
        return self.items


@dataclass
class _ListResult:
    items: list

    def scalars(self):
        return _Scalars(self.items)


@dataclass
class _ScalarOneOrNoneResult:
    item: object

    def scalar_one_or_none(self):
        return self.item


@dataclass
class _ScalarOneResult:
    item: object

    def scalar_one(self):
        return self.item


@dataclass
class _AllResult:
    items: list[tuple]

    def all(self):
        return self.items


@dataclass
class _FakeUser:
    id: str
    email: str
    full_name: str
    is_active: bool = True


@dataclass
class _FakeRole:
    id: str
    name: str
    display_name: str


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["org_admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_scim_users_list_happy_path(monkeypatch):
    _enable_scim(monkeypatch)

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM users JOIN user_roles" in sql:
            return _ListResult([
                _FakeUser(id="u1", email="user1@example.com", full_name="User One", is_active=True),
                _FakeUser(id="u2", email="user2@example.com", full_name="User Two", is_active=False),
            ])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/scim/v2/Users", headers=_auth_headers())

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["totalResults"] == 2
    assert payload["Resources"][0]["userName"] == "user1@example.com"


@pytest.mark.asyncio
async def test_scim_create_user_requires_user_name():
    app.state.feature_gate = _AllowAllGate()

    session = AsyncMock(spec=AsyncSession)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/v1/scim/v2/Users", headers=_auth_headers(), json={})

    app.dependency_overrides.clear()

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_scim_groups_list_and_create_happy_path():
    app.state.feature_gate = _AllowAllGate()

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM roles" in sql and "ORDER BY roles.name" in sql:
            return _ListResult([
                _FakeRole(id="r1", name="member", display_name="Member"),
            ])
        if "FROM users JOIN user_roles" in sql and "user_roles.role_id" in sql:
            return _AllResult([("u1", "user1@example.com")])
        if "count(*) AS count_1" in sql and "FROM roles" in sql:
            return _ScalarOneResult(0)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)
    session.commit = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        list_resp = await ac.get("/api/v1/scim/v2/Groups", headers=_auth_headers())
        create_resp = await ac.post(
            "/api/v1/scim/v2/Groups",
            headers=_auth_headers(),
            json={"displayName": "Finance Admins"},
        )

    app.dependency_overrides.clear()

    assert list_resp.status_code == 200
    assert list_resp.json()["totalResults"] == 1
    assert create_resp.status_code == 201
    assert create_resp.json()["displayName"] == "Finance Admins"


@pytest.mark.asyncio
async def test_scim_feature_gate_blocks_when_disabled():
    # Community gate default should deny enterprise.scim.
    app.state.feature_gate = CommunityFeatureGate()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/scim/v2/Users", headers=_auth_headers())

    assert resp.status_code == 403
