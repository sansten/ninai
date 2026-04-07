from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.endpoints.scim import _apply_scim_patch_op, _parse_scim_user_filter
from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


class _AllowAllGate:
    def is_enabled(self, *, org_id: str, feature: str) -> bool:
        return True


def _enable_scim(monkeypatch) -> None:
    monkeypatch.setattr(app.state, "feature_gate", _AllowAllGate(), raising=False)


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["org_admin"])
    return {"Authorization": f"Bearer {token}"}


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
    updated_at: object | None = None


@dataclass
class _FakeRole:
    id: str
    name: str
    display_name: str


def test_apply_scim_patch_active_true():
    user = _FakeUser(id="u1", email="a@b.com", full_name="A", is_active=False)
    _apply_scim_patch_op(user, {"op": "replace", "path": "active", "value": True})
    assert user.is_active is True


def test_apply_scim_patch_active_remove_false():
    user = _FakeUser(id="u1", email="a@b.com", full_name="A", is_active=True)
    _apply_scim_patch_op(user, {"op": "remove", "path": "active"})
    assert user.is_active is False


def test_apply_scim_patch_display_name():
    user = _FakeUser(id="u1", email="a@b.com", full_name="Old")
    _apply_scim_patch_op(user, {"op": "replace", "path": "displayName", "value": "New Name"})
    assert user.full_name == "New Name"


def test_apply_scim_patch_unknown_noop():
    user = _FakeUser(id="u1", email="a@b.com", full_name="Old")
    _apply_scim_patch_op(user, {"op": "replace", "path": "unknown", "value": "x"})
    assert user.full_name == "Old"


def test_parse_filter_user_name():
    assert _parse_scim_user_filter('userName eq "john@example.com"') == ("userName", "john@example.com")


def test_parse_filter_active_false():
    assert _parse_scim_user_filter("active eq false") == ("active", "false")


def test_parse_filter_invalid_raises_400():
    with pytest.raises(Exception):
        _parse_scim_user_filter("invalid eq x")


@pytest.mark.asyncio
async def test_list_users_with_filter_and_etag(monkeypatch):
    _enable_scim(monkeypatch)

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM users JOIN user_roles" in sql:
            return _ListResult([
                _FakeUser(id="u1", email="john@example.com", full_name="John", is_active=True),
            ])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/v1/scim/v2/Users?filter=userName%20eq%20%22john@example.com%22&startIndex=1&count=1",
            headers=_auth_headers(),
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.headers.get("etag") is not None
    assert resp.headers.get("content-type", "").startswith("application/scim+json")
    payload = resp.json()
    assert payload["totalResults"] == 1
    assert payload["itemsPerPage"] == 1


@pytest.mark.asyncio
async def test_patch_user_and_etag(monkeypatch):
    _enable_scim(monkeypatch)

    session = AsyncMock(spec=AsyncSession)
    user = _FakeUser(id="u1", email="old@example.com", full_name="Old", is_active=True)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM users JOIN user_roles" in sql:
            return _ScalarOneOrNoneResult(user)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)
    session.commit = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    payload = {"Operations": [{"op": "replace", "path": "displayName", "value": "Patched"}]}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.patch("/api/v1/scim/v2/Users/u1", headers=_auth_headers(), json=payload)

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.headers.get("etag") is not None
    assert resp.json()["name"]["formatted"] == "Patched"


@pytest.mark.asyncio
async def test_put_user_replace(monkeypatch):
    _enable_scim(monkeypatch)

    session = AsyncMock(spec=AsyncSession)
    user = _FakeUser(id="u1", email="old@example.com", full_name="Old", is_active=True)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM users JOIN user_roles" in sql:
            return _ScalarOneOrNoneResult(user)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)
    session.commit = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    payload = {"userName": "new@example.com", "name": {"formatted": "New Name"}, "active": False}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.put("/api/v1/scim/v2/Users/u1", headers=_auth_headers(), json=payload)

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["userName"] == "new@example.com"
    assert body["active"] is False


@pytest.mark.asyncio
async def test_patch_group_and_delete_group(monkeypatch):
    _enable_scim(monkeypatch)

    session = AsyncMock(spec=AsyncSession)
    role = _FakeRole(id="g1", name="team", display_name="Team")

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM roles" in sql and "roles.id" in sql:
            return _ScalarOneOrNoneResult(role)
        if "FROM user_roles" in sql:
            return _ScalarOneOrNoneResult(None)
        if "FROM users JOIN user_roles" in sql and "user_roles.role_id" in sql:
            return _AllResult([("u1", "user1@example.com")])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)
    session.commit = AsyncMock()
    session.delete = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        patch_resp = await ac.patch(
            "/api/v1/scim/v2/Groups/g1",
            headers=_auth_headers(),
            json={"Operations": [{"op": "add", "value": [{"value": "u1"}]}]},
        )
        delete_resp = await ac.delete("/api/v1/scim/v2/Groups/g1", headers=_auth_headers())

    app.dependency_overrides.clear()

    assert patch_resp.status_code == 200
    assert patch_resp.headers.get("content-type", "").startswith("application/scim+json")
    assert delete_resp.status_code == 204
