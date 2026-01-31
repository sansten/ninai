from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


@dataclass
class _FakeApiKey:
    id: str
    name: str
    prefix: str
    user_id: str
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


@dataclass
class _FakeWebhookSub:
    id: str
    url: str
    is_active: bool
    event_types: list[str]
    description: str | None
    created_at: datetime


class _Scalars:
    def __init__(self, items: list[Any]):
        self._items = items

    def all(self):
        return self._items


class _ListResult:
    def __init__(self, items: list[Any]):
        self._items = items

    def scalars(self):
        return _Scalars(self._items)


class _ScalarOneOrNoneResult:
    def __init__(self, item: Any | None):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


def _admin_headers(*, org_id: str = "o1", user_id: str = "u_admin") -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=["org_admin"])
    return {"Authorization": f"Bearer {token}"}


def _member_headers(*, org_id: str = "o1", user_id: str = "u_member") -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=["member"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_x_api_key_allows_admin_route_without_jwt(monkeypatch):
    now = datetime(2026, 1, 22, tzinfo=timezone.utc)

    key = _FakeApiKey(
        id="k1",
        name="Key",
        prefix="ninai_abc",
        user_id="u_admin",
        created_at=now,
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM api_keys" in sql:
            return _ListResult([key])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    import app.middleware.tenant_context as tenant_context

    async def _fake_authenticate_api_key(db: AsyncSession, plaintext: str):
        assert plaintext == "ninai_testkey"
        return ("u_admin", "o1", ["org_admin"], 0)

    monkeypatch.setattr(tenant_context.ApiKeyService, "authenticate_api_key", _fake_authenticate_api_key)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/admin/api-keys", headers={"X-API-Key": "ninai_testkey"})

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["id"] == "k1"


@pytest.mark.asyncio
async def test_admin_api_keys_requires_org_admin():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/admin/api-keys", headers=_member_headers())
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_webhooks_requires_org_admin():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/admin/webhooks", headers=_member_headers())
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_api_keys_list_create_revoke_happy_path(monkeypatch):
    now = datetime(2026, 1, 22, tzinfo=timezone.utc)

    created_key = _FakeApiKey(
        id="k1",
        name="My Key",
        prefix="ninai_abc",
        user_id="u_admin",
        created_at=now,
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "WHERE api_keys.id" in sql:
            # revoke_api_key lookup
            return _ScalarOneOrNoneResult(created_key)
        if "FROM api_keys" in sql:
            # list_api_keys
            return _ListResult([created_key])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)
    session.commit = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    import app.api.v1.endpoints.api_keys as api_keys_endpoints

    async def _fake_create_api_key(*, session: AsyncSession, organization_id: str, user_id: str, name: str):
        assert organization_id == "o1"
        assert user_id == "u_admin"
        assert name == "My Key"
        return created_key, "ninai_abc_plaintext"

    monkeypatch.setattr(api_keys_endpoints.ApiKeyService, "create_api_key", _fake_create_api_key)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create
        resp = await ac.post(
            "/api/v1/admin/api-keys",
            headers=_admin_headers(),
            json={"name": "My Key"},
        )
        assert resp.status_code == 201
        payload = resp.json()
        assert payload["id"] == "k1"
        assert payload["prefix"] == "ninai_abc"
        assert payload["api_key"] == "ninai_abc_plaintext"
        assert session.commit.await_count == 1

        # List
        resp = await ac.get("/api/v1/admin/api-keys", headers=_admin_headers())
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["id"] == "k1"
        assert items[0]["prefix"] == "ninai_abc"
        assert "api_key" not in items[0]

        # Revoke
        resp = await ac.post("/api/v1/admin/api-keys/k1/revoke", headers=_admin_headers())
        assert resp.status_code == 200
        revoked = resp.json()
        assert revoked["id"] == "k1"
        assert revoked["revoked_at"] is not None

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_admin_webhooks_list_create_delete_happy_path(monkeypatch):
    now = datetime(2026, 1, 22, tzinfo=timezone.utc)

    created_sub = _FakeWebhookSub(
        id="w1",
        url="https://example.com/hook",
        is_active=True,
        event_types=["audit.test"],
        description="desc",
        created_at=now,
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "WHERE webhook_subscriptions.id" in sql:
            # delete_webhook lookup
            return _ScalarOneOrNoneResult(created_sub)
        if "FROM webhook_subscriptions" in sql and "WHERE webhook_subscriptions.organization_id" in sql:
            # list_webhooks
            return _ListResult([created_sub])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)
    session.commit = AsyncMock()
    session.delete = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    import app.api.v1.endpoints.webhooks as webhooks_endpoints

    async def _fake_create_subscription(self, *, organization_id: str, url: str, event_types, description):
        assert organization_id == "o1"
        assert url == "https://example.com/hook"
        return created_sub, "whsec_test"

    monkeypatch.setattr(webhooks_endpoints.WebhookService, "create_subscription", _fake_create_subscription)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create
        resp = await ac.post(
            "/api/v1/admin/webhooks",
            headers=_admin_headers(),
            json={"url": "https://example.com/hook", "event_types": ["audit.test"], "description": "desc"},
        )
        assert resp.status_code == 201
        payload = resp.json()
        assert payload["id"] == "w1"
        assert payload["url"] == "https://example.com/hook"
        assert payload["secret"] == "whsec_test"
        assert session.commit.await_count == 1

        # List
        resp = await ac.get("/api/v1/admin/webhooks", headers=_admin_headers())
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["id"] == "w1"
        assert items[0]["url"] == "https://example.com/hook"
        assert "secret" not in items[0]

        # Delete
        resp = await ac.delete("/api/v1/admin/webhooks/w1", headers=_admin_headers())
        assert resp.status_code == 204
        assert session.delete.await_count == 1
        assert session.commit.await_count == 2

    app.dependency_overrides.clear()
