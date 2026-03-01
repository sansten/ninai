from __future__ import annotations

from typing import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import get_db
from app.main import app
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.organization import Organization
from app.models.user import User


@pytest_asyncio.fixture
async def episodes_client(test_engine) -> AsyncGenerator[AsyncClient, None]:
    org_id = str(uuid4())
    user_id = str(uuid4())

    session_factory = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory.begin() as session:
        await session.execute(
            insert(Organization),
            {
                "id": org_id,
                "name": "Episode Test Org",
                "slug": f"episode-test-org-{org_id[:8]}",
                "is_active": True,
            },
        )
        await session.execute(
            insert(User),
            {
                "id": user_id,
                "email": f"episode-user-{user_id[:8]}@example.com",
                "hashed_password": "$2b$12$placeholder",
                "full_name": "Episode Test User",
                "is_active": True,
                "role": "org_admin",
            },
        )

    async def override_get_db():
        async with session_factory() as session:
            yield session

    async def override_tenant_context() -> TenantContext:
        return TenantContext(
            user_id=user_id,
            org_id=org_id,
            roles=["org_admin"],
            clearance_level=4,
        )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_tenant_context] = override_tenant_context

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_tenant_context, None)


def _has_path(path: str) -> bool:
    return any(getattr(route, "path", None) == path for route in app.router.routes)


@pytest.mark.asyncio
async def test_episode_routes_registered():
    assert _has_path("/api/v1/episodes")
    assert _has_path("/api/v1/episodes/{episode_id}")
    assert _has_path("/api/v1/episodes/{episode_id}/events")
    assert _has_path("/api/v1/episodes/{episode_id}/resolve")


@pytest.mark.asyncio
async def test_create_and_get_episode(episodes_client: AsyncClient):
    create_resp = await episodes_client.post(
        "/api/v1/episodes",
        json={
            "scope_type": "personal",
            "episode_type": "support_case",
            "title": "Customer Network Issue",
            "tags": ["networking", "latency"],
            "entities": {"customer_id": "CUST-1001", "device_id": "DEV-22"},
        },
    )

    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["title"] == "Customer Network Issue"
    assert created["episode_type"] == "support_case"
    assert created["status"] == "open"
    assert created["organization_id"]

    episode_id = created["id"]

    get_resp = await episodes_client.get(f"/api/v1/episodes/{episode_id}")
    assert get_resp.status_code == 200

    fetched = get_resp.json()
    assert fetched["id"] == episode_id
    assert fetched["title"] == "Customer Network Issue"


@pytest.mark.asyncio
async def test_episode_events_are_sorted_by_event_ts(episodes_client: AsyncClient):
    create_episode_resp = await episodes_client.post(
        "/api/v1/episodes",
        json={
            "scope_type": "personal",
            "episode_type": "support_case",
            "title": "Timeline Ordering Test",
        },
    )
    assert create_episode_resp.status_code == 201
    episode_id = create_episode_resp.json()["id"]

    first_event_resp = await episodes_client.post(
        "/api/v1/episodes/events",
        json={
            "episode_id": episode_id,
            "event_type": "user_report",
            "event_ts": "2026-02-28T09:00:00Z",
            "actor_type": "user",
            "content": "Issue reported",
        },
    )
    assert first_event_resp.status_code == 201

    second_event_resp = await episodes_client.post(
        "/api/v1/episodes/events",
        json={
            "episode_id": episode_id,
            "event_type": "agent_action",
            "event_ts": "2026-02-28T10:00:00Z",
            "actor_type": "agent",
            "content": "Diagnostics run",
        },
    )
    assert second_event_resp.status_code == 201

    list_events_resp = await episodes_client.get(f"/api/v1/episodes/{episode_id}/events")
    assert list_events_resp.status_code == 200

    payload = list_events_resp.json()
    assert payload["total"] == 2
    assert payload["events"][0]["event_ts"] < payload["events"][1]["event_ts"]
    assert payload["events"][0]["content"] == "Issue reported"
    assert payload["events"][1]["content"] == "Diagnostics run"


@pytest.mark.asyncio
async def test_resolve_episode_updates_status(episodes_client: AsyncClient):
    create_resp = await episodes_client.post(
        "/api/v1/episodes",
        json={
            "scope_type": "personal",
            "episode_type": "support_case",
            "title": "Resolve Test",
        },
    )
    assert create_resp.status_code == 201

    episode_id = create_resp.json()["id"]

    resolve_resp = await episodes_client.post(f"/api/v1/episodes/{episode_id}/resolve")
    assert resolve_resp.status_code == 200

    resolved = resolve_resp.json()
    assert resolved["id"] == episode_id
    assert resolved["status"] == "resolved"
    assert resolved["resolved_at"] is not None


@pytest.mark.asyncio
async def test_episode_link_creation_is_idempotent(episodes_client: AsyncClient):
    first = await episodes_client.post(
        "/api/v1/episodes",
        json={"scope_type": "personal", "episode_type": "support_case", "title": "E1"},
    )
    second = await episodes_client.post(
        "/api/v1/episodes",
        json={"scope_type": "personal", "episode_type": "support_case", "title": "E2"},
    )

    assert first.status_code == 201
    assert second.status_code == 201

    e1 = first.json()["id"]
    e2 = second.json()["id"]

    link_payload = {
        "from_episode_id": e1,
        "to_episode_id": e2,
        "relation": "follow_on",
        "confidence": 0.88,
    }

    link_1 = await episodes_client.post("/api/v1/episodes/links", json=link_payload)
    link_2 = await episodes_client.post("/api/v1/episodes/links", json=link_payload)

    assert link_1.status_code == 201
    assert link_2.status_code == 201
    assert link_1.json()["id"] == link_2.json()["id"]

    links_resp = await episodes_client.get(f"/api/v1/episodes/{e1}/links")
    assert links_resp.status_code == 200
    links = links_resp.json()
    assert links["total"] == 1
