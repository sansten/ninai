from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.database import get_db
from app.main import app
from app.api.v1.endpoints import memories as memories_endpoints
from app.api.v1.endpoints.memories import get_memory_service


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_response_dict(*, memory_id: str, org_id: str, owner_id: str, content_preview: str) -> dict:
    now = _now_iso()
    return {
        "id": memory_id,
        "organization_id": org_id,
        "owner_id": owner_id,
        "scope": "personal",
        "scope_id": None,
        "memory_type": "long_term",
        "classification": "internal",
        "required_clearance": 0,
        "title": None,
        "content_preview": content_preview,
        "tags": [],
        "entities": {},
        "extra_metadata": {},
        "source_type": None,
        "source_id": None,
        "access_count": 0,
        "last_accessed_at": None,
        "is_promoted": False,
        "created_at": now,
        "updated_at": now,
    }


@pytest.mark.asyncio
async def test_batch_update_returns_per_item_results(client, auth_headers, test_org_id, test_user_id):
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    class StubMemoryService:
        async def update_memory(self, memory_id, data, new_embedding=None, request_id=None):
            if memory_id == "m_denied":
                raise PermissionError("no")
            return SimpleNamespace(
                **_memory_response_dict(
                    memory_id=memory_id,
                    org_id=test_org_id,
                    owner_id=test_user_id,
                    content_preview="preview",
                )
            )

    app.dependency_overrides[get_memory_service] = lambda: StubMemoryService()

    resp = await client.post(
        "/api/v1/memories/batch/update",
        headers=auth_headers,
        json={
            "items": [
                {"memory_id": "m_ok", "update": {"title": "t"}},
                {"memory_id": "m_denied", "update": {"title": "t"}},
            ]
        },
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["memory_id"] == "m_ok"
    assert data["results"][0]["success"] is True
    assert data["results"][0]["memory"]["id"] == "m_ok"
    assert data["results"][1]["memory_id"] == "m_denied"
    assert data["results"][1]["success"] is False
    assert data["results"][1]["error"]

    assert session.commit.await_count == 1

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_batch_delete_commits_only_if_any_success(client, auth_headers):
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    class StubMemoryService:
        async def delete_memory(self, memory_id, request_id=None):
            raise PermissionError("no")

    app.dependency_overrides[get_memory_service] = lambda: StubMemoryService()

    resp = await client.post(
        "/api/v1/memories/batch/delete",
        headers=auth_headers,
        json={"memory_ids": ["m1", "m2"]},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert all(r["success"] is False for r in data["results"])
    assert session.commit.await_count == 0
    assert session.rollback.await_count == 1

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_batch_share_returns_share_ids(client, auth_headers):
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    class StubMemoryService:
        async def share_memory(self, memory_id, request, request_id=None):
            return SimpleNamespace(id=f"s_{memory_id}")

    app.dependency_overrides[get_memory_service] = lambda: StubMemoryService()

    resp = await client.post(
        "/api/v1/memories/batch/share",
        headers=auth_headers,
        json={
            "memory_ids": ["m1", "m2"],
            "share": {"share_type": "user", "target_id": "u2", "permission": "read"},
        },
    )

    assert resp.status_code == 201
    data = resp.json()
    assert data["results"][0]["share_id"] == "s_m1"
    assert data["results"][1]["share_id"] == "s_m2"
    assert session.commit.await_count == 1

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_update_memory_content_persists_and_reenqueues(client, auth_headers, test_org_id, test_user_id, monkeypatch):
    session = AsyncMock()
    session.commit = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    captured: dict = {}

    class StubMemoryService:
        async def update_memory(self, memory_id, data, new_embedding=None, request_id=None):
            captured["memory_id"] = memory_id
            captured["content"] = data.content
            captured["request_id"] = request_id
            payload = _memory_response_dict(
                memory_id=memory_id,
                org_id=test_org_id,
                owner_id=test_user_id,
                content_preview=(data.content or "")[:2000] or "preview",
            )
            payload["title"] = "Updated"
            now = datetime.now(timezone.utc)
            payload["created_at"] = now
            payload["updated_at"] = now
            return SimpleNamespace(**payload)

    app.dependency_overrides[get_memory_service] = lambda: StubMemoryService()

    emitted: list[tuple[str, dict]] = []

    class FakeWebhookService:
        def __init__(self, db):
            self.db = db

        async def emit_event(self, *, organization_id: str, event_type: str, payload: dict) -> None:
            emitted.append((event_type, payload))

    monkeypatch.setattr(memories_endpoints, "WebhookService", FakeWebhookService)

    embed_calls: list[dict] = []
    memory_pipeline_calls: list[dict] = []
    episode_pipeline_calls: list[dict] = []
    fact_pipeline_calls: list[dict] = []
    monkeypatch.setattr(memories_endpoints, "enqueue_embed_and_index", lambda **kwargs: embed_calls.append(kwargs))
    monkeypatch.setattr(memories_endpoints, "enqueue_memory_pipeline", lambda **kwargs: memory_pipeline_calls.append(kwargs))
    monkeypatch.setattr(memories_endpoints, "enqueue_episode_pipeline", lambda **kwargs: episode_pipeline_calls.append(kwargs))
    monkeypatch.setattr(memories_endpoints, "enqueue_fact_pipeline", lambda **kwargs: fact_pipeline_calls.append(kwargs))

    resp = await client.patch(
        "/api/v1/memories/m-updated",
        headers=auth_headers,
        json={"content": "Updated memory body"},
    )

    assert resp.status_code == 200, resp.text
    assert captured["memory_id"] == "m-updated"
    assert captured["content"] == "Updated memory body"
    assert session.commit.await_count == 2
    assert emitted and emitted[0][0] == "memory.updated"
    assert embed_calls == [
        {
            "memory_id": "m-updated",
            "content": "Updated memory body",
            "org_id": test_org_id,
        }
    ]
    assert len(memory_pipeline_calls) == 1
    assert len(episode_pipeline_calls) == 1
    assert len(fact_pipeline_calls) == 1
    assert memory_pipeline_calls[0]["memory_id"] == "m-updated"
    assert episode_pipeline_calls[0]["memory_id"] == "m-updated"
    assert fact_pipeline_calls[0]["memory_id"] == "m-updated"

    app.dependency_overrides.clear()
