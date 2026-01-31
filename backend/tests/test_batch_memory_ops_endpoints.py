from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.database import get_db
from app.main import app
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
