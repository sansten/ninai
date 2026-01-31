from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.main import app
from app.api.v1.endpoints.memories import get_memory_service
import app.api.v1.endpoints.memories as memories_endpoints


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
async def test_create_memory_uses_embedding_service(client, auth_headers, test_org_id, test_user_id, monkeypatch):
    embed_mock = AsyncMock(return_value=[0.1, 0.2, 0.3])
    monkeypatch.setattr(memories_endpoints.EmbeddingService, "embed", embed_mock)

    captured = {"embedding": None, "content": None}

    class StubMemoryService:
        async def create_memory(self, data, embedding, request_id=None):
            captured["embedding"] = embedding
            captured["content"] = data.content
            return SimpleNamespace(
                **_memory_response_dict(
                    memory_id="m1",
                    org_id=test_org_id,
                    owner_id=test_user_id,
                    content_preview=(data.content or "")[:200],
                )
            )

    app.dependency_overrides[get_memory_service] = lambda: StubMemoryService()

    resp = await client.post(
        "/api/v1/memories",
        headers=auth_headers,
        json={
            "content": "hello world",
            "scope": "personal",
            "memory_type": "long_term",
            "classification": "internal",
        },
    )

    assert resp.status_code == 201, resp.text
    embed_mock.assert_awaited_once()
    assert embed_mock.await_args.args[0] == "hello world"
    assert captured["content"] == "hello world"
    assert captured["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_search_memories_uses_embedding_service(client, auth_headers, test_org_id, test_user_id, monkeypatch):
    embed_mock = AsyncMock(return_value=[0.4, 0.5, 0.6])
    monkeypatch.setattr(memories_endpoints.EmbeddingService, "embed", embed_mock)

    captured = {"embedding": None, "query": None}

    class StubMemoryService:
        async def search_memories(self, query_embedding, request, request_id=None):
            captured["embedding"] = query_embedding
            captured["query"] = request.query
            return [
                SimpleNamespace(
                    **_memory_response_dict(
                        memory_id="m1",
                        org_id=test_org_id,
                        owner_id=test_user_id,
                        content_preview="preview",
                    )
                )
            ]

        def get_search_ranking_meta(self, request):
            return {"hnms_mode_effective": "balanced"}

    app.dependency_overrides[get_memory_service] = lambda: StubMemoryService()

    resp = await client.get(
        "/api/v1/memories/search",
        headers=auth_headers,
        params={"query": "find this", "limit": 5},
    )

    assert resp.status_code == 200, resp.text
    embed_mock.assert_awaited_once()
    assert embed_mock.await_args.args[0] == "find this"
    assert captured["query"] == "find this"
    assert captured["embedding"] == [0.4, 0.5, 0.6]

    data = resp.json()
    assert data["query"] == "find this"
    assert data["total"] == 1
    assert len(data["results"]) == 1
    assert data.get("ranking_meta")
