from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.main import app
from app.api.v1.endpoints.memories import get_memory_service
import app.api.v1.endpoints.memories as memories_endpoints
from app.core.database import get_db


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
        async def create_memory(self, data, embedding, request_id=None, actor_ctx=None):
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
async def test_create_memory_accepts_occurred_at(client, auth_headers, test_org_id, test_user_id, monkeypatch):
    embed_mock = AsyncMock(return_value=[0.1, 0.2, 0.3])
    monkeypatch.setattr(memories_endpoints.EmbeddingService, "embed", embed_mock)

    captured = {"occurred_at": None}
    occurred_at_iso = "2026-04-15T09:02:00Z"

    class StubMemoryService:
        async def create_memory(self, data, embedding, request_id=None, actor_ctx=None):
            captured["occurred_at"] = data.occurred_at
            payload = _memory_response_dict(
                memory_id="m-ts",
                org_id=test_org_id,
                owner_id=test_user_id,
                content_preview=(data.content or "")[:200],
            )
            payload["occurred_at"] = data.occurred_at
            return SimpleNamespace(**payload)

    app.dependency_overrides[get_memory_service] = lambda: StubMemoryService()

    resp = await client.post(
        "/api/v1/memories",
        headers=auth_headers,
        json={
            "content": "incident detected",
            "scope": "personal",
            "memory_type": "long_term",
            "classification": "internal",
            "occurred_at": occurred_at_iso,
        },
    )

    assert resp.status_code == 201, resp.text
    assert captured["occurred_at"] is not None
    assert captured["occurred_at"].isoformat() == "2026-04-15T09:02:00+00:00"

    body = resp.json()
    assert body["occurred_at"].startswith("2026-04-15T09:02:00")


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


@pytest.mark.asyncio
async def test_search_memories_passes_use_graph_flag(client, auth_headers, test_org_id, test_user_id, monkeypatch):
    monkeypatch.setattr(memories_endpoints.EmbeddingService, "embed", AsyncMock(return_value=[0.4, 0.5, 0.6]))

    captured = {"use_graph": None}

    class StubMemoryService:
        async def search_memories(self, query_embedding, request, request_id=None):
            captured["use_graph"] = request.use_graph
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
        params={"query": "find this", "limit": 5, "use_graph": "true"},
    )

    assert resp.status_code == 200, resp.text
    assert captured["use_graph"] is True


@pytest.mark.asyncio
async def test_bulk_enrich_memories_queues_tasks(client, auth_headers, monkeypatch):
    class _Scalars:
        def __init__(self, items):
            self._items = items

        def all(self):
            return self._items

    class _Result:
        def __init__(self, items):
            self._items = items

        def scalars(self):
            return _Scalars(self._items)

    memories = [
        SimpleNamespace(id="m-1", memory_type="long_term"),
        SimpleNamespace(id="m-2", memory_type="long_term"),
    ]

    db = AsyncMock()

    async def _execute(stmt, *args, **kwargs):
        text = str(stmt)
        if "FROM memory_metadata" in text:
            return _Result(memories)
        return _Result([])

    db.execute = AsyncMock(side_effect=_execute)

    async def _override_db():
        yield db

    app.dependency_overrides[get_db] = _override_db

    captured_signatures = []
    queued = []

    class _TaskStub:
        def __init__(self, name):
            self.name = name

        def si(self, **kwargs):
            captured_signatures.append((self.name, kwargs))
            return (self.name, kwargs)

    class _ChainStub:
        def __init__(self, *signatures):
            self.signatures = signatures

        def apply_async(self):
            queued.append(self.signatures)

    monkeypatch.setattr("app.tasks.memory_pipeline.entity_resolution_task", _TaskStub("entity_resolution"))
    monkeypatch.setattr("app.tasks.memory_pipeline.world_model_task", _TaskStub("world_model"))
    monkeypatch.setattr("app.tasks.memory_pipeline.temporal_reasoning_task", _TaskStub("temporal_reasoning"))
    monkeypatch.setattr("app.tasks.memory_pipeline.episodic_grouping_task", _TaskStub("episodic_grouping"))
    monkeypatch.setattr("app.tasks.memory_pipeline.causal_reasoning_task", _TaskStub("causal_reasoning"))
    monkeypatch.setattr("celery.chain", lambda *signatures: _ChainStub(*signatures))

    resp = await client.post(
        "/api/v1/memories/enrich/bulk",
        headers=auth_headers,
        params={"limit": 10},
    )

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["queued_count"] == 2
    assert len(queued) == 2
    assert captured_signatures
