from __future__ import annotations

import asyncio
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
async def test_create_memory_enqueues_tenant_roles_and_clearance(client, auth_headers, test_org_id, test_user_id, monkeypatch):
    embed_mock = AsyncMock(return_value=[0.1, 0.2, 0.3])
    monkeypatch.setattr(memories_endpoints.EmbeddingService, "embed", embed_mock)

    captured_calls: list[tuple[str, dict]] = []

    def _capture(name: str):
        def _inner(**kwargs):
            captured_calls.append((name, kwargs))
            return None

        return _inner

    monkeypatch.setattr(memories_endpoints, "enqueue_memory_pipeline", _capture("memory"))
    monkeypatch.setattr(memories_endpoints, "enqueue_episode_pipeline", _capture("episode"))
    monkeypatch.setattr(memories_endpoints, "enqueue_fact_pipeline", _capture("fact"))

    class StubMemoryService:
        async def create_memory(self, data, embedding, request_id=None, actor_ctx=None):
            return SimpleNamespace(
                **_memory_response_dict(
                    memory_id="m-enqueue",
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
            "content": "hello tenant context",
            "scope": "personal",
            "memory_type": "long_term",
            "classification": "internal",
        },
    )

    assert resp.status_code == 201, resp.text
    assert [name for name, _ in captured_calls] == ["memory", "episode", "fact"]
    for _, kwargs in captured_calls:
        assert kwargs["initiator_user_id"] == test_user_id
        assert kwargs["initiator_roles"] == "org_admin"
        assert kwargs["initiator_clearance_level"] == 0


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
async def test_search_memories_falls_back_when_query_embedding_times_out(
    client, auth_headers, test_org_id, test_user_id, monkeypatch
):
    async def _raise_timeout(_query: str):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(memories_endpoints.EmbeddingService, "embed", _raise_timeout)
    monkeypatch.setattr(memories_endpoints.settings, "EMBEDDING_DIMENSIONS", 3)

    captured = {"embedding": None}

    class StubMemoryService:
        async def search_memories(self, query_embedding, request, request_id=None):
            captured["embedding"] = query_embedding
            return [
                SimpleNamespace(
                    **_memory_response_dict(
                        memory_id="m-timeout",
                        org_id=test_org_id,
                        owner_id=test_user_id,
                        content_preview="preview",
                    )
                )
            ]

        def get_search_ranking_meta(self, request):
            return {"hnms_mode_effective": "balanced"}

        def get_last_search_diagnostics(self):
            return {}

    app.dependency_overrides[get_memory_service] = lambda: StubMemoryService()

    resp = await client.get(
        "/api/v1/memories/search",
        headers=auth_headers,
        params={"query": "find this", "limit": 5},
    )

    assert resp.status_code == 200, resp.text
    assert captured["embedding"] == [0.0, 0.0, 0.0]
    data = resp.json()
    assert data["ranking_meta"]["embedding_fallback_reason"] == "TimeoutError"
    assert data["ranking_meta"]["query_embed_timeout_seconds"] == 5.0


@pytest.mark.asyncio
async def test_search_memories_recovers_with_fresh_session_when_primary_session_breaks(
    client, auth_headers, test_org_id, test_user_id, monkeypatch
):
    monkeypatch.setattr(memories_endpoints.EmbeddingService, "embed", AsyncMock(return_value=[0.4, 0.5, 0.6]))

    class PrimaryFailService:
        async def search_memories(self, query_embedding, request, request_id=None):
            raise RuntimeError("connection is closed")

        def get_search_ranking_meta(self, request):
            return {"hnms_mode_effective": "balanced"}

        def get_last_search_diagnostics(self):
            return {}

    class FreshSessionService:
        def __init__(self, session=None, user_id=None, org_id=None, clearance_level=0):
            self.permission_checker = AsyncMock()

        async def search_memories(self, query_embedding, request, request_id=None):
            return [
                SimpleNamespace(
                    **_memory_response_dict(
                        memory_id="m-fresh",
                        org_id=test_org_id,
                        owner_id=test_user_id,
                        content_preview="fresh-session-preview",
                    )
                )
            ]

    class _FreshSessionContext:
        def __init__(self, session):
            self._session = session

        async def __aenter__(self):
            return self._session

        async def __aexit__(self, exc_type, exc, tb):
            return False

    fresh_db = AsyncMock()
    monkeypatch.setattr(memories_endpoints, "MemoryService", FreshSessionService)
    monkeypatch.setattr(memories_endpoints, "set_tenant_context", AsyncMock())
    monkeypatch.setattr(
        memories_endpoints,
        "async_session_factory",
        lambda: _FreshSessionContext(fresh_db),
    )

    app.dependency_overrides[get_memory_service] = lambda: PrimaryFailService()

    resp = await client.get(
        "/api/v1/memories/search",
        headers=auth_headers,
        params={"query": "find this", "limit": 5},
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] == 1
    assert data["results"][0]["id"] == "m-fresh"
    assert data["ranking_meta"]["search_fallback_reason"] == "db_connection_closed:fresh_session_retry"


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
async def test_search_memories_normalizes_comma_delimited_tags(
    client, auth_headers, test_org_id, test_user_id, monkeypatch
):
    monkeypatch.setattr(memories_endpoints.EmbeddingService, "embed", AsyncMock(return_value=[0.4, 0.5, 0.6]))

    captured = {"tags": None}

    class StubMemoryService:
        async def search_memories(self, query_embedding, request, request_id=None):
            captured["tags"] = request.tags
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

        def get_last_search_diagnostics(self):
            return {}

    app.dependency_overrides[get_memory_service] = lambda: StubMemoryService()

    resp = await client.get(
        "/api/v1/memories/search",
        headers=auth_headers,
        params={"query": "find this", "tags": "locomo_001,locomo-full-abc"},
    )

    assert resp.status_code == 200, resp.text
    assert captured["tags"] == ["locomo_001", "locomo-full-abc"]


@pytest.mark.asyncio
async def test_list_memories_normalizes_comma_delimited_tags(client, auth_headers):
    captured = {"tags": None}

    class StubMemoryService:
        async def list_memories(self, scope=None, tags=None, memory_type=None, page=1, page_size=20):
            captured["tags"] = tags
            return [], 0, False

    app.dependency_overrides[get_memory_service] = lambda: StubMemoryService()

    resp = await client.get(
        "/api/v1/memories",
        headers=auth_headers,
        params={"tags": "locomo_001,locomo-full-abc"},
    )

    assert resp.status_code == 200, resp.text
    assert captured["tags"] == ["locomo_001", "locomo-full-abc"]


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

    queued = []

    class _ChainStub:
        def __init__(self, agent_names, kwargs):
            self.agent_names = agent_names
            self.kwargs = kwargs

        def apply_async(self):
            queued.append((self.agent_names, self.kwargs))

    def _build_canvas(*, agent_names, **kwargs):
        return _ChainStub(tuple(agent_names), kwargs)

    monkeypatch.setattr(memories_endpoints, "build_memory_enrichment_canvas", _build_canvas)

    resp = await client.post(
        "/api/v1/memories/enrich/bulk",
        headers=auth_headers,
        params={"limit": 10},
    )

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["queued_count"] == 2
    assert len(queued) == 2
    assert queued[0][0] == (
        "entity_resolution",
        "world_model",
        "temporal_reasoning",
        "episodic_grouping",
        "causal_reasoning",
    )
