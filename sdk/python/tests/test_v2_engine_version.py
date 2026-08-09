"""
Tests for v2 engine_version parameter support in the Ninai SDK.

Verifies:
- NinaiClient accepts engine_version in constructor
- client.v2 is a V2EngineResource
- memories.create() routes to /v2/interact when engine_version="v2"
- memories.search() routes to /v2/interact when engine_version="v2"
- Per-call engine_version overrides the client default
- v1 paths are unchanged
- V2InteractResult is returned for v2 calls
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Ensure the SDK package is importable
sdk_root = Path(__file__).parent.parent
if str(sdk_root) not in sys.path:
    sys.path.insert(0, str(sdk_root))

from ninai import NinaiClient
from ninai.models import Memory, SearchResult, V2InteractResult, V2GraphInspectResult, V2HealthResult
from ninai.resources import V2EngineResource, MemoriesResource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_V2_INTERACT_RESPONSE = {
    "response": "Project Alpha deadline is Friday.",
    "session_id": "sess-test",
    "user_utterance_id": "utt-u1",
    "assistant_utterance_id": "utt-a1",
    "cited_node_ids": ["e1", "e2"],
    "extracted_entities": [{"id": "project_alpha", "name": "Project Alpha", "type": "task"}],
    "graph_nodes_retrieved": 3,
    "qdrant_chunks_retrieved": 1,
    "graph_writes": 5,
    "decay_stats": {"decayed": 4, "pruned": 1},
    "latency_ms": 312,
    "error": "",
}

_V1_MEMORY_RESPONSE = {
    "id": "mem-001",
    "content": "meeting notes",
    "title": "Q4 meeting",
    "scope": "personal",
    "memory_type": "long_term",
    "classification": "internal",
    "tags": [],
    "entities": {},
    "extra_metadata": {},
    "organization_id": "org-001",
    "owner_id": "user-001",
    "created_at": "2026-05-21T10:00:00Z",
    "updated_at": "2026-05-21T10:00:00Z",
}

_V1_SEARCH_RESPONSE = {
    "items": [],
    "total": 0,
    "query": "hello",
}

_V2_GRAPH_INSPECT_RESPONSE = {
    "nodes": [
        {"id": "e1", "label": "Entity", "content": "Project Alpha", "weight": 0.8, "created_at": 1000},
    ],
    "seed_count": 1,
    "tenant_id": "t1",
}

_V2_HEALTH_RESPONSE = {
    "engine_version": "v2",
    "graph_available": True,
    "llm_available": True,
    "message": "all systems operational",
}


def _make_client(engine_version: str = "v1") -> NinaiClient:
    client = NinaiClient(
        api_key="nai_test",
        base_url="http://localhost:8000/api/v1",
        engine_version=engine_version,
    )
    return client


# ---------------------------------------------------------------------------
# NinaiClient constructor
# ---------------------------------------------------------------------------

class TestNinaiClientConstructor:
    def test_default_engine_version_is_v1(self):
        client = _make_client()
        assert client.engine_version == "v1"

    def test_engine_version_v2_accepted(self):
        client = _make_client(engine_version="v2")
        assert client.engine_version == "v2"

    def test_client_v2_resource_exists(self):
        client = _make_client()
        assert hasattr(client, "v2")
        assert isinstance(client.v2, V2EngineResource)

    def test_v2_resource_available_regardless_of_default_version(self):
        for version in ("v1", "v2"):
            client = _make_client(engine_version=version)
            assert isinstance(client.v2, V2EngineResource)

    def test_memories_resource_exists(self):
        client = _make_client()
        assert isinstance(client.memories, MemoriesResource)


# ---------------------------------------------------------------------------
# memories.create() routing
# ---------------------------------------------------------------------------

class TestMemoriesCreateRouting:
    def test_v1_default_posts_to_memories(self):
        client = _make_client(engine_version="v1")
        with patch.object(client, "_post", return_value=_V1_MEMORY_RESPONSE) as mock_post:
            result = client.memories.create(content="meeting notes")
        mock_post.assert_called_once()
        path, *_ = mock_post.call_args.args
        assert path == "/memories"
        assert isinstance(result, Memory)

    def test_v2_constructor_routes_to_v2_interact(self):
        client = _make_client(engine_version="v2")
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            result = client.memories.create(content="Project Alpha starts Monday")
        mock_post.assert_called_once()
        path = mock_post.call_args.args[0]
        assert path == "/v2/interact"
        assert isinstance(result, V2InteractResult)
        assert result.response == "Project Alpha deadline is Friday."

    def test_per_call_v2_overrides_v1_default(self):
        client = _make_client(engine_version="v1")
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            result = client.memories.create(
                content="Project Alpha update",
                engine_version="v2",
            )
        path = mock_post.call_args.args[0]
        assert path == "/v2/interact"
        assert isinstance(result, V2InteractResult)

    def test_per_call_v1_overrides_v2_default(self):
        client = _make_client(engine_version="v2")
        with patch.object(client, "_post", return_value=_V1_MEMORY_RESPONSE) as mock_post:
            result = client.memories.create(
                content="meeting notes",
                engine_version="v1",
            )
        path = mock_post.call_args.args[0]
        assert path == "/memories"
        assert isinstance(result, Memory)

    def test_v2_payload_contains_user_input(self):
        client = _make_client(engine_version="v2")
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            client.memories.create(content="hello world", engine_version="v2")
        payload = mock_post.call_args.kwargs.get("json", {})
        assert payload["user_input"] == "hello world"
        assert "session_id" in payload

    def test_v2_uses_provided_session_id(self):
        client = _make_client(engine_version="v2")
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            client.memories.create(
                content="hello", engine_version="v2", session_id="my-session"
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["session_id"] == "my-session"

    def test_v2_auto_generates_session_id_when_omitted(self):
        client = _make_client(engine_version="v2")
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            client.memories.create(content="hello", engine_version="v2")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["session_id"] != ""
        assert len(payload["session_id"]) > 0


# ---------------------------------------------------------------------------
# memories.search() routing
# ---------------------------------------------------------------------------

class TestMemoriesSearchRouting:
    def test_v1_default_gets_memories_search(self):
        client = _make_client(engine_version="v1")
        with patch.object(client, "_get", return_value=_V1_SEARCH_RESPONSE) as mock_get:
            result = client.memories.search("project alpha")
        mock_get.assert_called_once()
        path = mock_get.call_args.args[0]
        assert path == "/memories/search"
        assert isinstance(result, SearchResult)

    def test_v2_constructor_routes_to_v2_interact(self):
        client = _make_client(engine_version="v2")
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            result = client.memories.search("project alpha deadline")
        path = mock_post.call_args.args[0]
        assert path == "/v2/interact"
        assert isinstance(result, V2InteractResult)
        assert "e1" in result.cited_node_ids

    def test_per_call_v2_overrides_v1_default(self):
        client = _make_client(engine_version="v1")
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            result = client.memories.search("query", engine_version="v2")
        path = mock_post.call_args.args[0]
        assert path == "/v2/interact"
        assert isinstance(result, V2InteractResult)

    def test_per_call_v1_overrides_v2_default(self):
        client = _make_client(engine_version="v2")
        with patch.object(client, "_get", return_value=_V1_SEARCH_RESPONSE) as mock_get:
            result = client.memories.search("query", engine_version="v1")
        path = mock_get.call_args.args[0]
        assert path == "/memories/search"
        assert isinstance(result, SearchResult)

    def test_v2_payload_uses_query_as_user_input(self):
        client = _make_client(engine_version="v2")
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            client.memories.search("project deadline", engine_version="v2")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["user_input"] == "project deadline"

    def test_v2_search_uses_provided_session_id(self):
        client = _make_client(engine_version="v2")
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            client.memories.search("q", engine_version="v2", session_id="sess-abc")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["session_id"] == "sess-abc"


# ---------------------------------------------------------------------------
# client.v2 resource
# ---------------------------------------------------------------------------

class TestV2EngineResource:
    def test_interact_posts_to_v2_interact(self):
        client = _make_client()
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            result = client.v2.interact("What is the deadline?", session_id="sess-1")
        path = mock_post.call_args.args[0]
        assert path == "/v2/interact"
        assert isinstance(result, V2InteractResult)

    def test_interact_includes_prev_utterance_id(self):
        client = _make_client()
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            client.v2.interact("follow-up", session_id="s1", prev_utterance_id="utt-prev")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["prev_utterance_id"] == "utt-prev"

    def test_interact_auto_generates_session_id(self):
        client = _make_client()
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE) as mock_post:
            client.v2.interact("hello")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["session_id"] != ""

    def test_interact_returns_cited_nodes(self):
        client = _make_client()
        with patch.object(client, "_post", return_value=_V2_INTERACT_RESPONSE):
            result = client.v2.interact("deadline?")
        assert result.cited_node_ids == ["e1", "e2"]
        assert result.graph_nodes_retrieved == 3

    def test_graph_inspect_posts_to_v2_graph_inspect(self):
        client = _make_client()
        with patch.object(client, "_post", return_value=_V2_GRAPH_INSPECT_RESPONSE) as mock_post:
            result = client.v2.graph_inspect(entity_ids=["e1", "e2"], hops=2)
        path = mock_post.call_args.args[0]
        assert path == "/v2/graph/inspect"
        assert isinstance(result, V2GraphInspectResult)
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "e1"

    def test_graph_inspect_payload(self):
        client = _make_client()
        with patch.object(client, "_post", return_value=_V2_GRAPH_INSPECT_RESPONSE) as mock_post:
            client.v2.graph_inspect(entity_ids=["x"], hops=3, limit=50)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["entity_ids"] == ["x"]
        assert payload["hops"] == 3
        assert payload["limit"] == 50

    def test_health_gets_v2_health(self):
        client = _make_client()
        with patch.object(client, "_get", return_value=_V2_HEALTH_RESPONSE) as mock_get:
            result = client.v2.health()
        path = mock_get.call_args.args[0]
        assert path == "/v2/health"
        assert isinstance(result, V2HealthResult)
        assert result.graph_available is True
        assert result.llm_available is True


# ---------------------------------------------------------------------------
# V2InteractResult model
# ---------------------------------------------------------------------------

class TestV2InteractResultModel:
    def test_all_fields_populated(self):
        r = V2InteractResult(**_V2_INTERACT_RESPONSE)
        assert r.response == "Project Alpha deadline is Friday."
        assert r.cited_node_ids == ["e1", "e2"]
        assert r.graph_writes == 5
        assert r.latency_ms == 312
        assert r.error == ""

    def test_defaults_are_safe(self):
        r = V2InteractResult()
        assert r.response == ""
        assert r.cited_node_ids == []
        assert r.extracted_entities == []
        assert r.decay_stats == {}

    def test_extracted_entities_structure(self):
        r = V2InteractResult(**_V2_INTERACT_RESPONSE)
        assert len(r.extracted_entities) == 1
        assert r.extracted_entities[0]["name"] == "Project Alpha"
