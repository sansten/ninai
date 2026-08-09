from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.services.memory_service as memory_service_module
from app.schemas.memory import MemorySearchRequest
from app.services.memory_service import MemoryService


class _LexResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


def _execute_result_with_scalars(rows: list) -> MagicMock:
    result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = rows
    result.scalars.return_value = scalars
    return result


@pytest.mark.asyncio
async def test_core_search_falls_back_to_lexical_when_vector_fails(monkeypatch):
    monkeypatch.setattr(
        memory_service_module.QdrantService,
        "search",
        AsyncMock(side_effect=RuntimeError("qdrant down")),
    )
    monkeypatch.setattr(
        memory_service_module.settings,
        "SEARCH_ALLOW_LEXICAL_FALLBACK_ON_VECTOR_ERROR",
        True,
        raising=False,
    )

    now = datetime.now(timezone.utc)
    mem = SimpleNamespace(
        id="m1",
        is_active=True,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        content_hash="h1",
        title="doc",
        content_preview="doc preview",
        vector_id="v1",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
    )

    async def _execute(stmt):
        stmt_str = str(stmt)
        if "ts_rank" in stmt_str and "FROM memory_metadata" in stmt_str:
            return _LexResult([("m1", 0.42)])
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars([mem])
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute
    session.add = MagicMock()
    session.add_all = MagicMock()

    svc = MemoryService(session=session, user_id="user", org_id="org", clearance_level=0)
    svc.permission_checker.filter_memory_ids_with_access = AsyncMock(return_value=["m1"])

    req = MemorySearchRequest(query="error code 404", limit=10, hybrid=True)
    results = await svc.search_memories(query_embedding=[0.1, 0.2, 0.3], request=req, request_id="rid")

    assert [m.id for m in results] == ["m1"]


@pytest.mark.asyncio
async def test_core_search_expands_query_variants(monkeypatch):
    qdrant_search = AsyncMock(
        return_value=[{"id": "v1", "score": 0.95, "payload": {"memory_id": "m1"}}]
    )
    monkeypatch.setattr(memory_service_module.QdrantService, "search", qdrant_search)
    monkeypatch.setattr(
        memory_service_module.EmbeddingService,
        "embed",
        AsyncMock(return_value=[0.1, 0.2, 0.3]),
    )
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_QUERY_EXPANSION_ENABLED", True, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_QUERY_EXPANSION_MAX_VARIANTS", 2, raising=False)

    now = datetime.now(timezone.utc)
    mem = SimpleNamespace(
        id="m1",
        is_active=True,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        content_hash="h1",
        title="doc",
        content_preview="doc preview",
        vector_id="v1",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
    )

    async def _execute(stmt):
        stmt_str = str(stmt)
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars([mem])
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute
    session.add = MagicMock()
    session.add_all = MagicMock()

    svc = MemoryService(session=session, user_id="user", org_id="org", clearance_level=0)
    svc.permission_checker.filter_memory_ids_with_access = AsyncMock(return_value=["m1"])

    req = MemorySearchRequest(query="USA AI roadmap", limit=10, hybrid=False)
    results = await svc.search_memories(query_embedding=[0.1, 0.2, 0.3], request=req, request_id="rid")

    assert [m.id for m in results] == ["m1"]
    assert qdrant_search.await_count >= 2


@pytest.mark.asyncio
async def test_core_search_auto_hybrid_activates_lexical_when_vector_sparse(monkeypatch):
    qdrant_search = AsyncMock(
        return_value=[{"id": "v1", "score": 0.11, "payload": {"memory_id": "m1"}}]
    )
    monkeypatch.setattr(memory_service_module.QdrantService, "search", qdrant_search)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_AUTO_HYBRID_ENABLED", True, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_AUTO_HYBRID_MIN_VECTOR_HITS", 4, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_AUTO_HYBRID_MAX_TOP_SCORE", 0.55, raising=False)

    now = datetime.now(timezone.utc)
    mem = SimpleNamespace(
        id="m1",
        is_active=True,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        content_hash="h1",
        title="doc",
        content_preview="doc preview",
        vector_id="v1",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
        extra_metadata={"session_id": "s1", "speaker": "Caroline"},
    )

    lexical_seen = {"called": False}

    async def _execute(stmt):
        stmt_str = str(stmt)
        if "ts_rank" in stmt_str and "FROM memory_metadata" in stmt_str:
            lexical_seen["called"] = True
            return _LexResult([("m1", 0.51)])
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars([mem])
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute
    session.add = MagicMock()
    session.add_all = MagicMock()

    svc = MemoryService(session=session, user_id="user", org_id="org", clearance_level=0)
    svc.permission_checker.filter_memory_ids_with_access = AsyncMock(return_value=["m1"])

    req = MemorySearchRequest(query="what happened and why", limit=10, hybrid=False)
    results = await svc.search_memories(query_embedding=[0.1, 0.2, 0.3], request=req, request_id="rid")

    assert [m.id for m in results] == ["m1"]
    assert lexical_seen["called"] is True
    diagnostics = svc.get_last_search_diagnostics()
    assert diagnostics.get("lexical_mode") == "auto"


@pytest.mark.asyncio
async def test_core_search_multihop_subqueries_trigger_extra_vector_calls(monkeypatch):
    qdrant_search = AsyncMock(
        return_value=[{"id": "v1", "score": 0.88, "payload": {"memory_id": "m1"}}]
    )
    monkeypatch.setattr(memory_service_module.QdrantService, "search", qdrant_search)
    monkeypatch.setattr(
        memory_service_module.EmbeddingService,
        "embed",
        AsyncMock(return_value=[0.1, 0.2, 0.3]),
    )
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_QUERY_EXPANSION_ENABLED", False, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_MULTI_HOP_SUBQUERY_ENABLED", True, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_MULTI_HOP_MAX_SUBQUERIES", 3, raising=False)

    now = datetime.now(timezone.utc)
    mem = SimpleNamespace(
        id="m1",
        is_active=True,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        content_hash="h1",
        title="doc",
        content_preview="doc preview",
        vector_id="v1",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
        extra_metadata={"session_id": "s2", "speaker": "John"},
    )

    async def _execute(stmt):
        stmt_str = str(stmt)
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars([mem])
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute
    session.add = MagicMock()
    session.add_all = MagicMock()

    svc = MemoryService(session=session, user_id="user", org_id="org", clearance_level=0)
    svc.permission_checker.filter_memory_ids_with_access = AsyncMock(return_value=["m1"])

    req = MemorySearchRequest(query="what happened before and after the launch because errors increased", limit=10)
    results = await svc.search_memories(query_embedding=[0.1, 0.2, 0.3], request=req, request_id="rid")

    assert [m.id for m in results] == ["m1"]
    assert qdrant_search.await_count >= 2
    diagnostics = svc.get_last_search_diagnostics()
    assert int(diagnostics.get("multihop_subqueries_used", 0)) >= 1


@pytest.mark.asyncio
async def test_core_search_handles_graph_expansion_error_without_dropping_vector_hits(monkeypatch):
    monkeypatch.setattr(
        memory_service_module.QdrantService,
        "search",
        AsyncMock(return_value=[{"id": "v1", "score": 0.95, "payload": {"memory_id": "m1"}}]),
    )
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_QUERY_EXPANSION_ENABLED", False, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_MULTI_HOP_SUBQUERY_ENABLED", False, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_GRAPH_EXPANSION_ENABLED", True, raising=False)

    now = datetime.now(timezone.utc)
    mem = SimpleNamespace(
        id="m1",
        is_active=True,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        content_hash="h1",
        title="doc",
        content_preview="doc preview",
        vector_id="v1",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
        extra_metadata={"session_id": "s3", "speaker": "Caroline"},
    )

    async def _execute(stmt):
        stmt_str = str(stmt)
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars([mem])
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute
    session.add = MagicMock()
    session.add_all = MagicMock()

    svc = MemoryService(session=session, user_id="user", org_id="org", clearance_level=0)
    svc.permission_checker.filter_memory_ids_with_access = AsyncMock(return_value=["m1"])
    monkeypatch.setattr(
        svc,
        "_load_graph_neighbor_scores",
        AsyncMock(side_effect=RuntimeError("graph join failed")),
    )

    req = MemorySearchRequest(query="Where did Caroline move from?", limit=10, use_graph=True)
    results = await svc.search_memories(query_embedding=[0.1, 0.2, 0.3], request=req, request_id="rid")

    assert [m.id for m in results] == ["m1"]
    diagnostics = svc.get_last_search_diagnostics()
    assert diagnostics.get("fallback_reason") in {
        "vector_partial_error_degraded",
        "auto_hybrid_low_vector_confidence",
    }


@pytest.mark.asyncio
async def test_core_search_records_retrieval_diagnostics(monkeypatch):
    monkeypatch.setattr(
        memory_service_module.QdrantService,
        "search",
        AsyncMock(return_value=[{"id": "v1", "score": 0.95, "payload": {"memory_id": "m1"}}]),
    )
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_QUERY_EXPANSION_ENABLED", False, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_MULTI_HOP_SUBQUERY_ENABLED", False, raising=False)

    now = datetime.now(timezone.utc)
    mem = SimpleNamespace(
        id="m1",
        is_active=True,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        content_hash="h1",
        title="doc",
        content_preview="doc preview",
        vector_id="v1",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
        extra_metadata={"session_id": "s3", "speaker": "Caroline"},
    )

    async def _execute(stmt):
        stmt_str = str(stmt)
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars([mem])
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute
    session.add = MagicMock()
    session.add_all = MagicMock()

    svc = MemoryService(session=session, user_id="user", org_id="org", clearance_level=0)
    svc.permission_checker.filter_memory_ids_with_access = AsyncMock(return_value=["m1"])

    req = MemorySearchRequest(query="Where did Caroline move from?", limit=10)
    _ = await svc.search_memories(query_embedding=[0.1, 0.2, 0.3], request=req, request_id="rid")

    diagnostics = svc.get_last_search_diagnostics()
    assert "vector_hits" in diagnostics
    assert "lexical_hits" in diagnostics
    assert "graph_hits" in diagnostics
    assert "confidence_buckets" in diagnostics
    assert "lexical_mode" in diagnostics
