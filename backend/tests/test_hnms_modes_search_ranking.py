from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.services.memory_service as memory_service_module
from app.schemas.memory import MemorySearchRequest
from app.services.memory_service import MemoryService


def _execute_result_with_scalars(rows: list) -> MagicMock:
    result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = rows
    result.scalars.return_value = scalars
    return result


async def _build_service_with_memories(memories: list[SimpleNamespace]) -> MemoryService:
    async def _execute(stmt):
        stmt_str = str(stmt)
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars(memories)
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute

    svc = MemoryService(session=session, user_id="user", org_id="org", clearance_level=0)
    svc.permission_checker.check_memory_access = AsyncMock(
        return_value=SimpleNamespace(allowed=True, method="rls", reason="")
    )
    svc.audit_service.log_memory_access = AsyncMock()
    return svc


@pytest.mark.asyncio
async def test_hnms_mode_performance_overrides_base_decay_and_downranks_old(monkeypatch):
    now = datetime.now(timezone.utc)

    # Old has slightly better vector similarity, but should be downranked heavily.
    monkeypatch.setattr(
        memory_service_module.QdrantService,
        "search",
        AsyncMock(
            return_value=[
                {"id": "v_old", "score": 1.0, "payload": {"memory_id": "m_old"}},
                {"id": "v_new", "score": 0.9, "payload": {"memory_id": "m_new"}},
            ]
        ),
    )

    # Base temporal decay disabled, but mode=performance should still apply.
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_TEMPORAL_DECAY_ENABLED", False, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_HNMS_MODE_ALLOW_REQUEST_OVERRIDE", True, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_HNMS_MODE_PERFORMANCE_HALF_LIFE_DAYS", 1.0, raising=False)

    m_old = SimpleNamespace(
        id="m_old",
        is_active=True,
        created_at=now - timedelta(days=60),
        updated_at=now - timedelta(days=30),
        last_accessed_at=None,
        content_hash="h1",
        title="old",
        content_preview="old",
        vector_id="v_old",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
    )

    m_new = SimpleNamespace(
        id="m_new",
        is_active=True,
        created_at=now - timedelta(days=2),
        updated_at=now,
        last_accessed_at=None,
        content_hash="h2",
        title="new",
        content_preview="new",
        vector_id="v_new",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
    )

    svc = await _build_service_with_memories([m_old, m_new])

    req = MemorySearchRequest(query="hello", limit=10, hybrid=False, hnms_mode="performance")
    results = await svc.search_memories(query_embedding=[0.0] * 3, request=req, request_id="rid")

    assert [m.id for m in results] == ["m_new", "m_old"]


@pytest.mark.asyncio
async def test_hnms_mode_research_is_less_aggressive_than_performance(monkeypatch):
    now = datetime.now(timezone.utc)

    # Old has slightly better vector similarity.
    monkeypatch.setattr(
        memory_service_module.QdrantService,
        "search",
        AsyncMock(
            return_value=[
                {"id": "v_old", "score": 1.0, "payload": {"memory_id": "m_old"}},
                {"id": "v_new", "score": 0.9, "payload": {"memory_id": "m_new"}},
            ]
        ),
    )

    monkeypatch.setattr(memory_service_module.settings, "SEARCH_TEMPORAL_DECAY_ENABLED", False, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_HNMS_MODE_ALLOW_REQUEST_OVERRIDE", True, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_HNMS_MODE_RESEARCH_HALF_LIFE_DAYS", 1000.0, raising=False)

    m_old = SimpleNamespace(
        id="m_old",
        is_active=True,
        created_at=now - timedelta(days=60),
        updated_at=now - timedelta(days=30),
        last_accessed_at=None,
        content_hash="h1",
        title="old",
        content_preview="old",
        vector_id="v_old",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
    )

    m_new = SimpleNamespace(
        id="m_new",
        is_active=True,
        created_at=now - timedelta(days=2),
        updated_at=now,
        last_accessed_at=None,
        content_hash="h2",
        title="new",
        content_preview="new",
        vector_id="v_new",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
    )

    svc = await _build_service_with_memories([m_old, m_new])

    req = MemorySearchRequest(query="hello", limit=10, hybrid=False, hnms_mode="research")
    results = await svc.search_memories(query_embedding=[0.0] * 3, request=req, request_id="rid")

    assert [m.id for m in results] == ["m_old", "m_new"]
