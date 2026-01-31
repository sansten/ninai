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


def _execute_result_with_all(rows: list[tuple]) -> MagicMock:
    result = MagicMock()
    result.all.return_value = rows
    return result


@pytest.mark.asyncio
async def test_feedback_rerank_downranks_negative_feedback(monkeypatch):
    org_id = "org"
    user_id = "user"

    now = datetime.now(timezone.utc)

    # Old has slightly higher vector score initially.
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

    # Enable feedback rerank, keep temporal decay out of the way.
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_TEMPORAL_DECAY_ENABLED", False, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_FEEDBACK_RERANK_ENABLED", True, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_FEEDBACK_RERANK_NEGATIVE_MULTIPLIER", 0.1, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_FEEDBACK_RERANK_POSITIVE_MULTIPLIER", 1.0, raising=False)

    m_old = SimpleNamespace(
        id="m_old",
        is_active=True,
        created_at=now - timedelta(days=30),
        updated_at=now - timedelta(days=10),
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

    async def _execute(stmt):
        stmt_str = str(stmt)
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars([m_old, m_new])
        if "FROM memory_feedback" in stmt_str:
            return _execute_result_with_all([("m_old", {"value": -1})])
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute

    svc = MemoryService(session=session, user_id=user_id, org_id=org_id, clearance_level=0)
    svc.permission_checker.check_memory_access = AsyncMock(
        return_value=SimpleNamespace(allowed=True, method="rls", reason="")
    )
    svc.audit_service.log_memory_access = AsyncMock()

    req = MemorySearchRequest(query="hello", limit=10, hybrid=False)
    results = await svc.search_memories(query_embedding=[0.0] * 3, request=req, request_id="rid")

    assert [m.id for m in results] == ["m_new", "m_old"]


@pytest.mark.asyncio
async def test_feedback_rerank_boosts_positive_feedback(monkeypatch):
    org_id = "org"
    user_id = "user"

    now = datetime.now(timezone.utc)

    # New is slightly lower vector score initially.
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
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_FEEDBACK_RERANK_ENABLED", True, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_FEEDBACK_RERANK_NEGATIVE_MULTIPLIER", 1.0, raising=False)
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_FEEDBACK_RERANK_POSITIVE_MULTIPLIER", 2.0, raising=False)

    m_old = SimpleNamespace(
        id="m_old",
        is_active=True,
        created_at=now - timedelta(days=30),
        updated_at=now - timedelta(days=10),
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

    async def _execute(stmt):
        stmt_str = str(stmt)
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars([m_old, m_new])
        if "FROM memory_feedback" in stmt_str:
            return _execute_result_with_all([("m_new", {"relevant": True})])
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute

    svc = MemoryService(session=session, user_id=user_id, org_id=org_id, clearance_level=0)
    svc.permission_checker.check_memory_access = AsyncMock(
        return_value=SimpleNamespace(allowed=True, method="rls", reason="")
    )
    svc.audit_service.log_memory_access = AsyncMock()

    req = MemorySearchRequest(query="hello", limit=10, hybrid=False)
    results = await svc.search_memories(query_embedding=[0.0] * 3, request=req, request_id="rid")

    assert [m.id for m in results] == ["m_new", "m_old"]
