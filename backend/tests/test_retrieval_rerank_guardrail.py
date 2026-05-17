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
async def test_guardrail_penalizes_chatter_and_boosts_factual(monkeypatch):
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_HEURISTICS_ENABLED", True, raising=False)
    # Equal vector scores for both memories; guardrail should reorder.
    monkeypatch.setattr(
        memory_service_module.QdrantService,
        "search",
        AsyncMock(
            return_value=[
                {"id": "v1", "score": 0.9, "payload": {"memory_id": "m_chatter"}},
                {"id": "v2", "score": 0.9, "payload": {"memory_id": "m_fact"}},
            ]
        ),
    )

    now = datetime.now(timezone.utc)
    mem_chatter = SimpleNamespace(
        id="m_chatter",
        is_active=True,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        content_hash="h1",
        title="chat",
        content_preview="Hey Caroline, that's awesome! Any specific plans?",
        vector_id="v1",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
    )
    mem_fact = SimpleNamespace(
        id="m_fact",
        is_active=True,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        content_hash="h2",
        title="fact",
        content_preview="Caroline relationship status is single.",
        vector_id="v2",
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
            return _LexResult([])
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars([mem_chatter, mem_fact])
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute
    session.add = MagicMock()
    session.add_all = MagicMock()

    svc = MemoryService(session=session, user_id="user", org_id="org", clearance_level=0)
    svc.permission_checker.filter_memory_ids_with_access = AsyncMock(return_value=["m_chatter", "m_fact"])

    req = MemorySearchRequest(query="What is Caroline's relationship status?", limit=10, hybrid=False)
    results = await svc.search_memories(query_embedding=[0.1, 0.2, 0.3], request=req, request_id="rid")

    assert [m.id for m in results][:2] == ["m_fact", "m_chatter"]

    diag = svc.get_last_search_diagnostics()
    assert "guardrail_counts" in diag
    assert diag["guardrail_counts"].get("factual_boost", 0) >= 1
    assert (
        diag["guardrail_counts"].get("chatter_penalty", 0)
        + diag["guardrail_counts"].get("question_turn_penalty", 0)
    ) >= 1


@pytest.mark.asyncio
async def test_question_lexical_rescue_runs_when_hybrid_disabled(monkeypatch):
    monkeypatch.setattr(memory_service_module.settings, "SEARCH_HEURISTICS_ENABLED", True, raising=False)
    monkeypatch.setattr(
        memory_service_module.QdrantService,
        "search",
        AsyncMock(
            return_value=[
                {"id": "v1", "score": 0.95, "payload": {"memory_id": "m_chatter"}},
                {"id": "v2", "score": 0.83, "payload": {"memory_id": "m_fact"}},
            ]
        ),
    )

    now = datetime.now(timezone.utc)
    mem_chatter = SimpleNamespace(
        id="m_chatter",
        is_active=True,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        content_hash="h1",
        title="chat",
        content_preview="Hey Caroline, that's awesome! Any specific plans?",
        vector_id="v1",
        embedding_model="e",
        scope="personal",
        scope_id=None,
        classification="internal",
        source_type=None,
        source_id=None,
    )
    mem_fact = SimpleNamespace(
        id="m_fact",
        is_active=True,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        content_hash="h2",
        title="fact",
        content_preview="Caroline relationship status is single.",
        vector_id="v2",
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
            return _LexResult([("m_fact", 0.9)])
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars([mem_chatter, mem_fact])
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute
    session.add = MagicMock()
    session.add_all = MagicMock()

    svc = MemoryService(session=session, user_id="user", org_id="org", clearance_level=0)
    svc.permission_checker.filter_memory_ids_with_access = AsyncMock(return_value=["m_chatter", "m_fact"])

    req = MemorySearchRequest(query="What is Caroline's relationship status?", limit=10, hybrid=False)
    results = await svc.search_memories(query_embedding=[0.2, 0.1, 0.3], request=req, request_id="rid")

    assert [m.id for m in results][:2] == ["m_fact", "m_chatter"]

    diag = svc.get_last_search_diagnostics()
    assert diag.get("lexical_mode") == "rescue_question"
    assert int(diag.get("lexical_hits") or 0) >= 1
