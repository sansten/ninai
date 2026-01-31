from __future__ import annotations

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
async def test_hybrid_search_can_return_lexical_results_when_qdrant_empty(monkeypatch):
    org_id = "org"
    user_id = "user"

    monkeypatch.setattr(memory_service_module.QdrantService, "search", AsyncMock(return_value=[]))

    # Session.execute will be called twice:
    # 1) lexical select(id, rank)
    # 2) metadata select(MemoryMetadata)
    async def _execute(stmt):
        stmt_str = str(stmt)
        if "ts_rank" in stmt_str and "FROM memory_metadata" in stmt_str:
            return _LexResult([("m1", 0.42)])
        if "FROM memory_metadata" in stmt_str:
            return _execute_result_with_scalars([SimpleNamespace(id="m1")])
        return _execute_result_with_scalars([])

    session = AsyncMock()
    session.execute = _execute

    svc = MemoryService(session=session, user_id=user_id, org_id=org_id, clearance_level=0)
    svc.permission_checker.check_memory_access = AsyncMock(
        return_value=SimpleNamespace(allowed=True, method="rls", reason="")
    )
    svc.audit_service.log_memory_access = AsyncMock()

    req = MemorySearchRequest(query="hello", limit=10, hybrid=True)
    results = await svc.search_memories(query_embedding=[0.0] * 3, request=req, request_id="rid")

    assert [m.id for m in results] == ["m1"]
