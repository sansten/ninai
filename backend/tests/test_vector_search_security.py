from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import app.services.memory_service as memory_service_module
from app.schemas.memory import MemoryCreate, MemorySearchRequest
from app.services.memory_service import MemoryService


def _execute_result_with_scalars(rows: list) -> MagicMock:
    result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = rows
    result.scalars.return_value = scalars
    return result


@pytest.mark.asyncio
async def test_search_memories_postgres_recheck_filters_missing_ids(monkeypatch):
    org_id = "org"
    user_id = "user"

    qdrant_results = [
        {"id": "v1", "score": 0.9, "payload": {"memory_id": "m_allowed"}},
        {"id": "v2", "score": 0.8, "payload": {"memory_id": "m_missing"}},
    ]
    monkeypatch.setattr(memory_service_module.QdrantService, "search", AsyncMock(return_value=qdrant_results))

    captured = {}

    async def _execute(stmt):
        captured.setdefault("stmts", []).append(stmt)
        # Simulate Postgres/RLS re-check: only one row comes back.
        return _execute_result_with_scalars([SimpleNamespace(id="m_allowed")])

    session = AsyncMock()
    session.execute = _execute

    svc = MemoryService(session=session, user_id=user_id, org_id=org_id, clearance_level=0)
    svc.permission_checker.check_memory_access = AsyncMock(
        return_value=SimpleNamespace(allowed=True, method="rls", reason="")
    )
    svc.audit_service.log_memory_access = AsyncMock()

    req = MemorySearchRequest(query="hello", limit=10)
    results = await svc.search_memories(query_embedding=[0.0] * 3, request=req, request_id="rid")

    assert [m.id for m in results] == ["m_allowed"]
    assert hasattr(results[0], "provenance")
    assert results[0].provenance and results[0].provenance[0]["kind"] == "memory"
    # Ensure we re-query Postgres with defense-in-depth org constraint.
    stmt_strs = [str(s) for s in captured.get("stmts", [])]
    assert any("memory_metadata" in s for s in stmt_strs)
    assert any("organization_id" in s for s in stmt_strs)


@pytest.mark.asyncio
async def test_create_memory_includes_team_id_in_qdrant_payload(monkeypatch):
    org_id = "org"
    user_id = "user"
    team_id = str(uuid4())

    session = SimpleNamespace(add=MagicMock(), flush=AsyncMock())
    svc = MemoryService(session=session, user_id=user_id, org_id=org_id, clearance_level=0)
    svc.permission_checker.check_permission = AsyncMock(return_value=SimpleNamespace(allowed=True, reason=""))
    svc.audit_service.log_memory_operation = AsyncMock()

    upsert_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(memory_service_module.QdrantService, "upsert_memory", upsert_mock)

    data = MemoryCreate(content="x", scope="team", scope_id=team_id)
    await svc.create_memory(data=data, embedding=[0.0] * 3, request_id="rid")

    assert upsert_mock.call_count == 1
    payload = upsert_mock.call_args.kwargs["payload"]
    assert payload["scope"] == "team"
    assert payload["team_id"] == team_id
