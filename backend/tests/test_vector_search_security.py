from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import app.services.memory_service as memory_service_module
from app.schemas.memory import MemoryCreate, MemorySearchRequest
from app.services.memory_service import MemoryService
from app.services.identity_policy_service import ResolvedActorContext


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
    svc.permission_checker.filter_memory_ids_with_access = AsyncMock(return_value=["m_allowed"])
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


@pytest.mark.asyncio
async def test_create_memory_defaults_writer_context_to_anonymous(monkeypatch):
    org_id = "org"
    user_id = "user"

    session = SimpleNamespace(add=MagicMock(), flush=AsyncMock())
    svc = MemoryService(session=session, user_id=user_id, org_id=org_id, clearance_level=0)
    svc.permission_checker.check_permission = AsyncMock(return_value=SimpleNamespace(allowed=True, reason=""))
    svc.audit_service.log_memory_operation = AsyncMock()

    upsert_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(memory_service_module.QdrantService, "upsert_memory", upsert_mock)

    # No actor_ctx supplied → falls back to anonymous defaults
    data = MemoryCreate(content="x", scope="personal")
    await svc.create_memory(data=data, embedding=[0.0] * 3, request_id="rid")

    memory = session.add.call_args.args[0]
    assert memory.extra_metadata["write_actor_id"] == "anonymous"
    assert memory.extra_metadata["write_actor_type"] == "anonymous"
    assert memory.extra_metadata["write_role"] == "anonymous"


@pytest.mark.asyncio
async def test_create_memory_persists_explicit_writer_context(monkeypatch):
    org_id = "org"
    user_id = "user"

    session = SimpleNamespace(add=MagicMock(), flush=AsyncMock())
    svc = MemoryService(session=session, user_id=user_id, org_id=org_id, clearance_level=0)
    svc.permission_checker.check_permission = AsyncMock(return_value=SimpleNamespace(allowed=True, reason=""))
    svc.audit_service.log_memory_operation = AsyncMock()

    upsert_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(memory_service_module.QdrantService, "upsert_memory", upsert_mock)

    # Actor context now comes from auth layer, not from MemoryCreate fields
    actor_ctx = ResolvedActorContext(
        actor_id="bot-1",
        actor_type="bot",
        role="bot_operator",
        department=None,
        display_name=None,
        mode_applied="full",
        identity_confidence=1.0,
        mandate_was_active=False,
    )
    data = MemoryCreate(content="x", scope="personal")
    await svc.create_memory(data=data, embedding=[0.0] * 3, request_id="rid", actor_ctx=actor_ctx)

    memory = session.add.call_args.args[0]
    assert memory.extra_metadata["write_actor_id"] == "bot-1"
    assert memory.extra_metadata["write_actor_type"] == "bot"
    assert memory.extra_metadata["write_role"] == "bot_operator"
