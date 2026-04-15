"""
Tests for memory actor identity integration
=============================================

Covers:
- create_memory stores actor_ctx fields into extra_metadata
- create_memory without actor_ctx defaults to anonymous
- ANONYMOUS mode strips actor fields from extra_metadata  
- ROLE_ONLY mode strips actor_id but keeps role/type
- FULL mode with actor_ctx persists all identity fields
- create_memory_smart passes actor_ctx through to long-term path
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.services.memory_service as memory_service_module
from app.schemas.memory import MemoryCreate
from app.services.memory_service import MemoryService
from app.services.identity_policy_service import ResolvedActorContext


def _session_stub():
    return SimpleNamespace(add=MagicMock(), flush=AsyncMock())


def _make_service(session=None, user_id="user-1", org_id="org-1"):
    s = session or _session_stub()
    svc = MemoryService(session=s, user_id=user_id, org_id=org_id, clearance_level=0)
    svc.permission_checker.check_permission = AsyncMock(
        return_value=SimpleNamespace(allowed=True, reason="")
    )
    svc.audit_service.log_memory_operation = AsyncMock()
    return svc, s


def _full_ctx(
    actor_id="user-1",
    actor_type="employee",
    role="employee_engineering",
    department="Engineering",
    mode="full",
) -> ResolvedActorContext:
    return ResolvedActorContext(
        actor_id=actor_id,
        actor_type=actor_type,
        role=role,
        department=department,
        display_name=actor_id,
        mode_applied=mode,
        identity_confidence=1.0,
        mandate_was_active=False,
    )


@pytest.mark.asyncio
async def test_create_memory_with_full_actor_ctx(monkeypatch):
    svc, session = _make_service()
    monkeypatch.setattr(
        memory_service_module.QdrantService, "upsert_memory", AsyncMock(return_value=True)
    )

    data = MemoryCreate(content="test content", scope="personal")
    await svc.create_memory(
        data=data,
        embedding=[0.0] * 3,
        request_id="rid",
        actor_ctx=_full_ctx(),
    )

    memory = session.add.call_args.args[0]
    assert memory.extra_metadata["write_actor_id"] == "user-1"
    assert memory.extra_metadata["write_actor_type"] == "employee"
    assert memory.extra_metadata["write_role"] == "employee_engineering"
    assert memory.extra_metadata["write_department"] == "Engineering"
    assert memory.extra_metadata["write_identity_mode"] == "full"


@pytest.mark.asyncio
async def test_create_memory_no_actor_ctx_defaults_anonymous(monkeypatch):
    svc, session = _make_service()
    monkeypatch.setattr(
        memory_service_module.QdrantService, "upsert_memory", AsyncMock(return_value=True)
    )

    data = MemoryCreate(content="test content", scope="personal")
    await svc.create_memory(data=data, embedding=[0.0] * 3, request_id="rid")

    memory = session.add.call_args.args[0]
    assert memory.extra_metadata["write_actor_id"] == "anonymous"
    assert memory.extra_metadata["write_actor_type"] == "anonymous"
    assert memory.extra_metadata["write_role"] == "anonymous"


@pytest.mark.asyncio
async def test_create_memory_anonymous_mode_strips_identity(monkeypatch):
    svc, session = _make_service()
    monkeypatch.setattr(
        memory_service_module.QdrantService, "upsert_memory", AsyncMock(return_value=True)
    )

    anon_ctx = ResolvedActorContext(
        actor_id=None,
        actor_type=None,
        role=None,
        department=None,
        display_name=None,
        mode_applied="anonymous",
        identity_confidence=0.0,
        mandate_was_active=False,
    )
    data = MemoryCreate(content="test content", scope="personal")
    await svc.create_memory(data=data, embedding=[0.0] * 3, request_id="rid", actor_ctx=anon_ctx)

    memory = session.add.call_args.args[0]
    assert memory.extra_metadata["write_actor_id"] == "anonymous"
    assert memory.extra_metadata["write_actor_type"] == "anonymous"
    assert memory.extra_metadata["write_role"] == "anonymous"


@pytest.mark.asyncio
async def test_create_memory_role_only_mode(monkeypatch):
    svc, session = _make_service()
    monkeypatch.setattr(
        memory_service_module.QdrantService, "upsert_memory", AsyncMock(return_value=True)
    )

    role_only_ctx = ResolvedActorContext(
        actor_id=None,
        actor_type="employee",
        role="employee_engineering",
        department="Engineering",
        display_name=None,
        mode_applied="role_only",
        identity_confidence=0.8,
        mandate_was_active=False,
    )
    data = MemoryCreate(content="test content", scope="personal")
    await svc.create_memory(
        data=data, embedding=[0.0] * 3, request_id="rid", actor_ctx=role_only_ctx
    )

    memory = session.add.call_args.args[0]
    # actor_id should be "anonymous" since role_only_ctx.actor_id is None
    assert memory.extra_metadata["write_actor_id"] == "anonymous"
    assert memory.extra_metadata["write_actor_type"] == "employee"
    assert memory.extra_metadata["write_role"] == "employee_engineering"
    assert memory.extra_metadata["write_identity_mode"] == "role_only"


@pytest.mark.asyncio
async def test_create_memory_smart_force_long_term_passes_actor_ctx(monkeypatch):
    svc, session = _make_service()
    monkeypatch.setattr(
        memory_service_module.QdrantService, "upsert_memory", AsyncMock(return_value=True)
    )

    data = MemoryCreate(content="test content", scope="personal")
    ctx = _full_ctx(actor_id="svc-bot", actor_type="bot", role="bot_operator", department=None)
    result = await svc.create_memory_smart(
        data=data,
        embedding=[0.0] * 3,
        request_id="rid",
        force_long_term=True,
        actor_ctx=ctx,
    )

    memory = session.add.call_args.args[0]
    assert memory.extra_metadata["write_actor_id"] == "svc-bot"
    assert memory.extra_metadata["write_actor_type"] == "bot"


@pytest.mark.asyncio
async def test_qdrant_payload_reflects_actor_ctx(monkeypatch):
    svc, session = _make_service()
    upsert_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(memory_service_module.QdrantService, "upsert_memory", upsert_mock)

    data = MemoryCreate(content="test", scope="personal")
    await svc.create_memory(
        data=data, embedding=[0.0] * 3, request_id="rid", actor_ctx=_full_ctx()
    )

    payload = upsert_mock.call_args.kwargs["payload"]
    assert payload["write_actor_id"] == "user-1"
    assert payload["write_actor_type"] == "employee"
    assert payload["write_role"] == "employee_engineering"
