from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.memory import MemoryMetadata
from app.schemas.memory import MemoryCreate
from app.services.permission_checker import AccessDecision
from app.services.memory_service import MemoryService
from app.core.qdrant import QdrantService


def _make_service(session):
    svc = MemoryService(
        session=session,
        user_id="00000000-0000-0000-0000-000000000001",
        org_id="00000000-0000-0000-0000-000000000002",
        clearance_level=0,
    )
    svc.permission_checker.check_permission = AsyncMock(
        return_value=AccessDecision(allowed=True, reason="ok", method="test")
    )
    svc.audit_service.log_memory_operation = AsyncMock()
    return svc


def _existing_memory(**kwargs) -> MemoryMetadata:
    # MagicMock(spec=MemoryMetadata) passes isinstance checks and allows real attribute values.
    m = MagicMock(spec=MemoryMetadata)
    m.id = kwargs.get("id", "mem-existing-001")
    m.organization_id = kwargs.get("organization_id", "00000000-0000-0000-0000-000000000002")
    m.scope = kwargs.get("scope", "personal")
    m.scope_id = kwargs.get("scope_id", None)
    m.content_hash = kwargs.get("content_hash", "abc123")
    m.ingest_count = kwargs.get("ingest_count", 1)
    m.last_ingested_at = kwargs.get("last_ingested_at", None)
    m.unique_sources = list(kwargs.get("unique_sources", []))
    m.is_active = True
    return m


@pytest.mark.asyncio
async def test_duplicate_content_reinforces_existing_memory(monkeypatch):
    """Same content in same org+scope returns existing memory with incremented count."""
    existing = _existing_memory(ingest_count=1, unique_sources=[])

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = existing

    session = SimpleNamespace(
        execute=AsyncMock(return_value=result_mock),
        add=MagicMock(),
        flush=AsyncMock(),
    )
    svc = _make_service(session)

    body = MemoryCreate(content="Q4 revenue is $10M", scope="personal", source_id="etl-job-1")
    result = await svc.create_memory(body, embedding=[0.1, 0.2])

    assert result is existing
    assert result.ingest_count == 2
    assert result.last_ingested_at is not None
    assert "etl-job-1" in result.unique_sources
    # Must NOT insert a new row
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_reinforcement_tracks_unique_sources(monkeypatch):
    """Repeated source does not duplicate in unique_sources; new source is appended."""
    existing = _existing_memory(ingest_count=3, unique_sources=["etl-job-1", "api-v2"])

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = existing

    session = SimpleNamespace(
        execute=AsyncMock(return_value=result_mock),
        add=MagicMock(),
        flush=AsyncMock(),
    )
    svc = _make_service(session)

    # Same source again — should not duplicate
    body = MemoryCreate(content="same", scope="personal", source_id="etl-job-1")
    result = await svc.create_memory(body, embedding=[])
    assert result.unique_sources == ["etl-job-1", "api-v2"]
    assert result.ingest_count == 4

    # New source — should append
    existing.ingest_count = 4
    body2 = MemoryCreate(content="same", scope="personal", source_id="slack-connector")
    result2 = await svc.create_memory(body2, embedding=[])
    assert "slack-connector" in result2.unique_sources
    assert result2.ingest_count == 5


@pytest.mark.asyncio
async def test_new_content_creates_new_row(monkeypatch):
    """Content not seen before flows through normal insert path."""
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None

    session = SimpleNamespace(
        execute=AsyncMock(return_value=result_mock),
        add=MagicMock(),
        flush=AsyncMock(),
    )
    svc = _make_service(session)
    svc._ensure_search_vector = AsyncMock()
    monkeypatch.setattr(QdrantService, "upsert_memory", AsyncMock())

    body = MemoryCreate(content="brand new content never seen before", scope="personal")
    result = await svc.create_memory(body, embedding=[0.1, 0.2])

    session.add.assert_called_once()
    assert isinstance(result, MemoryMetadata)
    assert result.ingest_count == 1
    assert result.unique_sources == []


@pytest.mark.asyncio
async def test_different_scope_creates_new_row(monkeypatch):
    """Same content under different scope is a distinct memory — no reinforcement."""
    # Dedup query finds nothing (scope mismatch — different scope_id would filter it out)
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None

    session = SimpleNamespace(
        execute=AsyncMock(return_value=result_mock),
        add=MagicMock(),
        flush=AsyncMock(),
    )
    svc = _make_service(session)
    svc._ensure_search_vector = AsyncMock()
    monkeypatch.setattr(QdrantService, "upsert_memory", AsyncMock())

    body = MemoryCreate(
        content="CEO update", scope="department",
        scope_id="00000000-0000-0000-0000-000000000099",
    )
    result = await svc.create_memory(body, embedding=[0.1])

    session.add.assert_called_once()
    assert result.ingest_count == 1


@pytest.mark.asyncio
async def test_reinforcement_audit_logged(monkeypatch):
    """Reinforcement path logs a 'reinforce' audit operation."""
    existing = _existing_memory(ingest_count=2)

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = existing

    session = SimpleNamespace(
        execute=AsyncMock(return_value=result_mock),
        add=MagicMock(),
        flush=AsyncMock(),
    )
    svc = _make_service(session)

    body = MemoryCreate(content="repeated fact", scope="personal")
    await svc.create_memory(body, embedding=[])

    svc.audit_service.log_memory_operation.assert_awaited_once()
    call_kwargs = svc.audit_service.log_memory_operation.await_args.kwargs
    assert call_kwargs["operation"] == "reinforce"
    assert call_kwargs["memory_id"] == existing.id
    assert call_kwargs["details"]["ingest_count"] == 3


@pytest.mark.asyncio
async def test_dedup_error_falls_through_to_new_insert(monkeypatch):
    """If the dedup query raises (e.g. DB offline), ingest continues as normal new write."""
    session = SimpleNamespace(
        execute=AsyncMock(side_effect=Exception("db timeout")),
        add=MagicMock(),
        flush=AsyncMock(),
    )
    # Need a second execute for _ensure_search_vector — replace after first call
    call_count = {"n": 0}
    original_execute = AsyncMock(side_effect=Exception("db timeout"))
    search_execute = AsyncMock()

    async def _smart_execute(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("db timeout")
        return search_execute.return_value

    session.execute = _smart_execute

    svc = _make_service(session)
    svc._ensure_search_vector = AsyncMock()
    monkeypatch.setattr(QdrantService, "upsert_memory", AsyncMock())

    body = MemoryCreate(content="some content", scope="personal")
    result = await svc.create_memory(body, embedding=[0.1])

    # Falls through to new insert
    assert isinstance(result, MemoryMetadata)
    session.add.assert_called_once()
