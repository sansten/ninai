from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.redis import RedisClient
from app.schemas.memory import MemoryCreate
from app.services.permission_checker import AccessDecision
from app.services.memory_service import MemoryService
from app.services.short_term_memory import ShortTermMemoryService
from app.core.qdrant import QdrantService


@pytest.mark.asyncio
async def test_create_memory_smart_respects_ttl_override(monkeypatch):
    fake_redis = AsyncMock()
    fake_redis.setex = AsyncMock()
    fake_redis.sadd = AsyncMock()
    fake_redis.expire = AsyncMock()
    fake_redis.set = AsyncMock()

    async def _get_client(cls):
        return fake_redis

    monkeypatch.setattr(RedisClient, "get_client", classmethod(_get_client))

    service = MemoryService(
        session=AsyncMock(),
        user_id="00000000-0000-0000-0000-000000000001",
        org_id="00000000-0000-0000-0000-000000000002",
        clearance_level=0,
    )

    service.permission_checker.check_permission = AsyncMock(
        return_value=AccessDecision(allowed=True, reason="ok", method="test")
    )
    service.audit_service.log_memory_operation = AsyncMock()

    body = MemoryCreate(content="hello", scope="personal")

    await service.create_memory_smart(body, ttl=123)

    # ShortTermMemoryService.store writes two setex calls: the memory and access counter.
    assert fake_redis.setex.await_count >= 2
    for call in fake_redis.setex.await_args_list:
        assert call.args[1] == 123


@pytest.mark.asyncio
async def test_create_memory_smart_includes_occurred_at_in_metadata(monkeypatch):
    service = MemoryService(
        session=AsyncMock(),
        user_id="00000000-0000-0000-0000-000000000001",
        org_id="00000000-0000-0000-0000-000000000002",
        clearance_level=0,
    )

    service.permission_checker.check_permission = AsyncMock(
        return_value=AccessDecision(allowed=True, reason="ok", method="test")
    )
    service.audit_service.log_memory_operation = AsyncMock()

    store_mock = AsyncMock(
        return_value=SimpleNamespace(
            id="stm-1",
            importance_score=0.1,
            promotion_eligible=False,
            access_count=0,
        )
    )
    monkeypatch.setattr(ShortTermMemoryService, "store", store_mock)

    body = MemoryCreate(
        content="incident",
        scope="personal",
        occurred_at=datetime(2026, 4, 15, 9, 2, tzinfo=timezone.utc),
    )

    await service.create_memory_smart(body, ttl=321)

    kwargs = store_mock.await_args.kwargs
    metadata = kwargs["metadata"]
    assert metadata["event_time"] == "2026-04-15T09:02:00+00:00"
    assert metadata["event_date"] == "2026-04-15"
    assert "written_at" in metadata


@pytest.mark.asyncio
async def test_create_memory_persists_occurred_at_for_long_term(monkeypatch):
    session = SimpleNamespace(
        add=MagicMock(),
        flush=AsyncMock(),
        execute=AsyncMock(),
    )
    service = MemoryService(
        session=session,
        user_id="00000000-0000-0000-0000-000000000001",
        org_id="00000000-0000-0000-0000-000000000002",
        clearance_level=0,
    )

    service.permission_checker.check_permission = AsyncMock(
        return_value=AccessDecision(allowed=True, reason="ok", method="test")
    )
    service.audit_service.log_memory_operation = AsyncMock()
    service._ensure_search_vector = AsyncMock()

    upsert_mock = AsyncMock()
    monkeypatch.setattr(QdrantService, "upsert_memory", upsert_mock)

    body = MemoryCreate(
        content="incident",
        scope="personal",
        occurred_at=datetime(2026, 4, 15, 9, 2, tzinfo=timezone.utc),
    )

    memory = await service.create_memory(body, embedding=[0.1, 0.2])

    assert memory.extra_metadata["event_time"] == "2026-04-15T09:02:00+00:00"
    assert memory.extra_metadata["event_date"] == "2026-04-15"
    assert "written_at" in memory.extra_metadata

    payload = upsert_mock.await_args.kwargs["payload"]
    assert payload["event_time"] == "2026-04-15T09:02:00+00:00"
