from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.redis import RedisClient
from app.schemas.memory import MemoryCreate
from app.services.permission_checker import AccessDecision
from app.services.memory_service import MemoryService


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
