from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.services.memory_promoter as mod
from app.services.memory_promoter import MemoryPromoter
from app.services.short_term_memory import ShortTermMemory


class _FakeAudit:
    def __init__(self, session):
        self.session = session

    async def log_memory_operation(self, **kwargs):
        return None


class _FakeStmSvc:
    def __init__(self, user_id: str, org_id: str):
        self.user_id = user_id
        self.org_id = org_id

    async def delete(self, memory_id: str):
        return None


@pytest.mark.asyncio
async def test_promote_memory_records_promotion_history(monkeypatch):
    # Mock external side effects
    async def _fake_upsert_memory(**kwargs):
        return None

    monkeypatch.setattr(mod, "QdrantService", SimpleNamespace(upsert_memory=_fake_upsert_memory))
    monkeypatch.setattr(mod, "AuditService", _FakeAudit)
    monkeypatch.setattr(mod, "ShortTermMemoryService", _FakeStmSvc)

    added = []

    class _FakeSession:
        def add(self, obj):
            added.append(obj)

        async def flush(self):
            return None

    session = _FakeSession()

    promoter = MemoryPromoter(session=session, user_id="user", org_id="org")
    stm = ShortTermMemory(
        id="stm-1",
        organization_id="org",
        owner_id="user",
        content="Refund requested for order 123",
        title="Refund",
        scope="personal",
        tags=["billing"],
        entities={"order_id": ["123"]},
        metadata={"k": "v"},
        access_count=5,
        importance_score=0.9,
        promotion_eligible=True,
    )

    memory = await promoter.promote_memory(stm, embedding=[0.0], keep_in_cache=False, promotion_reason="agent:test")

    # MemoryMetadata first, then MemoryPromotionHistory.
    assert len(added) == 2
    assert getattr(memory, "extra_metadata")["original_stm_id"] == "stm-1"
    assert getattr(memory, "extra_metadata")["promotion_reason"] == "agent:test"

    history = added[1]
    assert getattr(history, "organization_id") == "org"
    assert getattr(history, "from_stm_id") == "stm-1"
    assert getattr(history, "to_memory_id") == getattr(memory, "id")
    assert getattr(history, "promotion_reason") == "agent:test"
