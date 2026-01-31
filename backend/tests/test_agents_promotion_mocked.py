import contextlib
from types import SimpleNamespace

import pytest

import app.agents.promotion_agent as mod
from app.agents.promotion_agent import PromotionAgent


class _FakeSession:
    pass


@contextlib.asynccontextmanager
async def _fake_tenant_session(**kwargs):
    yield _FakeSession()


class _FakePromoter:
    def __init__(self, session, user_id: str, org_id: str):
        self.session = session
        self.user_id = user_id
        self.org_id = org_id

    async def promote_by_id(self, stm_id: str, reason: str = "manual"):
        return SimpleNamespace(id="ltm-1")


class _FakeStmService:
    def __init__(self, user_id: str, org_id: str):
        self.user_id = user_id
        self.org_id = org_id

    async def get(self, memory_id: str):
        return SimpleNamespace(id=memory_id, content="Refund requested for order 123")


@pytest.mark.asyncio
async def test_promotion_agent_skips_when_not_short_term():
    agent = PromotionAgent()
    ctx = {
        "tenant": {"org_id": "org", "org_slug": None},
        "actor": {"user_id": "u", "roles": []},
        "memory": {"id": "m", "storage": "long_term", "content": "Refund requested", "enrichment": {}},
        "runtime": {"job_id": "t", "attempt": 1, "max_attempts": 1},
    }

    res = await agent.run("m", ctx)
    assert res.status == "success"
    assert res.outputs.get("promoted") is False


@pytest.mark.asyncio
async def test_promotion_agent_promotes_with_mocks(monkeypatch):
    monkeypatch.setattr(mod, "get_tenant_session", _fake_tenant_session)
    monkeypatch.setattr(mod, "MemoryPromoter", _FakePromoter)
    monkeypatch.setattr(mod, "ShortTermMemoryService", _FakeStmService)

    agent = PromotionAgent()
    ctx = {
        "tenant": {"org_id": "org", "org_slug": None},
        "actor": {"user_id": "u", "roles": []},
        "memory": {
            "id": "m",
            "storage": "short_term",
            "content": "Refund requested for order 123.",
            "enrichment": {"metadata": {"entities": {"order_id": ["123"]}}},
        },
        "runtime": {"job_id": "t", "attempt": 1, "max_attempts": 1},
    }

    res = await agent.run("m", ctx)
    assert res.status == "success"
    assert res.outputs.get("promoted") is True
    assert res.outputs.get("promoted_memory_id") == "ltm-1"
