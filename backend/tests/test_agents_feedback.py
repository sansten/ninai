from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from app.agents.feedback_learning_agent import FeedbackLearningAgent
import app.agents.feedback_learning_agent as feedback_module


@pytest.mark.asyncio
async def test_feedback_learning_agent_skips_for_non_long_term(monkeypatch):
    agent = FeedbackLearningAgent()

    get_tenant_session_mock = AsyncMock()
    monkeypatch.setattr(feedback_module, "get_tenant_session", get_tenant_session_mock)

    ctx = {
        "tenant": {"org_id": "org", "org_slug": None},
        "actor": {"user_id": "u", "roles": []},
        "memory": {"id": "m", "storage": "short_term", "content": "x", "classification": "internal", "enrichment": {}},
        "runtime": {"job_id": "t", "attempt": 1, "max_attempts": 1},
    }

    res = await agent.run("m", ctx)
    assert res.status == "success"
    assert res.outputs.get("applied") is False
    assert res.outputs.get("reason") == "not_long_term"
    assert get_tenant_session_mock.call_count == 0


@pytest.mark.asyncio
async def test_feedback_learning_agent_applies_pending_feedback(monkeypatch):
    agent = FeedbackLearningAgent()

    @asynccontextmanager
    async def _fake_tenant_session(**_kwargs):
        yield AsyncMock()

    class _FakeFeedbackService:
        def __init__(self, *_args, **_kwargs):
            pass

        async def apply_pending_feedback(self, *, memory_id: str, applied_by=None):
            assert memory_id == "m"
            return {"applied_count": 2, "updates": [{"type": "tag_add"}, {"type": "note"}]}

    monkeypatch.setattr(feedback_module, "get_tenant_session", _fake_tenant_session)
    monkeypatch.setattr(feedback_module, "MemoryFeedbackService", _FakeFeedbackService)

    ctx = {
        "tenant": {"org_id": "org", "org_slug": None},
        "actor": {"user_id": "u", "roles": []},
        "memory": {"id": "m", "storage": "long_term", "content": "x", "classification": "internal", "enrichment": {}},
        "runtime": {"job_id": "t", "attempt": 1, "max_attempts": 1},
    }

    res = await agent.run("m", ctx)
    assert res.status == "success"
    assert res.outputs.get("applied") is True
    assert res.outputs.get("applied_count") == 2
    assert isinstance(res.outputs.get("updates"), list)
