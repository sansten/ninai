from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.agents.classification_agent import ClassificationAgent


@pytest.mark.asyncio
async def test_classification_agent_heuristic_sensitive_biases_confidential_or_higher():
    agent = ClassificationAgent()

    ctx = {
        "tenant": {"org_id": "00000000-0000-0000-0000-000000000000", "org_slug": None},
        "memory": {
            "content": "Customer provided credit card 4111 1111 1111 1111 and password reset.",
            "classification": "internal",
        },
        "runtime": {"job_id": "t1"},
    }

    res = await agent.run("11111111-1111-1111-1111-111111111111", ctx)

    assert res.status == "success"
    assert res.outputs["is_sensitive"] is True
    assert res.outputs["classification"] in {"confidential", "restricted"}


@pytest.mark.asyncio
async def test_classification_agent_never_downgrades_existing_classification():
    agent = ClassificationAgent()

    ctx = {
        "tenant": {"org_id": "00000000-0000-0000-0000-000000000000", "org_slug": None},
        "memory": {
            "content": "Just a quick note about lunch.",
            "classification": "restricted",
        },
        "runtime": {"job_id": "t2"},
    }

    res = await agent.run("22222222-2222-2222-2222-222222222222", ctx)

    assert res.status == "success"
    assert res.outputs["classification"] == "restricted"


@pytest.mark.asyncio
async def test_classification_agent_returns_valid_timestamps():
    agent = ClassificationAgent()

    ctx = {
        "tenant": {"org_id": "00000000-0000-0000-0000-000000000000", "org_slug": None},
        "memory": {"content": "How to reset a device: Step 1, Step 2", "classification": "internal"},
        "runtime": {"job_id": "t3"},
    }

    res = await agent.run("33333333-3333-3333-3333-333333333333", ctx)

    assert isinstance(res.started_at, datetime)
    assert isinstance(res.finished_at, datetime)
    assert res.started_at.tzinfo is not None
    assert res.finished_at.tzinfo is not None
    assert res.finished_at >= res.started_at
