from __future__ import annotations

import pytest

from app.agents.topic_modeling_agent import TopicModelingAgent


@pytest.mark.asyncio
async def test_topic_modeling_agent_billing_topic_from_content_and_tags():
    agent = TopicModelingAgent()

    ctx = {
        "tenant": {"org_id": "00000000-0000-0000-0000-000000000000", "org_slug": None},
        "memory": {
            "content": "Customer asked for a refund due to duplicate payment on invoice 1001.",
            "classification": "internal",
            "enrichment": {"metadata": {"tags": ["billing"]}},
        },
        "runtime": {"job_id": "t1"},
    }

    res = await agent.run("11111111-1111-1111-1111-111111111111", ctx)

    assert res.status == "success"
    assert res.outputs["primary_topic"] == "billing"
    assert "billing" in res.outputs["topics"]
