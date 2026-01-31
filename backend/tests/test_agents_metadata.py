from __future__ import annotations

import pytest

from app.agents.metadata_extraction_agent import MetadataExtractionAgent


@pytest.mark.asyncio
async def test_metadata_extraction_agent_extracts_entities_and_tags():
    agent = MetadataExtractionAgent()

    ctx = {
        "tenant": {"org_id": "00000000-0000-0000-0000-000000000000", "org_slug": None},
        "memory": {
            "content": "Customer requested a refund for order #A123. Email: Alice@Example.com",
            "classification": "internal",
        },
        "runtime": {"job_id": "m1"},
    }

    res = await agent.run("11111111-1111-1111-1111-111111111111", ctx)

    assert res.status == "success"
    assert "billing" in (res.outputs.get("tags") or [])
    entities = res.outputs.get("entities") or {}
    assert entities.get("email") == ["alice@example.com"]
    assert "A123" in (entities.get("order_id") or [])
