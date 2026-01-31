import pytest

from app.agents.logseq_export_agent import LogseqExportAgent


@pytest.mark.asyncio
async def test_logseq_export_agent_renders_markdown():
    agent = LogseqExportAgent()
    ctx = {
        "tenant": {"org_id": "org", "org_slug": None},
        "actor": {"user_id": "u", "roles": []},
        "memory": {
            "id": "m",
            "storage": "long_term",
            "content": "Payment failed for order 123.",
            "classification": "internal",
            "enrichment": {"metadata": {"tags": ["billing"], "entities": {"order_id": ["123"]}}},
        },
        "runtime": {"job_id": "t", "attempt": 1, "max_attempts": 1},
    }

    res = await agent.run("m", ctx)
    md = res.outputs.get("markdown")
    assert isinstance(md, str)
    assert "billing" in md.lower()
