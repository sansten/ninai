import pytest

from app.agents.pattern_detection_agent import PatternDetectionAgent


@pytest.mark.asyncio
async def test_pattern_detection_heuristic_detects_resolution_pattern():
    agent = PatternDetectionAgent()
    ctx = {
        "tenant": {"org_id": "org", "org_slug": None},
        "actor": {"user_id": "u", "roles": []},
        "memory": {
            "id": "m",
            "storage": "long_term",
            "content": "Issue escalated. Root cause fixed and incident resolved. Next steps: monitor.",
            "classification": "internal",
            "enrichment": {
                "topics": {"topics": ["engineering"], "primary_topic": "engineering"},
                "metadata": {"tags": ["engineering"], "entities": {}},
            },
        },
        "runtime": {"job_id": "t", "attempt": 1, "max_attempts": 1},
    }

    res = await agent.run("m", ctx)
    assert res.status == "success"
    patterns = res.outputs.get("patterns")
    assert isinstance(patterns, list)
    assert any(p.get("pattern") == "issue_resolution" for p in patterns)
