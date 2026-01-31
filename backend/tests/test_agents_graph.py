import pytest

from app.agents.graph_linking_agent import GraphLinkingAgent


@pytest.mark.asyncio
async def test_graph_linking_builds_nodes_and_edges_from_enrichment():
    agent = GraphLinkingAgent()
    ctx = {
        "tenant": {"org_id": "org", "org_slug": None},
        "actor": {"user_id": "u", "roles": []},
        "memory": {
            "id": "m",
            "storage": "long_term",
            "content": "Refund requested for order 123.",
            "classification": "internal",
            "enrichment": {
                "metadata": {"tags": ["billing"], "entities": {"order_id": ["123"], "email": ["a@b.com"]}},
                "topics": {"topics": ["billing"], "primary_topic": "billing"},
            },
        },
        "runtime": {"job_id": "t", "attempt": 1, "max_attempts": 1},
    }

    res = await agent.run("m", ctx)
    graph = res.outputs.get("graph")
    assert isinstance(graph, dict)
    assert len(graph.get("nodes") or []) >= 3
    assert len(graph.get("edges") or []) >= 2
