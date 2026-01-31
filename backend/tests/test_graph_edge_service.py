from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.graph_edge_service import (
    GraphEdgeService,
    extract_edges_from_graph_outputs,
)


def test_extract_edges_from_graph_outputs_dedupes_and_normalizes():
    outputs = {
        "graph": {
            "nodes": [],
            "edges": [
                {"source": " memory:m1 ", "target": "tag:billing", "relation": "tagged", "weight": 2},
                {"source": "memory:m1", "target": "tag:billing", "relation": "tagged", "weight": 0.3},
                {"source": "", "target": "x", "relation": "r"},
                "not-a-dict",
            ],
        }
    }

    edges = extract_edges_from_graph_outputs(outputs)
    assert len(edges) == 1
    assert edges[0].from_node == "memory:m1"
    assert edges[0].to_node == "tag:billing"
    assert edges[0].relation == "tagged"
    # weight is clamped to 0..1
    assert edges[0].weight == 1.0


def test_extract_edges_from_graph_outputs_missing_graph_is_empty():
    assert extract_edges_from_graph_outputs(None) == []
    assert extract_edges_from_graph_outputs({}) == []
    assert extract_edges_from_graph_outputs({"graph": {}}) == []


@pytest.mark.asyncio
async def test_graph_edge_service_upsert_edges_calls_execute_and_flush():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=SimpleNamespace(rowcount=1))
    session.flush = AsyncMock()

    svc = GraphEdgeService(session)
    inserted = await svc.upsert_edges_for_memory(
        organization_id="org",
        memory_id="mem",
        outputs={"graph": {"edges": [{"source": "memory:mem", "target": "tag:x", "relation": "tagged"}]}} ,
    )

    assert inserted == 1
    assert session.execute.call_count == 1
    assert session.flush.call_count == 1
