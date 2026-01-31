from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.types import AgentResult
from app.services.agent_runner import AgentRunner, PipelineContext


@pytest.mark.asyncio
async def test_materialize_side_effects_emits_tool_events_for_graph_linking(monkeypatch):
    runner = AgentRunner(service_user_id="u")

    session = AsyncMock(spec=AsyncSession)

    events: list[dict] = []

    async def sink(e: dict) -> None:
        events.append(e)

    class _FakeGraphEdgeService:
        def __init__(self, _session):
            self._session = _session

        async def upsert_edges_for_memory(self, **kwargs):
            return None

    monkeypatch.setattr(
        "app.services.agent_runner.GraphEdgeService",
        _FakeGraphEdgeService,
        raising=True,
    )

    ctx = PipelineContext(org_id="o1", memory_id="m1", initiator_user_id="u1", trace_id="t1")
    result = AgentResult(
        agent_name="GraphLinkingAgent",
        agent_version="1",
        memory_id="m1",
        status="success",
        confidence=1.0,
        outputs={"edges": [{"source": "a", "target": "b"}]},
        warnings=[],
        errors=[],
        started_at=datetime(2026, 1, 23, tzinfo=timezone.utc),
        finished_at=datetime(2026, 1, 23, tzinfo=timezone.utc),
        trace_id="t1",
        provenance=[],
    )

    await runner._materialize_side_effects(
        session=session,
        ctx=ctx,
        agent_name="GraphLinkingAgent",
        result=result,
        scope="personal",
        scope_id=None,
        tool_event_sink=sink,
    )

    assert len(events) == 2
    assert events[0]["event_type"] == "tool_call"
    assert events[1]["event_type"] == "tool_result"
    assert events[0]["payload"]["tool"] == "GraphEdgeService.upsert_edges_for_memory"
    assert events[1]["payload"]["tool"] == "GraphEdgeService.upsert_edges_for_memory"
    assert events[1]["payload"]["ok"] is True
    assert "duration_ms" in events[1]["payload"]
