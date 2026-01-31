from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_run import AgentRun
from app.models.agent_run_event import AgentRunEvent
from app.services.agent_runner import AgentRunner


@pytest.mark.asyncio
async def test_tool_event_sink_appends_incrementing_events():
    runner = AgentRunner(service_user_id="u")

    session = AsyncMock(spec=AsyncSession)
    session.flush = AsyncMock()
    session.add = Mock()

    now = datetime(2026, 1, 23, tzinfo=timezone.utc)
    run_row = AgentRun(
        organization_id="o1",
        memory_id="m1",
        agent_name="A",
        agent_version="1",
        inputs_hash="h" * 64,
        status="retry",
        confidence=0.0,
        outputs={},
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id="t1",
    )
    run_row.id = "ar1"

    sink = runner._create_tool_event_sink(session=session, run_row=run_row)

    await sink({"event_type": "tool_call", "summary_text": "call", "payload": {"tool": "x"}})
    await sink({"event_type": "tool_result", "summary_text": "ok", "payload": {"ok": True}})

    assert session.add.call_count == 2

    e1 = session.add.call_args_list[0].args[0]
    e2 = session.add.call_args_list[1].args[0]
    assert isinstance(e1, AgentRunEvent)
    assert isinstance(e2, AgentRunEvent)

    assert e1.event_type == "tool_call"
    assert e2.event_type == "tool_result"
    assert e1.step_index == 10
    assert e2.step_index == 11
    assert e1.summary_text == "call"
    assert e2.summary_text == "ok"
    assert e1.trace_id == "t1"
