from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.types import AgentResult
from app.models.agent_run import AgentRun
from app.models.agent_run_event import AgentRunEvent
from app.services.agent_runner import AgentRunner


class _ScalarOneOrNoneResult:
    def __init__(self, item):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


@pytest.mark.asyncio
async def test_get_or_create_run_row_appends_run_started_event():
    runner = AgentRunner(service_user_id="u")
    session = AsyncMock(spec=AsyncSession)

    session.execute = AsyncMock(return_value=_ScalarOneOrNoneResult(None))
    session.flush = AsyncMock()
    session.add = Mock()

    started = datetime(2026, 1, 23, tzinfo=timezone.utc)

    row = await runner._get_or_create_run_row(
        session=session,
        org_id="o1",
        memory_id="m1",
        agent_name="A",
        agent_version="1",
        inputs_hash="h" * 64,
        trace_id="t1",
        started_at=started,
    )

    assert isinstance(row, AgentRun)
    assert session.add.call_count == 2

    added_types = [type(call.args[0]) for call in session.add.call_args_list]
    assert AgentRun in added_types
    assert AgentRunEvent in added_types

    # run_started event is added after the AgentRun row
    event_obj = [call.args[0] for call in session.add.call_args_list if isinstance(call.args[0], AgentRunEvent)][0]
    assert event_obj.event_type == "run_started"
    assert event_obj.step_index == 0
    assert event_obj.organization_id == "o1"
    assert event_obj.memory_id == "m1"
    assert "started" in event_obj.summary_text


@pytest.mark.asyncio
async def test_persist_result_appends_run_result_event():
    runner = AgentRunner(service_user_id="u")
    session = AsyncMock(spec=AsyncSession)

    session.flush = AsyncMock()
    session.add = Mock()

    started = datetime(2026, 1, 23, tzinfo=timezone.utc)
    finished = datetime(2026, 1, 23, 0, 0, 5, tzinfo=timezone.utc)

    row = AgentRun(
        organization_id="o1",
        memory_id="m1",
        agent_name="A",
        agent_version="1",
        inputs_hash="x" * 64,
        status="retry",
        confidence=0.0,
        outputs={},
        warnings=[],
        errors=[],
        started_at=started,
        finished_at=started,
        trace_id="t1",
    )
    # Give it an id to avoid reliance on DB flush
    row.id = "ar1"

    result = AgentResult(
        agent_name="A",
        agent_version="1",
        memory_id="m1",
        status="success",
        confidence=0.7,
        outputs={"k": 1},
        warnings=["w"],
        errors=[],
        started_at=started,
        finished_at=finished,
        trace_id="t1",
        provenance=[],
    )

    await runner._persist_result(session, row, result, "h" * 64)

    assert session.add.call_count == 1
    event_obj = session.add.call_args_list[0].args[0]
    assert isinstance(event_obj, AgentRunEvent)
    assert event_obj.event_type == "run_result"
    assert event_obj.step_index == 1
    assert event_obj.payload["status"] == "success"
    assert event_obj.payload["outputs_keys"] == ["k"]
    assert event_obj.trace_id == "t1"
    assert "success" in event_obj.summary_text
