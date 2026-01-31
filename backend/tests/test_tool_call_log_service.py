from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.cognitive_tooling.tool_call_log_service import ToolCallLogService


@pytest.mark.asyncio
async def test_tool_call_log_service_inserts_row() -> None:
    session = SimpleNamespace(add=Mock(), flush=AsyncMock())
    svc = ToolCallLogService(session)

    row = await svc.create(
        session_id="s1",
        iteration_id="i1",
        tool_name="memory.search",
        tool_input={"q": "x"},
        tool_output_summary={"count": 0},
        status="denied",
        denial_reason="missing permission",
    )

    assert row.tool_name == "memory.search"
    assert row.status == "denied"
    assert row.denial_reason == "missing permission"
    session.add.assert_called_once()
    session.flush.assert_awaited()
