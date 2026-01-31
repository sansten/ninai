from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.cognitive_tooling.tool_invoker import ToolInvoker
from app.services.cognitive_tooling.tool_registry import ToolRegistry, ToolSpec
from app.services.cognitive_tooling.policy_guard import ToolContext
from app.services.permission_checker import AccessDecision


@pytest.mark.asyncio
async def test_tool_invoker_propagates_reliability_warning_on_success():
    registry = ToolRegistry()

    async def handler(inp: dict):
        return {"ok": True}

    registry.register(ToolSpec(name="t1"), handler)

    guard = AsyncMock()
    guard.authorize = AsyncMock(
        return_value=AccessDecision(
            allowed=True,
            reason="Allowed",
            method="rbac",
            details={"tool": "t1", "reliability_warning": "low reliability"},
        )
    )

    log_service = AsyncMock()
    log_service.create = AsyncMock(return_value=SimpleNamespace(id="log1"))

    invoker = ToolInvoker(registry=registry, guard=guard, log_service=log_service)

    res = await invoker.invoke(
        session_id="s1",
        iteration_id="i1",
        tool_name="t1",
        tool_input={"a": 1},
        ctx=ToolContext(user_id="u", org_id="o"),
        swallow_exceptions=True,
    )

    assert res.status == "success"
    assert res.success is True
    assert res.warnings == ["low reliability"]


@pytest.mark.asyncio
async def test_tool_invoker_propagates_reliability_warning_on_denied():
    registry = ToolRegistry()

    async def handler(inp: dict):
        return {"ok": True}

    registry.register(ToolSpec(name="t1", require_justification=True), handler)

    guard = AsyncMock()
    guard.authorize = AsyncMock(
        return_value=AccessDecision(
            allowed=False,
            reason="Justification required",
            method="policy",
            details={"tool": "t1", "reliability_warning": "needs justification"},
        )
    )

    log_service = AsyncMock()
    log_service.create = AsyncMock(return_value=SimpleNamespace(id="log1"))

    invoker = ToolInvoker(registry=registry, guard=guard, log_service=log_service)

    res = await invoker.invoke(
        session_id="s1",
        iteration_id="i1",
        tool_name="t1",
        tool_input={"a": 1},
        ctx=ToolContext(user_id="u", org_id="o"),
        swallow_exceptions=True,
    )

    assert res.status == "denied"
    assert res.success is False
    assert res.warnings == ["needs justification"]
