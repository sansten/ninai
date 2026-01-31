from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.cognitive_tooling.policy_guard import PolicyGuard, ToolContext
from app.services.cognitive_tooling.tool_invoker import ToolInvoker
from app.services.cognitive_tooling.tool_registry import ToolRegistry, ToolSpec, ToolSensitivity
from app.services.permission_checker import AccessDecision


@pytest.mark.asyncio
async def test_tool_invoker_denied_does_not_call_tool_and_logs_summary_only() -> None:
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(return_value=AccessDecision(False, "nope", "rbac"))

    guard = PolicyGuard(permission_checker)
    registry = ToolRegistry()

    tool_called = {"called": False}

    async def handler(inp: dict) -> dict:
        tool_called["called"] = True
        return {"secret": "SHOULD_NOT_RUN"}

    registry.register(ToolSpec(name="memory.search", required_permissions=("memory:read:team",)), handler)

    log_service = AsyncMock()
    log_service.create = AsyncMock(return_value=type("Row", (), {"id": "tc1"})())

    invoker = ToolInvoker(registry=registry, guard=guard, log_service=log_service)

    res = await invoker.invoke(
        session_id="s1",
        iteration_id="i1",
        tool_name="memory.search",
        tool_input={"q": "hello", "secret": "dont_persist"},
        ctx=ToolContext(user_id="u", org_id="o"),
    )

    assert res.status == "denied"
    assert res.success is False
    assert res.tool_call_id == "tc1"
    assert tool_called["called"] is False

    log_service.create.assert_awaited_once()
    kwargs = log_service.create.await_args.kwargs
    assert kwargs["status"] == "denied"

    # Ensure raw tool_input values are not persisted by default
    persisted_input = kwargs["tool_input"]
    assert persisted_input["mode"] == "summary"
    assert "payload" not in persisted_input


@pytest.mark.asyncio
async def test_tool_invoker_success_logs_summary_only_by_default() -> None:
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(return_value=AccessDecision(True, "ok", "rbac"))

    guard = PolicyGuard(permission_checker)
    registry = ToolRegistry()

    async def handler(inp: dict) -> dict:
        return {"answer": 123, "secret": "dont_persist"}

    registry.register(ToolSpec(name="math.tool", required_permissions=("math:use",)), handler)

    log_service = AsyncMock()
    log_service.create = AsyncMock(return_value=type("Row", (), {"id": "tc2"})())

    invoker = ToolInvoker(registry=registry, guard=guard, log_service=log_service)

    res = await invoker.invoke(
        session_id="s1",
        iteration_id="i1",
        tool_name="math.tool",
        tool_input={"x": 1},
        ctx=ToolContext(user_id="u", org_id="o"),
    )

    assert res.status == "success"
    assert res.success is True
    assert res.tool_call_id == "tc2"
    assert res.output == {"answer": 123, "secret": "dont_persist"}

    kwargs = log_service.create.await_args.kwargs
    assert kwargs["status"] == "success"
    assert kwargs["tool_output_summary"]["mode"] == "summary"
    assert "payload" not in kwargs["tool_output_summary"]


@pytest.mark.asyncio
async def test_tool_invoker_can_persist_redacted_payload_when_allowed() -> None:
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(return_value=AccessDecision(True, "ok", "rbac"))

    guard = PolicyGuard(permission_checker)
    registry = ToolRegistry()

    async def handler(inp: dict) -> dict:
        return {"token": "abcd", "value": 1}

    spec = ToolSpec(
        name="persist.allowed",
        required_permissions=("x",),
        sensitivity=ToolSensitivity(
            allow_persist_input=True,
            allow_persist_output=True,
            redacted_output_fields=("token",),
        ),
    )
    registry.register(spec, handler)

    log_service = AsyncMock()
    log_service.create = AsyncMock(return_value=type("Row", (), {"id": "tc3"})())

    invoker = ToolInvoker(registry=registry, guard=guard, log_service=log_service)

    res = await invoker.invoke(
        session_id="s1",
        iteration_id="i1",
        tool_name="persist.allowed",
        tool_input={"a": 1},
        ctx=ToolContext(user_id="u", org_id="o"),
    )

    assert res.status == "success"
    assert res.success is True
    assert res.tool_call_id == "tc3"

    kwargs = log_service.create.await_args.kwargs
    assert kwargs["tool_input"]["mode"] == "persisted"
    assert kwargs["tool_output_summary"]["mode"] == "persisted"
    assert kwargs["tool_output_summary"]["payload"]["token"] == "[REDACTED]"
