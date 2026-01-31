from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.schemas.cognitive import PlannerOutput
from app.services.cognitive_loop.executor_agent import ExecutorAgent
from app.services.cognitive_tooling.policy_guard import PolicyGuard, ToolContext
from app.services.cognitive_tooling.tool_invoker import ToolInvoker
from app.services.cognitive_tooling.tool_registry import ToolRegistry, ToolSpec
from app.services.permission_checker import AccessDecision


def _plan_with_one_tool_step(tool_name: str | None) -> PlannerOutput:
    return PlannerOutput(
        objective="goal",
        assumptions=[],
        constraints=[],
        required_tools=[],
        steps=[
            {
                "step_id": "S1",
                "action": "do",
                "tool": tool_name,
                "tool_input_hint": {"x": 1},
                "expected_output": "y",
                "success_criteria": ["ok"],
                "risk_notes": [],
            }
        ],
        stop_conditions=[],
        confidence=0.6,
    )


@pytest.mark.asyncio
async def test_executor_agent_denied_tool_does_not_crash() -> None:
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(return_value=AccessDecision(False, "nope", "rbac"))
    guard = PolicyGuard(permission_checker)

    registry = ToolRegistry()

    async def handler(inp: dict) -> dict:
        return {"ok": True}

    registry.register(ToolSpec(name="memory.search", required_permissions=("memory:read:team",)), handler)

    log_service = AsyncMock()
    log_service.create = AsyncMock(return_value=type("Row", (), {"id": "tc1"})())

    invoker = ToolInvoker(registry=registry, guard=guard, log_service=log_service)
    agent = ExecutorAgent(tool_invoker=invoker)

    plan = _plan_with_one_tool_step("memory.search")
    out = await agent.execute(
        session_id="s1",
        iteration_id="i1",
        plan=plan,
        ctx=ToolContext(user_id="u", org_id="o"),
    )

    assert out.overall_status == "partial"
    assert out.step_results[0].status == "denied"
    assert out.step_results[0].tool_call_id == "tc1"


@pytest.mark.asyncio
async def test_executor_agent_success_step() -> None:
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(return_value=AccessDecision(True, "ok", "rbac"))
    guard = PolicyGuard(permission_checker)

    registry = ToolRegistry()

    async def handler(inp: dict) -> dict:
        return {"ok": True}

    registry.register(ToolSpec(name="math.tool", required_permissions=("math:use",)), handler)

    log_service = AsyncMock()
    log_service.create = AsyncMock(return_value=type("Row", (), {"id": "tc2"})())

    invoker = ToolInvoker(registry=registry, guard=guard, log_service=log_service)
    agent = ExecutorAgent(tool_invoker=invoker)

    plan = _plan_with_one_tool_step("math.tool")
    out = await agent.execute(
        session_id="s1",
        iteration_id="i1",
        plan=plan,
        ctx=ToolContext(user_id="u", org_id="o"),
    )

    assert out.overall_status == "success"
    assert out.step_results[0].status == "success"
    assert out.step_results[0].tool_call_id == "tc2"


@pytest.mark.asyncio
async def test_executor_agent_skips_step_without_tool() -> None:
    registry = ToolRegistry()
    guard = PolicyGuard(AsyncMock())
    log_service = AsyncMock()
    log_service.create = AsyncMock(return_value=type("Row", (), {"id": "tcX"})())

    invoker = ToolInvoker(registry=registry, guard=guard, log_service=log_service)
    agent = ExecutorAgent(tool_invoker=invoker)

    plan = _plan_with_one_tool_step(None)
    out = await agent.execute(session_id="s1", iteration_id="i1", plan=plan, ctx=ToolContext(user_id="u", org_id="o"))

    assert out.overall_status == "success"
    assert out.step_results[0].status == "skipped"
