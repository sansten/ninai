from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from app.services.cognitive_tooling.policy_guard import PolicyGuard, ToolContext
from app.services.cognitive_tooling.tool_registry import ToolSpec
from app.services.permission_checker import AccessDecision


@pytest.mark.asyncio
async def test_policy_guard_allows_when_permissions_granted() -> None:
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(return_value=AccessDecision(True, "ok", "rbac"))

    guard = PolicyGuard(permission_checker)
    tool = ToolSpec(name="memory.read", required_permissions=("memory:read:team",))

    decision = await guard.authorize(tool=tool, ctx=ToolContext(user_id="u", org_id="o"))
    assert decision.allowed is True


@pytest.mark.asyncio
async def test_policy_guard_denies_when_permission_missing() -> None:
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(return_value=AccessDecision(False, "no", "rbac"))

    guard = PolicyGuard(permission_checker)
    tool = ToolSpec(name="memory.delete", required_permissions=("memory:delete:team",))

    decision = await guard.authorize(tool=tool, ctx=ToolContext(user_id="u", org_id="o"))
    assert decision.allowed is False
    assert "no" in decision.reason


@pytest.mark.asyncio
async def test_policy_guard_denies_without_required_justification() -> None:
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(return_value=AccessDecision(True, "ok", "rbac"))

    guard = PolicyGuard(permission_checker)
    tool = ToolSpec(name="sensitive.read", required_permissions=(), require_justification=True)

    decision = await guard.authorize(tool=tool, ctx=ToolContext(user_id="u", org_id="o", justification=""))
    assert decision.allowed is False
    assert "Justification" in decision.reason
