"""Unit tests for PolicyGuard SelfModel integration."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.cognitive_tooling.policy_guard import PolicyGuard, ToolContext
from app.services.cognitive_tooling.tool_registry import ToolSpec
from app.services.permission_checker import AccessDecision


@pytest.mark.asyncio
async def test_policy_guard_requires_justification_for_unreliable_tool():
    """Test that PolicyGuard requires justification when tool has low reliability."""
    
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(
        return_value=AccessDecision(True, "ok", "rbac")
    )
    
    guard = PolicyGuard(permission_checker)
    tool = ToolSpec(name="crm.lookup", required_permissions=("crm:read",))
    
    # SelfModel indicates low reliability (< 80%)
    self_model = {
        "tool_reliability": {
            "crm.lookup": {
                "success_rate_30d": 0.65,  # Low reliability
                "sample_size_30d": 10,
            }
        }
    }
    
    ctx = ToolContext(
        user_id="u1",
        org_id="o1",
        justification="",  # No justification provided
        self_model=self_model,
    )
    
    decision = await guard.authorize(tool=tool, ctx=ctx)
    
    assert decision.allowed is False
    assert "justification required" in decision.reason.lower()
    assert decision.details.get("reliability_adjusted") is True


@pytest.mark.asyncio
async def test_policy_guard_allows_with_justification_for_unreliable_tool():
    """Test that PolicyGuard allows unreliable tool when justification provided."""
    
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(
        return_value=AccessDecision(True, "ok", "rbac")
    )
    
    guard = PolicyGuard(permission_checker)
    tool = ToolSpec(name="crm.lookup", required_permissions=("crm:read",))
    
    self_model = {
        "tool_reliability": {
            "crm.lookup": {
                "success_rate_30d": 0.70,
                "sample_size_30d": 8,
            }
        }
    }
    
    ctx = ToolContext(
        user_id="u1",
        org_id="o1",
        justification="Need customer data for sales report",
        self_model=self_model,
    )
    
    decision = await guard.authorize(tool=tool, ctx=ctx)
    
    assert decision.allowed is True
    assert "warning" in decision.reason.lower()
    assert decision.details.get("reliability_warning") is not None


@pytest.mark.asyncio
async def test_policy_guard_no_extra_requirement_for_reliable_tool():
    """Test that PolicyGuard doesn't add extra requirements for reliable tools."""
    
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(
        return_value=AccessDecision(True, "ok", "rbac")
    )
    
    guard = PolicyGuard(permission_checker)
    tool = ToolSpec(name="memory.search", required_permissions=("memory:read:team",))
    
    self_model = {
        "tool_reliability": {
            "memory.search": {
                "success_rate_30d": 0.95,  # High reliability
                "sample_size_30d": 50,
            }
        }
    }
    
    ctx = ToolContext(
        user_id="u1",
        org_id="o1",
        justification="",
        self_model=self_model,
    )
    
    decision = await guard.authorize(tool=tool, ctx=ctx)
    
    assert decision.allowed is True
    assert "warning" not in decision.reason.lower()
    assert decision.details.get("reliability_warning") is None


@pytest.mark.asyncio
async def test_policy_guard_ignores_insufficient_sample_size():
    """Test that PolicyGuard doesn't adjust requirements when sample size too small."""
    
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(
        return_value=AccessDecision(True, "ok", "rbac")
    )
    
    guard = PolicyGuard(permission_checker)
    tool = ToolSpec(name="new.tool", required_permissions=())
    
    self_model = {
        "tool_reliability": {
            "new.tool": {
                "success_rate_30d": 0.50,  # Low but only 2 samples
                "sample_size_30d": 2,  # Too few samples
            }
        }
    }
    
    ctx = ToolContext(
        user_id="u1",
        org_id="o1",
        justification="",
        self_model=self_model,
    )
    
    decision = await guard.authorize(tool=tool, ctx=ctx)
    
    # Should allow since sample size < 3 (not statistically significant)
    assert decision.allowed is True
    assert decision.details.get("reliability_warning") is None


@pytest.mark.asyncio
async def test_policy_guard_handles_missing_self_model():
    """Test that PolicyGuard works without SelfModel data."""
    
    permission_checker = AsyncMock()
    permission_checker.check_permission = AsyncMock(
        return_value=AccessDecision(True, "ok", "rbac")
    )
    
    guard = PolicyGuard(permission_checker)
    tool = ToolSpec(name="some.tool", required_permissions=())
    
    ctx = ToolContext(
        user_id="u1",
        org_id="o1",
        self_model=None,  # No SelfModel data
    )
    
    decision = await guard.authorize(tool=tool, ctx=ctx)
    
    assert decision.allowed is True
