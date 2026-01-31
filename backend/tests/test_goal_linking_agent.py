"""Unit tests for GoalLinkingAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.goal_agents import GoalLinkingAgentOutput
from app.services.goal_linking_agent import GoalLinkingAgent


@pytest.mark.asyncio
async def test_goal_linking_agent_suggests_evidence_link():
    """Test that GoalLinkingAgent suggests relevant evidence links."""
    
    mock_output = GoalLinkingAgentOutput(
        links=[
            {
                "goal_id": "goal-123",
                "node_id": "node-456",
                "memory_id": "mem-789",
                "link_type": "evidence",
                "confidence": 0.92,
                "reason": "Memory contains supporting data for sales analysis",
            }
        ],
        confidence=0.92,
    )
    
    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(return_value=mock_output.model_dump())
    
    agent = GoalLinkingAgent(llm_client=mock_llm)
    
    result = await agent.suggest_links(
        memory={
            "id": "mem-789",
            "content": "Q1 sales increased by 15% compared to Q4",
            "scope": "team",
        },
        active_goals=[
            {
                "id": "goal-123",
                "title": "Analyze quarterly sales trends",
                "nodes": [{"id": "node-456", "title": "Compile Q1 data"}],
            }
        ],
    )
    
    assert len(result.links) == 1
    assert result.links[0].link_type == "evidence"
    assert result.links[0].confidence == 0.92


@pytest.mark.asyncio
async def test_goal_linking_agent_no_relevant_links():
    """Test that agent returns empty links when memory not relevant."""
    
    mock_output = GoalLinkingAgentOutput(
        links=[],
        confidence=0.0,
    )
    
    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(return_value=mock_output.model_dump())
    
    agent = GoalLinkingAgent(llm_client=mock_llm)
    
    result = await agent.suggest_links(
        memory={
            "id": "mem-001",
            "content": "Random unrelated information",
            "scope": "personal",
        },
        active_goals=[
            {"id": "goal-999", "title": "Complete project X", "nodes": []}
        ],
    )
    
    assert len(result.links) == 0
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_goal_linking_agent_fail_closed_on_error():
    """Test fail-closed behavior when LLM call fails."""
    
    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(side_effect=Exception("LLM service unavailable"))
    
    agent = GoalLinkingAgent(llm_client=mock_llm)
    
    result = await agent.suggest_links(
        memory={"id": "m1", "content": "test", "scope": "personal"},
        active_goals=[],
    )
    
    # Fail-closed: no links on error
    assert len(result.links) == 0
    assert result.confidence == 0.0
