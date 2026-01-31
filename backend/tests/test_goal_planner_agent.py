"""Unit tests for GoalPlannerAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.goal_agents import GoalPlannerAgentOutput
from app.services.goal_planner_agent import GoalPlannerAgent


@pytest.mark.asyncio
async def test_goal_planner_agent_proposes_goal():
    """Test that GoalPlannerAgent proposes a valid goal from user request."""
    
    # Mock OllamaClient response
    mock_output = GoalPlannerAgentOutput(
        create_goal=True,
        goal={
            "title": "Complete Q1 Sales Report",
            "description": "Prepare comprehensive sales report for Q1 2026",
            "goal_type": "task",
            "visibility_scope": "team",
            "scope_id": None,
            "priority": 3,
            "due_at": None,
        },
        nodes=[
            {
                "temp_id": "N1",
                "parent_temp_id": None,
                "node_type": "task",
                "title": "Gather sales data",
                "description": "Collect Q1 sales metrics from CRM",
                "success_criteria": ["Data exported to CSV"],
                "expected_outputs": {"format": "csv"},
            },
        ],
        edges=[],
        confidence=0.85,
        evidence_memory_ids=[],
    )
    
    # Mock the LLM client
    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(return_value=mock_output.model_dump())
    
    agent = GoalPlannerAgent(llm_client=mock_llm)
    
    result = await agent.propose_goal(
        user_request="I need to prepare the Q1 sales report",
        session_context={"outcome": "pass", "evidence_count": 3},
        existing_goals=[],
    )
    
    assert result.create_goal is True
    assert result.goal.title == "Complete Q1 Sales Report"
    assert result.confidence == 0.85
    assert len(result.nodes) == 1


@pytest.mark.asyncio
async def test_goal_planner_agent_skips_duplicate_goal():
    """Test that GoalPlannerAgent doesn't create duplicate goals."""
    
    mock_output = GoalPlannerAgentOutput(
        create_goal=False,
        goal={
            "title": "Placeholder",
            "goal_type": "task",
            "visibility_scope": "personal",
        },
        nodes=[],
        edges=[],
        confidence=0.0,
        evidence_memory_ids=[],
    )
    
    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(return_value=mock_output.model_dump())
    
    agent = GoalPlannerAgent(llm_client=mock_llm)
    
    result = await agent.propose_goal(
        user_request="I need to prepare the Q1 sales report",
        session_context={},
        existing_goals=[
            {"title": "Complete Q1 Sales Report", "status": "active"}
        ],
    )
    
    assert result.create_goal is False
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_goal_planner_agent_fail_closed_on_error():
    """Test fail-closed behavior when LLM call fails."""
    
    mock_llm = AsyncMock()
    mock_llm.complete_json = AsyncMock(side_effect=Exception("LLM service unavailable"))
    
    agent = GoalPlannerAgent(llm_client=mock_llm)
    
    result = await agent.propose_goal(
        user_request="Create a new project",
        session_context={},
        existing_goals=[],
    )
    
    # Fail-closed: should not create goal on error
    assert result.create_goal is False
    assert result.confidence == 0.0
