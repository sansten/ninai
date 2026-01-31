"""GoalGraph agent schema validation tests."""

from __future__ import annotations

import pytest

from app.schemas.goal_agents import GoalLinkingAgentOutput, GoalPlannerAgentOutput


def test_goal_planner_schema_valid_minimal():
    payload = {
        "create_goal": True,
        "goal": {
            "title": "Ship feature X",
            "description": "Do the thing",
            "goal_type": "project",
            "visibility_scope": "personal",
            "scope_id": None,
            "priority": 2,
            "due_at": None,
        },
        "nodes": [
            {
                "temp_id": "N1",
                "parent_temp_id": None,
                "node_type": "task",
                "title": "Gather evidence",
                "description": "",
                "success_criteria": ["2 evidence memories"],
                "expected_outputs": {},
            }
        ],
        "edges": [],
        "confidence": 0.6,
        "evidence_memory_ids": [],
    }

    parsed = GoalPlannerAgentOutput.model_validate(payload)
    assert parsed.create_goal is True
    assert parsed.goal is not None
    assert parsed.goal.goal_type == "project"


def test_goal_linking_schema_valid():
    payload = {
        "links": [
            {
                "goal_id": "00000000-0000-0000-0000-000000000001",
                "node_id": None,
                "memory_id": "00000000-0000-0000-0000-000000000002",
                "link_type": "evidence",
                "confidence": 0.7,
                "reason": "Tag overlap",
            }
        ],
        "confidence": 0.7,
    }

    parsed = GoalLinkingAgentOutput.model_validate(payload)
    assert len(parsed.links) == 1
    assert parsed.links[0].link_type == "evidence"


def test_goal_linking_schema_rejects_bad_confidence():
    with pytest.raises(Exception):
        GoalLinkingAgentOutput.model_validate({"links": [], "confidence": 2.0})
