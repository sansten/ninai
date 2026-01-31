from __future__ import annotations

import pytest

from app.agents.llm.base import LLMClient
from app.schemas.cognitive import PlannerOutput
from app.services.cognitive_loop.planner_agent import PlannerAgent


class FakeLLM(LLMClient):
    def __init__(self, payload: dict):
        self.payload = payload

    async def complete_json(self, *, prompt: str, schema_hint: dict, tool_event_sink=None) -> dict:
        return dict(self.payload)


@pytest.mark.asyncio
async def test_planner_agent_validates_llm_output() -> None:
    llm_payload = {
        "objective": "Do X",
        "assumptions": ["A"],
        "constraints": ["C"],
        "required_tools": [],
        "steps": [
            {
                "step_id": "S1",
                "action": "A1",
                "tool": None,
                "tool_input_hint": {},
                "expected_output": "E",
                "success_criteria": ["SC"],
                "risk_notes": [],
            }
        ],
        "stop_conditions": ["Stop"],
        "confidence": 0.7,
    }

    agent = PlannerAgent(llm_client=FakeLLM(llm_payload))
    plan = await agent.plan(goal="Do X", evidence_cards=[{"id": "m1", "summary": "s"}], available_tools=["memory.search"])
    assert isinstance(plan, PlannerOutput)
    assert plan.objective == "Do X"
    assert plan.confidence == 0.7


@pytest.mark.asyncio
async def test_planner_agent_falls_back_on_invalid_output() -> None:
    agent = PlannerAgent(llm_client=FakeLLM({"not": "valid"}))
    plan = await agent.plan(goal="Do Y", evidence_cards=[], available_tools=["memory.search"])
    assert plan.objective == "Do Y"
    assert plan.confidence <= 0.3
    assert len(plan.steps) >= 1
