"""Tests for PlannerAgent tool_recommendations and strategy_hints prompt injection."""
from __future__ import annotations

import json

import pytest

from app.agents.llm.base import LLMClient
from app.schemas.cognitive import PlannerOutput
from app.services.cognitive_loop.planner_agent import PlannerAgent

_VALID_PLAN = {
    "objective": "Do X",
    "assumptions": [],
    "constraints": [],
    "required_tools": ["memory.search"],
    "steps": [
        {
            "step_id": "S1",
            "action": "search",
            "tool": "memory.search",
            "tool_input_hint": {"query": "x"},
            "expected_output": "results",
            "success_criteria": ["found"],
            "risk_notes": [],
        }
    ],
    "stop_conditions": [],
    "confidence": 0.8,
}


class CapturingLLM(LLMClient):
    """LLM stub that captures the last prompt and returns a valid plan."""

    def __init__(self, payload: dict | None = None):
        self.last_prompt: str = ""
        self._payload = payload or dict(_VALID_PLAN)

    async def complete_json(self, *, prompt: str, schema_hint: dict, tool_event_sink=None) -> dict:
        self.last_prompt = prompt
        return dict(self._payload)


@pytest.mark.asyncio
async def test_tool_recommendations_serialized_into_prompt(monkeypatch):
    """tool_recommendations appear in the constructed LLM prompt."""
    from app.core import config as cfg
    monkeypatch.setattr(cfg.settings, "AGENT_STRATEGY", "llm", raising=False)

    llm = CapturingLLM()
    agent = PlannerAgent(llm_client=llm)

    recs = [
        {"tool_name": "memory.search", "score": 0.95, "success_rate": 0.9, "total_uses": 20},
        {"tool_name": "memory.get", "score": 0.80, "success_rate": 0.75, "total_uses": 10},
    ]
    await agent.plan(
        goal="analyze something",
        evidence_cards=[],
        available_tools=["memory.search", "memory.get"],
        tool_recommendations=recs,
    )

    assert "memory.search" in llm.last_prompt
    assert "0.95" in llm.last_prompt  # score present
    # Compact form excludes total_uses
    assert "total_uses" not in llm.last_prompt


@pytest.mark.asyncio
async def test_strategy_hints_serialized_into_prompt(monkeypatch):
    """strategy_hints appear in the constructed LLM prompt."""
    from app.core import config as cfg
    monkeypatch.setattr(cfg.settings, "AGENT_STRATEGY", "llm", raising=False)

    llm = CapturingLLM()
    agent = PlannerAgent(llm_client=llm)

    hints = {
        "goal_type": "analysis",
        "domain": "finance",
        "top_strategies": [
            {
                "tool_sequence": ["memory.search", "memory.get"],
                "success_rate": 0.9,
                "avg_iterations": 2.0,
                "avg_plan_confidence": 0.8,
                "sample_count": 5,
            }
        ],
        "note": "use these",
    }
    await agent.plan(
        goal="analyze financials",
        evidence_cards=[],
        available_tools=["memory.search"],
        strategy_hints=hints,
    )

    assert "top_strategies" in llm.last_prompt
    assert "memory.search" in llm.last_prompt
    assert "0.9" in llm.last_prompt


@pytest.mark.asyncio
async def test_none_tool_recommendations_produces_empty_json_in_prompt(monkeypatch):
    """None tool_recommendations → '[]' in prompt, no error."""
    from app.core import config as cfg
    monkeypatch.setattr(cfg.settings, "AGENT_STRATEGY", "llm", raising=False)

    llm = CapturingLLM()
    agent = PlannerAgent(llm_client=llm)

    await agent.plan(
        goal="do something",
        evidence_cards=[],
        available_tools=["memory.search"],
        tool_recommendations=None,
    )

    # Empty list serialises to [] in the prompt
    assert "[]" in llm.last_prompt


@pytest.mark.asyncio
async def test_none_strategy_hints_produces_empty_json_in_prompt(monkeypatch):
    """None strategy_hints → '{}' in prompt, no error."""
    from app.core import config as cfg
    monkeypatch.setattr(cfg.settings, "AGENT_STRATEGY", "llm", raising=False)

    llm = CapturingLLM()
    agent = PlannerAgent(llm_client=llm)

    await agent.plan(
        goal="do something",
        evidence_cards=[],
        available_tools=["memory.search"],
        strategy_hints=None,
    )

    assert "{}" in llm.last_prompt


@pytest.mark.asyncio
async def test_heuristic_mode_ignores_tool_recommendations(monkeypatch):
    """In heuristic mode, tool_recommendations and strategy_hints are ignored gracefully."""
    from app.core import config as cfg
    monkeypatch.setattr(cfg.settings, "AGENT_STRATEGY", "heuristic", raising=False)

    agent = PlannerAgent()

    plan = await agent.plan(
        goal="search for something",
        evidence_cards=[],
        available_tools=["memory.search"],
        tool_recommendations=[{"tool_name": "memory.search", "score": 0.9, "success_rate": 0.8, "total_uses": 5}],
        strategy_hints={"top_strategies": []},
    )

    assert isinstance(plan, PlannerOutput)
    assert plan.confidence == 0.25  # heuristic confidence


@pytest.mark.asyncio
async def test_compact_tool_recommendations_excludes_total_uses(monkeypatch):
    """Only tool_name, score, success_rate are forwarded; total_uses is stripped."""
    from app.core import config as cfg
    monkeypatch.setattr(cfg.settings, "AGENT_STRATEGY", "llm", raising=False)

    llm = CapturingLLM()
    agent = PlannerAgent(llm_client=llm)

    recs = [{"tool_name": "tool.A", "score": 0.7, "success_rate": 0.6, "total_uses": 999}]
    await agent.plan(
        goal="some goal",
        evidence_cards=[],
        available_tools=["tool.A"],
        tool_recommendations=recs,
    )

    # Parse the tool_recommendations section of the prompt
    prompt = llm.last_prompt
    # The compact JSON is between the TOOL PERFORMANCE CONTEXT header and the next section
    assert "tool.A" in prompt
    assert "999" not in prompt  # total_uses stripped
