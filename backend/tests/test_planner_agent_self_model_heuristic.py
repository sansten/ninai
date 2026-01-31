from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.config import settings
from app.services.cognitive_loop.planner_agent import PlannerAgent


@pytest.mark.asyncio
async def test_heuristic_planner_adds_extra_evidence_step_when_low_confidence_domain(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_STRATEGY", "heuristic", raising=False)

    agent = PlannerAgent()

    plan = await agent.plan(
        goal="Handle a tricky legal question",
        evidence_cards=[],
        available_tools=["memory.search"],
        self_model_summary={
            "low_confidence_domains": ["legal"],
            "recommended_evidence_multiplier": 2,
        },
    )

    step_ids = [s.step_id for s in plan.steps]
    assert step_ids == ["S1", "S2"]
    assert plan.steps[0].tool == "memory.search"
    assert plan.steps[1].tool == "memory.search"
    assert int(plan.steps[0].tool_input_hint.get("limit")) == 20


@pytest.mark.asyncio
async def test_heuristic_planner_scales_limit_by_multiplier_without_extra_step(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_STRATEGY", "heuristic", raising=False)

    agent = PlannerAgent()

    plan = await agent.plan(
        goal="Summarize the last sprint",
        evidence_cards=[],
        available_tools=["memory.search"],
        self_model_summary={
            "low_confidence_domains": [],
            "recommended_evidence_multiplier": 3,
        },
    )

    step_ids = [s.step_id for s in plan.steps]
    assert step_ids == ["S1"]
    assert int(plan.steps[0].tool_input_hint.get("limit")) == 30
