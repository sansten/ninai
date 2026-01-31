from __future__ import annotations

import pytest

from app.agents.llm.base import LLMClient
from app.schemas.cognitive import CriticOutput
from app.services.cognitive_loop.critic_agent import CriticAgent


class FakeLLM(LLMClient):
    def __init__(self, payload: dict):
        self.payload = payload

    async def complete_json(self, *, prompt: str, schema_hint: dict, tool_event_sink=None) -> dict:
        return dict(self.payload)


@pytest.mark.asyncio
async def test_critic_agent_validates_llm_output() -> None:
    llm_payload = {
        "evaluation": "pass",
        "strengths": ["ok"],
        "issues": [],
        "followup_questions": [],
        "confidence": 0.8,
    }

    agent = CriticAgent(llm_client=FakeLLM(llm_payload))
    out = await agent.critique(
        goal="Do X",
        plan={"objective": "Do X"},
        execution={"overall_status": "success"},
        evidence_cards=[{"id": "m1", "summary": "s"}],
    )

    assert isinstance(out, CriticOutput)
    assert out.evaluation == "pass"
    assert out.confidence == 0.8


@pytest.mark.asyncio
async def test_critic_agent_falls_back_on_invalid_output() -> None:
    agent = CriticAgent(llm_client=FakeLLM({"not": "valid"}))
    out = await agent.critique(goal="Do Y", plan={}, execution={}, evidence_cards=[])

    assert out.evaluation == "needs_evidence"
    assert out.confidence <= 0.3
