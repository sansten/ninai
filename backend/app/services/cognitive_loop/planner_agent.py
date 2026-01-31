"""PlannerAgent for CognitiveLoop.

Produces a strict PlannerOutput JSON plan based on:
- user goal
- RLS-filtered evidence cards (summary-only)
- available tool names

Fail-closed behavior:
- If LLM is disabled or returns invalid output, fall back to a conservative
  heuristic plan with low confidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from app.agents.llm.base import LLMClient
from app.agents.llm.ollama import OllamaClient
from app.core.config import settings
from app.schemas.cognitive import PlannerOutput
from app.services.cognitive_loop.prompt_loader import load_prompt_text


class PlannerAgent:
    name = "planner_agent"
    version = "v1"
    prompt_version = "planner_v1"

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.llm_client = llm_client

    def _default_llm(self) -> LLMClient:
        return OllamaClient(
            base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
            model=str(getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")),
            timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
            max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
        )

    def _heuristic_plan(
        self,
        *,
        goal: str,
        available_tools: list[str],
        self_model_summary: dict[str, Any] | None = None,
    ) -> PlannerOutput:
        # Conservative baseline that always asks for more evidence.
        tool = "memory.search" if "memory.search" in set(available_tools) else None
        required_tools = [tool] if tool else []

        low_domains: list[str] = []
        multiplier = 1
        if isinstance(self_model_summary, dict):
            try:
                low_domains = list(self_model_summary.get("low_confidence_domains") or [])
            except Exception:
                low_domains = []
            try:
                multiplier = int(self_model_summary.get("recommended_evidence_multiplier") or 1)
            except Exception:
                multiplier = 1
        multiplier = max(1, min(3, multiplier))

        base_limit = 10
        scaled_limit = max(1, min(30, base_limit * multiplier))

        return PlannerOutput(
            objective=goal,
            assumptions=["Evidence may be incomplete."],
            constraints=[
                "Fail closed if uncertain.",
                "Do not access data without authorization.",
                "Prefer evidence citations over speculation.",
            ],
            required_tools=required_tools,
            steps=[
                {
                    "step_id": "S1",
                    "action": "Retrieve supporting evidence from memory.",
                    "tool": tool,
                    "tool_input_hint": {"query": goal, "limit": scaled_limit},
                    "expected_output": "A set of relevant evidence cards with summaries.",
                    "success_criteria": ["At least 3 relevant evidence cards."],
                    "risk_notes": ["Tool may be denied by policy; ask for user clarification if denied."],
                }
            ]
            + (
                [
                    {
                        "step_id": "S2",
                        "action": "Because domain confidence is low, gather additional evidence before proceeding.",
                        "tool": tool,
                        "tool_input_hint": {"query": goal, "limit": max(12, scaled_limit)},
                        "expected_output": "More evidence cards to reduce uncertainty.",
                        "success_criteria": ["At least 5 relevant evidence cards."],
                        "risk_notes": [
                            "Fail closed if evidence remains insufficient; request clarification.",
                        ],
                    }
                ]
                if (tool and low_domains)
                else []
            ),
            stop_conditions=["If evidence is insufficient, request follow-up questions."],
            confidence=0.25,
        )

    async def plan(
        self,
        *,
        goal: str,
        evidence_cards: list[dict[str, Any]],
        available_tools: list[str],
        self_model_summary: dict[str, Any] | None = None,
        tool_event_sink=None,
    ) -> PlannerOutput:
        cleaned_goal = (goal or "").strip()
        if not cleaned_goal:
            return self._heuristic_plan(
                goal="(empty goal)",
                available_tools=available_tools,
                self_model_summary=self_model_summary,
            )

        if str(getattr(settings, "AGENT_STRATEGY", "llm")).lower() != "llm":
            return self._heuristic_plan(
                goal=cleaned_goal,
                available_tools=available_tools,
                self_model_summary=self_model_summary,
            )

        prompt_template = load_prompt_text("cognitive_loop", "planner_v1.txt")
        prompt = prompt_template.format(
            goal=cleaned_goal,
            evidence_json=json.dumps(evidence_cards, ensure_ascii=False, indent=2),
            tools_json=json.dumps(sorted(list(set(available_tools or []))), ensure_ascii=False, indent=2),
            self_model_json=json.dumps(self_model_summary or {}, ensure_ascii=False, indent=2),
        )

        client = self.llm_client or self._default_llm()
        started_at = datetime.now(timezone.utc)
        resp = await client.complete_json(
            prompt=prompt,
            schema_hint={"schema": PlannerOutput.model_json_schema()},
            tool_event_sink=tool_event_sink,
        )
        _ = started_at

        try:
            return PlannerOutput.model_validate(resp)
        except ValidationError:
            # Fail closed: return conservative plan instead of passing through invalid JSON.
            return self._heuristic_plan(
                goal=cleaned_goal,
                available_tools=available_tools,
                self_model_summary=self_model_summary,
            )
