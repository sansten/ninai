"""GoalPlannerAgent.

LLM-based agent that proposes creating goals from user requests or cognitive session outcomes.
Outputs conform to GoalPlannerAgentOutput schema.

Fail-closed behavior:
- If LLM is disabled or returns invalid output, returns empty suggestion with low confidence.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.agents.llm.base import LLMClient
from app.agents.llm.ollama import OllamaClient
from app.core.config import settings
from app.schemas.goal_agents import GoalPlannerAgentOutput
from app.services.cognitive_loop.prompt_loader import load_prompt_text


class GoalPlannerAgent:
    name = "goal_planner_agent"
    version = "v1"
    prompt_version = "goal_planner_v1"

    def __init__(self, *, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def _default_llm(self) -> LLMClient:
        return OllamaClient(
            base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
            model=str(getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")),
            timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
            max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
        )

    def _fail_closed(self) -> GoalPlannerAgentOutput:
        """Fail-closed: no goal proposal."""
        return GoalPlannerAgentOutput(
            create_goal=False,
            confidence=0.0,
        )

    async def propose_goal(
        self,
        *,
        user_request: str,
        session_context: dict[str, Any] | None = None,
        existing_goals: list[dict[str, Any]] | None = None,
        tool_event_sink=None,
    ) -> GoalPlannerAgentOutput:
        """Propose a goal based on user request or session outcomes.
        
        Args:
            user_request: User's goal description or session outcome summary
            session_context: Optional cognitive session context (evaluation, evidence)
            existing_goals: Optional list of existing active goals to avoid duplication
            tool_event_sink: Optional event sink for tool call logging
            
        Returns:
            GoalPlannerAgentOutput with goal proposal or empty if not needed
        """
        cleaned_request = (user_request or "").strip()
        if not cleaned_request:
            return self._fail_closed()

        if str(getattr(settings, "AGENT_STRATEGY", "llm")).lower() != "llm":
            return self._fail_closed()

        try:
            prompt_template = load_prompt_text("goal_agents", "goal_planner_v1.txt")
        except Exception:
            return self._fail_closed()

        prompt = prompt_template.format(
            user_request=cleaned_request,
            session_context=json.dumps(session_context or {}, ensure_ascii=False, indent=2),
            existing_goals=json.dumps(existing_goals or [], ensure_ascii=False, indent=2),
        )

        llm = self.llm_client or self._default_llm()
        try:
            resp = await llm.complete_json(
                prompt=prompt,
                schema_hint={
                    "create_goal": "bool",
                    "goal": {
                        "title": "string",
                        "description": "string",
                        "goal_type": "task|project|objective|policy|research",
                        "visibility_scope": "personal|team|department|division|organization",
                        "priority": "int 0-5",
                    },
                    "nodes": [
                        {
                            "temp_id": "string",
                            "node_type": "subgoal|task|milestone",
                            "title": "string",
                        }
                    ],
                    "edges": [{"from_temp_id": "string", "to_temp_id": "string", "edge_type": "depends_on|blocks|related_to"}],
                    "confidence": "float 0-1",
                },
                tool_event_sink=tool_event_sink,
            )
            return GoalPlannerAgentOutput(**resp)
        except Exception:
            # Fail-closed on any error
            return self._fail_closed()
