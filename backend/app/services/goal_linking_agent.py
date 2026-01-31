"""GoalLinkingAgent.

LLM-based agent that suggests linking memories to active goals based on semantic relevance.
Outputs conform to GoalLinkingAgentOutput schema.

Fail-closed behavior:
- If LLM is disabled or returns invalid output, returns empty suggestions.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.agents.llm.base import LLMClient
from app.agents.llm.ollama import OllamaClient
from app.core.config import settings
from app.schemas.goal_agents import GoalLinkingAgentOutput
from app.services.cognitive_loop.prompt_loader import load_prompt_text


class GoalLinkingAgent:
    name = "goal_linking_agent"
    version = "v1"
    prompt_version = "goal_linking_v1"

    def __init__(self, *, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def _default_llm(self) -> LLMClient:
        return OllamaClient(
            base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
            model=str(getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")),
            timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
            max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
        )

    def _fail_closed(self) -> GoalLinkingAgentOutput:
        """Fail-closed: no linking suggestions."""
        return GoalLinkingAgentOutput(links=[], confidence=0.0)

    async def suggest_links(
        self,
        *,
        memory: dict[str, Any],
        active_goals: list[dict[str, Any]],
        tool_event_sink=None,
    ) -> GoalLinkingAgentOutput:
        """Suggest goal-memory links based on semantic relevance.
        
        Args:
            memory: Memory metadata (id, title, content_preview, tags, classification)
            active_goals: List of active goals with their nodes
            tool_event_sink: Optional event sink for tool call logging
            
        Returns:
            GoalLinkingAgentOutput with suggested links or empty if none
        """
        if not memory or not active_goals:
            return self._fail_closed()

        if str(getattr(settings, "AGENT_STRATEGY", "llm")).lower() != "llm":
            return self._fail_closed()

        try:
            prompt_template = load_prompt_text("goal_agents", "goal_linking_v1.txt")
        except Exception:
            return self._fail_closed()

        prompt = prompt_template.format(
            memory=json.dumps(memory, ensure_ascii=False, indent=2),
            active_goals=json.dumps(active_goals, ensure_ascii=False, indent=2),
        )

        llm = self.llm_client or self._default_llm()
        try:
            resp = await llm.complete_json(
                prompt=prompt,
                schema_hint={
                    "links": [
                        {
                            "goal_id": "string",
                            "node_id": "string|null",
                            "memory_id": "string",
                            "link_type": "evidence|progress|blocker|reference",
                            "confidence": "float 0-1",
                            "reason": "string",
                        }
                    ],
                    "confidence": "float 0-1",
                },
                tool_event_sink=tool_event_sink,
            )
            return GoalLinkingAgentOutput(**resp)
        except Exception:
            # Fail-closed on any error
            return self._fail_closed()
