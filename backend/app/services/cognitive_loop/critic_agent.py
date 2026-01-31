"""CriticAgent for CognitiveLoop.

Evaluates an execution against the plan and available evidence.
Outputs strict JSON validated by CriticOutput.

Fail-closed behavior:
- If LLM is disabled or returns invalid output, return needs_evidence with low confidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from app.agents.llm.base import LLMClient
from app.agents.llm.ollama import OllamaClient
from app.core.config import settings
from app.schemas.cognitive import CriticOutput
from app.services.cognitive_loop.prompt_loader import load_prompt_text


class CriticAgent:
    name = "critic_agent"
    version = "v1"
    prompt_version = "critic_v1"

    def __init__(self, *, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def _default_llm(self) -> LLMClient:
        return OllamaClient(
            base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
            model=str(getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")),
            timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
            max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
        )

    def _fail_closed(self, reason: str | None = None) -> CriticOutput:
        strengths: list[str] = []
        issues = []
        if reason:
            issues = [
                {
                    "type": "missing_evidence",
                    "description": reason,
                    "affected_steps": [],
                    "recommended_fix": "Provide more evidence or clarify constraints.",
                }
            ]
        return CriticOutput(
            evaluation="needs_evidence",
            strengths=strengths,
            issues=issues,
            followup_questions=["What additional evidence or constraints should be considered?"],
            confidence=0.25,
        )

    async def critique(
        self,
        *,
        goal: str,
        plan: dict[str, Any],
        execution: dict[str, Any],
        evidence_cards: list[dict[str, Any]],
        simulation: dict[str, Any] | None = None,
        tool_event_sink=None,
    ) -> CriticOutput:
        if str(getattr(settings, "AGENT_STRATEGY", "llm")).lower() != "llm":
            return self._fail_closed("LLM disabled; cannot critique reliably.")

        prompt_template = load_prompt_text("cognitive_loop", "critic_v1.txt")
        prompt = prompt_template.format(
            goal=(goal or "").strip(),
            plan_json=json.dumps(plan or {}, ensure_ascii=False, indent=2),
            execution_json=json.dumps(execution or {}, ensure_ascii=False, indent=2),
            evidence_json=json.dumps(evidence_cards or [], ensure_ascii=False, indent=2),
            simulation_json=json.dumps(simulation or {}, ensure_ascii=False, indent=2),
        )

        client = self.llm_client or self._default_llm()
        started_at = datetime.now(timezone.utc)
        resp = await client.complete_json(
            prompt=prompt,
            schema_hint={"schema": CriticOutput.model_json_schema()},
            tool_event_sink=tool_event_sink,
        )
        _ = started_at

        try:
            return CriticOutput.model_validate(resp)
        except ValidationError:
            return self._fail_closed("Invalid critic JSON output.")
