"""Adaptive persona agent (Phase 52)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


_ACRONYM_EXPANSIONS = {
    "SLA": "service level agreement (SLA)",
    "MTTR": "mean time to recovery (MTTR)",
    "API": "application programming interface (API)",
    "RLS": "row-level security (RLS)",
}


def _strip_parentheticals(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", text)


def _truncate_to_sentences(text: str, max_sentences: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return " ".join(parts[:max_sentences]).strip()


def _expand_acronyms(text: str) -> tuple[str, bool]:
    changed = False
    adapted = text
    for acronym, expansion in _ACRONYM_EXPANSIONS.items():
        pattern = rf"\b{re.escape(acronym)}\b"
        if re.search(pattern, adapted):
            adapted = re.sub(pattern, expansion, adapted)
            changed = True
    return adapted, changed


class AdaptivePersonaAgent(BaseAgent):
    """Phase 52: adapt response style to user's expertise and verbosity profile."""

    name = "AdaptivePersonaAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("adapted_content"), str):
            raise ValueError("adapted_content must be a string")
        if not isinstance(outputs.get("persona_applied"), str):
            raise ValueError("persona_applied must be a string")
        if not isinstance(outputs.get("changes_made"), list):
            raise ValueError("changes_made must be a list")

    def _heuristic(
        self,
        *,
        content: str,
        persona: dict[str, Any],
        context_type: str,
    ) -> dict[str, Any]:
        expertise_level = str(persona.get("expertise_level") or "intermediate").lower()
        preferred_verbosity = str(persona.get("preferred_verbosity") or "normal").lower()
        if context_type == "alert":
            preferred_verbosity = "brief"

        adapted = content
        changes: list[str] = []

        if expertise_level == "novice" and preferred_verbosity == "detailed":
            adapted, expanded = _expand_acronyms(adapted)
            if expanded:
                changes.append("expanded acronyms")
            adapted = adapted.strip() + "\n\nWhat this means: This is the practical impact in plain terms."
            changes.append("added context")

        elif expertise_level == "expert" and preferred_verbosity == "brief":
            stripped = _strip_parentheticals(adapted)
            if stripped != adapted:
                changes.append("removed parentheticals")
            adapted = _truncate_to_sentences(stripped, max_sentences=2)
            changes.append("truncated to brief format")

        elif expertise_level == "intermediate" and preferred_verbosity == "normal":
            pass
        else:
            if preferred_verbosity == "brief":
                adapted = _truncate_to_sentences(adapted, max_sentences=2)
                changes.append("truncated to brief format")

        persona_applied = f"{expertise_level}_{preferred_verbosity}"
        confidence = 0.8 if changes else 0.7
        return {
            "adapted_content": adapted,
            "persona_applied": persona_applied,
            "changes_made": changes,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}

        content = str(enrichment.get("content") or memory.get("content") or "")
        persona = dict(enrichment.get("persona") or {})
        context_type = str(enrichment.get("context_type") or "memory_read")

        strategy = getattr(settings, "ADAPTIVE_PERSONA_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(content=content, persona=persona, context_type=context_type)
        else:
            prompt = (
                "You are a response style adaptation engine. Output JSON only.\n\n"
                f"CONTENT: {content[:800]}\n"
                f"PERSONA: {persona}\n"
                f"CONTEXT_TYPE: {context_type}\n\n"
                "Return JSON with keys: adapted_content, persona_applied, changes_made, confidence"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_ollama_model("agents")),
                timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
            )
            resp = await client.complete_json(prompt=prompt, schema_hint={}, tool_event_sink=context.get("tool_event_sink"))
            if (
                isinstance(resp, dict)
                and isinstance(resp.get("adapted_content"), str)
                and isinstance(resp.get("persona_applied"), str)
                and isinstance(resp.get("changes_made"), list)
            ):
                outputs = {
                    **resp,
                    "rationale": "llm",
                }
            else:
                outputs = self._heuristic(content=content, persona=persona, context_type=context_type)

        finished_at = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.7)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result