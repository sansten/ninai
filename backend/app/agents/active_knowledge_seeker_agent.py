"""Active knowledge seeker agent (Phase 63)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.llm_breaker import create_llm_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


_WORD_RE = re.compile(r"\b[a-z0-9_]+\b")


def _tokenize(value: str) -> set[str]:
    return set(_WORD_RE.findall((value or "").lower()))


def _memory_text(memory: dict[str, Any]) -> str:
    parts = [
        str(memory.get("content") or ""),
        str(memory.get("content_preview") or ""),
        str(memory.get("title") or ""),
    ]
    tags = memory.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    parts.extend(str(t) for t in tags)
    return " ".join(parts)


def _pick_top_question(knowledge_gaps: list[dict[str, Any]]) -> str | None:
    for gap in knowledge_gaps:
        if gap.get("priority") == "critical":
            return str(gap.get("question_to_ask") or "") or None
    for gap in knowledge_gaps:
        if gap.get("priority") == "high":
            return str(gap.get("question_to_ask") or "") or None
    return str(knowledge_gaps[0].get("question_to_ask") or "") or None if knowledge_gaps else None


def run_heuristic(
    *,
    goal: str,
    available_memories: list[dict[str, Any]],
    required_entities: list[str],
    confidence_threshold: float = 0.4,
) -> dict[str, Any]:
    req = [str(e).strip() for e in (required_entities or []) if str(e).strip()]
    threshold = float(confidence_threshold if confidence_threshold is not None else 0.4)

    memory_tokens: list[set[str]] = []
    for m in available_memories or []:
        memory_tokens.append(_tokenize(_memory_text(m)))

    goal_tokens = _tokenize(goal or "")

    covered = 0
    knowledge_gaps: list[dict[str, Any]] = []

    for entity in req:
        entity_tokens = _tokenize(entity)
        if not entity_tokens:
            continue

        is_covered = any(entity_tokens <= tokens for tokens in memory_tokens)
        if is_covered:
            covered += 1
            continue

        in_goal = bool(entity_tokens & goal_tokens)
        priority = "critical" if in_goal else "high"
        coverage_score = 0.0
        knowledge_gaps.append(
            {
                "gap_description": f"No information found about '{entity}'",
                "question_to_ask": f"What is the current status of {entity}?",
                "required_for": goal or "current objective",
                "priority": priority,
                "coverage_score": coverage_score,
            }
        )

    if not req:
        coverage_score = 1.0
    else:
        coverage_score = round(covered / len(req), 4)
    is_sufficient = coverage_score >= threshold
    top_question = _pick_top_question(knowledge_gaps)

    return {
        "knowledge_gaps": knowledge_gaps,
        "coverage_score": coverage_score,
        "is_sufficient": is_sufficient,
        "top_question": top_question,
        "confidence": 0.85,
        "rationale": "heuristic",
    }


class ActiveKnowledgeSeekerAgent(BaseAgent):
    name = "ActiveKnowledgeSeekerAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("knowledge_gaps"), list):
            raise ValueError("knowledge_gaps must be a list")

        coverage_score = outputs.get("coverage_score")
        if not isinstance(coverage_score, (int, float)) or not (0.0 <= float(coverage_score) <= 1.0):
            raise ValueError("coverage_score must be a float between 0 and 1")

        if not isinstance(outputs.get("is_sufficient"), bool):
            raise ValueError("is_sufficient must be bool")

        top_question = outputs.get("top_question")
        if top_question is not None and not isinstance(top_question, str):
            raise ValueError("top_question must be str or None")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        goal = str(enrichment.get("goal") or "")
        available_memories = list(enrichment.get("available_memories") or [])
        required_entities = list(enrichment.get("required_entities") or [])
        confidence_threshold = float(enrichment.get("confidence_threshold") or 0.4)

        strategy = getattr(settings, "ACTIVE_KNOWLEDGE_SEEKER_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(
                goal=goal,
                available_memories=available_memories,
                required_entities=required_entities,
                confidence_threshold=confidence_threshold,
            )
        else:
            prompt = (
                "You identify knowledge gaps required to accomplish a goal. Output JSON only.\n\n"
                f"GOAL: {goal}\n"
                f"AVAILABLE_MEMORIES: {available_memories[:80]}\n"
                f"REQUIRED_ENTITIES: {required_entities[:50]}\n"
                f"CONFIDENCE_THRESHOLD: {confidence_threshold}\n\n"
                "Return JSON with keys:\n"
                "- knowledge_gaps: list[{gap_description, question_to_ask, required_for, priority, coverage_score}]\n"
                "- coverage_score: float\n"
                "- is_sufficient: bool\n"
                "- top_question: str or null\n"
                "- confidence: float\n"
                "- rationale: str"
            )
            client = create_llm_client(
                base_url=str(getattr(settings, "VLLM_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_llm_model("agents")),
                timeout_seconds=float(getattr(settings, "VLLM_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "VLLM_MAX_CONCURRENCY", 2)),
            )
            resp = await client.complete_json(
                prompt=prompt,
                schema_hint={},
                tool_event_sink=context.get("tool_event_sink"),
            )
            if (
                isinstance(resp, dict)
                and isinstance(resp.get("knowledge_gaps"), list)
                and isinstance(resp.get("coverage_score"), (int, float))
                and isinstance(resp.get("is_sufficient"), bool)
                and (resp.get("top_question") is None or isinstance(resp.get("top_question"), str))
            ):
                outputs = resp
            else:
                outputs = run_heuristic(
                    goal=goal,
                    available_memories=available_memories,
                    required_entities=required_entities,
                    confidence_threshold=confidence_threshold,
                )

        finished_at = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence") or 0.5),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=trace_id,
        )
        self.validate_outputs(result)
        return result
