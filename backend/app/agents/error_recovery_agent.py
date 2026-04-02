"""Error recovery and replan agent (Phase 68)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


_CONFIDENCE_BY_STRATEGY = {
    "retry": 0.8,
    "skip": 0.75,
    "substitute": 0.65,
    "replan": 0.5,
    "escalate": 0.4,
}


def _step_title(step: dict[str, Any]) -> str:
    return str(step.get("title") or "").strip()


def _choose_fallback_tool(*, failed_step: dict[str, Any], available_tools: list[str]) -> str | None:
    title = _step_title(failed_step).lower()
    for tool in available_tools or []:
        tool_name = str(tool or "").strip()
        if not tool_name:
            continue
        if tool_name.lower() not in title:
            return tool_name
    for tool in available_tools or []:
        tool_name = str(tool or "").strip()
        if tool_name:
            return tool_name
    return None


def run_heuristic(
    *,
    failed_step: dict[str, Any],
    remaining_plan: list[dict[str, Any]],
    completed_steps: list[dict[str, Any]],
    available_tools: list[str],
) -> dict[str, Any]:
    step = dict(failed_step or {})
    remaining = list(remaining_plan or [])
    attempts = int(step.get("attempts") or 0)
    error_type = str(step.get("error_type") or "unknown").strip().lower()

    recovery_strategy = "escalate"
    revised_plan = remaining
    substitute_step = None
    skip_justification = None
    escalation_reason = None

    if error_type == "transient" and attempts < 3:
        retry_step = dict(step)
        retry_step["attempts"] = 0
        recovery_strategy = "retry"
        revised_plan = [retry_step] + remaining
    elif error_type in {"not_found", "permission_denied"} and attempts >= 1:
        recovery_strategy = "skip"
        skip_justification = f"Step '{_step_title(step)}' skipped: {error_type}"
        revised_plan = remaining
    elif error_type == "service_unavailable":
        fallback_tool = _choose_fallback_tool(failed_step=step, available_tools=available_tools)
        substitute_step = {
            "title": f"[SUBSTITUTE] {_step_title(step)} via fallback",
            "tool": fallback_tool,
            "step_index": step.get("step_index"),
        }
        recovery_strategy = "substitute"
        revised_plan = [substitute_step] + remaining
    elif error_type == "data_corruption":
        recovery_strategy = "replan"
        revised_plan = []
    else:
        recovery_strategy = "escalate"
        escalation_reason = f"Unrecoverable: {error_type} after {attempts} attempts"
        revised_plan = remaining

    return {
        "recovery_strategy": recovery_strategy,
        "revised_plan": revised_plan,
        "substitute_step": substitute_step,
        "skip_justification": skip_justification,
        "escalation_reason": escalation_reason,
        "confidence": _CONFIDENCE_BY_STRATEGY[recovery_strategy],
        "rationale": "heuristic",
        "completed_step_count": len(completed_steps or []),
    }


class ErrorRecoveryAgent(BaseAgent):
    name = "ErrorRecoveryAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if outputs.get("recovery_strategy") not in _CONFIDENCE_BY_STRATEGY:
            raise ValueError("recovery_strategy must be a known strategy")
        if not isinstance(outputs.get("revised_plan"), list):
            raise ValueError("revised_plan must be a list")
        substitute_step = outputs.get("substitute_step")
        if substitute_step is not None and not isinstance(substitute_step, dict):
            raise ValueError("substitute_step must be dict or None")
        skip_justification = outputs.get("skip_justification")
        if skip_justification is not None and not isinstance(skip_justification, str):
            raise ValueError("skip_justification must be str or None")
        escalation_reason = outputs.get("escalation_reason")
        if escalation_reason is not None and not isinstance(escalation_reason, str):
            raise ValueError("escalation_reason must be str or None")
        confidence = outputs.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            raise ValueError("confidence must be float between 0 and 1")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        failed_step = dict(enrichment.get("failed_step") or {})
        remaining_plan = list(enrichment.get("remaining_plan") or [])
        completed_steps = list(enrichment.get("completed_steps") or [])
        available_tools = list(enrichment.get("available_tools") or [])

        strategy = getattr(settings, "ERROR_RECOVERY_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(
                failed_step=failed_step,
                remaining_plan=remaining_plan,
                completed_steps=completed_steps,
                available_tools=available_tools,
            )
        else:
            prompt = (
                "You choose recovery actions for failed execution steps. Output JSON only.\n\n"
                f"FAILED_STEP: {failed_step}\n"
                f"REMAINING_PLAN: {remaining_plan[:40]}\n"
                f"COMPLETED_STEPS: {completed_steps[:40]}\n"
                f"AVAILABLE_TOOLS: {available_tools[:40]}\n\n"
                "Return JSON with keys:\n"
                "- recovery_strategy: str\n"
                "- revised_plan: list[dict]\n"
                "- substitute_step: dict or null\n"
                "- skip_justification: str or null\n"
                "- escalation_reason: str or null\n"
                "- confidence: float\n"
                "- rationale: str"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_ollama_model("agents")),
                timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
            )
            resp = await client.complete_json(
                prompt=prompt,
                schema_hint={},
                tool_event_sink=context.get("tool_event_sink"),
            )
            if (
                isinstance(resp, dict)
                and resp.get("recovery_strategy") in _CONFIDENCE_BY_STRATEGY
                and isinstance(resp.get("revised_plan"), list)
                and (resp.get("substitute_step") is None or isinstance(resp.get("substitute_step"), dict))
                and (resp.get("skip_justification") is None or isinstance(resp.get("skip_justification"), str))
                and (resp.get("escalation_reason") is None or isinstance(resp.get("escalation_reason"), str))
                and isinstance(resp.get("confidence"), (int, float))
            ):
                outputs = resp
            else:
                outputs = run_heuristic(
                    failed_step=failed_step,
                    remaining_plan=remaining_plan,
                    completed_steps=completed_steps,
                    available_tools=available_tools,
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
