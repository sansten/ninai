"""Recursive self-improvement planner agent (Phase 60)."""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _proposal_from_failure_type(*, failure_rate: float, errors: list[str]) -> tuple[str, str, float]:
    lowered = " ".join(errors).lower()
    if "timeout" in lowered or "graph_too_large" in lowered:
        proposal_type = "data_preprocessing"
        description = "pre-filter input to reduce size before agent run"
    elif "low_confidence" in lowered:
        proposal_type = "parameter_tune"
        description = "lower confidence threshold or increase evidence limit"
    else:
        proposal_type = "routing_change"
        description = "route to human review when confidence < 0.4"

    expected_gain = min(0.5, float(failure_rate) * 1.5)
    return proposal_type, description, expected_gain


def run_heuristic(
    *,
    performance_metrics: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
    current_config: dict[str, Any],
    improvement_threshold: float = 0.15,
) -> dict[str, Any]:
    threshold = float(improvement_threshold)
    metrics = list(performance_metrics or [])
    failures = list(failure_records or [])

    proposals: list[dict[str, Any]] = []

    for metric in metrics:
        agent_name = str(metric.get("agent_name") or "")
        failure_rate = float(metric.get("failure_rate") or 0.0)
        if not agent_name or failure_rate <= 0.2:
            continue

        related = [f for f in failures if str(f.get("agent_name") or "") == agent_name]
        error_types = [str(f.get("error_type") or "") for f in related]

        proposal_type, description, expected_gain = _proposal_from_failure_type(
            failure_rate=failure_rate,
            errors=error_types,
        )
        if expected_gain < threshold:
            continue

        proposals.append(
            {
                "target_agent": agent_name,
                "proposal_type": proposal_type,
                "description": description,
                "expected_gain": round(expected_gain, 4),
                "evidence_count": len(related),
            }
        )

    high_priority = [p for p in proposals if float(p.get("expected_gain") or 0.0) >= 0.3]

    rates = [float(m.get("failure_rate") or 0.0) for m in metrics] if metrics else [0.0]
    system_health_score = _clip01(1.0 - mean(rates))

    return {
        "proposals": proposals,
        "high_priority_proposals": high_priority,
        "system_health_score": round(system_health_score, 4),
        "confidence": 0.7 if proposals else 0.5,
        "rationale": "heuristic",
    }


class SelfImprovementPlannerAgent(BaseAgent):
    name = "SelfImprovementPlannerAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("proposals"), list):
            raise ValueError("proposals must be a list")
        if not isinstance(outputs.get("high_priority_proposals"), list):
            raise ValueError("high_priority_proposals must be a list")
        shs = outputs.get("system_health_score")
        if not isinstance(shs, (int, float)) or not (0.0 <= float(shs) <= 1.0):
            raise ValueError("system_health_score must be float in [0,1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        performance_metrics = list(enrichment.get("performance_metrics") or [])
        failure_records = list(enrichment.get("failure_records") or [])
        current_config = dict(enrichment.get("current_config") or {})
        improvement_threshold = float(enrichment.get("improvement_threshold") or 0.15)

        strategy = getattr(settings, "SELF_IMPROVEMENT_PLANNER_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(
                performance_metrics=performance_metrics,
                failure_records=failure_records,
                current_config=current_config,
                improvement_threshold=improvement_threshold,
            )
        else:
            prompt = (
                "You propose recursive self-improvements for AI agents. Output JSON only.\n\n"
                f"PERFORMANCE_METRICS: {performance_metrics[:50]}\n"
                f"FAILURE_RECORDS: {failure_records[:100]}\n"
                f"CURRENT_CONFIG: {current_config}\n"
                f"IMPROVEMENT_THRESHOLD: {improvement_threshold}\n\n"
                "Return JSON with keys:\n"
                "- proposals: list[{target_agent, proposal_type, description, expected_gain, evidence_count}]\n"
                "- high_priority_proposals: list[same shape]\n"
                "- system_health_score: float 0..1\n"
                "- confidence: float 0..1\n"
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
                and isinstance(resp.get("proposals"), list)
                and isinstance(resp.get("high_priority_proposals"), list)
                and isinstance(resp.get("system_health_score"), (int, float))
            ):
                outputs = resp
            else:
                outputs = run_heuristic(
                    performance_metrics=performance_metrics,
                    failure_records=failure_records,
                    current_config=current_config,
                    improvement_threshold=improvement_threshold,
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
