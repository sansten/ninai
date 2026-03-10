"""Human Review Queue — Phase 32.

Surfaces memories flagged by UncertaintyReportingAgent's review_recommended
output into a structured review queue. Each queued memory receives a priority
and a compact reviewer context snapshot assembled from prior-agent enrichment.

Depends on (reads enrichment from):
  - UncertaintyReportingAgent (Phase 22) — review_recommended, uncertainty_level,
                                            uncertainty_score, unresolved_conflicts,
                                            low_confidence_fields
  - AnomalyDetectionAgent (Phase 25)     — anomaly_detected, anomaly_type, severity
  - AuditTrailAgent (Phase 31)           — audit_log (reviewer context snapshot)

Outputs:
  - queue_eligible:   bool        — whether this memory enters the review queue
  - priority:         str         — "low" | "normal" | "high" | "urgent"
  - review_reasons:   list[str]   — human-readable reasons for flagging
  - review_context:   dict        — compact snapshot for the reviewer
  - queue_entry_id:   str | None  — stable entry ID (== memory_id when eligible)
  - confidence:       float
  - rationale:        "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Priority levels
# ---------------------------------------------------------------------------

_PRIORITY_LEVELS = ("low", "normal", "high", "urgent")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def compute_priority(
    *,
    uncertainty_level: str,
    anomaly_detected: bool,
    uncertainty_score: float,
) -> str:
    """Map uncertainty signals to a review priority."""
    level = str(uncertainty_level or "low").lower()
    if level == "critical":
        return "urgent"
    if level == "high" or anomaly_detected:
        return "high"
    if level == "medium":
        return "normal"
    # low uncertainty but score still triggers review
    if uncertainty_score >= 0.35:
        return "normal"
    return "low"


def build_review_reasons(
    *,
    unresolved_conflicts: list[dict[str, Any]],
    low_confidence_fields: list[str],
    anomaly_detected: bool,
    anomaly_type: str | None,
    severity: str | None,
    uncertainty_level: str,
    uncertainty_score: float,
) -> list[str]:
    """Build a human-readable list of reasons this memory needs review."""
    reasons: list[str] = []

    if unresolved_conflicts:
        count = len(unresolved_conflicts)
        reasons.append(f"{count} unresolved conflict{'s' if count > 1 else ''}")

    if low_confidence_fields:
        field_list = ", ".join(low_confidence_fields[:5])
        reasons.append(f"low confidence in: {field_list}")

    if anomaly_detected:
        atype = anomaly_type or "unknown"
        sev = severity or "unclassified"
        reasons.append(f"anomaly detected: {atype} ({sev})")

    level = str(uncertainty_level or "").lower()
    if level in {"high", "critical"}:
        reasons.append(f"high uncertainty score: {uncertainty_score:.2f}")

    if not reasons:
        reasons.append("flagged for review by UncertaintyReportingAgent")

    return reasons


def build_review_context(enrichment: dict[str, Any]) -> dict[str, Any]:
    """Assemble a compact reviewer context from enrichment signals."""
    unresolved = enrichment.get("unresolved_conflicts") or []
    low_fields = enrichment.get("low_confidence_fields") or []
    audit_log = enrichment.get("audit_log") or []

    # Include the last 3 audit entries for traceability
    audit_snapshot: list[dict[str, Any]] = []
    if isinstance(audit_log, list):
        for entry in audit_log[-3:]:
            if isinstance(entry, dict):
                audit_snapshot.append({
                    "agent": entry.get("agent_name"),
                    "decision": entry.get("decision"),
                    "confidence": entry.get("confidence"),
                })

    ctx: dict[str, Any] = {
        "uncertainty_level": enrichment.get("uncertainty_level"),
        "uncertainty_score": enrichment.get("uncertainty_score"),
        "unresolved_conflict_count": len(unresolved) if isinstance(unresolved, list) else 0,
        "low_confidence_fields": low_fields if isinstance(low_fields, list) else [],
        "anomaly_detected": bool(enrichment.get("anomaly_detected")),
        "anomaly_type": enrichment.get("anomaly_type"),
    }
    if audit_snapshot:
        ctx["audit_snapshot"] = audit_snapshot

    return ctx


def run_heuristic(memory_id: str, enrichment: dict[str, Any]) -> dict[str, Any]:
    """Determine queue eligibility and build the queue entry from enrichment."""
    review_recommended = bool(enrichment.get("review_recommended"))
    uncertainty_level = str(enrichment.get("uncertainty_level") or "low").lower()

    score_raw = enrichment.get("uncertainty_score")
    uncertainty_score = float(score_raw) if score_raw is not None else 0.0

    anomaly_detected = bool(enrichment.get("anomaly_detected"))
    anomaly_type = enrichment.get("anomaly_type")
    severity_val = enrichment.get("severity")

    unresolved: list[dict[str, Any]] = []
    raw_unresolved = enrichment.get("unresolved_conflicts")
    if isinstance(raw_unresolved, list):
        unresolved = [c for c in raw_unresolved if isinstance(c, dict)]

    low_fields: list[str] = []
    raw_low = enrichment.get("low_confidence_fields")
    if isinstance(raw_low, list):
        low_fields = [f for f in raw_low if isinstance(f, str)]

    # Eligible if uncertainty flagged it, or if an anomaly pushes over the threshold
    queue_eligible = review_recommended or (anomaly_detected and uncertainty_score >= 0.20)

    priority = compute_priority(
        uncertainty_level=uncertainty_level,
        anomaly_detected=anomaly_detected,
        uncertainty_score=uncertainty_score,
    )

    review_reasons = build_review_reasons(
        unresolved_conflicts=unresolved,
        low_confidence_fields=low_fields,
        anomaly_detected=anomaly_detected,
        anomaly_type=anomaly_type,
        severity=str(severity_val) if severity_val else None,
        uncertainty_level=uncertainty_level,
        uncertainty_score=uncertainty_score,
    )

    review_context = build_review_context(enrichment)
    queue_entry_id = str(memory_id) if queue_eligible else None

    if queue_eligible:
        base_conf = {"urgent": 0.90, "high": 0.85, "normal": 0.75, "low": 0.65}.get(priority, 0.70)
    else:
        base_conf = 0.80  # confident it does not need review

    return {
        "queue_eligible": queue_eligible,
        "priority": priority if queue_eligible else "low",
        "review_reasons": review_reasons if queue_eligible else [],
        "review_context": review_context if queue_eligible else {},
        "queue_entry_id": queue_entry_id,
        "confidence": round(base_conf, 4),
        "rationale": "heuristic",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class HumanReviewQueueAgent(BaseAgent):
    """Phase 32: Route review_recommended memories into a structured queue.

    Reads enrichment from UncertaintyReportingAgent and AnomalyDetectionAgent
    to decide whether a memory enters the human review queue, what priority it
    receives, and what context the reviewer needs to make a decision.
    Resolved verdicts feed back into FeedbackIntegrationAgent via /review/resolve.
    """

    name = "HumanReviewQueueAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return [
            "UncertaintyReportingAgent",
            "AnomalyDetectionAgent",
            "AuditTrailAgent",
        ]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("queue_eligible"), bool):
            raise ValueError("queue_eligible must be a bool")

        priority = outputs.get("priority")
        if priority not in _PRIORITY_LEVELS:
            raise ValueError(f"priority must be one of {_PRIORITY_LEVELS}, got {priority!r}")

        if not isinstance(outputs.get("review_reasons"), list):
            raise ValueError("review_reasons must be a list")

        if not isinstance(outputs.get("review_context"), dict):
            raise ValueError("review_context must be a dict")

        entry_id = outputs.get("queue_entry_id")
        if entry_id is not None and not isinstance(entry_id, str):
            raise ValueError("queue_entry_id must be a string or None")

        conf = outputs.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            raise ValueError("confidence must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment: dict[str, Any] = memory.get("enrichment") or {}

        strategy = getattr(settings, "HUMAN_REVIEW_QUEUE_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "heuristic")
        strategy = str(strategy or "heuristic").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = run_heuristic(memory_id, enrichment)
        else:
            base_outputs = run_heuristic(memory_id, enrichment)
            prompt = (
                "You are an enterprise human review queue agent. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Assess whether this memory should be routed to the human review queue "
                "and determine its priority. Use the enrichment signals below.\n\n"
                f"REVIEW_RECOMMENDED: {enrichment.get('review_recommended')}\n"
                f"UNCERTAINTY_LEVEL: {enrichment.get('uncertainty_level')}\n"
                f"UNCERTAINTY_SCORE: {enrichment.get('uncertainty_score')}\n"
                f"UNRESOLVED_CONFLICTS: {enrichment.get('unresolved_conflicts', [])}\n"
                f"LOW_CONFIDENCE_FIELDS: {enrichment.get('low_confidence_fields', [])}\n"
                f"ANOMALY_DETECTED: {enrichment.get('anomaly_detected')}\n"
                f"ANOMALY_TYPE: {enrichment.get('anomaly_type')}\n"
                f"SEVERITY: {enrichment.get('severity')}\n\n"
                "Return JSON with keys:\n"
                "- queue_eligible: bool\n"
                "- priority: one of low|normal|high|urgent\n"
                "- review_reasons: list of strings\n"
                "- review_context: dict\n"
                f"- queue_entry_id: string or null (use {memory_id!r} if eligible)\n"
                "- confidence: float 0..1\n"
                "- rationale: 'llm'"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")),
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
                and isinstance(resp.get("queue_eligible"), bool)
                and resp.get("priority") in _PRIORITY_LEVELS
                and isinstance(resp.get("review_reasons"), list)
                and isinstance(resp.get("review_context"), dict)
            ):
                outputs = resp
                if "rationale" not in outputs:
                    outputs["rationale"] = "llm"
            else:
                outputs = base_outputs

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.70)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
