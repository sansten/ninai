"""Uncertainty Reporting — Phase 22.

Aggregates confidence scores and conflict flags from all prior agents into
a single uncertainty surface. Reports what the system doesn't know, what
it's conflicted on, and where human review is needed.

Depends on (reads enrichment from):
  - ConflictDetectionAgent (Phase 13)          — conflicts, conflict_count, high_severity_conflicts
  - AdaptiveConflictResolutionAgent (Phase 14) — resolution_rate, escalation_targets
  - MemoryDecayAgent (Phase 15)                — decay_factor, is_stale
  - CredibilityAgent (Phase 19)                — credibility_score, credibility_flags, source_tier
  - GoalDecompositionAgent (Phase 21)          — goal_detected, blocking_subtask, completion_fraction
  - PlaybookAgent (Phase 20)                   — playbook_confidence, matched_playbook_id
  - TemporalReasoningAgent (Phase 17)          — temporal_confidence
  - CausalReasoningAgent (Phase 12)            — causal_confidence

Outputs:
  - uncertainty_level:      "low" | "medium" | "high" | "critical"
  - unresolved_conflicts:   list[{entity, severity, conflict_type}]
  - low_confidence_fields:  list[str]          — fields whose confidence is below threshold
  - review_recommended:     bool
  - uncertainty_score:      float 0..1         — overall uncertainty magnitude
  - confidence:             float
  - rationale:              "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_LOW_CONFIDENCE_THRESHOLD = 0.55  # below this → field is "low confidence"
_REVIEW_SCORE_THRESHOLD = 0.35    # uncertainty_score above this → review_recommended
_STALE_DECAY_THRESHOLD = 0.30     # decay_factor below this → memory is effectively stale

_LEVEL_THRESHOLDS: list[tuple[float, str]] = [
    (0.80, "critical"),
    (0.55, "high"),
    (0.30, "medium"),
    (0.00, "low"),
]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def collect_unresolved_conflicts(enrichment: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract conflicts that have not been fully resolved.

    Uses Phase 13's `conflicts` list and Phase 14's `resolution_rate` to
    decide which conflicts remain outstanding.  If resolution_rate is 1.0
    all conflicts are considered resolved.  Otherwise any conflict whose
    severity is "high" or that appears in `escalation_targets` is kept.
    """
    conflicts: list[dict] = enrichment.get("conflicts") or []
    resolution_rate: float = float(enrichment.get("resolution_rate") or 0.0)
    escalation_targets: list[str] = list(enrichment.get("escalation_targets") or [])

    if not conflicts:
        return []

    if resolution_rate >= 1.0:
        return []

    unresolved: list[dict[str, Any]] = []
    for c in conflicts:
        if not isinstance(c, dict):
            continue
        severity = str(c.get("severity") or "low").lower()
        entity = str(c.get("entity") or "unknown")
        conflict_type = str(c.get("conflict_type") or "unknown")

        is_high = severity == "high"
        is_escalated = entity in escalation_targets or conflict_type in escalation_targets

        if is_high or is_escalated or resolution_rate < 0.5:
            unresolved.append({
                "entity": entity,
                "severity": severity,
                "conflict_type": conflict_type,
            })

    return unresolved


def collect_low_confidence_fields(enrichment: dict[str, Any]) -> list[str]:
    """Return a list of named fields whose associated confidence is too low.

    Checks well-known confidence signals written by prior agents.
    """
    low: list[str] = []
    threshold = _LOW_CONFIDENCE_THRESHOLD

    # Phase 19 — CredibilityAgent
    cred = enrichment.get("credibility_score")
    if cred is not None and _is_float(cred) and float(cred) < threshold:
        low.append("credibility_score")

    # Phase 20 — PlaybookAgent
    pb_conf = enrichment.get("playbook_confidence")
    if pb_conf is not None and _is_float(pb_conf) and float(pb_conf) < threshold:
        low.append("playbook_confidence")

    # Phase 17 — TemporalReasoningAgent
    temp_conf = enrichment.get("temporal_confidence")
    if temp_conf is not None and _is_float(temp_conf) and float(temp_conf) < threshold:
        low.append("temporal_confidence")

    # Phase 12 — CausalReasoningAgent
    causal_conf = enrichment.get("causal_confidence")
    if causal_conf is not None and _is_float(causal_conf) and float(causal_conf) < threshold:
        low.append("causal_confidence")

    # Phase 15 — MemoryDecayAgent — stale memory implies low confidence in relevance
    decay = enrichment.get("decay_factor")
    if decay is not None and _is_float(decay) and float(decay) < _STALE_DECAY_THRESHOLD:
        low.append("decay_factor")

    # Phase 21 — GoalDecompositionAgent — blocked goal is uncertain
    blocking = enrichment.get("blocking_subtask")
    if blocking:
        low.append("goal_completion")

    return low


def _is_float(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def compute_uncertainty_score(
    *,
    unresolved_count: int,
    low_confidence_count: int,
    high_severity_count: int,
    credibility_score: float | None,
    is_stale: bool,
) -> float:
    """Aggregate signals into a single uncertainty score in [0, 1].

    Higher = more uncertain.
    """
    score = 0.0

    # Unresolved conflicts are the strongest uncertainty signal
    score += min(0.30, unresolved_count * 0.10)

    # High-severity conflicts add extra weight
    score += min(0.20, high_severity_count * 0.10)

    # Low-confidence fields each contribute a small amount
    score += min(0.20, low_confidence_count * 0.05)

    # Poor credibility raises uncertainty
    if credibility_score is not None:
        credibility_gap = max(0.0, _LOW_CONFIDENCE_THRESHOLD - credibility_score)
        score += round(credibility_gap * 0.30, 4)

    # Stale memory is uncertain regardless of other signals
    if is_stale:
        score += 0.10

    return round(min(1.0, score), 4)


def classify_uncertainty_level(score: float) -> str:
    """Map uncertainty score to a named level."""
    for threshold, label in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "low"


def compute_agent_confidence(
    *,
    uncertainty_level: str,
    unresolved_count: int,
    low_confidence_count: int,
) -> float:
    """Confidence that the uncertainty assessment itself is accurate.

    Ironic but necessary: we are more confident about high uncertainty
    when many signals agree, less confident when signals are sparse.
    """
    if uncertainty_level == "critical":
        base = 0.85
    elif uncertainty_level == "high":
        base = 0.80
    elif uncertainty_level == "medium":
        base = 0.70
    else:
        base = 0.65

    signal_count = unresolved_count + low_confidence_count
    if signal_count == 0:
        base -= 0.10
    elif signal_count >= 4:
        base += 0.05

    return round(min(0.90, max(0.40, base)), 4)


def run_heuristic(enrichment: dict[str, Any]) -> dict[str, Any]:
    """Full heuristic analysis of the enrichment dict."""
    unresolved = collect_unresolved_conflicts(enrichment)
    low_fields = collect_low_confidence_fields(enrichment)

    high_sev_conflicts: list[dict] = enrichment.get("high_severity_conflicts") or []
    high_severity_count = len([c for c in high_sev_conflicts if isinstance(c, dict)])

    cred_raw = enrichment.get("credibility_score")
    credibility_score: float | None = float(cred_raw) if _is_float(cred_raw) else None

    decay_raw = enrichment.get("decay_factor")
    is_stale: bool = bool(enrichment.get("is_stale")) or (
        _is_float(decay_raw) and float(decay_raw) < _STALE_DECAY_THRESHOLD
    )

    score = compute_uncertainty_score(
        unresolved_count=len(unresolved),
        low_confidence_count=len(low_fields),
        high_severity_count=high_severity_count,
        credibility_score=credibility_score,
        is_stale=is_stale,
    )

    level = classify_uncertainty_level(score)
    review = score >= _REVIEW_SCORE_THRESHOLD

    confidence = compute_agent_confidence(
        uncertainty_level=level,
        unresolved_count=len(unresolved),
        low_confidence_count=len(low_fields),
    )

    return {
        "uncertainty_level": level,
        "unresolved_conflicts": unresolved,
        "low_confidence_fields": low_fields,
        "review_recommended": review,
        "uncertainty_score": score,
        "confidence": confidence,
        "rationale": "heuristic",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class UncertaintyReportingAgent(BaseAgent):
    """Phase 22: Aggregate uncertainty signals from all prior agents.

    Reads enrichment written by Phases 12–21, computes an overall uncertainty
    score, classifies it, and flags whether human review is recommended.
    """

    name = "UncertaintyReportingAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return [
            "ConflictDetectionAgent",
            "AdaptiveConflictResolutionAgent",
            "CredibilityAgent",
            "GoalDecompositionAgent",
        ]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        level = outputs.get("uncertainty_level")
        if level not in {"low", "medium", "high", "critical"}:
            raise ValueError(f"uncertainty_level must be one of low/medium/high/critical, got {level!r}")

        if not isinstance(outputs.get("unresolved_conflicts"), list):
            raise ValueError("unresolved_conflicts must be a list")

        if not isinstance(outputs.get("low_confidence_fields"), list):
            raise ValueError("low_confidence_fields must be a list")

        if not isinstance(outputs.get("review_recommended"), bool):
            raise ValueError("review_recommended must be a bool")

        score = outputs.get("uncertainty_score")
        if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
            raise ValueError("uncertainty_score must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment: dict[str, Any] = memory.get("enrichment") or {}

        strategy = getattr(settings, "UNCERTAINTY_REPORTING_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = run_heuristic(enrichment)
        else:
            prompt = (
                "You are an enterprise uncertainty analysis engine for a memory OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Analyze the enrichment signals below and produce an uncertainty report. "
                "Identify unresolved conflicts, fields with low confidence, and whether "
                "human review is recommended.\n\n"
                f"CONFLICTS: {enrichment.get('conflicts', [])}\n"
                f"HIGH_SEVERITY_CONFLICTS: {enrichment.get('high_severity_conflicts', [])}\n"
                f"RESOLUTION_RATE: {enrichment.get('resolution_rate')}\n"
                f"ESCALATION_TARGETS: {enrichment.get('escalation_targets', [])}\n"
                f"CREDIBILITY_SCORE: {enrichment.get('credibility_score')}\n"
                f"CREDIBILITY_FLAGS: {enrichment.get('credibility_flags', [])}\n"
                f"DECAY_FACTOR: {enrichment.get('decay_factor')}\n"
                f"IS_STALE: {enrichment.get('is_stale')}\n"
                f"PLAYBOOK_CONFIDENCE: {enrichment.get('playbook_confidence')}\n"
                f"TEMPORAL_CONFIDENCE: {enrichment.get('temporal_confidence')}\n"
                f"CAUSAL_CONFIDENCE: {enrichment.get('causal_confidence')}\n"
                f"BLOCKING_SUBTASK: {enrichment.get('blocking_subtask')}\n\n"
                "Return JSON with keys:\n"
                "- uncertainty_level: one of low|medium|high|critical\n"
                "- unresolved_conflicts: list of {entity, severity, conflict_type}\n"
                "- low_confidence_fields: list of field name strings\n"
                "- review_recommended: bool\n"
                "- uncertainty_score: float 0..1\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(getattr(settings, "OLLAMA_MODEL", "llama3.1:8b")),
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
                and resp.get("uncertainty_level") in {"low", "medium", "high", "critical"}
                and isinstance(resp.get("unresolved_conflicts"), list)
                and isinstance(resp.get("review_recommended"), bool)
            ):
                outputs = resp
            else:
                outputs = run_heuristic(enrichment)

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.65)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
