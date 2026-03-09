"""Anomaly Detection — Phase 25.

Detects unusual patterns in a memory's enrichment by comparing signals
against learned thresholds.  Flags memories whose enrichment values
deviate from expected baselines so they can be investigated before
propagating bad data downstream.

Depends on (reads enrichment from):
  - ConflictDetectionAgent (Phase 13)     — conflict_count, high_severity_conflicts
  - CredibilityAgent (Phase 19)           — credibility_score, source_tier
  - MemoryDecayAgent (Phase 15)           — decay_factor, is_stale
  - TemporalReasoningAgent (Phase 17)     — temporal_confidence, sequence_position
  - UncertaintyReportingAgent (Phase 22)  — uncertainty_score, uncertainty_level
  - EpisodicGroupingAgent (Phase 18)      — episode_id

Outputs:
  - anomaly_detected:  bool
  - anomaly_type:      str | None  — one of credibility_spike | uncertainty_surge |
                                     conflict_flood | temporal_drift | stale_revival | None
  - severity:          "low" | "medium" | "high" | None
  - affected_fields:   list[str]
  - anomaly_score:     float  0..1   — composite severity
  - confidence:        float
  - rationale:         "heuristic" | "llm"
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

_CREDIBILITY_SPIKE_THRESHOLD = 0.25     # credibility below this is anomalous
_UNCERTAINTY_SURGE_THRESHOLD = 0.80     # uncertainty_score above this is anomalous
_CONFLICT_FLOOD_THRESHOLD = 5           # conflict_count at or above this is anomalous
_TEMPORAL_DRIFT_THRESHOLD = 0.20        # temporal_confidence below this is anomalous
_STALE_REVIVAL_DECAY = 0.10             # decay_factor below this but memory not old

_SEVERITY_HIGH = 0.70
_SEVERITY_MEDIUM = 0.40

_VALID_ANOMALY_TYPES = {
    "credibility_spike",
    "uncertainty_surge",
    "conflict_flood",
    "temporal_drift",
    "stale_revival",
    None,
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _is_float(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def detect_credibility_spike(enrichment: dict[str, Any]) -> tuple[bool, float]:
    """Return (detected, contribution) for credibility anomaly."""
    cred = enrichment.get("credibility_score")
    if cred is None or not _is_float(cred):
        return False, 0.0
    val = float(cred)
    if val < _CREDIBILITY_SPIKE_THRESHOLD:
        # Scale: 0.25 → 0.30, 0.0 → 0.50
        contribution = 0.30 + ((_CREDIBILITY_SPIKE_THRESHOLD - val) / _CREDIBILITY_SPIKE_THRESHOLD) * 0.20
        return True, round(min(0.50, contribution), 4)
    return False, 0.0


def detect_uncertainty_surge(enrichment: dict[str, Any]) -> tuple[bool, float]:
    """Return (detected, contribution) for uncertainty surge."""
    score = enrichment.get("uncertainty_score")
    if score is None or not _is_float(score):
        # Fall back to level
        level = str(enrichment.get("uncertainty_level") or "").lower()
        if level == "critical":
            return True, 0.40
        return False, 0.0
    val = float(score)
    if val >= _UNCERTAINTY_SURGE_THRESHOLD:
        contribution = 0.30 + (val - _UNCERTAINTY_SURGE_THRESHOLD) / (1.0 - _UNCERTAINTY_SURGE_THRESHOLD) * 0.20
        return True, round(min(0.50, contribution), 4)
    return False, 0.0


def detect_conflict_flood(enrichment: dict[str, Any]) -> tuple[bool, float]:
    """Return (detected, contribution) for conflict flood."""
    count = enrichment.get("conflict_count")
    if count is None:
        count = len(enrichment.get("conflicts") or [])
    try:
        count = int(count)
    except (TypeError, ValueError):
        return False, 0.0
    if count >= _CONFLICT_FLOOD_THRESHOLD:
        contribution = min(0.50, 0.30 + (count - _CONFLICT_FLOOD_THRESHOLD) * 0.04)
        return True, round(contribution, 4)
    return False, 0.0


def detect_temporal_drift(enrichment: dict[str, Any]) -> tuple[bool, float]:
    """Return (detected, contribution) for temporal drift."""
    conf = enrichment.get("temporal_confidence")
    if conf is None or not _is_float(conf):
        return False, 0.0
    val = float(conf)
    if val < _TEMPORAL_DRIFT_THRESHOLD:
        contribution = 0.25 + ((_TEMPORAL_DRIFT_THRESHOLD - val) / _TEMPORAL_DRIFT_THRESHOLD) * 0.15
        return True, round(min(0.40, contribution), 4)
    return False, 0.0


def detect_stale_revival(enrichment: dict[str, Any]) -> tuple[bool, float]:
    """Return (detected, contribution) for stale revival.

    A stale_revival is when decay_factor is extremely low but the memory
    was recently written (sequence_position is small or explicitly not old).
    Without access to timestamps, we approximate by checking that
    decay_factor is below the threshold but is_stale is not set — suggesting
    a contradiction between freshness indicators.
    """
    decay = enrichment.get("decay_factor")
    if decay is None or not _is_float(decay):
        return False, 0.0
    val = float(decay)
    is_stale = bool(enrichment.get("is_stale"))
    # Anomalous if decay is very low but memory not flagged stale
    if val < _STALE_REVIVAL_DECAY and not is_stale:
        contribution = 0.20 + ((_STALE_REVIVAL_DECAY - val) / _STALE_REVIVAL_DECAY) * 0.15
        return True, round(min(0.35, contribution), 4)
    return False, 0.0


def pick_primary_anomaly_type(detections: dict[str, tuple[bool, float]]) -> str | None:
    """Select the anomaly type with the highest contribution."""
    best: str | None = None
    best_score = 0.0
    for name, (detected, contribution) in detections.items():
        if detected and contribution > best_score:
            best = name
            best_score = contribution
    return best


def classify_severity(anomaly_score: float) -> str | None:
    if anomaly_score <= 0.0:
        return None
    if anomaly_score >= _SEVERITY_HIGH:
        return "high"
    if anomaly_score >= _SEVERITY_MEDIUM:
        return "medium"
    return "low"


def compute_anomaly_confidence(
    detected: bool,
    affected_count: int,
    anomaly_score: float,
) -> float:
    """Confidence in the anomaly assessment."""
    if not detected:
        base = 0.70  # confident there is no anomaly
    else:
        base = 0.60
        if affected_count >= 2:
            base += 0.10  # multiple signals agree
        if anomaly_score >= _SEVERITY_HIGH:
            base += 0.05  # strong signal

    return round(min(0.90, max(0.40, base)), 4)


def run_heuristic(enrichment: dict[str, Any]) -> dict[str, Any]:
    """Full threshold-based anomaly detection."""
    detections: dict[str, tuple[bool, float]] = {
        "credibility_spike": detect_credibility_spike(enrichment),
        "uncertainty_surge": detect_uncertainty_surge(enrichment),
        "conflict_flood": detect_conflict_flood(enrichment),
        "temporal_drift": detect_temporal_drift(enrichment),
        "stale_revival": detect_stale_revival(enrichment),
    }

    affected_fields: list[str] = []
    total_score = 0.0

    field_map = {
        "credibility_spike": "credibility_score",
        "uncertainty_surge": "uncertainty_score",
        "conflict_flood": "conflict_count",
        "temporal_drift": "temporal_confidence",
        "stale_revival": "decay_factor",
    }

    for anomaly_name, (detected, contribution) in detections.items():
        if detected:
            affected_fields.append(field_map[anomaly_name])
            total_score += contribution

    anomaly_score = round(min(1.0, total_score), 4)
    anomaly_detected = anomaly_score > 0.0
    anomaly_type = pick_primary_anomaly_type(detections) if anomaly_detected else None
    severity = classify_severity(anomaly_score)
    confidence = compute_anomaly_confidence(anomaly_detected, len(affected_fields), anomaly_score)

    return {
        "anomaly_detected": anomaly_detected,
        "anomaly_type": anomaly_type,
        "severity": severity,
        "affected_fields": affected_fields,
        "anomaly_score": anomaly_score,
        "confidence": confidence,
        "rationale": "heuristic",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AnomalyDetectionAgent(BaseAgent):
    """Phase 25: Detect unusual patterns in enriched memory signals.

    Compares credibility, uncertainty, conflict, temporal, and decay signals
    against threshold baselines and flags memories that deviate significantly
    so downstream systems can quarantine or investigate them.
    """

    name = "AnomalyDetectionAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return [
            "ConflictDetectionAgent",
            "CredibilityAgent",
            "MemoryDecayAgent",
            "TemporalReasoningAgent",
            "UncertaintyReportingAgent",
        ]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("anomaly_detected"), bool):
            raise ValueError("anomaly_detected must be a bool")

        atype = outputs.get("anomaly_type")
        if atype not in _VALID_ANOMALY_TYPES:
            raise ValueError(
                f"anomaly_type must be one of {_VALID_ANOMALY_TYPES}, got {atype!r}"
            )

        severity = outputs.get("severity")
        if severity not in {"low", "medium", "high", None}:
            raise ValueError(f"severity must be low/medium/high/None, got {severity!r}")

        if not isinstance(outputs.get("affected_fields"), list):
            raise ValueError("affected_fields must be a list")

        score = outputs.get("anomaly_score")
        if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
            raise ValueError("anomaly_score must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment: dict[str, Any] = memory.get("enrichment") or {}

        strategy = getattr(settings, "ANOMALY_DETECTION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = run_heuristic(enrichment)
        else:
            prompt = (
                "You are an enterprise memory anomaly detection engine. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Analyse the enrichment signals below for unusual patterns. "
                "Anomaly types: credibility_spike, uncertainty_surge, conflict_flood, "
                "temporal_drift, stale_revival.\n\n"
                f"CREDIBILITY_SCORE: {enrichment.get('credibility_score')}\n"
                f"UNCERTAINTY_SCORE: {enrichment.get('uncertainty_score')}\n"
                f"UNCERTAINTY_LEVEL: {enrichment.get('uncertainty_level')}\n"
                f"CONFLICT_COUNT: {enrichment.get('conflict_count')}\n"
                f"TEMPORAL_CONFIDENCE: {enrichment.get('temporal_confidence')}\n"
                f"DECAY_FACTOR: {enrichment.get('decay_factor')}\n"
                f"IS_STALE: {enrichment.get('is_stale')}\n\n"
                "Return JSON with keys:\n"
                "- anomaly_detected: bool\n"
                "- anomaly_type: one of credibility_spike|uncertainty_surge|"
                "conflict_flood|temporal_drift|stale_revival|null\n"
                "- severity: low|medium|high|null\n"
                "- affected_fields: list of field name strings\n"
                "- anomaly_score: float 0..1\n"
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
                and isinstance(resp.get("anomaly_detected"), bool)
                and isinstance(resp.get("affected_fields"), list)
                and resp.get("anomaly_type") in _VALID_ANOMALY_TYPES
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
