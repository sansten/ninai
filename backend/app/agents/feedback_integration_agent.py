"""Feedback Integration & Self-Correction — Phase 24.

Collects human-review outcomes (triggered by Phase 22's review_recommended flag)
and applies corrections back into the enrichment pipeline.  Adjusts credibility
weights, conflict resolution verdicts, and playbook success rates using EMA so
that repeated human signals compound over time.

Depends on (reads enrichment from):
  - UncertaintyReportingAgent (Phase 22) — uncertainty_level, review_recommended
  - CredibilityAgent (Phase 19)          — credibility_score, source_tier
  - AdaptiveConflictResolutionAgent (14) — resolution_rate, escalation_targets
  - PlaybookAgent (Phase 20)             — matched_playbook_id, playbook_confidence

Reads from context["memory"]["human_verdict"]:
  - verdict:              "approve" | "reject" | "partial" | None
  - credibility_override: float | None   — explicit credibility value from reviewer
  - resolution_confirmed: bool | None    — True=fully resolved, False=still open
  - playbook_correct:     bool | None    — False=playbook match was wrong
  - reviewer_id:          str | None

Outputs:
  - feedback_applied:     bool
  - corrected_fields:     list[str]
  - correction_delta:     dict[str, float]   — field → signed magnitude of change
  - adjusted_credibility: float | None       — new credibility score after EMA
  - feedback_confidence:  float
  - rationale:            "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMA_ALPHA = 0.25                  # weight given to new human signal
_PLAYBOOK_PENALTY = 0.10           # subtracted when playbook is marked wrong
_VERDICT_BOOST = 0.90              # target credibility on approve with no override
_VERDICT_PENALTY = 0.10            # target credibility on reject with no override
_VALID_VERDICTS = {"approve", "reject", "partial", None}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _is_float(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _ema(current: float, new_signal: float, alpha: float = _EMA_ALPHA) -> float:
    """Exponential moving average: blend new signal into current value."""
    return round((1.0 - alpha) * current + alpha * new_signal, 4)


def apply_credibility_correction(
    enrichment: dict[str, Any],
    human_verdict: dict[str, Any],
) -> tuple[float | None, float]:
    """Compute adjusted credibility and the delta.

    Returns (adjusted_credibility, delta).  Returns (None, 0.0) if no
    credibility correction is warranted.
    """
    current_raw = enrichment.get("credibility_score")
    current: float = float(current_raw) if _is_float(current_raw) else 0.5

    override = human_verdict.get("credibility_override")
    verdict = human_verdict.get("verdict")

    if override is not None and _is_float(override):
        target = max(0.0, min(1.0, float(override)))
    elif verdict == "approve":
        target = _VERDICT_BOOST
    elif verdict == "reject":
        target = _VERDICT_PENALTY
    else:
        return None, 0.0  # partial or None with no override — no credibility change

    adjusted = _ema(current, target)
    delta = round(adjusted - current, 4)
    return adjusted, delta


def apply_resolution_correction(
    enrichment: dict[str, Any],
    human_verdict: dict[str, Any],
) -> float | None:
    """Return the resolution_rate delta if resolution_confirmed is set."""
    confirmed = human_verdict.get("resolution_confirmed")
    if confirmed is None:
        return None

    current_raw = enrichment.get("resolution_rate")
    current: float = float(current_raw) if _is_float(current_raw) else 0.0
    target = 1.0 if confirmed else 0.0
    return round(target - current, 4)


def apply_playbook_correction(
    enrichment: dict[str, Any],
    human_verdict: dict[str, Any],
) -> float | None:
    """Return the playbook_confidence delta if playbook_correct is False."""
    if human_verdict.get("playbook_correct") is not False:
        return None

    current_raw = enrichment.get("playbook_confidence")
    if current_raw is None or not _is_float(current_raw):
        return None

    current = float(current_raw)
    new_val = max(0.0, current - _PLAYBOOK_PENALTY)
    return round(new_val - current, 4)


def compute_feedback_confidence(
    corrected_fields: list[str],
    reviewer_id: str | None,
    verdict: str | None,
) -> float:
    """Confidence that the feedback was meaningful and correctly applied."""
    base = 0.65

    if reviewer_id:
        base += 0.05  # identified reviewer adds trust

    field_count = len(corrected_fields)
    if field_count == 0:
        base -= 0.15  # no corrections → low confidence feedback did anything
    elif field_count >= 2:
        base += 0.05  # multiple corrections → richer signal

    if verdict in {"approve", "reject"}:
        base += 0.05  # clear verdict boosts confidence

    return round(min(0.90, max(0.30, base)), 4)


def run_heuristic(
    enrichment: dict[str, Any],
    human_verdict: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply all corrections from human_verdict to enrichment signals."""
    if not human_verdict or not isinstance(human_verdict, dict):
        return {
            "feedback_applied": False,
            "corrected_fields": [],
            "correction_delta": {},
            "adjusted_credibility": None,
            "feedback_confidence": compute_feedback_confidence([], None, None),
            "rationale": "heuristic",
        }

    corrected_fields: list[str] = []
    correction_delta: dict[str, float] = {}
    adjusted_credibility: float | None = None
    verdict = human_verdict.get("verdict")

    # Credibility
    adj_cred, cred_delta = apply_credibility_correction(enrichment, human_verdict)
    if adj_cred is not None:
        adjusted_credibility = adj_cred
        correction_delta["credibility_score"] = cred_delta
        corrected_fields.append("credibility_score")

    # Resolution rate
    res_delta = apply_resolution_correction(enrichment, human_verdict)
    if res_delta is not None:
        correction_delta["resolution_rate"] = res_delta
        corrected_fields.append("resolution_rate")

    # Playbook confidence
    pb_delta = apply_playbook_correction(enrichment, human_verdict)
    if pb_delta is not None:
        correction_delta["playbook_confidence"] = pb_delta
        corrected_fields.append("playbook_confidence")

    reviewer_id = human_verdict.get("reviewer_id") or None
    confidence = compute_feedback_confidence(corrected_fields, reviewer_id, verdict)

    return {
        "feedback_applied": len(corrected_fields) > 0,
        "corrected_fields": corrected_fields,
        "correction_delta": correction_delta,
        "adjusted_credibility": adjusted_credibility,
        "feedback_confidence": confidence,
        "rationale": "heuristic",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FeedbackIntegrationAgent(BaseAgent):
    """Phase 24: Apply human review corrections to enrichment signals.

    Reads human_verdict from context and EMA-adjusts credibility, resolution
    rate, and playbook confidence so the system improves with each expert review.
    """

    name = "FeedbackIntegrationAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return [
            "UncertaintyReportingAgent",
            "CredibilityAgent",
            "AdaptiveConflictResolutionAgent",
            "PlaybookAgent",
        ]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("feedback_applied"), bool):
            raise ValueError("feedback_applied must be a bool")

        if not isinstance(outputs.get("corrected_fields"), list):
            raise ValueError("corrected_fields must be a list")

        if not isinstance(outputs.get("correction_delta"), dict):
            raise ValueError("correction_delta must be a dict")

        adj = outputs.get("adjusted_credibility")
        if adj is not None and not isinstance(adj, (int, float)):
            raise ValueError("adjusted_credibility must be a float or None")

        if adj is not None and not (0.0 <= float(adj) <= 1.0):
            raise ValueError("adjusted_credibility must be in [0, 1]")

        conf = outputs.get("feedback_confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            raise ValueError("feedback_confidence must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment: dict[str, Any] = memory.get("enrichment") or {}
        human_verdict: dict[str, Any] | None = memory.get("human_verdict") or None

        strategy = getattr(settings, "FEEDBACK_INTEGRATION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = run_heuristic(enrichment, human_verdict)
        else:
            verdict_str = str(human_verdict) if human_verdict else "null"
            prompt = (
                "You are an enterprise memory feedback integration engine. "
                "Output JSON only. Do not hallucinate.\n\n"
                "A human reviewer has assessed a memory enrichment. Apply their corrections "
                "to the enrichment signals below. Use EMA (alpha=0.25) to blend overrides.\n\n"
                f"ENRICHMENT_CREDIBILITY: {enrichment.get('credibility_score')}\n"
                f"ENRICHMENT_RESOLUTION_RATE: {enrichment.get('resolution_rate')}\n"
                f"ENRICHMENT_PLAYBOOK_CONFIDENCE: {enrichment.get('playbook_confidence')}\n"
                f"HUMAN_VERDICT: {verdict_str}\n\n"
                "Return JSON with keys:\n"
                "- feedback_applied: bool\n"
                "- corrected_fields: list of field name strings\n"
                "- correction_delta: dict of field -> signed float delta\n"
                "- adjusted_credibility: float or null\n"
                "- feedback_confidence: float 0..1\n"
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
                and isinstance(resp.get("feedback_applied"), bool)
                and isinstance(resp.get("corrected_fields"), list)
                and isinstance(resp.get("correction_delta"), dict)
            ):
                outputs = resp
            else:
                outputs = run_heuristic(enrichment, human_verdict)

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("feedback_confidence", 0.65)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
