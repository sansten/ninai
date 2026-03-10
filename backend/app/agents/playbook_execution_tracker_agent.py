"""Playbook Execution Tracker — Phase 33.

Tracks whether a playbook suggested by PlaybookAgent was actually followed and
records the outcome.  Computes a coverage-based outcome score, identifies which
steps were skipped or run out of order, and signals the result back so
StrategyLearningService can adjust future playbook rankings.

Depends on:
  - PlaybookAgent (Phase 20)            — matched_playbook_id, step_position,
                                          playbook_confidence
  - FeedbackIntegrationAgent (Phase 24) — feedback_applied, correction_delta

Inputs (via context["memory"]["enrichment"]):
  - matched_playbook_id:  str | None
  - step_position:        int | None
  - playbook_confidence:  float

Inputs (via context["memory"]["execution_trace"]):
  - executed_steps:  list[int]  — 1-based step indices actually run
  - outcome:         "success" | "failure" | "partial" | None
  - total_steps:     int | None — expected steps in the playbook
  - session_id:      str | None

Outputs:
  - playbook_followed:    bool
  - outcome_score:        float 0..1
  - deviation_steps:      list[int] — steps skipped or missing
  - tracked_playbook_id:  str | None
  - confidence:           float
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

_FOLLOW_THRESHOLD = 0.50   # min step coverage to call playbook "followed"
_VALID_OUTCOMES = {"success", "failure", "partial", None}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def compute_step_coverage(
    executed_steps: list[int],
    total_steps: int | None,
    step_position: int | None,
) -> float:
    """Fraction of playbook steps that were executed.

    Falls back to a weaker estimate when total_steps is unknown.
    """
    if not executed_steps:
        return 0.0

    if total_steps and total_steps > 0:
        unique = set(executed_steps)
        covered = len([s for s in unique if 1 <= s <= total_steps])
        return round(covered / total_steps, 4)

    # No total — use step_position as a rough denominator
    if step_position and step_position > 0:
        unique = set(executed_steps)
        covered = len([s for s in unique if s <= step_position])
        return round(min(1.0, covered / step_position), 4)

    return round(min(1.0, len(set(executed_steps)) / max(len(executed_steps), 1)), 4)


def find_deviation_steps(
    executed_steps: list[int],
    total_steps: int | None,
    step_position: int | None,
) -> list[int]:
    """Return step indices that were expected but not executed.

    When total_steps is known, expected range is [1..total_steps].
    When only step_position is known, expected range is [1..step_position].
    When neither is known, returns an empty list.
    """
    upper = None
    if total_steps and total_steps > 0:
        upper = total_steps
    elif step_position and step_position > 0:
        upper = step_position

    if upper is None:
        return []

    executed_set = set(executed_steps)
    return [i for i in range(1, upper + 1) if i not in executed_set]


def compute_outcome_score(
    outcome: str | None,
    coverage: float,
) -> float:
    """Score 0..1 blending declared outcome with step coverage.

    success  → anchored at 0.85, lifted by coverage
    partial  → anchored at 0.45, lifted by coverage
    failure  → anchored at 0.10, small coverage lift
    None     → pure coverage estimate
    """
    if outcome == "success":
        return round(min(1.0, 0.85 + 0.15 * coverage), 4)
    if outcome == "partial":
        return round(min(1.0, 0.45 + 0.35 * coverage), 4)
    if outcome == "failure":
        return round(min(1.0, 0.10 + 0.10 * coverage), 4)
    # outcome is None — coverage alone drives the score
    return round(coverage, 4)


def compute_agent_confidence(
    playbook_matched: bool,
    outcome: str | None,
    coverage: float,
    feedback_applied: bool,
) -> float:
    """Confidence in the tracking assessment."""
    base = 0.50

    if playbook_matched:
        base += 0.10
    if outcome in {"success", "failure"}:
        base += 0.10
    if coverage >= 0.80:
        base += 0.08
    elif coverage >= 0.50:
        base += 0.04
    if feedback_applied:
        base += 0.05

    return round(min(0.90, base), 4)


def run_heuristic(
    enrichment: dict[str, Any],
    execution_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    """Heuristic playbook execution tracking."""
    matched_id = enrichment.get("matched_playbook_id") or None
    step_position = enrichment.get("step_position")
    playbook_confidence = float(enrichment.get("playbook_confidence") or 0.0)
    feedback_applied = bool(enrichment.get("feedback_applied") or False)

    trace: dict[str, Any] = execution_trace if isinstance(execution_trace, dict) else {}
    executed_steps: list[int] = [
        int(s) for s in (trace.get("executed_steps") or [])
        if str(s).lstrip("-").isdigit()
    ]
    outcome_raw = trace.get("outcome")
    outcome = outcome_raw if outcome_raw in _VALID_OUTCOMES else None
    total_steps_raw = trace.get("total_steps")
    total_steps = int(total_steps_raw) if total_steps_raw and str(total_steps_raw).isdigit() else None

    if step_position is not None:
        try:
            step_position = int(step_position)
        except (TypeError, ValueError):
            step_position = None

    coverage = compute_step_coverage(executed_steps, total_steps, step_position)
    playbook_followed = bool(
        matched_id
        and len(executed_steps) > 0
        and playbook_confidence >= 0.30
        and coverage >= _FOLLOW_THRESHOLD
    )

    deviation_steps = find_deviation_steps(executed_steps, total_steps, step_position)
    outcome_score = compute_outcome_score(outcome, coverage)
    confidence = compute_agent_confidence(
        playbook_matched=bool(matched_id),
        outcome=outcome,
        coverage=coverage,
        feedback_applied=feedback_applied,
    )

    return {
        "playbook_followed": playbook_followed,
        "outcome_score": outcome_score,
        "deviation_steps": deviation_steps,
        "tracked_playbook_id": matched_id,
        "confidence": confidence,
        "rationale": "heuristic",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PlaybookExecutionTrackerAgent(BaseAgent):
    """Phase 33: Track whether a suggested playbook was followed and record outcome.

    Reads PlaybookAgent enrichment and an execution_trace from context to
    determine step coverage, deviations, and outcome quality.  Produces an
    outcome_score that downstream services (StrategyLearningService) can use
    to adjust playbook rankings.
    """

    name = "PlaybookExecutionTrackerAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["PlaybookAgent", "FeedbackIntegrationAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("playbook_followed"), bool):
            raise ValueError("playbook_followed must be a bool")

        outcome_score = outputs.get("outcome_score")
        if not isinstance(outcome_score, (int, float)) or not (0.0 <= float(outcome_score) <= 1.0):
            raise ValueError("outcome_score must be a float in [0, 1]")

        if not isinstance(outputs.get("deviation_steps"), list):
            raise ValueError("deviation_steps must be a list")

        tracked_id = outputs.get("tracked_playbook_id")
        if tracked_id is not None and not isinstance(tracked_id, str):
            raise ValueError("tracked_playbook_id must be a str or None")

        confidence = outputs.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            raise ValueError("confidence must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment: dict[str, Any] = memory.get("enrichment") or {}
        execution_trace: dict[str, Any] | None = memory.get("execution_trace") or None

        strategy = getattr(settings, "PLAYBOOK_TRACKER_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = run_heuristic(enrichment, execution_trace)
        else:
            trace_str = str(execution_trace) if execution_trace else "null"
            prompt = (
                "You are an enterprise playbook execution tracker for a memory OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Determine whether the suggested playbook was followed, compute an "
                "outcome score, and list deviation steps.\n\n"
                f"MATCHED_PLAYBOOK_ID: {enrichment.get('matched_playbook_id')}\n"
                f"STEP_POSITION: {enrichment.get('step_position')}\n"
                f"PLAYBOOK_CONFIDENCE: {enrichment.get('playbook_confidence')}\n"
                f"EXECUTION_TRACE: {trace_str}\n\n"
                "Return JSON with keys:\n"
                "- playbook_followed: bool\n"
                "- outcome_score: float 0..1\n"
                "- deviation_steps: list of int step indices skipped\n"
                "- tracked_playbook_id: str or null\n"
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
                and isinstance(resp.get("playbook_followed"), bool)
                and isinstance(resp.get("deviation_steps"), list)
            ):
                outputs = resp
            else:
                outputs = run_heuristic(enrichment, execution_trace)

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.50)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
