"""Multi-Turn Goal Tracking — Phase 34.

Tracks goal state across a user session or time window by extending
GoalDecompositionAgent outputs.  Detects when a user returns to a prior goal,
surfaces stalled or blocked goals proactively, and computes a continuity score
showing how connected the current memory is to goals seen earlier in the
session.

Depends on:
  - GoalDecompositionAgent (Phase 21)   — goal_detected, subtasks,
                                          completion_fraction, blocking_subtask
  - TemporalReasoningAgent (Phase 17)   — temporal_phase
  - ProactiveMemoryPushAgent (Phase 11) — push_candidates (urgency signal)

Inputs (via context["memory"]["enrichment"]):
  - goal_detected:       bool
  - subtasks:            list[str]
  - completion_fraction: float
  - blocking_subtask:    str | None
  - temporal_phase:      str

Inputs (via context["session_context"]):
  - goals: list[dict]  — prior goals seen this session; each entry may contain:
      goal_id, description, subtasks, completion_fraction, last_seen,
      blocking_subtask

Outputs:
  - active_goals:          list[dict]   — goals currently in progress
  - stalled_goals:         list[dict]   — goals that appear blocked / stuck
  - goal_continuity_score: float 0..1
  - is_returning_goal:     bool         — current memory continues a prior goal
  - confidence:            float
  - rationale:             "heuristic" | "llm"
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.llm_breaker import create_llm_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTINUITY_MATCH_THRESHOLD = 2   # min shared tokens to treat as same goal
_STALL_FRACTION_THRESHOLD = 0.15  # completion fraction below this is suspect
_MAX_GOALS_LISTED = 20


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w{3,}\b", text.lower()))


def _goal_description(subtasks: list[str], content: str = "") -> str:
    if subtasks:
        return subtasks[0][:200]
    return (content or "")[:200].strip()


def match_prior_goal(
    current_subtasks: list[str],
    session_goals: list[dict[str, Any]],
    threshold: int = _CONTINUITY_MATCH_THRESHOLD,
) -> dict[str, Any] | None:
    """Return the session goal that best overlaps with current_subtasks, or None."""
    if not current_subtasks or not session_goals:
        return None

    current_tokens: set[str] = set()
    for s in current_subtasks:
        current_tokens |= _tokenize(s)

    best_match: dict[str, Any] | None = None
    best_overlap = threshold - 1

    for goal in session_goals:
        prior_tokens: set[str] = set()
        for s in goal.get("subtasks") or []:
            prior_tokens |= _tokenize(s)
        prior_tokens |= _tokenize(goal.get("description") or "")

        overlap = len(current_tokens & prior_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = goal

    return best_match


def classify_goal_stalled(
    goal: dict[str, Any],
    current_blocking: str | None,
    temporal_phase: str,
) -> bool:
    """True when the goal shows signs of being stuck or blocked."""
    if goal.get("blocking_subtask"):
        return True
    if current_blocking:
        return True
    fraction = float(goal.get("completion_fraction") or 0.0)
    if temporal_phase == "planning" and fraction < _STALL_FRACTION_THRESHOLD:
        return True
    return False


def compute_continuity_score(
    *,
    is_returning: bool,
    active_count: int,
    stalled_count: int,
    completion_fractions: list[float],
) -> float:
    """Continuity score in [0, 1] reflecting session goal coherence."""
    if not is_returning and active_count == 0:
        return 0.0

    base = 0.40
    if is_returning:
        base += 0.30
    if active_count > 0:
        base += min(0.15, active_count * 0.05)
    if stalled_count == 0 and active_count > 0:
        base += 0.05
    if completion_fractions:
        avg_cf = sum(completion_fractions) / len(completion_fractions)
        base += round(avg_cf * 0.10, 4)

    return round(min(1.0, base), 4)


def compute_agent_confidence(
    *,
    goal_detected: bool,
    has_session_goals: bool,
    is_returning: bool,
    active_count: int,
) -> float:
    """Confidence in the multi-turn goal tracking assessment."""
    base = 0.50
    if goal_detected:
        base += 0.10
    if has_session_goals:
        base += 0.08
    if is_returning:
        base += 0.10
    if active_count > 0:
        base += 0.05
    return round(min(0.88, base), 4)


def run_heuristic(
    enrichment: dict[str, Any],
    session_goals: list[dict[str, Any]],
    content: str,
) -> dict[str, Any]:
    """Heuristic multi-turn goal tracking."""
    goal_detected = bool(enrichment.get("goal_detected"))
    subtasks: list[str] = list(enrichment.get("subtasks") or [])
    completion_fraction = float(enrichment.get("completion_fraction") or 0.0)
    blocking_subtask = enrichment.get("blocking_subtask") or None
    temporal_phase = str(enrichment.get("temporal_phase") or "unknown").strip().lower()

    now_iso = datetime.now(timezone.utc).isoformat()

    active_goals: list[dict[str, Any]] = []
    stalled_goals: list[dict[str, Any]] = []
    is_returning_goal = False
    matched_prior: dict[str, Any] | None = None

    if goal_detected and subtasks:
        matched_prior = match_prior_goal(subtasks, session_goals)
        if matched_prior:
            is_returning_goal = True
            current_goal: dict[str, Any] = dict(matched_prior)
            current_goal["completion_fraction"] = completion_fraction
            current_goal["last_seen"] = now_iso
            current_goal["blocking_subtask"] = blocking_subtask
        else:
            current_goal = {
                "goal_id": str(uuid.uuid4()),
                "description": _goal_description(subtasks, content),
                "subtasks": subtasks,
                "completion_fraction": completion_fraction,
                "last_seen": now_iso,
                "blocking_subtask": blocking_subtask,
            }

        if classify_goal_stalled(current_goal, blocking_subtask, temporal_phase):
            stalled_goals.append({
                "goal_id": current_goal.get("goal_id"),
                "description": current_goal.get("description"),
                "stall_reason": str(blocking_subtask or "low_progress"),
            })
        else:
            active_goals.append({
                "goal_id": current_goal.get("goal_id"),
                "description": current_goal.get("description"),
                "completion_fraction": current_goal.get("completion_fraction"),
                "last_seen": current_goal.get("last_seen"),
            })

    # Carry forward prior goals that were not matched by the current memory.
    matched_id = matched_prior.get("goal_id") if matched_prior else None
    for g in session_goals:
        if g.get("goal_id") and g.get("goal_id") == matched_id:
            continue
        if classify_goal_stalled(g, None, temporal_phase):
            stalled_goals.append({
                "goal_id": g.get("goal_id"),
                "description": g.get("description"),
                "stall_reason": str(g.get("blocking_subtask") or "low_progress"),
            })
        else:
            active_goals.append({
                "goal_id": g.get("goal_id"),
                "description": g.get("description"),
                "completion_fraction": g.get("completion_fraction"),
                "last_seen": g.get("last_seen"),
            })

    active_goals = active_goals[:_MAX_GOALS_LISTED]
    stalled_goals = stalled_goals[:_MAX_GOALS_LISTED]

    fractions = [float(g.get("completion_fraction") or 0.0) for g in active_goals]
    continuity = compute_continuity_score(
        is_returning=is_returning_goal,
        active_count=len(active_goals),
        stalled_count=len(stalled_goals),
        completion_fractions=fractions,
    )
    confidence = compute_agent_confidence(
        goal_detected=goal_detected,
        has_session_goals=bool(session_goals),
        is_returning=is_returning_goal,
        active_count=len(active_goals),
    )

    return {
        "active_goals": active_goals,
        "stalled_goals": stalled_goals,
        "goal_continuity_score": continuity,
        "is_returning_goal": is_returning_goal,
        "confidence": confidence,
        "rationale": "heuristic",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MultiTurnGoalTrackingAgent(BaseAgent):
    """Phase 34: Track goal state across a session or time window.

    Merges current memory's goal signals with prior session goals to produce
    an up-to-date picture of active, stalled, and returning goals.
    """

    name = "MultiTurnGoalTrackingAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["GoalDecompositionAgent", "TemporalReasoningAgent", "ProactiveMemoryPushAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("active_goals"), list):
            raise ValueError("active_goals must be a list")

        if not isinstance(outputs.get("stalled_goals"), list):
            raise ValueError("stalled_goals must be a list")

        gcs = outputs.get("goal_continuity_score")
        if not isinstance(gcs, (int, float)) or not (0.0 <= float(gcs) <= 1.0):
            raise ValueError("goal_continuity_score must be a float in [0, 1]")

        if not isinstance(outputs.get("is_returning_goal"), bool):
            raise ValueError("is_returning_goal must be a bool")

        confidence = outputs.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            raise ValueError("confidence must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment: dict[str, Any] = memory.get("enrichment") or {}
        session_ctx: dict[str, Any] = context.get("session_context") or {}
        session_goals: list[dict[str, Any]] = list(session_ctx.get("goals") or [])
        content = str(memory.get("content") or "")

        strategy = getattr(settings, "MULTI_TURN_GOAL_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = run_heuristic(enrichment, session_goals, content)
        else:
            prompt = (
                "You are an enterprise multi-turn goal tracking engine for a Cognitive OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Given the current memory enrichment and prior session goals, identify "
                "which goals are active, which are stalled, and whether the user is "
                "returning to a prior goal.\n\n"
                f"GOAL_DETECTED: {enrichment.get('goal_detected')}\n"
                f"SUBTASKS: {enrichment.get('subtasks')}\n"
                f"COMPLETION_FRACTION: {enrichment.get('completion_fraction')}\n"
                f"BLOCKING_SUBTASK: {enrichment.get('blocking_subtask')}\n"
                f"TEMPORAL_PHASE: {enrichment.get('temporal_phase')}\n"
                f"SESSION_GOALS: {session_goals[:10]}\n\n"
                "Return JSON with keys:\n"
                "- active_goals: list of {goal_id, description, completion_fraction, last_seen}\n"
                "- stalled_goals: list of {goal_id, description, stall_reason}\n"
                "- goal_continuity_score: float 0..1\n"
                "- is_returning_goal: bool\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation"
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
                and isinstance(resp.get("active_goals"), list)
                and isinstance(resp.get("stalled_goals"), list)
                and isinstance(resp.get("is_returning_goal"), bool)
            ):
                outputs = resp
            else:
                outputs = run_heuristic(enrichment, session_goals, content)

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
