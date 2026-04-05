"""Adaptive Conflict Resolution — Phase 14.

Receives detected conflicts from Phase 13 (ConflictDetectionAgent) and
orchestrates resolution: assigns action types, routes to owning teams,
scores priority, and adapts hint weights based on resolution feedback.

Inputs (via context["memory"]["enrichment"]):
  - conflicts:           list[{entity, conflict_type, severity, source_silos,
                               resolution_hint, ...}]   — Phase 13
  - resolution_feedback: list[{hint_used, resolved}]   — optional feedback loop

Outputs:
  - resolution_actions:  list[{conflict_ref, entity, action_type, assigned_team,
                               priority, hint, hint_effectiveness}]
  - escalation_targets:  list[str]
  - resolution_rate:     float
  - hint_effectiveness:  dict[str, float]
  - conflict_count:      int
  - confidence:          float
  - rationale:           "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# Action type matrix: (conflict_type, severity) -> action_type
_ACTION_MAP: dict[tuple[str, str], str] = {
    ("state_mismatch", "high"): "escalate",
    ("state_mismatch", "medium"): "sync_teams",
    ("state_mismatch", "low"): "monitor",
    ("silo_gap", "high"): "investigate",
    ("silo_gap", "medium"): "investigate",
    ("silo_gap", "low"): "monitor",
    ("causal_loop", "high"): "escalate",
    ("causal_loop", "medium"): "monitor",
    ("causal_loop", "low"): "archive",
}
_DEFAULT_ACTION = "monitor"

_VALID_ACTION_TYPES: frozenset[str] = frozenset(
    ["escalate", "sync_teams", "investigate", "monitor", "archive"]
)

# Priority base score per severity
_SEVERITY_PRIORITY: dict[str, float] = {
    "high": 0.90,
    "medium": 0.55,
    "low": 0.25,
}

# Small additive boost per conflict type
_CONFLICT_TYPE_BOOST: dict[str, float] = {
    "state_mismatch": 0.05,
    "causal_loop": 0.03,
    "silo_gap": 0.02,
}

# Default hint effectiveness when no feedback is available
_DEFAULT_HINT_EFFECTIVENESS: dict[str, float] = {
    "escalate": 0.80,
    "sync_teams": 0.70,
    "investigate": 0.65,
    "monitor": 0.40,
    "archive": 0.30,
}

# Action types considered "active" (non-passive) for resolution_rate
_ACTIVE_ACTION_TYPES: frozenset[str] = frozenset(["escalate", "sync_teams", "investigate"])


def assign_action_type(*, conflict_type: str, severity: str) -> str:
    """Map (conflict_type, severity) to a resolution action type."""
    key = (str(conflict_type or "").lower(), str(severity or "").lower())
    return _ACTION_MAP.get(key, _DEFAULT_ACTION)


def assign_team(*, source_silos: list[str]) -> str:
    """Pick the owning team. First listed silo is treated as the authority."""
    if not source_silos:
        return "unknown"
    return source_silos[0]


def score_priority(*, severity: str, conflict_type: str) -> float:
    """Compute a 0..1 priority score for a conflict.

    Base: severity weight (0.25 / 0.55 / 0.90)
    Boost: conflict_type additive (0.02..0.05)
    Capped at 1.0.
    """
    base = _SEVERITY_PRIORITY.get(str(severity or "").lower(), 0.25)
    boost = _CONFLICT_TYPE_BOOST.get(str(conflict_type or "").lower(), 0.0)
    return min(round(base + boost, 3), 1.0)


def compute_hint_effectiveness(
    *, resolution_feedback: list[dict[str, Any]]
) -> dict[str, float]:
    """Compute per-hint effectiveness from historical resolution feedback.

    Falls back to built-in defaults when feedback is empty or a hint has
    no matching feedback entries.
    """
    result: dict[str, float] = dict(_DEFAULT_HINT_EFFECTIVENESS)

    if not resolution_feedback:
        return result

    counts: dict[str, list[int]] = {}  # hint -> [resolved_count, total_count]
    for fb in resolution_feedback:
        hint = str(fb.get("hint_used") or "").lower()
        if not hint:
            continue
        if hint not in counts:
            counts[hint] = [0, 0]
        counts[hint][1] += 1
        if fb.get("resolved"):
            counts[hint][0] += 1

    for hint, (resolved, total) in counts.items():
        if total > 0:
            result[hint] = round(resolved / total, 3)

    return result


def build_resolution_actions(
    *,
    conflicts: list[dict[str, Any]],
    hint_effectiveness: dict[str, float],
) -> list[dict[str, Any]]:
    """Build resolution action records for each conflict.

    Returns list sorted by priority descending, capped at 100.
    """
    actions: list[dict[str, Any]] = []

    for i, conflict in enumerate(conflicts):
        entity = conflict.get("entity", "")
        conflict_type = str(conflict.get("conflict_type") or "").lower()
        severity = str(conflict.get("severity") or "low").lower()
        source_silos: list[str] = conflict.get("source_silos", [])
        resolution_hint = str(conflict.get("resolution_hint") or "")

        action_type = assign_action_type(conflict_type=conflict_type, severity=severity)
        assigned_team = assign_team(source_silos=source_silos)
        priority = score_priority(severity=severity, conflict_type=conflict_type)
        effectiveness = hint_effectiveness.get(action_type, _DEFAULT_HINT_EFFECTIVENESS.get(action_type, 0.40))

        actions.append(
            {
                "conflict_ref": i,
                "entity": entity,
                "action_type": action_type,
                "assigned_team": assigned_team,
                "priority": priority,
                "hint": resolution_hint,
                "hint_effectiveness": effectiveness,
            }
        )

    actions.sort(key=lambda x: -x["priority"])
    return actions[:100]


def extract_escalation_targets(*, actions: list[dict[str, Any]]) -> list[str]:
    """Return unique assigned teams for escalate-type actions, in order."""
    seen: set[str] = set()
    targets: list[str] = []
    for action in actions:
        if action.get("action_type") == "escalate":
            team = action.get("assigned_team", "")
            if team and team not in seen:
                seen.add(team)
                targets.append(team)
    return targets


class AdaptiveConflictResolutionAgent(BaseAgent):
    """Phase 14: Orchestrate resolution of detected cross-silo conflicts.

    Inputs (via context["memory"]["enrichment"]):
      - conflicts:           list[{entity, conflict_type, severity, source_silos,
                                   resolution_hint, ...}]   — Phase 13
      - resolution_feedback: list[{hint_used, resolved}]

    Outputs:
      - resolution_actions:  list[{conflict_ref, entity, action_type, assigned_team,
                                   priority, hint, hint_effectiveness}]
      - escalation_targets:  list[str]
      - resolution_rate:     float
      - hint_effectiveness:  dict[str, float]
      - conflict_count:      int
      - confidence:          float
      - rationale:           "heuristic" | "llm"
    """

    name = "AdaptiveConflictResolutionAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["ConflictDetectionAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("resolution_actions", []), list):
            raise ValueError("resolution_actions must be a list")
        if not isinstance(outputs.get("escalation_targets", []), list):
            raise ValueError("escalation_targets must be a list")
        for action in outputs.get("resolution_actions", []):
            if not isinstance(action, dict):
                raise ValueError("each resolution_actions item must be a dict")
            if "entity" not in action:
                raise ValueError("each resolution_actions item must have entity")
            if "action_type" not in action:
                raise ValueError("each resolution_actions item must have action_type")
            if action["action_type"] not in _VALID_ACTION_TYPES:
                raise ValueError(f"invalid action_type: {action['action_type']}")

    def _heuristic(
        self,
        *,
        conflicts: list[dict[str, Any]],
        resolution_feedback: list[dict[str, Any]],
    ) -> dict[str, Any]:
        hint_effectiveness = compute_hint_effectiveness(
            resolution_feedback=resolution_feedback
        )
        actions = build_resolution_actions(
            conflicts=conflicts, hint_effectiveness=hint_effectiveness
        )
        escalation_targets = extract_escalation_targets(actions=actions)

        active_count = sum(
            1 for a in actions if a.get("action_type") in _ACTIVE_ACTION_TYPES
        )
        resolution_rate = round(active_count / len(actions), 3) if actions else 0.0

        n = len(conflicts)
        confidence = min(0.85, round(0.40 + 0.05 * min(n, 9), 3)) if n > 0 else 0.40

        return {
            "resolution_actions": actions,
            "escalation_targets": escalation_targets,
            "resolution_rate": resolution_rate,
            "hint_effectiveness": hint_effectiveness,
            "conflict_count": n,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}
        content = (context.get("memory") or {}).get("content", "")

        conflicts: list[dict[str, Any]] = enrichment.get("conflicts", [])
        resolution_feedback: list[dict[str, Any]] = enrichment.get(
            "resolution_feedback", []
        )

        strategy = getattr(settings, "CONFLICT_RESOLUTION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                conflicts=conflicts, resolution_feedback=resolution_feedback
            )
        else:
            prompt = (
                "You are an adaptive conflict resolution engine for an enterprise Cognitive OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Given the detected conflicts, assign resolution actions and routing.\n\n"
                f"CONFLICTS: {conflicts}\n"
                f"RESOLUTION_FEEDBACK: {resolution_feedback}\n\n"
                "Return JSON with keys:\n"
                "- resolution_actions: list of {conflict_ref, entity, action_type, "
                "assigned_team, priority, hint, hint_effectiveness}\n"
                "- escalation_targets: list of team name strings\n"
                "- resolution_rate: float 0..1\n"
                "- hint_effectiveness: dict mapping action_type to float\n"
                "- conflict_count: int\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation"
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
                and isinstance(resp.get("resolution_actions"), list)
                and isinstance(resp.get("escalation_targets"), list)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    conflicts=conflicts, resolution_feedback=resolution_feedback
                )

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.5)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
