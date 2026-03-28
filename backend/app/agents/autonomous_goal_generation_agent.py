"""Autonomous Goal Generation Agent — Phase 37.

Self-directed goal generation driven by intrinsic motivation.  The agent
inspects upstream signals (knowledge gaps, usage patterns, tool reliability)
and generates goals the system should pursue on its own — without waiting for
a user request.

Depends on:
  - GoalDecompositionAgent  (Phase 21) — subtask signals in enrichment
  - ProactiveMemoryPushAgent (Phase 11) — push_candidates, triggers_detected
  - MetaCognitivePlanningAgent (Phase 36) — complexity_score, reasoning_mode

Inputs (via context["memory"]["enrichment"]):
  - knowledge_gaps:    list[dict]         — detected knowledge gaps
  - tool_reliability:  dict[str, dict]    — tool_name → {success_rate_30d, sample_size_30d}
  - activity_domains:  list[dict]         — [{domain, event_count}] from audit history
  - complexity_score:  float              — from MetaCognitivePlanningAgent

Outputs:
  - generated_goals:  list[dict]   — [{id, title, description, motivation_type,
                                        priority_score, trigger_ids, confidence}]
  - motivation_type:  str          — dominant: "curiosity"|"prediction"|"self_improvement"
  - priority_score:   float        — 0..1 aggregate urgency
  - goal_count:       int
  - rationale:        "heuristic" | "llm"
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_MOTIVATION_TYPES = frozenset({"curiosity", "prediction", "self_improvement"})

_LOW_RELIABILITY_THRESHOLD = 0.60   # tools below this trigger self-improvement goals
_MIN_SAMPLE_SIZE = 3                  # ignore tools with fewer samples
_MAX_GOALS = 10                       # cap on generated goals per run
_DEFAULT_PRIORITY = 0.50


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _build_curiosity_goals(knowledge_gaps: list[dict]) -> list[dict]:
    """Convert knowledge gaps into curiosity-driven goals."""
    goals: list[dict] = []
    for gap in knowledge_gaps:
        gap_type = str(gap.get("gap_type") or "unknown")
        domain = str(gap.get("domain") or "unknown")
        description = str(gap.get("description") or "")
        confidence = float(gap.get("confidence_in_gap", 0.7))
        related = gap.get("related_memories") or []

        goals.append({
            "id": str(uuid.uuid4()),
            "title": f"Investigate: {gap_type} gap in {domain}",
            "description": description or f"Knowledge gap detected in domain '{domain}'.",
            "motivation_type": "curiosity",
            "priority_score": round(min(0.9, max(0.3, confidence * 0.9)), 3),
            "trigger_ids": list(related),
            "confidence": round(confidence, 3),
            "metadata": {
                "gap_type": gap_type,
                "domain": domain,
                "suggested_approach": gap.get("suggested_learning_approach"),
            },
        })
    return goals


def _build_prediction_goals(activity_domains: list[dict]) -> list[dict]:
    """Convert activity domain counts into anticipatory preparation goals."""
    goals: list[dict] = []
    total_events = sum(int(d.get("event_count", 0)) for d in activity_domains)
    if total_events == 0:
        return goals

    for entry in activity_domains[:5]:  # top-5 domains
        domain = str(entry.get("domain") or "unknown")
        count = int(entry.get("event_count", 0))
        if count == 0:
            continue

        freq_ratio = count / total_events
        confidence = round(min(0.90, 0.40 + freq_ratio * 0.50), 3)

        goals.append({
            "id": str(uuid.uuid4()),
            "title": f"Prepare: Likely request about {domain}",
            "description": (
                f"Activity analysis: {count} recent events in '{domain}'. "
                f"Pre-loading context may reduce response latency."
            ),
            "motivation_type": "prediction",
            "priority_score": round(min(0.85, 0.35 + freq_ratio * 0.50), 3),
            "trigger_ids": [],
            "confidence": confidence,
            "metadata": {
                "domain": domain,
                "event_count": count,
                "frequency_ratio": round(freq_ratio, 3),
            },
        })
    return goals


def _build_improvement_goals(tool_reliability: dict[str, dict]) -> list[dict]:
    """Convert low-reliability tool stats into self-improvement goals."""
    goals: list[dict] = []
    for tool_name, stats in tool_reliability.items():
        success_rate = float(stats.get("success_rate_30d", 1.0))
        sample_size = int(stats.get("sample_size_30d", 0))
        if sample_size < _MIN_SAMPLE_SIZE:
            continue
        if success_rate >= _LOW_RELIABILITY_THRESHOLD:
            continue

        target_rate = round(min(1.0, success_rate + 0.25), 3)
        confidence = round(min(0.95, 0.50 + sample_size / 100.0), 3)
        urgency = round(1.0 - success_rate, 3)

        goals.append({
            "id": str(uuid.uuid4()),
            "title": f"Improve: {tool_name} reliability",
            "description": (
                f"Tool '{tool_name}' has {success_rate:.0%} 30-day success rate "
                f"({sample_size} calls). Target: {target_rate:.0%}."
            ),
            "motivation_type": "self_improvement",
            "priority_score": urgency,
            "trigger_ids": [],
            "confidence": confidence,
            "metadata": {
                "tool_name": tool_name,
                "current_success_rate": success_rate,
                "target_success_rate": target_rate,
                "sample_size_30d": sample_size,
            },
        })

    goals.sort(key=lambda g: g["priority_score"], reverse=True)
    return goals


def _rank_goals(goals: list[dict]) -> list[dict]:
    """Sort all goals by priority_score descending, cap at _MAX_GOALS."""
    ranked = sorted(goals, key=lambda g: float(g.get("priority_score", 0.0)), reverse=True)
    return ranked[:_MAX_GOALS]


def _dominant_motivation(goals: list[dict]) -> str:
    """Return the motivation_type with the highest average priority_score."""
    if not goals:
        return "curiosity"

    buckets: dict[str, list[float]] = {}
    for g in goals:
        mt = str(g.get("motivation_type", "curiosity"))
        buckets.setdefault(mt, []).append(float(g.get("priority_score", 0.0)))

    best = max(buckets, key=lambda k: sum(buckets[k]) / len(buckets[k]))
    return best


def _aggregate_priority(goals: list[dict]) -> float:
    """Weighted aggregate: top goal counts most."""
    if not goals:
        return 0.0
    scores = [float(g.get("priority_score", 0.0)) for g in goals]
    total = sum(s * (0.5 ** i) for i, s in enumerate(scores))
    norm = sum(0.5 ** i for i in range(len(scores)))
    return round(min(1.0, total / norm), 3) if norm > 0 else 0.0


async def run_heuristic_async(
    knowledge_gaps: list[dict],
    activity_domains: list[dict],
    tool_reliability: dict[str, dict],
) -> dict[str, Any]:
    """Heuristic path — pure logic, no LLM."""
    curiosity = _build_curiosity_goals(knowledge_gaps)
    prediction = _build_prediction_goals(activity_domains)
    improvement = _build_improvement_goals(tool_reliability)

    all_goals = _rank_goals(curiosity + prediction + improvement)
    motivation_type = _dominant_motivation(all_goals)
    priority_score = _aggregate_priority(all_goals)

    return {
        "generated_goals": all_goals,
        "motivation_type": motivation_type,
        "priority_score": priority_score,
        "goal_count": len(all_goals),
        "rationale": "heuristic",
    }


def _safe_default() -> dict[str, Any]:
    return {
        "generated_goals": [],
        "motivation_type": "curiosity",
        "priority_score": 0.0,
        "goal_count": 0,
        "rationale": "heuristic",
    }


def _parse_llm_output(resp: Any) -> dict[str, Any] | None:
    """Validate and normalise LLM JSON; return None if unusable."""
    if not isinstance(resp, dict):
        return None

    goals_raw = resp.get("generated_goals")
    if not isinstance(goals_raw, list):
        return None

    motivation_type = resp.get("motivation_type")
    if motivation_type not in _VALID_MOTIVATION_TYPES:
        return None

    priority_score = resp.get("priority_score")
    if not isinstance(priority_score, (int, float)) or not (0.0 <= float(priority_score) <= 1.0):
        return None

    # Normalise each goal — keep only structurally valid entries
    goals: list[dict] = []
    for g in goals_raw:
        if not isinstance(g, dict):
            continue
        title = g.get("title")
        if not title:
            continue
        goals.append({
            "id": str(g.get("id") or uuid.uuid4()),
            "title": str(title),
            "description": str(g.get("description") or ""),
            "motivation_type": str(g.get("motivation_type") or motivation_type),
            "priority_score": round(float(g.get("priority_score") or priority_score), 3),
            "trigger_ids": list(g.get("trigger_ids") or []),
            "confidence": round(float(g.get("confidence") or 0.5), 3),
            "metadata": g.get("metadata") or {},
        })

    if not goals:
        return None

    return {
        "generated_goals": goals,
        "motivation_type": str(motivation_type),
        "priority_score": round(float(priority_score), 3),
        "goal_count": len(goals),
        "rationale": "llm",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AutonomousGoalGenerationAgent(BaseAgent):
    """Phase 37: Generate self-directed goals from intrinsic motivation signals.

    The agent inspects knowledge gaps, usage patterns, and tool reliability to
    proactively surface goals the system should pursue without user prompting.
    """

    name = "AutonomousGoalGenerationAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return [
            "GoalDecompositionAgent",
            "ProactiveMemoryPushAgent",
            "MetaCognitivePlanningAgent",
        ]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        goals = outputs.get("generated_goals")
        if not isinstance(goals, list):
            raise ValueError("generated_goals must be a list")

        motivation_type = outputs.get("motivation_type")
        if motivation_type not in _VALID_MOTIVATION_TYPES:
            raise ValueError(
                f"motivation_type must be one of {_VALID_MOTIVATION_TYPES}, got {motivation_type!r}"
            )

        priority_score = outputs.get("priority_score")
        if not isinstance(priority_score, (int, float)) or not (0.0 <= float(priority_score) <= 1.0):
            raise ValueError("priority_score must be a float in [0, 1]")

        goal_count = outputs.get("goal_count")
        if not isinstance(goal_count, int) or goal_count < 0:
            raise ValueError("goal_count must be a non-negative int")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        enrichment: dict[str, Any] = (context.get("memory") or {}).get("enrichment") or {}

        knowledge_gaps: list[dict] = enrichment.get("knowledge_gaps") or []
        tool_reliability: dict[str, dict] = enrichment.get("tool_reliability") or {}
        activity_domains: list[dict] = enrichment.get("activity_domains") or []

        strategy = str(getattr(settings, "AGENT_STRATEGY", "llm") or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = await run_heuristic_async(knowledge_gaps, activity_domains, tool_reliability)
        else:
            prompt = (
                "You are an autonomous goal generation engine for an enterprise AI OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Given the signals below, generate self-directed goals the system should pursue.\n\n"
                f"KNOWLEDGE_GAPS: {knowledge_gaps}\n"
                f"ACTIVITY_DOMAINS: {activity_domains}\n"
                f"TOOL_RELIABILITY: {tool_reliability}\n\n"
                "Return JSON with keys:\n"
                "  generated_goals:  list of goal objects, each with:\n"
                "    id             str  (uuid)\n"
                "    title          str\n"
                "    description    str\n"
                "    motivation_type str (curiosity|prediction|self_improvement)\n"
                "    priority_score float (0-1)\n"
                "    trigger_ids    list[str]\n"
                "    confidence     float (0-1)\n"
                "    metadata       dict\n"
                "  motivation_type:  str  (dominant: curiosity|prediction|self_improvement)\n"
                "  priority_score:   float (0-1, aggregate urgency)\n"
            )
            try:
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
                parsed = _parse_llm_output(resp)
            except Exception:
                parsed = None

            if parsed is not None:
                outputs = parsed
            else:
                outputs = await run_heuristic_async(
                    knowledge_gaps, activity_domains, tool_reliability
                )

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("priority_score", _DEFAULT_PRIORITY)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
