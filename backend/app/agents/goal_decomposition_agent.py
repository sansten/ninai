"""Goal Decomposition — Phase 21.

Breaks a high-level goal embedded in memory content into ordered subtasks,
estimates completion fraction using episodic and temporal signals, and
identifies any blocking subtask that is preventing forward progress.

Depends on:
  - EpisodicGroupingAgent (Phase 18)  — episode_key, episode_role
  - TemporalReasoningAgent (Phase 17) — temporal_phase
  - PlaybookAgent (Phase 20)          — matched_playbook_id, step_position

Inputs (via context["memory"]):
  - content: raw text of the memory

Inputs (via context["memory"]["enrichment"]):
  - episode_key:         str      — stable episode cluster key (Phase 18)
  - episode_role:        str      — opener|body|climax|closer (Phase 18)
  - temporal_phase:      str      — planning|execution|review|unknown (Phase 17)
  - matched_playbook_id: str|None — best matching playbook (Phase 20)
  - step_position:       int|None — current step in playbook (Phase 20)

Outputs:
  - goal_detected:       bool
  - subtasks:            list[str]
  - completion_fraction: float 0..1
  - blocking_subtask:    str | None
  - confidence:          float
  - rationale:           "heuristic" | "llm"
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOAL_PHRASES = re.compile(
    r"\b(plan|goal|objective|aim|target|milestone|intend|propose|"
    r"strategy|roadmap|initiative|mission|step[s]?|phase[s]?|stage[s]?)\b",
    re.IGNORECASE,
)

_NEED_TO = re.compile(r"\b(need\s+to|going\s+to|want\s+to|have\s+to|will|should|must)\b", re.IGNORECASE)

_NUMBERED_ITEM = re.compile(r"^\s*\d+[.)]\s+(.+)$", re.MULTILINE)
_BULLET_ITEM = re.compile(r"^\s*[-*•]\s+(.+)$", re.MULTILINE)
_STEP_ITEM = re.compile(r"\bstep\s*\d+[:\s]+([^\n.!?]{3,})", re.IGNORECASE)

_ACTION_VERBS = re.compile(
    r"\b(analyze|create|build|deploy|fix|update|review|test|design|"
    r"implement|check|verify|run|setup|configure|migrate|monitor|"
    r"evaluate|schedule|contact|send|document|prepare|plan|define|"
    r"resolve|investigate|refactor|optimize|integrate|validate|launch)\b",
    re.IGNORECASE,
)

_BLOCKING_RE = re.compile(
    r"\b(blocked|blocking|stuck|waiting\s+on|pending|cannot|can'?t|failed|"
    r"failure|error|issue|problem|delayed|delay|on\s+hold|stalled|missing|"
    r"incomplete|not\s+done|unresolved|blocker)\b",
    re.IGNORECASE,
)

_EPISODE_ROLE_FRACTION: dict[str, float] = {
    "opener": 0.10,
    "body": 0.40,
    "climax": 0.80,
    "closer": 1.00,
}

_TEMPORAL_PHASE_ADJUSTMENT: dict[str, float] = {
    "planning": -0.10,
    "execution": 0.00,
    "review": 0.15,
    "unknown": 0.00,
}

_MAX_SUBTASKS = 10
_MAX_SUBTASK_LEN = 200


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def detect_goal(content: str) -> bool:
    """True when content appears to contain a high-level goal or plan."""
    if not content or len(content.strip()) < 5:
        return False
    if _GOAL_PHRASES.search(content):
        return True
    if _NEED_TO.search(content) and _ACTION_VERBS.search(content):
        return True
    # A numbered list implies a structured plan even without explicit keywords.
    if _NUMBERED_ITEM.search(content):
        return True
    return False


def _is_action_sentence(text: str) -> bool:
    """True when text looks like an actionable task description."""
    text = text.strip()
    if len(text) < 8:
        return False
    return bool(_ACTION_VERBS.search(text))


def extract_subtasks(content: str, max_subtasks: int = _MAX_SUBTASKS) -> list[str]:
    """Extract ordered subtasks from content.

    Priority: numbered list > step pattern > bullet list > action sentences.
    Each subtask is stripped and truncated to _MAX_SUBTASK_LEN chars.
    """
    if not content:
        return []

    def _clean(items: list[str]) -> list[str]:
        seen: list[str] = []
        for item in items:
            s = item.strip()[:_MAX_SUBTASK_LEN]
            if s and s not in seen:
                seen.append(s)
        return seen[:max_subtasks]

    numbered = _NUMBERED_ITEM.findall(content)
    if numbered:
        return _clean(numbered)

    steps = _STEP_ITEM.findall(content)
    if steps:
        return _clean(steps)

    bullets = _BULLET_ITEM.findall(content)
    if bullets:
        return _clean(bullets)

    # Fallback: split into sentences and keep action sentences.
    sentences = re.split(r"[.!?;]\s+", content)
    action = [s for s in sentences if _is_action_sentence(s)]
    return _clean(action)


def estimate_completion_fraction(
    *,
    episode_role: str,
    temporal_phase: str,
    step_position: int | None,
    total_steps: int,
) -> float:
    """Estimate how far through the goal we are as a value in [0, 1].

    Uses episode_role as the primary signal, adjusted by temporal_phase.
    When step_position is available it blends in the per-step progress.
    """
    role = str(episode_role or "body").lower().strip()
    phase = str(temporal_phase or "unknown").lower().strip()

    base = _EPISODE_ROLE_FRACTION.get(role, 0.40)
    adjustment = _TEMPORAL_PHASE_ADJUSTMENT.get(phase, 0.00)

    if step_position is not None and total_steps > 0:
        step_frac = max(0.0, (step_position - 1) / total_steps)
        base = round(0.5 * base + 0.5 * step_frac, 4)

    fraction = round(base + adjustment, 4)
    return max(0.0, min(1.0, fraction))


def detect_blocking_subtask(subtasks: list[str], content: str) -> str | None:
    """Return the first subtask that appears blocked, or None.

    Checks each subtask for blocking language first, then falls back to
    finding the subtask most associated with blocking context in the content.
    """
    if not subtasks:
        return None

    for subtask in subtasks:
        if _BLOCKING_RE.search(subtask):
            return subtask[:_MAX_SUBTASK_LEN]

    block_match = _BLOCKING_RE.search(content)
    if not block_match:
        return None

    block_pos = block_match.start()
    window = content[max(0, block_pos - 120): block_pos + 120].lower()

    best_subtask: str | None = None
    best_overlap = 0
    for subtask in subtasks:
        tokens = set(re.findall(r"\b\w{3,}\b", subtask.lower()))
        overlap = sum(1 for t in tokens if t in window)
        if overlap > best_overlap:
            best_overlap = overlap
            best_subtask = subtask

    return best_subtask[:_MAX_SUBTASK_LEN] if best_subtask and best_overlap > 0 else None


def compute_agent_confidence(
    *,
    goal_detected: bool,
    subtask_count: int,
    has_blocking: bool,
    completion_fraction: float,
    has_playbook: bool,
) -> float:
    """Confidence in the goal decomposition output."""
    if not goal_detected:
        return 0.45
    if subtask_count == 0:
        return 0.50
    base = 0.60
    if subtask_count >= 3:
        base += 0.08
    if has_playbook:
        base += 0.07
    if completion_fraction > 0.0:
        base += 0.05
    return round(min(0.85, base), 4)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GoalDecompositionAgent(BaseAgent):
    """Phase 21: Decompose goal-bearing memories into ordered subtasks.

    Detects whether a memory encodes a high-level goal, breaks it into
    subtasks, estimates completion fraction, and flags any blocking step.
    """

    name = "GoalDecompositionAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["EpisodicGroupingAgent", "TemporalReasoningAgent", "PlaybookAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("goal_detected"), bool):
            raise ValueError("goal_detected must be a bool")

        if not isinstance(outputs.get("subtasks"), list):
            raise ValueError("subtasks must be a list")

        cf = outputs.get("completion_fraction")
        if not isinstance(cf, (int, float)) or not (0.0 <= float(cf) <= 1.0):
            raise ValueError("completion_fraction must be a float in [0, 1]")

        blocking = outputs.get("blocking_subtask")
        if blocking is not None and not isinstance(blocking, str):
            raise ValueError("blocking_subtask must be a str or None")

    def _heuristic(
        self,
        *,
        content: str,
        episode_role: str,
        temporal_phase: str,
        step_position: int | None,
        has_playbook: bool,
    ) -> dict[str, Any]:
        goal = detect_goal(content)
        subtasks = extract_subtasks(content) if goal else []
        total_steps = len(subtasks)
        cf = estimate_completion_fraction(
            episode_role=episode_role,
            temporal_phase=temporal_phase,
            step_position=step_position,
            total_steps=total_steps,
        )
        blocking = detect_blocking_subtask(subtasks, content)
        confidence = compute_agent_confidence(
            goal_detected=goal,
            subtask_count=total_steps,
            has_blocking=blocking is not None,
            completion_fraction=cf,
            has_playbook=has_playbook,
        )
        return {
            "goal_detected": goal,
            "subtasks": subtasks,
            "completion_fraction": cf,
            "blocking_subtask": blocking,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}

        content = str(memory.get("content") or "")
        episode_role = str(enrichment.get("episode_role") or "body").strip()
        temporal_phase = str(enrichment.get("temporal_phase") or "unknown").strip()

        step_position = enrichment.get("step_position")
        if step_position is not None:
            try:
                step_position = int(step_position)
            except (TypeError, ValueError):
                step_position = None

        matched_playbook_id = enrichment.get("matched_playbook_id")
        has_playbook = bool(matched_playbook_id)

        strategy = getattr(settings, "GOAL_DECOMPOSITION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                content=content,
                episode_role=episode_role,
                temporal_phase=temporal_phase,
                step_position=step_position,
                has_playbook=has_playbook,
            )
        else:
            prompt = (
                "You are an enterprise goal decomposition engine for a memory OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Analyze this memory and detect whether it contains a high-level goal. "
                "If so, decompose it into ordered subtasks, estimate how complete the "
                "goal appears to be, and identify any blocking subtask.\n\n"
                f"CONTENT: {content[:500]}\n"
                f"EPISODE_ROLE: {episode_role}\n"
                f"TEMPORAL_PHASE: {temporal_phase}\n"
                f"STEP_POSITION: {step_position}\n"
                f"HAS_PLAYBOOK: {has_playbook}\n\n"
                "Return JSON with keys:\n"
                "- goal_detected: bool\n"
                "- subtasks: list of strings (ordered)\n"
                "- completion_fraction: float 0..1\n"
                "- blocking_subtask: str or null\n"
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
                and isinstance(resp.get("goal_detected"), bool)
                and isinstance(resp.get("subtasks"), list)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    content=content,
                    episode_role=episode_role,
                    temporal_phase=temporal_phase,
                    step_position=step_position,
                    has_playbook=has_playbook,
                )

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
