"""Proactive Memory Push — Phase 11.

Shifts from reactive to proactive: detects triggers and recommends which teams
should receive a memory push before they ask for it.

Depends on Phase 10 (OrgAttentionAgent) for per-team attention signals and
overlap data.

Triggers:
  project_start      — new project / kickoff detected in content
  onboarding         — new team member detected in content
  overlap_alert      — cross-team duplicate work from Phase 10
  recurring_pattern  — same topic appears in multiple Phase 10 overlap alerts

Outputs:
  push_candidates    list[{recipient_team, topic, trigger_type,
                           urgency_score, delivery_hint}]
  triggers_detected  list[str]
  push_count         int
  confidence         float
  rationale          "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


_DEFAULT_THRESHOLD = 0.5

_PROJECT_START_KEYWORDS: frozenset[str] = frozenset([
    "new project",
    "kickoff",
    "kick off",
    "kick-off",
    "project launch",
    "project started",
    "project initiated",
    "starting project",
    "initiat",
])

_ONBOARDING_KEYWORDS: frozenset[str] = frozenset([
    "onboard",
    "new hire",
    "new member",
    "joins the team",
    "joined the team",
    "welcome",
    "new employee",
    "just joined",
    "recently joined",
])

_TRIGGER_WEIGHTS: dict[str, float] = {
    "recurring_pattern": 0.35,
    "overlap_alert": 0.30,
    "project_start": 0.25,
    "onboarding": 0.20,
}
_DEFAULT_TRIGGER_WEIGHT = 0.15


def detect_triggers(*, content: str) -> list[str]:
    """Return trigger types detected from memory content keywords."""
    lower = (content or "").lower()
    triggers: list[str] = []
    if any(kw in lower for kw in _PROJECT_START_KEYWORDS):
        triggers.append("project_start")
    if any(kw in lower for kw in _ONBOARDING_KEYWORDS):
        triggers.append("onboarding")
    return triggers


def score_push_candidate(*, trigger_type: str, relevance: float) -> float:
    """Compute push urgency score.

    Base: 0.40
    Trigger weight: 0.15..0.35 depending on trigger type
    Relevance boost: relevance * 0.10 (up to 0.10)
    Capped at 1.0.
    """
    weight = _TRIGGER_WEIGHTS.get((trigger_type or "").lower(), _DEFAULT_TRIGGER_WEIGHT)
    score = 0.40 + weight + float(relevance) * 0.10
    return min(round(score, 3), 1.0)


def _delivery_hint(urgency: float) -> str:
    if urgency >= 0.75:
        return "immediate_notification"
    if urgency >= 0.50:
        return "next_query_injection"
    return "webhook"


def build_push_candidates(
    *,
    triggers_detected: list[str],
    overlap_alerts: list[dict[str, Any]],
    attention_signals: list[dict[str, Any]],
    threshold: float,
    recurring_topics: set[str],
) -> list[dict[str, Any]]:
    """Assemble push candidates from overlap alerts and content triggers.

    Returns candidates sorted by urgency_score descending, capped at 50.
    """
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    # Push from overlap alerts (overlap_alert or recurring_pattern per topic)
    for alert in overlap_alerts:
        topic = alert.get("topic", "")
        teams: list[str] = alert.get("teams", [])
        trigger = "recurring_pattern" if topic in recurring_topics else "overlap_alert"

        for team in teams:
            key = (team, topic, trigger)
            if key in seen:
                continue
            seen.add(key)

            relevance = 0.0
            for sig in attention_signals:
                if sig.get("team") == team and sig.get("topic") == topic:
                    relevance = max(relevance, float(sig.get("relevance", 0.0)))

            urgency = score_push_candidate(trigger_type=trigger, relevance=relevance)
            if urgency < threshold:
                continue
            candidates.append(
                {
                    "recipient_team": team,
                    "topic": topic,
                    "trigger_type": trigger,
                    "urgency_score": urgency,
                    "delivery_hint": _delivery_hint(urgency),
                }
            )

    # Push from content triggers (project_start / onboarding)
    content_triggers = [t for t in triggers_detected if t in ("project_start", "onboarding")]
    for trigger in content_triggers:
        for sig in attention_signals:
            team = sig.get("team", "")
            topic = sig.get("topic", "")
            if not team or not topic:
                continue
            key = (team, topic, trigger)
            if key in seen:
                continue
            seen.add(key)

            relevance = float(sig.get("relevance", 0.0))
            urgency = score_push_candidate(trigger_type=trigger, relevance=relevance)
            if urgency < threshold:
                continue
            candidates.append(
                {
                    "recipient_team": team,
                    "topic": topic,
                    "trigger_type": trigger,
                    "urgency_score": urgency,
                    "delivery_hint": _delivery_hint(urgency),
                }
            )

    candidates.sort(key=lambda x: -x["urgency_score"])
    return candidates[:50]


class ProactiveMemoryPushAgent(BaseAgent):
    """Phase 11: Detect triggers and recommend proactive memory pushes.

    Inputs (via context["memory"]["enrichment"]):
      - attention_signals:  list[{team, topic, entity_type, relevance}] — Phase 10
      - overlap_alerts:     list[{topic, entity_type, teams, ...}]       — Phase 10

    Inputs (via context["memory"]["content"]):
      - Raw memory content for keyword-based trigger detection.

    Outputs:
      - push_candidates:    list[{recipient_team, topic, trigger_type,
                                  urgency_score, delivery_hint}]
      - triggers_detected:  list[str]
      - push_count:         int
      - confidence:         float
      - rationale:          "heuristic" | "llm"
    """

    name = "ProactiveMemoryPushAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["OrgAttentionAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("push_candidates", []), list):
            raise ValueError("push_candidates must be a list")
        if not isinstance(outputs.get("triggers_detected", []), list):
            raise ValueError("triggers_detected must be a list")
        for candidate in outputs.get("push_candidates", []):
            if not isinstance(candidate, dict):
                raise ValueError("each push_candidates item must be a dict")
            if "recipient_team" not in candidate:
                raise ValueError("each push_candidates item must have recipient_team")
            if "trigger_type" not in candidate:
                raise ValueError("each push_candidates item must have trigger_type")

    def _get_threshold(self) -> float:
        raw = getattr(settings, "PROACTIVE_PUSH_THRESHOLD", None)
        try:
            return float(raw) if raw is not None else _DEFAULT_THRESHOLD
        except (TypeError, ValueError):
            return _DEFAULT_THRESHOLD

    def _heuristic(
        self,
        *,
        content: str,
        attention_signals: list[dict[str, Any]],
        overlap_alerts: list[dict[str, Any]],
        threshold: float,
    ) -> dict[str, Any]:
        triggers_detected = detect_triggers(content=content)

        if overlap_alerts:
            if "overlap_alert" not in triggers_detected:
                triggers_detected.append("overlap_alert")

            topic_alert_counts: dict[str, int] = {}
            for alert in overlap_alerts:
                topic = alert.get("topic", "")
                topic_alert_counts[topic] = topic_alert_counts.get(topic, 0) + 1
            recurring_topics = {t for t, c in topic_alert_counts.items() if c > 1}
            if recurring_topics and "recurring_pattern" not in triggers_detected:
                triggers_detected.append("recurring_pattern")
        else:
            recurring_topics = set()

        candidates = build_push_candidates(
            triggers_detected=triggers_detected,
            overlap_alerts=overlap_alerts,
            attention_signals=attention_signals,
            threshold=threshold,
            recurring_topics=recurring_topics,
        )

        push_count = len(candidates)
        if push_count == 0:
            confidence = 0.4
        else:
            confidence = min(0.9, round(0.4 + 0.4 * min(push_count / 5.0, 1.0), 3))

        return {
            "push_candidates": candidates,
            "triggers_detected": triggers_detected,
            "push_count": push_count,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}
        content = (context.get("memory") or {}).get("content", "")

        attention_signals: list[dict[str, Any]] = enrichment.get("attention_signals", [])
        overlap_alerts: list[dict[str, Any]] = enrichment.get("overlap_alerts", [])

        threshold = self._get_threshold()

        strategy = getattr(settings, "PROACTIVE_PUSH_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                content=content,
                attention_signals=attention_signals,
                overlap_alerts=overlap_alerts,
                threshold=threshold,
            )
        else:
            prompt = (
                "You are a proactive memory push engine for an enterprise memory OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Detect triggers from the memory content and attention context, then "
                "recommend which teams should receive a proactive push of this memory.\n\n"
                f"THRESHOLD: {threshold}\n"
                f"CONTENT: {content[:500]}\n"
                f"ATTENTION SIGNALS: {attention_signals}\n"
                f"OVERLAP ALERTS: {overlap_alerts}\n\n"
                "Return JSON with keys:\n"
                "- push_candidates: list of {recipient_team, topic, trigger_type, "
                "urgency_score, delivery_hint}\n"
                "- triggers_detected: list of trigger type strings "
                "(project_start, onboarding, overlap_alert, recurring_pattern)\n"
                "- push_count: int\n"
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
                and isinstance(resp.get("push_candidates"), list)
                and isinstance(resp.get("triggers_detected"), list)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    content=content,
                    attention_signals=attention_signals,
                    overlap_alerts=overlap_alerts,
                    threshold=threshold,
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
