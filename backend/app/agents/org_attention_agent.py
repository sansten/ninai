"""Organizational Attention Model — Phase 10.

Tracks active context per org unit (team/department) and detects when two or
more teams are independently working on the same topic.  Overlap alerts surface
before duplicate effort crystallises.

Depends on Phase 9 (SiloPropagationAgent) for cross-silo propagation signals
and Phase 6 (SemanticNormalizationAgent) for primary domain + entities.

Outputs:
  - attention_signals:  list[{team, topic, entity_type, relevance}]
  - overlap_alerts:     list[{topic, entity_type, teams, team_count,
                              overlap_score, alert_type}]
  - active_topics:      list[str]
  - overlap_count:      int
  - confidence:         float
  - rationale:          "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


_DEFAULT_THRESHOLD = 0.5

_ENTITY_TYPE_OVERLAP_WEIGHT: dict[str, float] = {
    "person": 0.20,
    "org": 0.20,
    "organization": 0.20,
    "customer": 0.15,
    "project": 0.12,
    "system": 0.10,
    "metric": 0.08,
    "role": 0.08,
    "concept": 0.05,
}
_DEFAULT_ENTITY_OVERLAP_WEIGHT = 0.05


def build_team_topic_map(
    *,
    propagation_signals: list[dict[str, Any]],
    primary_domain: str,
    entities: list[str],
) -> dict[str, dict[str, Any]]:
    """Map topic → {teams: set[str], entity_type: str}.

    Teams are collected from source_domain and target_domain in propagation
    signals, plus primary_domain for entities from Phase 6.
    """
    topic_map: dict[str, dict[str, Any]] = {}

    for sig in propagation_signals:
        entity = (sig.get("entity") or "").strip()
        if not entity:
            continue
        entity_type = (sig.get("entity_type") or "concept").lower()
        if entity not in topic_map:
            topic_map[entity] = {"teams": set(), "entity_type": entity_type}
        topic_map[entity]["teams"].add(sig.get("source_domain", ""))
        topic_map[entity]["teams"].add(sig.get("target_domain", ""))
        topic_map[entity]["teams"].discard("")

    for ent in entities:
        ent = (ent or "").strip()
        if not ent:
            continue
        if ent not in topic_map:
            topic_map[ent] = {"teams": set(), "entity_type": "concept"}
        topic_map[ent]["teams"].add(primary_domain)

    return topic_map


def score_overlap(*, team_count: int, entity_type: str) -> float:
    """Compute overlap score based on team count and entity type.

    Base: 0.40
    Each additional team beyond 1: +0.10 (capped at +0.30 for 4+ teams)
    Entity type weight: 0.05..0.20
    Capped at 1.0.
    """
    score = 0.40
    extra_teams = min(team_count - 1, 3)
    score += extra_teams * 0.10
    weight = _ENTITY_TYPE_OVERLAP_WEIGHT.get(
        (entity_type or "concept").lower(), _DEFAULT_ENTITY_OVERLAP_WEIGHT
    )
    score += weight
    return min(round(score, 3), 1.0)


def detect_overlaps(
    *,
    topic_map: dict[str, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Return overlap alerts for topics with 2+ teams above threshold."""
    alerts: list[dict[str, Any]] = []

    for topic, info in sorted(topic_map.items()):
        teams = sorted(info["teams"])
        if len(teams) < 2:
            continue
        entity_type = info["entity_type"]
        overlap_score = score_overlap(team_count=len(teams), entity_type=entity_type)
        if overlap_score < threshold:
            continue
        alerts.append(
            {
                "topic": topic,
                "entity_type": entity_type,
                "teams": teams,
                "team_count": len(teams),
                "overlap_score": overlap_score,
                "alert_type": "parallel_work",
            }
        )

    return alerts[:50]


class OrgAttentionAgent(BaseAgent):
    """Phase 10: Track per-team attention context and surface overlap alerts.

    Inputs (via context["memory"]["enrichment"]):
      - propagation_signals: list[{entity, entity_type, source_domain,
                                   target_domain, relevance_score, ...}]  — Phase 9
      - business_domain:     str — primary domain from Phase 6
      - entities:            list[str] — extracted entities from Phase 6

    Outputs:
      - attention_signals:  list[{team, topic, entity_type, relevance}]
      - overlap_alerts:     list[{topic, entity_type, teams, team_count,
                                  overlap_score, alert_type}]
      - active_topics:      list[str]
      - overlap_count:      int
      - confidence:         float
      - rationale:          "heuristic" | "llm"
    """

    name = "OrgAttentionAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["SiloPropagationAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("attention_signals", []), list):
            raise ValueError("attention_signals must be a list")
        if not isinstance(outputs.get("overlap_alerts", []), list):
            raise ValueError("overlap_alerts must be a list")
        for alert in outputs.get("overlap_alerts", []):
            if not isinstance(alert, dict):
                raise ValueError("each overlap_alerts item must be a dict")
            if "topic" not in alert:
                raise ValueError("each overlap_alerts item must have topic")
            if "teams" not in alert:
                raise ValueError("each overlap_alerts item must have teams")

    def _get_threshold(self) -> float:
        raw = getattr(settings, "ORG_ATTENTION_THRESHOLD", None)
        try:
            return float(raw) if raw is not None else _DEFAULT_THRESHOLD
        except (TypeError, ValueError):
            return _DEFAULT_THRESHOLD

    def _heuristic(
        self,
        *,
        propagation_signals: list[dict[str, Any]],
        primary_domain: str,
        entities: list[str],
        threshold: float,
    ) -> dict[str, Any]:
        topic_map = build_team_topic_map(
            propagation_signals=propagation_signals,
            primary_domain=primary_domain,
            entities=entities,
        )

        overlap_alerts = detect_overlaps(topic_map=topic_map, threshold=threshold)

        attention_signals: list[dict[str, Any]] = []
        for topic, info in sorted(topic_map.items()):
            entity_type = info["entity_type"]
            for team in sorted(info["teams"]):
                relevance = 0.5
                for sig in propagation_signals:
                    if sig.get("entity") == topic and (
                        sig.get("source_domain") == team or sig.get("target_domain") == team
                    ):
                        relevance = max(relevance, float(sig.get("relevance_score", 0.5)))
                attention_signals.append(
                    {
                        "team": team,
                        "topic": topic,
                        "entity_type": entity_type,
                        "relevance": round(relevance, 3),
                    }
                )

        active_topics = sorted(
            topic
            for topic, info in topic_map.items()
            if primary_domain in info["teams"]
        )

        overlap_count = len(overlap_alerts)
        total_topics = len(topic_map)
        if total_topics == 0:
            confidence = 0.4
        else:
            confidence = min(0.9, round(0.4 + 0.4 * (overlap_count / max(total_topics, 1)), 3))

        return {
            "attention_signals": attention_signals[:100],
            "overlap_alerts": overlap_alerts,
            "active_topics": active_topics,
            "overlap_count": overlap_count,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}
        content = (context.get("memory") or {}).get("content", "")

        propagation_signals: list[dict[str, Any]] = enrichment.get("propagation_signals", [])
        primary_domain: str = enrichment.get("business_domain", "general")
        entities: list[str] = enrichment.get("entities", [])

        threshold = self._get_threshold()

        strategy = getattr(settings, "ORG_ATTENTION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                propagation_signals=propagation_signals,
                primary_domain=primary_domain,
                entities=entities,
                threshold=threshold,
            )
        else:
            prompt = (
                "You are an organizational attention model for an enterprise memory OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Track which teams are working on which topics and detect overlap "
                "when 2+ teams independently work on the same entity.\n\n"
                f"PRIMARY DOMAIN: {primary_domain}\n"
                f"THRESHOLD: {threshold}\n"
                f"PROPAGATION SIGNALS: {propagation_signals}\n"
                f"ENTITIES: {entities}\n\n"
                "Return JSON with keys:\n"
                "- attention_signals: list of {team, topic, entity_type, relevance}\n"
                "- overlap_alerts: list of {topic, entity_type, teams, team_count, "
                "overlap_score, alert_type}\n"
                "- active_topics: list of topic strings for the primary domain\n"
                "- overlap_count: int\n"
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
                and isinstance(resp.get("attention_signals"), list)
                and isinstance(resp.get("overlap_alerts"), list)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    propagation_signals=propagation_signals,
                    primary_domain=primary_domain,
                    entities=entities,
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
