"""Predictive World State Monitor — Phase 5.

Pattern-matches the current enterprise world model snapshot (from Phase 4)
against known risk configurations and alerts on anomalous or high-risk
state before it manifests.

Depends on Phase 4 (WorldModelAgent) for world_nodes, entity_edges,
and state_changes.

Risk patterns detected:
  rapid_domain_expansion    — entity is active across 3+ silos at state_version >= 3
  low_confidence_cross_silo — entity crosses silos but node_confidence < 0.55
  high_weight_edge_cluster  — 2+ edges with weight >= 0.80 (tight coupling risk)
  unresolved_expansion      — "concept"-type entities crossing 2+ silos
  state_change_cascade      — 2+ simultaneous domain_expansion events in one write

Outputs:
  state_alerts   list[{pattern, risk_score, affected_entities,
                        predicted_outcome, severity}]
  risk_summary   {total_alerts, high_risk_alerts, patterns_detected,
                  monitored_nodes}
  confidence     float
  rationale      "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


_DEFAULT_THRESHOLD = 0.45

_PATTERN_WEIGHTS: dict[str, float] = {
    "rapid_domain_expansion": 0.30,
    "state_change_cascade": 0.25,
    "low_confidence_cross_silo": 0.25,
    "high_weight_edge_cluster": 0.20,
    "unresolved_expansion": 0.15,
}

_PREDICTED_OUTCOMES: dict[str, str] = {
    "rapid_domain_expansion": "multi_team_blast_radius_if_entity_fails",
    "state_change_cascade": "cross_silo_write_storm_risk",
    "low_confidence_cross_silo": "entity_resolution_degradation_risk",
    "high_weight_edge_cluster": "tightly_coupled_failure_cascade_risk",
    "unresolved_expansion": "semantic_drift_across_silos_risk",
}

_HIGH_RISK_THRESHOLD = 0.65


def _severity(risk_score: float) -> str:
    if risk_score >= 0.75:
        return "critical"
    if risk_score >= 0.65:
        return "high"
    if risk_score >= 0.45:
        return "medium"
    return "low"


def score_alert(*, pattern: str, silo_count: int = 1, confidence_penalty: float = 0.0) -> float:
    """Compute risk score for a detected pattern.

    Base: 0.35
    Pattern weight: from _PATTERN_WEIGHTS (0.15..0.30)
    Silo spread bonus: min(0.15, silo_count * 0.04)
    Confidence penalty (for low-confidence entities): reduces score by up to 0.05
    Capped at 1.0.
    """
    weight = _PATTERN_WEIGHTS.get(pattern, 0.10)
    spread_bonus = min(0.15, silo_count * 0.04)
    score = 0.35 + weight + spread_bonus - confidence_penalty
    return min(round(score, 3), 1.0)


def detect_rapid_domain_expansion(
    nodes: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Alert when an entity spans 3+ silos at state_version >= 3.

    These are well-established but widely-spread entities; any failure
    creates a large blast radius.
    """
    alerts: list[dict[str, Any]] = []
    for node in nodes:
        silo_count = int(node.get("silo_count", 0))
        state_version = int(node.get("state_version", 1))
        if silo_count >= 3 and state_version >= 3:
            risk_score = score_alert(pattern="rapid_domain_expansion", silo_count=silo_count)
            if risk_score < threshold:
                continue
            alerts.append(
                {
                    "pattern": "rapid_domain_expansion",
                    "risk_score": risk_score,
                    "affected_entities": [node["entity"]],
                    "predicted_outcome": _PREDICTED_OUTCOMES["rapid_domain_expansion"],
                    "severity": _severity(risk_score),
                }
            )
    return alerts


def detect_low_confidence_cross_silo(
    nodes: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Alert when an entity crosses 2+ silos but node_confidence < 0.55.

    Entity is spreading but resolution quality is weak — likely the same entity
    described differently in each silo, risking downstream mis-linking.
    """
    alerts: list[dict[str, Any]] = []
    for node in nodes:
        silo_count = int(node.get("silo_count", 0))
        node_confidence = float(node.get("node_confidence", 0.5))
        if silo_count >= 2 and node_confidence < 0.55:
            penalty = round((0.55 - node_confidence) * 0.10, 3)
            risk_score = score_alert(
                pattern="low_confidence_cross_silo",
                silo_count=silo_count,
                confidence_penalty=penalty,
            )
            if risk_score < threshold:
                continue
            alerts.append(
                {
                    "pattern": "low_confidence_cross_silo",
                    "risk_score": risk_score,
                    "affected_entities": [node["entity"]],
                    "predicted_outcome": _PREDICTED_OUTCOMES["low_confidence_cross_silo"],
                    "severity": _severity(risk_score),
                }
            )
    return alerts


def detect_high_weight_edge_cluster(
    edges: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Alert when 2+ edges have weight >= 0.80.

    Tightly coupled high-confidence entities form a dependency cluster;
    a failure in any one node is likely to propagate to the others.
    """
    heavy_edges = [e for e in edges if float(e.get("weight", 0.0)) >= 0.80]
    if len(heavy_edges) < 2:
        return []

    entities: set[str] = set()
    for edge in heavy_edges:
        entities.add(edge.get("from_entity", ""))
        entities.add(edge.get("to_entity", ""))
    entities.discard("")

    risk_score = score_alert(
        pattern="high_weight_edge_cluster", silo_count=min(len(entities), 4)
    )
    if risk_score < threshold:
        return []

    return [
        {
            "pattern": "high_weight_edge_cluster",
            "risk_score": risk_score,
            "affected_entities": sorted(entities),
            "predicted_outcome": _PREDICTED_OUTCOMES["high_weight_edge_cluster"],
            "severity": _severity(risk_score),
        }
    ]


def detect_unresolved_expansion(
    nodes: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Alert when 'concept'-type entities cross 2+ silos.

    Ambiguous concepts spreading across departments indicate semantic drift —
    the same word meaning different things to different teams.
    """
    alerts: list[dict[str, Any]] = []
    for node in nodes:
        entity_type = (node.get("entity_type") or "").lower()
        silo_count = int(node.get("silo_count", 0))
        if entity_type == "concept" and silo_count >= 2:
            risk_score = score_alert(pattern="unresolved_expansion", silo_count=silo_count)
            if risk_score < threshold:
                continue
            alerts.append(
                {
                    "pattern": "unresolved_expansion",
                    "risk_score": risk_score,
                    "affected_entities": [node["entity"]],
                    "predicted_outcome": _PREDICTED_OUTCOMES["unresolved_expansion"],
                    "severity": _severity(risk_score),
                }
            )
    return alerts


def detect_state_change_cascade(
    state_changes: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Alert when a single write triggers 2+ simultaneous domain_expansion events.

    Multiple entities expanding across silos in one write indicates a large-scope
    cross-silo operation with elevated risk of inconsistency or write amplification.
    """
    expansions = [c for c in state_changes if c.get("change_type") == "domain_expansion"]
    if len(expansions) < 2:
        return []

    entities = [c.get("entity", "") for c in expansions if c.get("entity")]
    risk_score = score_alert(
        pattern="state_change_cascade", silo_count=min(len(entities), 4)
    )
    if risk_score < threshold:
        return []

    return [
        {
            "pattern": "state_change_cascade",
            "risk_score": risk_score,
            "affected_entities": entities,
            "predicted_outcome": _PREDICTED_OUTCOMES["state_change_cascade"],
            "severity": _severity(risk_score),
        }
    ]


def build_risk_summary(alerts: list[dict[str, Any]], node_count: int) -> dict[str, Any]:
    high_risk = sum(1 for a in alerts if a.get("risk_score", 0.0) >= _HIGH_RISK_THRESHOLD)
    patterns = sorted({a["pattern"] for a in alerts})
    return {
        "total_alerts": len(alerts),
        "high_risk_alerts": high_risk,
        "patterns_detected": patterns,
        "monitored_nodes": node_count,
    }


class PredictiveMonitorAgent(BaseAgent):
    """Phase 5: Pattern-match world model state and alert on risk before it manifests.

    Inputs (via context["memory"]["enrichment"]):
      - world_nodes:    list[{entity, entity_type, domains, silo_count,
                              state_version, node_confidence}]  — Phase 4
      - entity_edges:   list[{from_entity, to_entity, relationship_type, weight}] — Phase 4
      - state_changes:  list[{entity, change_type, description}]                   — Phase 4

    Outputs:
      - state_alerts:  list[{pattern, risk_score, affected_entities,
                             predicted_outcome, severity}]
      - risk_summary:  {total_alerts, high_risk_alerts, patterns_detected,
                        monitored_nodes}
      - confidence:    float
      - rationale:     "heuristic" | "llm"
    """

    name = "PredictiveMonitorAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["WorldModelAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("state_alerts", []), list):
            raise ValueError("state_alerts must be a list")
        if not isinstance(outputs.get("risk_summary", {}), dict):
            raise ValueError("risk_summary must be a dict")
        for alert in outputs.get("state_alerts", []):
            if not isinstance(alert, dict):
                raise ValueError("each state_alert must be a dict")
            if "pattern" not in alert:
                raise ValueError("each state_alert must have pattern")
            if "risk_score" not in alert:
                raise ValueError("each state_alert must have risk_score")
            if "affected_entities" not in alert:
                raise ValueError("each state_alert must have affected_entities")

    def _get_threshold(self) -> float:
        raw = getattr(settings, "PREDICTIVE_MONITOR_THRESHOLD", None)
        try:
            return float(raw) if raw is not None else _DEFAULT_THRESHOLD
        except (TypeError, ValueError):
            return _DEFAULT_THRESHOLD

    def _heuristic(
        self,
        *,
        world_nodes: list[dict[str, Any]],
        entity_edges: list[dict[str, Any]],
        state_changes: list[dict[str, Any]],
        threshold: float,
    ) -> dict[str, Any]:
        alerts: list[dict[str, Any]] = []
        alerts.extend(detect_rapid_domain_expansion(world_nodes, threshold))
        alerts.extend(detect_low_confidence_cross_silo(world_nodes, threshold))
        alerts.extend(detect_high_weight_edge_cluster(entity_edges, threshold))
        alerts.extend(detect_unresolved_expansion(world_nodes, threshold))
        alerts.extend(detect_state_change_cascade(state_changes, threshold))

        alerts.sort(key=lambda x: -x["risk_score"])

        summary = build_risk_summary(alerts, len(world_nodes))

        total = len(world_nodes)
        if total == 0:
            confidence = 0.4
        else:
            high_risk = summary["high_risk_alerts"]
            confidence = min(0.9, round(0.4 + 0.4 * min(high_risk / max(total, 1), 1.0), 3))

        return {
            "state_alerts": alerts,
            "risk_summary": summary,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        world_nodes: list[dict[str, Any]] = enrichment.get("world_nodes", [])
        entity_edges: list[dict[str, Any]] = enrichment.get("entity_edges", [])
        state_changes: list[dict[str, Any]] = enrichment.get("state_changes", [])

        threshold = self._get_threshold()

        strategy = getattr(settings, "PREDICTIVE_MONITOR_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not world_nodes:
            outputs = self._heuristic(
                world_nodes=world_nodes,
                entity_edges=entity_edges,
                state_changes=state_changes,
                threshold=threshold,
            )
        else:
            prompt = (
                "You are a predictive risk monitor for an enterprise memory OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Analyse the world model snapshot and detect risk patterns that "
                "could cause downstream incidents before they manifest.\n\n"
                f"THRESHOLD: {threshold}\n"
                f"WORLD NODES: {world_nodes}\n"
                f"ENTITY EDGES: {entity_edges}\n"
                f"STATE CHANGES: {state_changes}\n\n"
                "Detect patterns from: rapid_domain_expansion, low_confidence_cross_silo, "
                "high_weight_edge_cluster, unresolved_expansion, state_change_cascade.\n\n"
                "Return JSON with keys:\n"
                "- state_alerts: list of {pattern, risk_score, affected_entities, "
                "predicted_outcome, severity}\n"
                "- risk_summary: {total_alerts, high_risk_alerts, patterns_detected, "
                "monitored_nodes}\n"
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
                and isinstance(resp.get("state_alerts"), list)
                and isinstance(resp.get("risk_summary"), dict)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    world_nodes=world_nodes,
                    entity_edges=entity_edges,
                    state_changes=state_changes,
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
