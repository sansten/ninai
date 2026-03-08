"""World Model Agent — Phase 4.

Builds a unified world model graph snapshot from resolved entities and
cross-silo links produced by Phase 7 (EntityResolutionAgent).

Produces a graph-structured view of enterprise entities as they appear in
this memory: world nodes (canonical entities with silo spread and state
version) and entity edges (co-occurrence relationships between nodes).

Depends on Phase 7 (EntityResolutionAgent) via memory enrichment
resolved_entities and cross_silo_links fields.

Outputs power downstream phases:
  - Phase 5 (Predictive World State Monitor): world_nodes for pattern matching
  - Phase 8 (Context Amplifier): entity graph available for richer fan-out
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Entity-type importance weights
# ---------------------------------------------------------------------------

_ENTITY_TYPE_WEIGHT: dict[str, float] = {
    "person": 0.20,
    "org": 0.20,
    "organization": 0.20,
    "customer": 0.15,
    "project": 0.10,
    "system": 0.10,
    "document": 0.10,
    "event": 0.08,
    "metric": 0.08,
    "role": 0.08,
    "time_period": 0.05,
    "concept": 0.05,
}

_DEFAULT_ENTITY_WEIGHT = 0.05


def compute_state_version(entity: dict[str, Any]) -> int:
    """Derive a lightweight state version for an entity node.

    Higher version = entity is better-known / more cross-silo:
      - base: 1
      - +1 if silo_count >= 2  (cross-silo discovery)
      - +1 if silo_count >= 3  (well-established multi-silo node)
      - +1 if confidence >= 0.85 (high-confidence resolution)
    """
    version = 1
    silo_count = int(entity.get("silo_count", len(entity.get("domains", []))))
    if silo_count >= 2:
        version += 1
    if silo_count >= 3:
        version += 1
    confidence = float(entity.get("confidence", 0.5))
    if confidence >= 0.85:
        version += 1
    return version


def compute_node_confidence(entity: dict[str, Any]) -> float:
    """Combine resolution confidence + entity-type weight into a node score."""
    base_conf = float(entity.get("confidence", 0.5))
    entity_type = (entity.get("entity_type") or "concept").lower()
    type_weight = _ENTITY_TYPE_WEIGHT.get(entity_type, _DEFAULT_ENTITY_WEIGHT)
    return min(round(base_conf + type_weight * 0.2, 3), 1.0)


def build_world_nodes(
    resolved_entities: list[dict[str, Any]],
    cross_silo_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build world model nodes from resolved entities, deduplicated by canonical name.

    Cross-silo links enrich the silo_count of matching nodes.
    """
    silo_lookup: dict[str, int] = {
        link["canonical"]: link.get("silo_count", 2)
        for link in cross_silo_links
        if "canonical" in link
    }

    seen: set[str] = set()
    nodes: list[dict[str, Any]] = []

    for entity in resolved_entities:
        canonical = entity.get("canonical", "")
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)

        domains = list(entity.get("domains", []))
        silo_count = silo_lookup.get(canonical, max(1, len(domains)))
        enriched = {**entity, "silo_count": silo_count}

        nodes.append(
            {
                "entity": canonical,
                "entity_type": entity.get("entity_type", "concept"),
                "domains": domains,
                "silo_count": silo_count,
                "state_version": compute_state_version(enriched),
                "node_confidence": compute_node_confidence(enriched),
            }
        )

    return nodes


def build_entity_edges(
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build co-occurrence edges between all pairs of world nodes in this memory.

    Weight = average of both nodes' node_confidence.
    Capped at 100 edges to avoid combinatorial explosion.
    """
    edges: list[dict[str, Any]] = []
    for a, b in combinations(nodes, 2):
        weight = round((a["node_confidence"] + b["node_confidence"]) / 2, 3)
        edges.append(
            {
                "from_entity": a["entity"],
                "to_entity": b["entity"],
                "relationship_type": "co_occurrence",
                "weight": weight,
            }
        )
    return edges[:100]


def detect_state_changes(
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect notable state transitions encoded in the current world model snapshot.

    Since this agent is stateless, changes are inferred from the snapshot:
      - domain_expansion: entity spans 2+ silos (crossing department boundaries)
      - high_confidence_node: entity resolved with node_confidence >= 0.85
    """
    changes: list[dict[str, Any]] = []
    for node in nodes:
        if node["silo_count"] >= 2:
            changes.append(
                {
                    "entity": node["entity"],
                    "change_type": "domain_expansion",
                    "description": (
                        f"{node['entity']} is active across {node['silo_count']} silos: "
                        + ", ".join(node["domains"])
                    ),
                }
            )
        if node["node_confidence"] >= 0.85:
            changes.append(
                {
                    "entity": node["entity"],
                    "change_type": "high_confidence_node",
                    "description": (
                        f"{node['entity']} resolved with node_confidence {node['node_confidence']}"
                    ),
                }
            )
    return changes


def build_graph_summary(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    state_changes: list[dict[str, Any]],
) -> dict[str, Any]:
    cross_silo_nodes = sum(1 for n in nodes if n["silo_count"] >= 2)
    high_confidence_nodes = sum(1 for n in nodes if n["node_confidence"] >= 0.85)
    expansion_events = sum(1 for c in state_changes if c["change_type"] == "domain_expansion")
    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "cross_silo_nodes": cross_silo_nodes,
        "high_confidence_nodes": high_confidence_nodes,
        "expansion_events": expansion_events,
    }


class WorldModelAgent(BaseAgent):
    """Phase 4: Build a unified world model graph from resolved entities.

    Inputs (via context["memory"]["enrichment"]):
      - resolved_entities:  list[{canonical, entity_type, domains, confidence, ...}] — Phase 7
      - cross_silo_links:   list[{canonical, entity_type, domains, silo_count}]       — Phase 7

    Outputs:
      - world_nodes:    list[{entity, entity_type, domains, silo_count,
                              state_version, node_confidence}]
      - entity_edges:   list[{from_entity, to_entity, relationship_type, weight}]
      - state_changes:  list[{entity, change_type, description}]
      - graph_summary:  {total_nodes, total_edges, cross_silo_nodes,
                         high_confidence_nodes, expansion_events}
      - confidence:     float
      - rationale:      "heuristic" | "llm"
    """

    name = "WorldModelAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["EntityResolutionAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("world_nodes", []), list):
            raise ValueError("world_nodes must be a list")
        if not isinstance(outputs.get("entity_edges", []), list):
            raise ValueError("entity_edges must be a list")
        if not isinstance(outputs.get("state_changes", []), list):
            raise ValueError("state_changes must be a list")
        if not isinstance(outputs.get("graph_summary", {}), dict):
            raise ValueError("graph_summary must be a dict")
        for node in outputs.get("world_nodes", []):
            if not isinstance(node, dict):
                raise ValueError("each world_node must be a dict")
            if "entity" not in node or "entity_type" not in node:
                raise ValueError("each world_node must have entity and entity_type")
        for edge in outputs.get("entity_edges", []):
            if not isinstance(edge, dict):
                raise ValueError("each entity_edge must be a dict")
            if "from_entity" not in edge or "to_entity" not in edge:
                raise ValueError("each entity_edge must have from_entity and to_entity")

    def _heuristic(
        self,
        *,
        resolved_entities: list[dict[str, Any]],
        cross_silo_links: list[dict[str, Any]],
    ) -> dict[str, Any]:
        nodes = build_world_nodes(resolved_entities, cross_silo_links)
        edges = build_entity_edges(nodes)
        changes = detect_state_changes(nodes)
        summary = build_graph_summary(nodes, edges, changes)

        total = len(resolved_entities)
        if total == 0:
            confidence = 0.4
        else:
            cross_silo = sum(1 for n in nodes if n["silo_count"] >= 2)
            confidence = min(0.9, round(0.4 + 0.5 * (cross_silo / max(total, 1)), 3))

        return {
            "world_nodes": nodes,
            "entity_edges": edges,
            "state_changes": changes,
            "graph_summary": summary,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        content = (context.get("memory") or {}).get("content", "")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        resolved_entities: list[dict[str, Any]] = enrichment.get("resolved_entities", [])
        cross_silo_links: list[dict[str, Any]] = enrichment.get("cross_silo_links", [])

        strategy = getattr(settings, "WORLD_MODEL_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                resolved_entities=resolved_entities,
                cross_silo_links=cross_silo_links,
            )
        else:
            prompt = (
                "You are an enterprise knowledge graph builder. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Given resolved entities and cross-silo links, build a unified world model:\n"
                "- world_nodes: list of {entity, entity_type, domains, silo_count, "
                "state_version, node_confidence}\n"
                "- entity_edges: list of {from_entity, to_entity, relationship_type, weight}\n"
                "- state_changes: list of {entity, change_type, description}\n"
                "- graph_summary: {total_nodes, total_edges, cross_silo_nodes, "
                "high_confidence_nodes, expansion_events}\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation\n\n"
                f"RESOLVED ENTITIES: {resolved_entities}\n"
                f"CROSS SILO LINKS: {cross_silo_links}\n\n"
                "Return JSON with keys: world_nodes, entity_edges, state_changes, "
                "graph_summary, confidence, rationale"
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
                and isinstance(resp.get("world_nodes"), list)
                and isinstance(resp.get("entity_edges"), list)
                and isinstance(resp.get("state_changes"), list)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    resolved_entities=resolved_entities,
                    cross_silo_links=cross_silo_links,
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
