"""Causal Reasoning Agent — Phase 12.

Traces cause-effect chains across enterprise silos using the world model
graph (Phase 4) and cross-silo signals (Phase 9). Identifies root causes
for high-urgency push candidates (Phase 11) and produces counterfactual
intervention hints.

Depends on Phase 11 (ProactiveMemoryPushAgent).

Inputs (via context["memory"]["enrichment"]):
  - world_nodes:        list[{entity, entity_type, domains, silo_count,
                              state_version, node_confidence}]  — Phase 4
  - entity_edges:       list[{from_entity, to_entity,
                              relationship_type, weight}]        — Phase 4
  - propagated_signals: list[{content, source_silo, target_domains,
                              relevance_score, entities, ...}]   — Phase 9
  - push_candidates:    list[{recipient_team, topic, trigger_type,
                              urgency_score, delivery_hint}]     — Phase 11

Outputs:
  - causal_chains:       list[{effect_entity, causal_path, hop_count,
                               chain_strength, silo_span, narrative}]
  - root_causes:         list[{entity, entity_type, cause_score, domains}]
  - counterfactual_hint: str | None
  - top_effect:          str | None
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


# ---------------------------------------------------------------------------
# Entity-type cause weights: higher = entity more often acts as a root cause.
# Metrics and incidents are typically effects; people and orgs are causes.
# ---------------------------------------------------------------------------

_CAUSE_TYPE_WEIGHT: dict[str, float] = {
    "person": 1.0,
    "org": 0.95,
    "organization": 0.95,
    "customer": 0.90,
    "project": 0.80,
    "system": 0.75,
    "deployment": 0.70,
    "sprint": 0.65,
    "event": 0.60,
    "incident": 0.55,
    "document": 0.50,
    "concept": 0.40,
    "metric": 0.30,  # metrics are typically effects, not causes
}

_DEFAULT_CAUSE_WEIGHT = 0.45
_MAX_CHAINS = 10
_MAX_HOPS = 3


def _cause_weight(entity_type: str) -> float:
    return _CAUSE_TYPE_WEIGHT.get((entity_type or "").lower(), _DEFAULT_CAUSE_WEIGHT)


def identify_effect_candidates(
    *,
    world_nodes: list[dict[str, Any]],
    push_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Identify entities that are likely effects to explain causally.

    Seeds from two sources:
      1. Push candidate topics with urgency_score >= 0.5 (Phase 11 flagged them)
      2. World nodes with silo_count >= 2 (cross-silo spread = effect of multi-team activity)

    Returns list of {entity, entity_type, domains, effect_score} sorted descending.
    Capped at 20.
    """
    node_lookup: dict[str, dict[str, Any]] = {n["entity"]: n for n in world_nodes}
    scores: dict[str, float] = {}
    entity_meta: dict[str, dict[str, Any]] = {}

    for cand in push_candidates:
        topic = cand.get("topic", "")
        urgency = float(cand.get("urgency_score", 0.0))
        if topic and urgency >= 0.5:
            scores[topic] = max(scores.get(topic, 0.0), urgency)
            entity_meta[topic] = node_lookup.get(
                topic, {"entity": topic, "entity_type": "concept", "domains": []}
            )

    for node in world_nodes:
        entity = node["entity"]
        silo_count = int(node.get("silo_count", 1))
        if silo_count >= 2:
            cross_score = min(0.9, 0.4 + 0.15 * (silo_count - 1))
            scores[entity] = max(scores.get(entity, 0.0), cross_score)
            entity_meta[entity] = node

    candidates = [
        {
            "entity": entity,
            "entity_type": entity_meta[entity].get("entity_type", "concept"),
            "domains": entity_meta[entity].get("domains", []),
            "effect_score": round(score, 3),
        }
        for entity, score in scores.items()
    ]
    candidates.sort(key=lambda x: -x["effect_score"])
    return candidates[:20]


def build_adjacency(
    entity_edges: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build an undirected adjacency map from entity_edges.

    Returns {entity: [{neighbor, weight}, ...]}
    """
    adj: dict[str, list[dict[str, Any]]] = {}
    for edge in entity_edges:
        a = edge.get("from_entity", "")
        b = edge.get("to_entity", "")
        w = float(edge.get("weight", 0.5))
        if not a or not b:
            continue
        adj.setdefault(a, []).append({"neighbor": b, "weight": w})
        adj.setdefault(b, []).append({"neighbor": a, "weight": w})
    return adj


def trace_causal_chain(
    *,
    effect_entity: str,
    adjacency: dict[str, list[dict[str, Any]]],
    node_lookup: dict[str, dict[str, Any]],
    propagated_signal_entities: set[str],
    max_hops: int = _MAX_HOPS,
) -> list[dict[str, Any]]:
    """BFS backward from effect_entity to discover causal paths.

    Direction heuristic: a neighbor is a plausible cause if:
      - its cause_weight > current node's cause_weight  (type ordering: person > metric)
      - OR it appears in propagated_signals (came from another silo)

    Path strength = product of edge weights × cause_weight of the terminal cause node.
    Returns list of {causal_path, hop_count, chain_strength} sorted by chain_strength
    descending. Capped at _MAX_CHAINS.
    """
    effect_node = node_lookup.get(effect_entity, {})
    effect_cw = _cause_weight(effect_node.get("entity_type", "concept"))

    completed: list[dict[str, Any]] = []
    # frontier: (current_entity, path_so_far, cumulative_edge_strength)
    frontier: list[tuple[str, list[str], float]] = [(effect_entity, [effect_entity], 1.0)]
    visited_globally: set[tuple[str, ...]] = set()

    for _hop in range(max_hops):
        if not frontier:
            break
        next_frontier: list[tuple[str, list[str], float]] = []
        for current, path, strength in frontier:
            current_node = node_lookup.get(current, {})
            current_cw = _cause_weight(current_node.get("entity_type", "concept"))

            for adj_entry in adjacency.get(current, []):
                neighbor = adj_entry["neighbor"]
                if neighbor in path:
                    continue  # no cycles

                edge_w = adj_entry["weight"]
                neighbor_node = node_lookup.get(neighbor, {})
                neighbor_cw = _cause_weight(neighbor_node.get("entity_type", "concept"))

                is_causal = (
                    neighbor_cw > current_cw
                    or neighbor in propagated_signal_entities
                )
                if not is_causal:
                    continue

                new_path = path + [neighbor]
                path_key = tuple(new_path)
                if path_key in visited_globally:
                    continue
                visited_globally.add(path_key)

                new_strength = round(strength * edge_w * neighbor_cw, 4)
                completed.append(
                    {
                        "causal_path": new_path,
                        "hop_count": len(new_path) - 1,
                        "chain_strength": new_strength,
                    }
                )
                next_frontier.append((neighbor, new_path, new_strength))

        frontier = next_frontier

    # Compute silo_span for each completed path
    results = []
    for item in completed:
        silos: set[str] = set()
        for entity in item["causal_path"]:
            node = node_lookup.get(entity, {})
            for domain in node.get("domains", []):
                silos.add(domain)
        item["silo_span"] = len(silos)
        results.append(item)

    results.sort(key=lambda x: -x["chain_strength"])
    return results[:_MAX_CHAINS]


def extract_root_causes(
    chains: list[dict[str, Any]],
    node_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract distinct root cause entities from causal chains.

    Root cause = last element of causal_path (farthest from effect).
    Score = max chain_strength across chains where this entity is the root.
    Returns list sorted by cause_score descending.
    """
    cause_scores: dict[str, float] = {}
    for chain in chains:
        path = chain.get("causal_path", [])
        if len(path) < 2:
            continue
        root = path[-1]
        cause_scores[root] = max(cause_scores.get(root, 0.0), chain["chain_strength"])

    results = []
    for entity, score in cause_scores.items():
        node = node_lookup.get(entity, {})
        results.append(
            {
                "entity": entity,
                "entity_type": node.get("entity_type", "concept"),
                "cause_score": round(score, 3),
                "domains": node.get("domains", []),
            }
        )
    results.sort(key=lambda x: -x["cause_score"])
    return results


def build_chain_narrative(
    *,
    effect_entity: str,
    chain: dict[str, Any],
    node_lookup: dict[str, dict[str, Any]],
) -> str:
    """Generate a plain-English description of a causal chain.

    Path is stored as [effect, ..., root_cause]; we reverse to read cause → effect.
    """
    path = chain.get("causal_path", [])
    if not path:
        return ""
    cause_to_effect = list(reversed(path))
    parts = []
    for entity in cause_to_effect:
        node = node_lookup.get(entity, {})
        etype = node.get("entity_type", "")
        label = f"{entity} ({etype})" if etype else entity
        parts.append(label)
    return " → ".join(parts)


def generate_counterfactual_hint(
    root_causes: list[dict[str, Any]],
    top_effect: str | None,
) -> str | None:
    """Produce the highest-value intervention suggestion based on the top root cause."""
    if not root_causes or not top_effect:
        return None
    root = root_causes[0]
    domains = root.get("domains", [])
    domain_str = f" [{', '.join(domains)}]" if domains else ""
    return (
        f"Addressing {root['entity']}{domain_str} could mitigate "
        f"the impact on {top_effect} "
        f"(cause_score={root['cause_score']:.2f})"
    )


class CausalReasoningAgent(BaseAgent):
    """Phase 12: Trace cause-effect chains across silos using the world model graph.

    Inputs (via context["memory"]["enrichment"]):
      - world_nodes:        list[{entity, entity_type, domains, silo_count,
                                  state_version, node_confidence}]  — Phase 4
      - entity_edges:       list[{from_entity, to_entity,
                                  relationship_type, weight}]        — Phase 4
      - propagated_signals: list[{content, source_silo, target_domains,
                                  relevance_score, entities, ...}]   — Phase 9
      - push_candidates:    list[{recipient_team, topic, trigger_type,
                                  urgency_score, delivery_hint}]     — Phase 11

    Outputs:
      - causal_chains:       list[{effect_entity, causal_path, hop_count,
                                   chain_strength, silo_span, narrative}]
      - root_causes:         list[{entity, entity_type, cause_score, domains}]
      - counterfactual_hint: str | None
      - top_effect:          str | None
      - confidence:          float
      - rationale:           "heuristic" | "llm"
    """

    name = "CausalReasoningAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["ProactiveMemoryPushAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("causal_chains", []), list):
            raise ValueError("causal_chains must be a list")
        if not isinstance(outputs.get("root_causes", []), list):
            raise ValueError("root_causes must be a list")
        for chain in outputs.get("causal_chains", []):
            if not isinstance(chain, dict):
                raise ValueError("each causal_chains item must be a dict")
            if "effect_entity" not in chain:
                raise ValueError("each causal_chains item must have effect_entity")
            if "causal_path" not in chain:
                raise ValueError("each causal_chains item must have causal_path")

    def _heuristic(
        self,
        *,
        world_nodes: list[dict[str, Any]],
        entity_edges: list[dict[str, Any]],
        propagated_signals: list[dict[str, Any]],
        push_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        node_lookup: dict[str, dict[str, Any]] = {n["entity"]: n for n in world_nodes}
        adjacency = build_adjacency(entity_edges)

        # Collect entity names that arrived via cross-silo propagation
        propagated_signal_entities: set[str] = set()
        for sig in propagated_signals:
            for entity in sig.get("entities", []):
                propagated_signal_entities.add(entity)
            # Best-effort: match world node names against signal content keywords
            content_lower = (sig.get("content") or "").lower()
            for node in world_nodes:
                if node["entity"].lower() in content_lower:
                    propagated_signal_entities.add(node["entity"])

        effect_candidates = identify_effect_candidates(
            world_nodes=world_nodes,
            push_candidates=push_candidates,
        )

        all_chains: list[dict[str, Any]] = []
        top_effect: str | None = effect_candidates[0]["entity"] if effect_candidates else None

        for eff_cand in effect_candidates[:5]:
            effect_entity = eff_cand["entity"]
            chains = trace_causal_chain(
                effect_entity=effect_entity,
                adjacency=adjacency,
                node_lookup=node_lookup,
                propagated_signal_entities=propagated_signal_entities,
            )
            for chain in chains:
                narrative = build_chain_narrative(
                    effect_entity=effect_entity,
                    chain=chain,
                    node_lookup=node_lookup,
                )
                all_chains.append(
                    {
                        "effect_entity": effect_entity,
                        "causal_path": chain["causal_path"],
                        "hop_count": chain["hop_count"],
                        "chain_strength": chain["chain_strength"],
                        "silo_span": chain["silo_span"],
                        "narrative": narrative,
                    }
                )

        all_chains.sort(key=lambda x: -x["chain_strength"])
        all_chains = all_chains[:_MAX_CHAINS]

        root_causes = extract_root_causes(all_chains, node_lookup)
        counterfactual_hint = generate_counterfactual_hint(root_causes, top_effect)

        n_chains = len(all_chains)
        cross_silo_chains = sum(1 for c in all_chains if c["silo_span"] >= 2)
        if n_chains == 0:
            confidence = 0.3
        else:
            confidence = min(0.9, round(0.35 + 0.45 * (cross_silo_chains / max(n_chains, 1)), 3))

        return {
            "causal_chains": all_chains,
            "root_causes": root_causes,
            "counterfactual_hint": counterfactual_hint,
            "top_effect": top_effect,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        content = (context.get("memory") or {}).get("content", "")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        world_nodes: list[dict[str, Any]] = enrichment.get("world_nodes", [])
        entity_edges: list[dict[str, Any]] = enrichment.get("entity_edges", [])
        propagated_signals: list[dict[str, Any]] = enrichment.get("propagated_signals", [])
        push_candidates: list[dict[str, Any]] = enrichment.get("push_candidates", [])

        strategy = getattr(settings, "CAUSAL_REASONING_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                world_nodes=world_nodes,
                entity_edges=entity_edges,
                propagated_signals=propagated_signals,
                push_candidates=push_candidates,
            )
        else:
            prompt = (
                "You are an enterprise causal reasoning engine. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Given a world model graph and cross-silo signals, identify cause-effect "
                "chains that explain why high-urgency entities are in their current state.\n\n"
                f"WORLD NODES: {world_nodes[:20]}\n"
                f"ENTITY EDGES: {entity_edges[:20]}\n"
                f"PROPAGATED SIGNALS: {propagated_signals[:10]}\n"
                f"PUSH CANDIDATES: {push_candidates[:10]}\n\n"
                "Return JSON with keys:\n"
                "- causal_chains: list of {effect_entity, causal_path (list of entity names), "
                "hop_count, chain_strength (0..1), silo_span (int), narrative (str)}\n"
                "- root_causes: list of {entity, entity_type, cause_score (0..1), domains}\n"
                "- counterfactual_hint: str or null\n"
                "- top_effect: str or null\n"
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
                and isinstance(resp.get("causal_chains"), list)
                and isinstance(resp.get("root_causes"), list)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    world_nodes=world_nodes,
                    entity_edges=entity_edges,
                    propagated_signals=propagated_signals,
                    push_candidates=push_candidates,
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
