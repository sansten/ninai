"""Counterfactual memory simulation agent (Phase 51)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def intervention_magnitude(intervention: dict[str, Any]) -> float:
    """Compute intervention magnitude from an intervention payload."""
    source = intervention.get("from")
    target = intervention.get("to")

    if isinstance(source, bool) or isinstance(target, bool):
        return 1.0

    if isinstance(source, (int, float)) and isinstance(target, (int, float)):
        denominator = max(abs(float(target)), 1.0)
        return abs(float(target) - float(source)) / denominator

    return 1.0 if str(source) != str(target) else 0.0


def _node_from_edge(edge: dict[str, Any], key_options: list[str]) -> str:
    for key in key_options:
        value = edge.get(key)
        if value is not None:
            return str(value)
    return ""


def build_adjacency(edges: list[dict[str, Any]]) -> dict[str, list[tuple[str, float]]]:
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for edge in edges:
        source = _node_from_edge(edge, ["from", "source", "from_memory_id", "src"])
        target = _node_from_edge(edge, ["to", "target", "to_memory_id", "dst"])
        if not source or not target:
            continue
        weight = _safe_float(edge.get("weight", edge.get("edge_weight", 0.5)), 0.5)
        adjacency.setdefault(source, []).append((target, weight))
    return adjacency


def bfs_affected_nodes(
    *,
    start_node: str,
    adjacency: dict[str, list[tuple[str, float]]],
    max_depth: int = 3,
) -> tuple[list[str], list[float]]:
    """Traverse reachable nodes with BFS up to max_depth and collect edge weights."""
    if not start_node:
        return [], []

    queue: deque[tuple[str, int]] = deque([(start_node, 0)])
    visited = {start_node}
    affected_nodes: list[str] = []
    traversed_weights: list[float] = []

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue

        for nxt, weight in adjacency.get(node, []):
            traversed_weights.append(weight)
            if nxt in visited:
                continue
            visited.add(nxt)
            affected_nodes.append(nxt)
            queue.append((nxt, depth + 1))

    return affected_nodes, traversed_weights


class CounterfactualMemoryAgent(BaseAgent):
    """Phase 51: simulate what-if effects from an intervention across causal edges."""

    name = "CounterfactualMemoryAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["CausalReasoningAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return

        outputs = result.outputs or {}
        if not isinstance(outputs.get("counterfactual_outcome"), str):
            raise ValueError("counterfactual_outcome must be a string")
        if not isinstance(outputs.get("affected_nodes"), list):
            raise ValueError("affected_nodes must be a list")
        conf = outputs.get("confidence")
        if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
            raise ValueError("confidence must be a float between 0 and 1")
        if not isinstance(outputs.get("counterfactual_delta"), dict):
            raise ValueError("counterfactual_delta must be a dict")
        if not isinstance(outputs.get("assumptions"), list):
            raise ValueError("assumptions must be a list")

    def _heuristic(
        self,
        *,
        memory_id: str,
        intervention: dict[str, Any],
        causal_graph: list[dict[str, Any]],
    ) -> dict[str, Any]:
        magnitude = intervention_magnitude(intervention)
        adjacency = build_adjacency(causal_graph)
        affected_nodes, traversed_weights = bfs_affected_nodes(
            start_node=memory_id,
            adjacency=adjacency,
            max_depth=3,
        )

        probability_change = round(sum(w * magnitude for w in traversed_weights), 4)

        field = str(intervention.get("field") or "field")
        from_value = intervention.get("from")
        to_value = intervention.get("to")
        severity_shift = f"{from_value}->{to_value}"

        outcome = (
            f"Had {field} been {to_value} instead of {from_value}, "
            f"{len(affected_nodes)} downstream effects would likely have occurred including "
            f"{affected_nodes[:2]}."
        )

        confidence = max(0.0, min(0.9, 0.5 + 0.1 * len(affected_nodes)))

        return {
            "counterfactual_outcome": outcome,
            "affected_nodes": affected_nodes,
            "confidence": round(confidence, 4),
            "counterfactual_delta": {
                "probability_change": probability_change,
                "severity_shift": severity_shift,
            },
            "assumptions": [
                "assumes no concurrent incidents",
                "assumes causal graph edge weights are calibrated",
            ],
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        intervention = dict(enrichment.get("intervention") or {})
        causal_graph = list(enrichment.get("causal_graph") or [])
        related_memories = list(enrichment.get("related_memories") or [])

        strategy = getattr(settings, "COUNTERFACTUAL_MEMORY_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]
        if strategy == "heuristic":
            outputs = self._heuristic(
                memory_id=memory_id,
                intervention=intervention,
                causal_graph=causal_graph,
            )
        else:
            prompt = (
                "You are a counterfactual simulation engine for an enterprise Cognitive OS. "
                "Output JSON only.\n\n"
                f"MEMORY_ID: {memory_id}\n"
                f"INTERVENTION: {intervention}\n"
                f"CAUSAL_GRAPH: {causal_graph[:60]}\n"
                f"RELATED_MEMORIES: {related_memories[:20]}\n\n"
                "Return JSON with keys:\n"
                "- counterfactual_outcome: str\n"
                "- affected_nodes: list[str]\n"
                "- confidence: float 0..1\n"
                "- counterfactual_delta: dict with probability_change and severity_shift\n"
                "- assumptions: list[str]"
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
                and isinstance(resp.get("counterfactual_outcome"), str)
                and isinstance(resp.get("affected_nodes"), list)
                and isinstance(resp.get("counterfactual_delta"), dict)
                and isinstance(resp.get("assumptions"), list)
            ):
                outputs = {
                    **resp,
                    "rationale": "llm",
                }
            else:
                outputs = self._heuristic(
                    memory_id=memory_id,
                    intervention=intervention,
                    causal_graph=causal_graph,
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