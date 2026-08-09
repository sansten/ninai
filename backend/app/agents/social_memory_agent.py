"""Social memory and team dynamics agent (Phase 66)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.llm_breaker import create_llm_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def run_heuristic(*, memories: list[dict[str, Any]], org_users: list[str]) -> dict[str, Any]:
    mem_list = list(memories or [])
    users = [str(u) for u in (org_users or []) if str(u)]

    pair_counts: Counter[tuple[str, str]] = Counter()
    actor_total_interactions: Counter[str] = Counter()
    undirected_neighbors: defaultdict[str, set[str]] = defaultdict(set)

    for memory in mem_list:
        actor = str(memory.get("user_id") or "").strip()
        linked = memory.get("linked_user_ids") or []
        if isinstance(linked, str):
            linked = [linked]
        if not actor:
            continue

        for collaborator_raw in linked:
            collaborator = str(collaborator_raw or "").strip()
            if not collaborator or collaborator == actor:
                continue
            pair_counts[(actor, collaborator)] += 1
            actor_total_interactions[actor] += 1
            undirected_neighbors[actor].add(collaborator)
            undirected_neighbors[collaborator].add(actor)

    collaboration_edges: list[dict[str, Any]] = []
    for (actor, collaborator), count in pair_counts.items():
        denom = max(1, actor_total_interactions.get(actor, 0))
        strength = _clamp01(count / denom)
        collaboration_edges.append(
            {
                "actor": actor,
                "collaborator": collaborator,
                "interaction_count": int(count),
                "strength": round(strength, 4),
            }
        )

    collaboration_edges.sort(key=lambda e: (-e["interaction_count"], e["actor"], e["collaborator"]))

    all_users = set(users)
    for actor, collaborator in pair_counts:
        all_users.add(actor)
        all_users.add(collaborator)

    degrees: dict[str, int] = {u: len(undirected_neighbors.get(u, set())) for u in all_users}

    knowledge_silos = sorted([u for u in users if degrees.get(u, 0) == 0])

    most_connected = None
    if degrees:
        max_degree = max(degrees.values())
        if max_degree > 0:
            most_connected = sorted([u for u, d in degrees.items() if d == max_degree])[0]

    least_connected = None
    non_zero = {u: d for u, d in degrees.items() if d > 0}
    if non_zero:
        min_degree = min(non_zero.values())
        least_connected = sorted([u for u, d in non_zero.items() if d == min_degree])[0]

    n = len(users)
    max_possible_edges = (n * (n - 1)) / 2 if n >= 2 else 0
    undirected_edges = set(tuple(sorted((a, b))) for (a, b) in pair_counts.keys())
    actual_edge_count = len(undirected_edges)
    team_cohesion_score = 0.0 if max_possible_edges == 0 else _clamp01(actual_edge_count / max_possible_edges)

    return {
        "collaboration_edges": collaboration_edges,
        "knowledge_silos": knowledge_silos,
        "most_connected": most_connected,
        "least_connected": least_connected,
        "team_cohesion_score": round(team_cohesion_score, 4),
        "confidence": 0.8,
        "rationale": "heuristic",
    }


class SocialMemoryAgent(BaseAgent):
    name = "SocialMemoryAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("collaboration_edges"), list):
            raise ValueError("collaboration_edges must be a list")
        if not isinstance(outputs.get("knowledge_silos"), list):
            raise ValueError("knowledge_silos must be a list")

        most_connected = outputs.get("most_connected")
        if most_connected is not None and not isinstance(most_connected, str):
            raise ValueError("most_connected must be str or None")

        least_connected = outputs.get("least_connected")
        if least_connected is not None and not isinstance(least_connected, str):
            raise ValueError("least_connected must be str or None")

        cohesion = outputs.get("team_cohesion_score")
        if not isinstance(cohesion, (int, float)) or not (0.0 <= float(cohesion) <= 1.0):
            raise ValueError("team_cohesion_score must be float between 0 and 1")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        memories = list(enrichment.get("memories") or [])
        org_users = list(enrichment.get("org_users") or [])

        strategy = getattr(settings, "SOCIAL_MEMORY_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(memories=memories, org_users=org_users)
        else:
            prompt = (
                "You analyze team collaboration patterns from memory events. Output JSON only.\n\n"
                f"MEMORIES: {memories[:100]}\n"
                f"ORG_USERS: {org_users[:200]}\n\n"
                "Return JSON with keys:\n"
                "- collaboration_edges: list[{actor, collaborator, interaction_count, strength}]\n"
                "- knowledge_silos: list[str]\n"
                "- most_connected: str or null\n"
                "- least_connected: str or null\n"
                "- team_cohesion_score: float\n"
                "- confidence: float\n"
                "- rationale: str"
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
                and isinstance(resp.get("collaboration_edges"), list)
                and isinstance(resp.get("knowledge_silos"), list)
                and (resp.get("most_connected") is None or isinstance(resp.get("most_connected"), str))
                and (resp.get("least_connected") is None or isinstance(resp.get("least_connected"), str))
                and isinstance(resp.get("team_cohesion_score"), (int, float))
            ):
                outputs = resp
            else:
                outputs = run_heuristic(memories=memories, org_users=org_users)

        finished_at = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence") or 0.5),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=trace_id,
        )
        self.validate_outputs(result)
        return result
