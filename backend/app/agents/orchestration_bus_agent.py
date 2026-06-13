"""Cross-Agent Orchestration Bus — Phase 29.

Wires enrichment agents into a dependency-aware execution graph so they run in
the correct topological order instead of being called ad hoc per memory.  The
bus resolves agent dependencies at call time via each agent's ``dependencies()``
method, performs a Kahn topological sort, executes agents sequentially, and
propagates each agent's outputs into the shared enrichment context so downstream
agents can consume them.

Typical enrichment order enforced by the graph:
  MetadataExtraction → SemanticNormalization → EntityResolution
  → ContextAmplifier → SiloPropagation → OrgAttention
  → ProactiveMemoryPush → CausalReasoning → ConflictDetection
  → AdaptiveConflictResolution → CredibilityAgent → ...

Outputs:
  - execution_plan:   list[dict]  — per-agent status, latency_ms, outputs
  - agent_order:      list[str]   — agent names in execution order
  - skipped_agents:   list[str]   — agents skipped (failed dep / should_run=False)
  - bus_latency_ms:   float       — total wall-clock time for the bus run
"""

from __future__ import annotations

import json
import re
from collections import deque
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.llm_breaker import create_llm_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Default enrichment pipeline — ordered list of agent names the bus runs when
# the caller does not supply an explicit list.  Topo-sort will re-order them
# based on declared dependencies, so the list order here is advisory only.
# ---------------------------------------------------------------------------

_DEFAULT_AGENTS: list[str] = [
    "MetadataExtractionAgent",
    "SemanticNormalizationAgent",
    "EntityResolutionAgent",
    "ContextAmplifierAgent",
    "SiloPropagationAgent",
    "OrgAttentionAgent",
    "ProactiveMemoryPushAgent",
    "WorldModelAgent",
    "PredictiveMonitorAgent",
    "CausalReasoningAgent",
    "ConflictDetectionAgent",
    "AdaptiveConflictResolutionAgent",
    "MemoryDecayAgent",
    "MemoryConsolidationAgent",
    "TemporalReasoningAgent",
    "EpisodicGroupingAgent",
    "CredibilityAgent",
    "PlaybookAgent",
    "GoalDecompositionAgent",
    "UncertaintyReportingAgent",
    "NarrativeSynthesisAgent",
    "FeedbackIntegrationAgent",
    "AnomalyDetectionAgent",
]

# Agents the LLM may select when filtering the pipeline for a given memory.
_SELECTABLE_AGENTS = set(_DEFAULT_AGENTS)

_LLM_PROMPT = """You are an orchestration planner for an enterprise memory system.

Given the memory content and existing enrichment below, select the minimal set of
enrichment agents that are relevant for this memory.  Only choose from the
provided candidate list.  Return a JSON object with a single key "agents" whose
value is a list of agent names.  Do not add explanation.

Memory content: {content}
Existing enrichment keys: {enrichment_keys}
Candidate agents: {candidates}

Response (JSON only):"""

_MAX_LLM_CHARS = 800


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------

def _topo_sort(agents: dict[str, BaseAgent]) -> list[BaseAgent]:
    """Return agents in dependency-safe execution order.

    Deps that are not in the ``agents`` dict are treated as already satisfied
    (they ran in a previous bus invocation or are not part of this pipeline).

    Raises ``ValueError`` if a dependency cycle is detected.
    """
    in_degree: dict[str, int] = {name: 0 for name in agents}
    adj: dict[str, list[str]] = {name: [] for name in agents}

    for name, agent in agents.items():
        for dep in agent.dependencies():
            if dep not in agents:
                continue  # satisfied externally
            adj[dep].append(name)
            in_degree[name] += 1

    queue: deque[str] = deque(
        name for name, deg in in_degree.items() if deg == 0
    )
    ordered: list[BaseAgent] = []

    while queue:
        name = queue.popleft()
        ordered.append(agents[name])
        for neighbor in adj[name]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) != len(agents):
        raise ValueError(
            "orchestration bus: dependency cycle detected among agents: "
            + str(list(agents.keys()))
        )

    return ordered


# ---------------------------------------------------------------------------
# Heuristic agent selection
# ---------------------------------------------------------------------------

def _heuristic_agent_names(content: str, enrichment: dict[str, Any]) -> list[str]:
    """Return a filtered subset of the default pipeline based on simple signals."""
    content_lower = (content or "").lower()
    selected: list[str] = []

    # Always include the foundational pipeline
    base = [
        "MetadataExtractionAgent",
        "SemanticNormalizationAgent",
        "EntityResolutionAgent",
        "ContextAmplifierAgent",
    ]
    selected.extend(base)

    # Anomaly / conflict signals → conflict pipeline
    if any(kw in content_lower for kw in ("conflict", "dispute", "contradiction", "mismatch")):
        selected += [
            "SiloPropagationAgent",
            "CausalReasoningAgent",
            "ConflictDetectionAgent",
            "AdaptiveConflictResolutionAgent",
            "CredibilityAgent",
        ]

    # Goal / task signals → goal + playbook pipeline
    if any(kw in content_lower for kw in ("goal", "task", "objective", "plan", "milestone")):
        selected += [
            "GoalDecompositionAgent",
            "PlaybookAgent",
            "EpisodicGroupingAgent",
        ]

    # Temporal signals
    if any(kw in content_lower for kw in ("deadline", "schedule", "when", "date", "history")):
        selected += ["TemporalReasoningAgent", "EpisodicGroupingAgent", "MemoryDecayAgent"]

    # Org / cross-silo signals
    if any(kw in content_lower for kw in ("team", "department", "org", "silo", "cross")):
        selected += ["SiloPropagationAgent", "OrgAttentionAgent", "ProactiveMemoryPushAgent"]

    # Always close with synthesis / reporting at the end
    selected += [
        "UncertaintyReportingAgent",
        "NarrativeSynthesisAgent",
        "AnomalyDetectionAgent",
    ]

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for name in selected:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


# ---------------------------------------------------------------------------
# LLM agent selection
# ---------------------------------------------------------------------------

def _parse_llm_agent_names(raw: str, candidates: set[str]) -> list[str]:
    """Extract agent list from raw LLM JSON response."""
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
        names = data.get("agents") or []
        return [n for n in names if isinstance(n, str) and n in candidates]
    except Exception:
        return []


async def _llm_agent_names(
    content: str,
    enrichment: dict[str, Any],
    client: Any,
    model: str,
) -> list[str]:
    prompt = _LLM_PROMPT.format(
        content=(content or "")[:_MAX_LLM_CHARS],
        enrichment_keys=list(enrichment.keys())[:20],
        candidates=sorted(_SELECTABLE_AGENTS),
    )
    try:
        response = await client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        raw = (response.get("message") or {}).get("content") or ""
        names = _parse_llm_agent_names(raw, _SELECTABLE_AGENTS)
        return names if names else list(_DEFAULT_AGENTS)
    except Exception:
        return list(_DEFAULT_AGENTS)


# ---------------------------------------------------------------------------
# OrchestrationBusAgent
# ---------------------------------------------------------------------------

class OrchestrationBusAgent(BaseAgent):
    """Dependency-aware enrichment orchestration bus.

    Resolves agent dependencies, topologically sorts the pipeline, and executes
    agents in order while propagating outputs as enrichment signals.
    """

    name = "OrchestrationBusAgent"
    version = "1.0"

    def dependencies(self) -> list[str]:
        return []

    def should_run(self, memory_id: str, context: AgentContext) -> bool:
        return bool(memory_id)

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        from app.agents.registry import get_agent  # local import to avoid circularity

        t0 = datetime.now(timezone.utc)

        config: dict[str, Any] = dict(context.get("config") or {})
        memory: dict[str, Any] = dict(context.get("memory") or {})
        content: str = str(memory.get("content") or "")
        enrichment: dict[str, Any] = dict(memory.get("enrichment") or {})

        # Determine which agents to run -------------------------------------------
        explicit_names: list[str] | None = config.get("agent_names") or None
        strategy = str(config.get("agent_strategy") or getattr(settings, "AGENT_STRATEGY", "heuristic"))

        if explicit_names:
            agent_names = list(explicit_names)
        elif strategy == "llm":
            try:
                model = str(getattr(settings, "VLLM_MODEL", "qwen2.5:7b"))
                client = create_llm_client()
                agent_names = await _llm_agent_names(content, enrichment, client, model)
            except Exception:
                agent_names = list(_DEFAULT_AGENTS)
        else:
            agent_names = _heuristic_agent_names(content, enrichment)

        # Build agent instances ---------------------------------------------------
        agents: dict[str, BaseAgent] = {}
        for name in agent_names:
            agent = get_agent(name.lower())
            if agent is not None:
                agents[agent.name] = agent

        # Topological sort --------------------------------------------------------
        try:
            ordered = _topo_sort(agents)
        except ValueError as exc:
            t_err = datetime.now(timezone.utc)
            return AgentResult(
                agent_name=self.name,
                agent_version=self.version,
                memory_id=memory_id,
                status="failed",
                confidence=0.0,
                errors=[str(exc)],
                outputs={
                    "execution_plan": [],
                    "agent_order": [],
                    "skipped_agents": list(agents.keys()),
                    "bus_latency_ms": (t_err - t0).total_seconds() * 1000,
                },
                started_at=t0,
                finished_at=t_err,
            )

        # Execute in order --------------------------------------------------------
        completed: dict[str, str] = {}  # agent_name -> status
        skipped_agents: list[str] = []
        execution_plan: list[dict[str, Any]] = []

        # Work on a mutable copy of context so propagation doesn't mutate caller's ctx
        run_memory: dict[str, Any] = dict(memory)
        run_enrichment: dict[str, Any] = dict(enrichment)
        run_memory["enrichment"] = run_enrichment

        run_context: AgentContext = dict(context)  # type: ignore[assignment]
        run_context["memory"] = run_memory  # type: ignore[index]

        for agent in ordered:
            # Check that all in-scope dependencies succeeded
            deps_failed = [
                dep
                for dep in agent.dependencies()
                if dep in agents and completed.get(dep) != "success"
            ]
            if deps_failed:
                skipped_agents.append(agent.name)
                execution_plan.append(
                    {
                        "agent_name": agent.name,
                        "status": "skipped",
                        "reason": f"dep_failed:{','.join(deps_failed)}",
                        "latency_ms": 0.0,
                        "outputs": {},
                    }
                )
                completed[agent.name] = "skipped"
                continue

            if not agent.should_run(memory_id, run_context):
                skipped_agents.append(agent.name)
                execution_plan.append(
                    {
                        "agent_name": agent.name,
                        "status": "skipped",
                        "reason": "should_run=False",
                        "latency_ms": 0.0,
                        "outputs": {},
                    }
                )
                completed[agent.name] = "skipped"
                continue

            t_a = datetime.now(timezone.utc)
            try:
                result = await agent.run(memory_id, run_context)
                agent_status = result.status
                agent_outputs = dict(result.outputs or {})
            except Exception as exc:
                agent_status = "failed"
                agent_outputs = {}
                execution_plan.append(
                    {
                        "agent_name": agent.name,
                        "status": "failed",
                        "reason": str(exc),
                        "latency_ms": (datetime.now(timezone.utc) - t_a).total_seconds() * 1000,
                        "outputs": {},
                    }
                )
                completed[agent.name] = "failed"
                continue

            t_b = datetime.now(timezone.utc)
            latency = (t_b - t_a).total_seconds() * 1000

            # Propagate outputs into shared enrichment for downstream agents
            if agent_status == "success" and agent_outputs:
                run_enrichment.update(agent_outputs)

            execution_plan.append(
                {
                    "agent_name": agent.name,
                    "status": agent_status,
                    "latency_ms": round(latency, 2),
                    "outputs": agent_outputs,
                }
            )
            completed[agent.name] = agent_status

        t1 = datetime.now(timezone.utc)
        bus_latency = (t1 - t0).total_seconds() * 1000

        return AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=1.0,
            outputs={
                "execution_plan": execution_plan,
                "agent_order": [p["agent_name"] for p in execution_plan],
                "skipped_agents": skipped_agents,
                "bus_latency_ms": round(bus_latency, 2),
            },
            started_at=t0,
            finished_at=t1,
        )

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("execution_plan"), list):
            raise ValueError("execution_plan must be a list")
        if not isinstance(outputs.get("agent_order"), list):
            raise ValueError("agent_order must be a list")
        if not isinstance(outputs.get("skipped_agents"), list):
            raise ValueError("skipped_agents must be a list")
        if not isinstance(outputs.get("bus_latency_ms"), (int, float)):
            raise ValueError("bus_latency_ms must be numeric")
