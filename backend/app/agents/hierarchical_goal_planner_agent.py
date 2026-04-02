"""Hierarchical goal planner agent (Phase 65)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


_SPLIT_RE = re.compile(r"\b(?:and|then|after|also|as well as)\b", re.IGNORECASE)


def _effort_for_depth(depth: int) -> str:
    if depth <= 0:
        return "large"
    if depth == 1:
        return "medium"
    if depth == 2:
        return "small"
    return "trivial"


def _split_subgoals(root_goal: str) -> list[str]:
    parts = [p.strip(" .") for p in _SPLIT_RE.split(root_goal or "") if p.strip(" .")]
    return parts


def _leaf_tasks(nodes: list[dict[str, Any]]) -> list[str]:
    parent_indices = {n["parent_index"] for n in nodes if n.get("parent_index") is not None}
    out: list[str] = []
    for idx, node in enumerate(nodes):
        if idx not in parent_indices:
            out.append(str(node.get("title") or ""))
    return out


def _critical_path(nodes: list[dict[str, Any]]) -> list[str]:
    if not nodes:
        return []
    max_depth = max(int(n.get("depth", 0)) for n in nodes)
    candidates = [i for i, n in enumerate(nodes) if int(n.get("depth", 0)) == max_depth]
    if not candidates:
        return [str(nodes[0].get("title") or "")]

    leaf_index = candidates[0]
    path: list[str] = []
    current = leaf_index
    while current is not None:
        node = nodes[current]
        path.append(str(node.get("title") or ""))
        current = node.get("parent_index")
    path.reverse()
    return path


def run_heuristic(
    *,
    root_goal: str,
    depth_limit: int = 3,
    domain_context: str = "",
) -> dict[str, Any]:
    goal = str(root_goal or "").strip() or "Untitled goal"
    limit = max(0, int(depth_limit if depth_limit is not None else 3))

    nodes: list[dict[str, Any]] = [
        {
            "title": goal,
            "depth": 0,
            "parent_index": None,
            "estimated_effort": _effort_for_depth(0),
            "status": "pending",
        }
    ]

    if limit >= 1:
        subgoals = _split_subgoals(goal)
        if len(subgoals) < 2:
            subgoals = [
                f"Gather information for: {goal}",
                f"Execute plan for: {goal}",
            ]
        for sg in subgoals[:6]:
            if domain_context:
                title = f"{sg} ({domain_context})"
            else:
                title = sg
            nodes.append(
                {
                    "title": title,
                    "depth": 1,
                    "parent_index": 0,
                    "estimated_effort": _effort_for_depth(1),
                    "status": "pending",
                }
            )

    if limit >= 2:
        depth1 = [(i, n) for i, n in enumerate(nodes) if n["depth"] == 1]
        for parent_index, node in depth1:
            sub_goal_title = str(node["title"])
            tasks = [
                f"Verify preconditions for: {sub_goal_title}",
                f"Perform: {sub_goal_title}",
                f"Validate outcome of: {sub_goal_title}",
            ]
            for t in tasks:
                nodes.append(
                    {
                        "title": t,
                        "depth": 2,
                        "parent_index": parent_index,
                        "estimated_effort": _effort_for_depth(2),
                        "status": "pending",
                    }
                )

    if limit >= 3:
        depth2 = [(i, n) for i, n in enumerate(nodes) if n["depth"] == 2]
        for parent_index, node in depth2:
            task_title = str(node["title"])
            actions = [
                f"Start: {task_title}",
                f"Complete: {task_title}",
            ]
            for a in actions:
                nodes.append(
                    {
                        "title": a,
                        "depth": 3,
                        "parent_index": parent_index,
                        "estimated_effort": _effort_for_depth(3),
                        "status": "pending",
                    }
                )

    max_depth_reached = max(int(n["depth"]) for n in nodes) if nodes else 0

    return {
        "hierarchy": nodes,
        "total_nodes": len(nodes),
        "max_depth_reached": max_depth_reached,
        "critical_path": _critical_path(nodes),
        "leaf_tasks": _leaf_tasks(nodes),
        "confidence": 0.7,
        "rationale": "heuristic",
    }


class HierarchicalGoalPlannerAgent(BaseAgent):
    name = "HierarchicalGoalPlannerAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("hierarchy"), list):
            raise ValueError("hierarchy must be a list")
        if not isinstance(outputs.get("total_nodes"), int):
            raise ValueError("total_nodes must be int")
        if not isinstance(outputs.get("max_depth_reached"), int):
            raise ValueError("max_depth_reached must be int")
        if not isinstance(outputs.get("critical_path"), list):
            raise ValueError("critical_path must be a list")
        if not isinstance(outputs.get("leaf_tasks"), list):
            raise ValueError("leaf_tasks must be a list")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        root_goal = str(enrichment.get("root_goal") or "")
        depth_limit = int(enrichment.get("depth_limit") or 3)
        domain_context = str(enrichment.get("domain_context") or "")

        strategy = getattr(settings, "HIERARCHICAL_GOAL_PLANNER_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(root_goal=root_goal, depth_limit=depth_limit, domain_context=domain_context)
        else:
            prompt = (
                "You decompose goals into hierarchical plans. Output JSON only.\n\n"
                f"ROOT_GOAL: {root_goal}\n"
                f"DEPTH_LIMIT: {depth_limit}\n"
                f"DOMAIN_CONTEXT: {domain_context}\n\n"
                "Return JSON with keys:\n"
                "- hierarchy: list[{title, depth, parent_index, estimated_effort, status}]\n"
                "- total_nodes: int\n"
                "- max_depth_reached: int\n"
                "- critical_path: list[str]\n"
                "- leaf_tasks: list[str]\n"
                "- confidence: float\n"
                "- rationale: str"
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
                and isinstance(resp.get("hierarchy"), list)
                and isinstance(resp.get("total_nodes"), int)
                and isinstance(resp.get("max_depth_reached"), int)
                and isinstance(resp.get("critical_path"), list)
                and isinstance(resp.get("leaf_tasks"), list)
            ):
                outputs = resp
            else:
                outputs = run_heuristic(root_goal=root_goal, depth_limit=depth_limit, domain_context=domain_context)

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
