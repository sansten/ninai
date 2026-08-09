"""P2P Agent Coordination Service — Phase 93.

Decentralised agent coordination without a central sequencer.

Problem: OrchestrationBus is a central hub — if it's the bottleneck or single
point of failure, all agents queue behind it. True P2P coordination lets agents
discover each other and negotiate task ownership via an open auction protocol.

Architecture:
  - Agents register capability manifests in a shared store (Redis or in-memory).
  - Tasks are posted to a priority-ordered queue with a type and payload.
  - Agents self-select tasks that match their capabilities and post bids.
  - The highest-confidence bidder atomically claims the task.
  - Completion is reported; the coordination layer tracks provenance.

This does NOT require a central sequencer — any agent can post tasks, any
matching agent can bid, and the claim is resolved by the store's atomic ops.

Redis key scheme (when redis_client is provided):
  p2p:agents:{agent_id}          → JSON manifest (with TTL)
  p2p:tasks:queue                → sorted set, score = priority
  p2p:task:{task_id}             → JSON task details
  p2p:bids:{task_id}             → sorted set, score = confidence
  p2p:owner:{task_id}            → string (atomic SETNX)
  p2p:completed:{task_id}        → JSON completion record

Usage::

    svc = P2PCoordinationService()  # in-memory mode

    svc.register_agent("agent-research", capabilities=["research", "synthesis"])
    svc.register_agent("agent-action",   capabilities=["email", "jira"])

    task_id = svc.post_task(
        task_type="research",
        payload={"query": "Q3 performance summary"},
        priority=0.8,
    )
    svc.bid_for_task(task_id, agent_id="agent-research", confidence=0.90)
    claimed = svc.claim_task(task_id, agent_id="agent-research")
    svc.resolve_task(task_id, agent_id="agent-research", outcome="success")
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentCapabilityManifest:
    agent_id: str
    capabilities: list[str]
    registered_at: float = field(default_factory=time.time)
    ttl_seconds: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_alive(self) -> bool:
        return (time.time() - self.registered_at) < self.ttl_seconds

    def can_handle(self, task_type: str) -> bool:
        task_lower = task_type.lower()
        return any(cap.lower() in task_lower or task_lower in cap.lower()
                   for cap in self.capabilities)


@dataclass
class Task:
    task_id: str
    task_type: str
    payload: dict[str, Any]
    priority: float        # 0–1, higher = more urgent
    posted_at: float
    status: str            # pending | claimed | completed | failed
    owner: str | None = None
    bids: dict[str, float] = field(default_factory=dict)  # agent_id → confidence
    completion: dict[str, Any] = field(default_factory=dict)


@dataclass
class BidResult:
    task_id: str
    agent_id: str
    confidence: float
    accepted: bool         # True = bid registered; False = task already claimed
    is_leading_bid: bool   # True = this bid currently leads


@dataclass
class ClaimResult:
    task_id: str
    agent_id: str
    claimed: bool          # True = atomic claim succeeded
    reason: str


class P2PCoordinationService:
    """In-memory P2P coordination with optional Redis persistence.

    The in-memory mode is suitable for single-pod deployments or tests.
    Pass a Redis client (sync redis.Redis) to enable cross-pod coordination.
    """

    def __init__(self, redis_client: Any = None) -> None:
        self._agents: dict[str, AgentCapabilityManifest] = {}
        self._tasks: dict[str, Task] = {}
        self._r = redis_client
        self._prefix = "p2p"

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        *,
        capabilities: list[str],
        ttl_seconds: int = 300,
        metadata: dict[str, Any] | None = None,
    ) -> AgentCapabilityManifest:
        """Register an agent's capabilities so it can be discovered by task posters."""
        manifest = AgentCapabilityManifest(
            agent_id=str(agent_id),
            capabilities=[str(c).lower() for c in capabilities],
            ttl_seconds=ttl_seconds,
            metadata=dict(metadata or {}),
        )
        self._agents[agent_id] = manifest

        if self._r is not None:
            import json
            data = {
                "agent_id": manifest.agent_id,
                "capabilities": manifest.capabilities,
                "registered_at": manifest.registered_at,
                "ttl_seconds": ttl_seconds,
                "metadata": manifest.metadata,
            }
            key = f"{self._prefix}:agents:{agent_id}"
            self._r.setex(key, ttl_seconds, json.dumps(data))

        return manifest

    def deregister_agent(self, agent_id: str) -> None:
        self._agents.pop(str(agent_id), None)
        if self._r is not None:
            self._r.delete(f"{self._prefix}:agents:{agent_id}")

    def discover_agents(self, capability: str | None = None) -> list[AgentCapabilityManifest]:
        """Return all live agents, optionally filtered by capability."""
        # Evict expired
        self._agents = {k: v for k, v in self._agents.items() if v.is_alive()}
        if capability is None:
            return list(self._agents.values())
        return [a for a in self._agents.values() if a.can_handle(capability)]

    # ------------------------------------------------------------------
    # Task posting
    # ------------------------------------------------------------------

    def post_task(
        self,
        *,
        task_type: str,
        payload: dict[str, Any],
        priority: float = 0.5,
        task_id: str | None = None,
    ) -> str:
        """Post a task to the coordination queue. Returns the task_id."""
        tid = str(task_id or uuid.uuid4())
        task = Task(
            task_id=tid,
            task_type=str(task_type),
            payload=dict(payload),
            priority=max(0.0, min(1.0, float(priority))),
            posted_at=time.time(),
            status="pending",
        )
        self._tasks[tid] = task

        if self._r is not None:
            import json
            self._r.zadd(f"{self._prefix}:tasks:queue", {tid: priority})
            self._r.set(f"{self._prefix}:task:{tid}", json.dumps({
                "task_id": tid, "task_type": task.task_type,
                "payload": task.payload, "priority": task.priority,
                "posted_at": task.posted_at, "status": task.status,
            }))

        return tid

    def get_pending_tasks(self, capability: str | None = None) -> list[Task]:
        """Return pending tasks sorted by priority (highest first)."""
        tasks = [t for t in self._tasks.values() if t.status == "pending"]
        if capability is not None:
            tasks = [t for t in tasks if capability.lower() in t.task_type.lower()
                     or t.task_type.lower() in capability.lower()]
        return sorted(tasks, key=lambda t: t.priority, reverse=True)

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(str(task_id))

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------

    def bid_for_task(
        self,
        task_id: str,
        *,
        agent_id: str,
        confidence: float,
    ) -> BidResult:
        """Submit a bid for a pending task.

        An agent bids with a confidence score (0–1) indicating how well it
        can handle this task type. The highest bidder earns the right to claim.
        """
        tid = str(task_id)
        task = self._tasks.get(tid)
        if task is None:
            return BidResult(task_id=tid, agent_id=agent_id,
                             confidence=confidence, accepted=False,
                             is_leading_bid=False)

        if task.status != "pending":
            return BidResult(task_id=tid, agent_id=agent_id,
                             confidence=confidence, accepted=False,
                             is_leading_bid=False)

        conf = max(0.0, min(1.0, float(confidence)))
        task.bids[str(agent_id)] = conf

        if self._r is not None:
            self._r.zadd(f"{self._prefix}:bids:{tid}", {str(agent_id): conf})

        # Is this the leading bid?
        max_conf = max(task.bids.values()) if task.bids else 0.0
        is_leading = conf >= max_conf

        return BidResult(
            task_id=tid,
            agent_id=str(agent_id),
            confidence=conf,
            accepted=True,
            is_leading_bid=is_leading,
        )

    def leading_bidder(self, task_id: str) -> tuple[str, float] | None:
        """Return (agent_id, confidence) of the current leading bidder, or None."""
        task = self._tasks.get(str(task_id))
        if not task or not task.bids:
            return None
        best = max(task.bids.items(), key=lambda kv: kv[1])
        return best

    # ------------------------------------------------------------------
    # Claiming (atomic)
    # ------------------------------------------------------------------

    def claim_task(self, task_id: str, *, agent_id: str) -> ClaimResult:
        """Atomically claim a task for the given agent.

        Returns ClaimResult.claimed=True only if the agent wins the claim.
        """
        tid = str(task_id)
        task = self._tasks.get(tid)
        if task is None:
            return ClaimResult(tid, agent_id, False, "task not found")
        if task.status != "pending":
            return ClaimResult(tid, agent_id, False,
                               f"task is {task.status!r}, not pending")

        # Redis atomic claim via SETNX
        if self._r is not None:
            import json
            key = f"{self._prefix}:owner:{tid}"
            set_result = self._r.setnx(key, str(agent_id))
            if not set_result:
                current_owner = self._r.get(key)
                return ClaimResult(tid, agent_id, False,
                                   f"already claimed by {current_owner!r}")
            task.status = "claimed"
            task.owner = str(agent_id)
            self._r.set(f"{self._prefix}:task:{tid}:status", "claimed")
            return ClaimResult(tid, agent_id, True, "claimed via Redis SETNX")

        # In-memory claim — check if already claimed (no true atomicity in-process)
        if task.owner is not None:
            return ClaimResult(tid, agent_id, False,
                               f"already claimed by {task.owner!r}")
        task.status = "claimed"
        task.owner = str(agent_id)
        return ClaimResult(tid, agent_id, True, "claimed in-memory")

    # ------------------------------------------------------------------
    # Task resolution
    # ------------------------------------------------------------------

    def resolve_task(
        self,
        task_id: str,
        *,
        agent_id: str,
        outcome: str,
        result_data: dict[str, Any] | None = None,
    ) -> Task:
        """Mark a claimed task as completed or failed."""
        tid = str(task_id)
        task = self._tasks.get(tid)
        if task is None:
            raise KeyError(f"Unknown task_id: {tid}")
        if task.owner != str(agent_id):
            raise ValueError(f"Agent {agent_id!r} does not own task {tid!r} (owner={task.owner!r})")

        task.status = "completed" if outcome in ("success", "completed", "done") else "failed"
        task.completion = {
            "outcome": outcome,
            "completed_at": time.time(),
            "result_data": dict(result_data or {}),
        }

        if self._r is not None:
            import json
            self._r.set(f"{self._prefix}:completed:{tid}", json.dumps(task.completion))

        return task

    # ------------------------------------------------------------------
    # Coordination stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        all_tasks = list(self._tasks.values())
        return {
            "live_agents": len(self.discover_agents()),
            "pending_tasks": sum(1 for t in all_tasks if t.status == "pending"),
            "claimed_tasks": sum(1 for t in all_tasks if t.status == "claimed"),
            "completed_tasks": sum(1 for t in all_tasks if t.status == "completed"),
            "failed_tasks": sum(1 for t in all_tasks if t.status == "failed"),
            "total_tasks": len(all_tasks),
        }
