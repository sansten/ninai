"""Goal-Memory Binding Service — Gap A.

The cognitive loop already links evidence memories (from evaluation reports)
to goals, but memories *created* during goal execution — e.g. findings written
by executor tool calls — are never bound to the goal that caused them.

This service closes that gap:
  new memory written during session → bind to active goal with link_type="created_during"

Design rules:
- Pure service: wraps GoalService.link_memory() with fire-and-forget semantics.
- Operates without raising: returns a BindingResult so callers can log failures
  without disrupting the main execution path.
- Handles the common case where goal_id or memory_id is missing (returns no-op).
- Uses link_type="created_during" to distinguish from "evidence" links.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BindingResult:
    memory_id: str
    goal_id: str | None
    bound: bool
    error: str | None = None


class GoalMemoryBindingService:
    """Binds a newly-created memory to the active goal for the session.

    Typical usage (inside executor after memory.write tool call succeeds):
        binding_svc = GoalMemoryBindingService(goal_service=goal_svc)
        result = await binding_svc.bind(
            org_id=org_id,
            goal_id=session_goal_id,
            memory_id=new_memory_id,
            actor_user_id=user_id,
        )
    """

    def __init__(self, *, goal_service: Any) -> None:
        """
        Parameters
        ----------
        goal_service:  A GoalService instance with link_memory().
                       Type is Any to avoid coupling to DB session at import time.
        """
        self._svc = goal_service

    async def bind(
        self,
        *,
        org_id: str,
        goal_id: str | None,
        memory_id: str | None,
        actor_user_id: str | None = None,
        confidence: float = 0.6,
    ) -> BindingResult:
        """Bind a memory to a goal with link_type='created_during'.

        Returns a no-op BindingResult if goal_id or memory_id is absent.
        """
        if not goal_id or not memory_id:
            return BindingResult(
                memory_id=str(memory_id or ""),
                goal_id=goal_id,
                bound=False,
                error="missing goal_id or memory_id",
            )
        try:
            await self._svc.link_memory(
                org_id=org_id,
                actor_user_id=actor_user_id,
                goal_id=goal_id,
                memory_id=memory_id,
                link_type="created_during",
                confidence=confidence,
                node_id=None,
                linked_by="agent",
            )
            return BindingResult(memory_id=memory_id, goal_id=goal_id, bound=True)
        except Exception as exc:
            return BindingResult(
                memory_id=memory_id,
                goal_id=goal_id,
                bound=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def bind_batch(
        self,
        *,
        org_id: str,
        goal_id: str | None,
        memory_ids: list[str],
        actor_user_id: str | None = None,
        confidence: float = 0.6,
    ) -> list[BindingResult]:
        """Bind multiple memory IDs to a goal in sequence.

        Returns one BindingResult per memory_id.  Failures are recorded per-item
        without stopping the batch.
        """
        results: list[BindingResult] = []
        for mid in memory_ids:
            result = await self.bind(
                org_id=org_id,
                goal_id=goal_id,
                memory_id=mid,
                actor_user_id=actor_user_id,
                confidence=confidence,
            )
            results.append(result)
        return results
