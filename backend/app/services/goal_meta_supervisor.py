"""GoalGraph Meta Supervision Helper.

Determines when GoalGraph operations require Meta review before proceeding.

Fail-closed: If Meta supervision is required but cannot be performed, the operation is blocked.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.goal import Goal
from app.models.meta_agent import MetaAgentRun
from app.models.memory import MemoryMetadata
from app.services.meta_agent.meta_supervisor import MetaSupervisor


class GoalMetaSupervisor:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.meta = MetaSupervisor()

    async def requires_review_for_status_change(
        self,
        *,
        goal: Goal,
        old_status: str,
        new_status: str,
    ) -> bool:
        """Check if status transition requires Meta review.
        
        Triggers:
        - Low confidence goal (<0.60) changing status
        - Any transition to/from "active" state
        - Transition to "completed" or "abandoned"
        """
        if goal.confidence < 0.60:
            return True

        status_transitions = {
            ("proposed", "active"),
            ("active", "blocked"),
            ("active", "completed"),
            ("active", "abandoned"),
            ("blocked", "active"),
        }

        if (old_status, new_status) in status_transitions or new_status in {"completed", "abandoned"}:
            return True

        return False

    async def requires_review_for_memory_link(
        self,
        *,
        goal: Goal,
        memory: MemoryMetadata,
        link_type: str,
    ) -> bool:
        """Check if memory link requires Meta review.
        
        Triggers:
        - Cross-scope memory linking (memory classification higher than goal scope)
        - Sensitive memory (restricted/confidential) linked to team/personal goals
        - Low confidence goal (<0.60) linking any memory
        """
        if goal.confidence < 0.60:
            return True

        # Cross-scope check
        memory_class = str(getattr(memory, "classification", "internal") or "internal").lower()
        goal_scope = str(getattr(goal, "visibility_scope", "personal") or "personal").lower()

        # If memory is restricted/confidential and goal is not organization-wide, require review
        if memory_class in {"restricted", "confidential"} and goal_scope not in {"organization", "division"}:
            return True

        # If memory is confidential and link_type is "evidence" (critical link), require review
        if memory_class == "confidential" and link_type == "evidence":
            return True

        return False

    async def review_status_change(
        self,
        *,
        org_id: str,
        goal: Goal,
        old_status: str,
        new_status: str,
        trace_id: str | None = None,
    ) -> MetaAgentRun:
        """Perform Meta review of goal status change.
        
        Returns:
            MetaAgentRun with status: accepted|contested|escalated
            
        Raises:
            ValueError if review fails and operation should be blocked
        """
        # Create a lightweight review context for Meta
        run = MetaAgentRun(
            organization_id=org_id,
            resource_type="goal",
            resource_id=goal.id,
            supervision_type="review",
            status="contested",
            evidence={
                "old_status": old_status,
                "new_status": new_status,
                "goal_confidence": goal.confidence,
                "goal_type": goal.goal_type,
                "visibility_scope": goal.visibility_scope,
            },
        )
        self.session.add(run)
        await self.session.flush()

        try:
            # Simple deterministic review based on confidence and transition
            if goal.confidence < 0.50:
                run.status = "escalated"
                run.final_confidence = 0.0
                run.risk_score = 1.0
                run.reasoning_summary = "Goal confidence too low for status change"
            elif new_status in {"completed", "abandoned"} and goal.confidence < 0.70:
                run.status = "contested"
                run.final_confidence = goal.confidence
                run.risk_score = 0.6
                run.reasoning_summary = "Completion requires higher confidence"
            else:
                run.status = "accepted"
                run.final_confidence = goal.confidence
                run.risk_score = 0.2
                run.reasoning_summary = "Status change approved"

            await self.session.flush()

            if run.status == "escalated":
                raise ValueError(f"Meta review escalated status change for goal {goal.id}: requires human approval")

            return run

        except Exception as exc:
            run.status = "escalated"
            run.final_confidence = 0.0
            run.risk_score = 1.0
            run.reasoning_summary = f"Review failed: {str(exc)}"
            await self.session.flush()
            raise ValueError(f"Meta review failed for goal {goal.id}: {str(exc)}")

    async def review_memory_link(
        self,
        *,
        org_id: str,
        goal: Goal,
        memory: MemoryMetadata,
        link_type: str,
        trace_id: str | None = None,
    ) -> MetaAgentRun:
        """Perform Meta review of goal-memory link.
        
        Returns:
            MetaAgentRun with status: accepted|contested|escalated
            
        Raises:
            ValueError if review fails and operation should be blocked
        """
        run = MetaAgentRun(
            organization_id=org_id,
            resource_type="goal_memory_link",
            # resource_id is a UUID column; keep it stable and store the composite key in evidence.
            resource_id=goal.id,
            supervision_type="review",
            status="contested",
            evidence={
                "goal_id": goal.id,
                "memory_id": memory.id,
                "link_type": link_type,
                "goal_confidence": goal.confidence,
                "memory_classification": getattr(memory, "classification", None),
                "goal_visibility_scope": goal.visibility_scope,
            },
        )
        self.session.add(run)
        await self.session.flush()

        try:
            memory_class = str(getattr(memory, "classification", "internal") or "internal").lower()
            
            # Fail-closed: restricted memory cannot be linked to personal/team goals
            if memory_class == "restricted" and goal.visibility_scope in {"personal", "team"}:
                run.status = "escalated"
                run.final_confidence = 0.0
                run.risk_score = 1.0
                run.reasoning_summary = "Restricted memory cannot be linked to personal/team goals"
            elif memory_class == "confidential" and goal.confidence < 0.70:
                run.status = "contested"
                run.final_confidence = goal.confidence
                run.risk_score = 0.7
                run.reasoning_summary = "Confidential memory link requires high goal confidence"
            else:
                run.status = "accepted"
                run.final_confidence = goal.confidence
                run.risk_score = 0.2
                run.reasoning_summary = "Memory link approved"

            await self.session.flush()

            if run.status == "escalated":
                raise ValueError(f"Meta review escalated memory link for goal {goal.id}: requires human approval")

            return run

        except Exception as exc:
            run.status = "escalated"
            run.final_confidence = 0.0
            run.risk_score = 1.0
            run.reasoning_summary = f"Review failed: {str(exc)}"
            await self.session.flush()
            raise ValueError(f"Meta review failed for goal-memory link {goal.id}:{memory.id}: {str(exc)}")
