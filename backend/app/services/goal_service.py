"""GoalGraph services.

GoalService is the authoritative interface for GoalGraph CRUD and logging.
All DB access assumes the tenant context (RLS variables) is already set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import and_, desc, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.goal import Goal, GoalActivityLog, GoalEdge, GoalMemoryLink, GoalNode
from app.models.memory import MemoryMetadata
from app.schemas.base import PaginatedResponse
from app.services.goal_meta_supervisor import GoalMetaSupervisor


@dataclass(frozen=True)
class GoalProgress:
    percent_complete: float
    completed_nodes: int
    total_nodes: int
    confidence: float


def compute_goal_progress(*, nodes: list[GoalNode], goal_confidence: float) -> GoalProgress:
    actionable = [n for n in nodes if n.node_type in {"subgoal", "task", "milestone"}]
    total = len(actionable)
    completed = sum(1 for n in actionable if n.status == "done")
    percent = 0.0 if total == 0 else round((completed / total) * 100.0, 2)

    # Conservative: cap progress confidence by goal confidence.
    confidence = float(min(max(goal_confidence or 0.5, 0.0), 1.0))

    return GoalProgress(
        percent_complete=percent,
        completed_nodes=completed,
        total_nodes=total,
        confidence=confidence,
    )


class GoalService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.meta_supervisor = GoalMetaSupervisor(session)

    async def create_goal(
        self,
        *,
        org_id: str,
        actor_user_id: str | None = None,
        created_by_user_id: str | None = None,  # Deprecated alias, use actor_user_id
        title: str | None = None,
        description: str | None = None,
        goal_type: str | None = None,
        status: str = "proposed",
        visibility_scope: str | None = None,
        scope_id: str | None = None,
        owner_type: str = "user",
        owner_id: str | None = None,
        priority: int = 0,
        confidence: float = 0.5,
        due_at: Any = None,
        tags: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,  # Legacy support
    ) -> Goal:
        """Create a new goal with flexible parameter support.
        
        Supports both direct kwargs and legacy payload dict format.
        """
        # Handle legacy payload format
        if payload:
            return await self._create_goal_from_payload(
                org_id=org_id,
                created_by_user_id=created_by_user_id or actor_user_id,
                payload=payload,
            )

        # Modern keyword argument format
        user_id = actor_user_id or created_by_user_id

        if not (title and goal_type and visibility_scope):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing required fields: title, goal_type, visibility_scope",
            )
        
        goal = Goal(
            organization_id=org_id,
            created_by_user_id=user_id,
            owner_type=owner_type,
            owner_id=owner_id or user_id,
            title=title,
            description=description,
            goal_type=goal_type,
            status=status,
            priority=priority,
            due_at=due_at,
            confidence=confidence,
            visibility_scope=visibility_scope,
            scope_id=scope_id,
            tags=tags or [],
            extra_metadata=extra_metadata or {},
        )
        self.session.add(goal)
        await self.session.flush()

        await self._log(
            org_id=org_id,
            goal_id=goal.id,
            node_id=None,
            actor_type="user" if user_id else "system",
            actor_id=user_id,
            action="create_goal",
            details={"title": goal.title, "visibility_scope": goal.visibility_scope},
        )

        return goal

    async def _create_goal_from_payload(
        self,
        *,
        org_id: str,
        created_by_user_id: str | None,
        payload: dict[str, Any],
    ) -> Goal:
        """Legacy payload-based goal creation."""
        goal = Goal(
            organization_id=org_id,
            created_by_user_id=created_by_user_id,
            owner_type=payload["owner_type"],
            owner_id=payload.get("owner_id"),
            title=payload["title"],
            description=payload.get("description"),
            goal_type=payload["goal_type"],
            status=payload["status"],
            priority=payload.get("priority", 0),
            due_at=payload.get("due_at"),
            confidence=payload.get("confidence", 0.5),
            visibility_scope=payload["visibility_scope"],
            scope_id=payload.get("scope_id"),
            tags=payload.get("tags") or [],
            extra_metadata=payload.get("metadata") or {},
        )
        self.session.add(goal)
        await self.session.flush()

        await self._log(
            org_id=org_id,
            goal_id=goal.id,
            node_id=None,
            actor_type="user" if created_by_user_id else "system",
            actor_id=created_by_user_id,
            action="create_goal",
            details={"title": goal.title, "visibility_scope": goal.visibility_scope},
        )

        return goal

    async def get_goal(self, *, goal_id: str, org_id: str) -> Goal:
        result = await self.session.execute(
            select(Goal).where(and_(Goal.id == goal_id, Goal.organization_id == org_id))
        )
        goal = result.scalar_one_or_none()
        if not goal:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Goal not found")
        return goal

    async def list_goals(
        self,
        *,
        org_id: str,
        page: int,
        page_size: int,
        status_filter: str | None = None,
    ) -> PaginatedResponse[Goal]:
        query = select(Goal).where(Goal.organization_id == org_id)
        count_query = select(func.count(Goal.id)).where(Goal.organization_id == org_id)

        if status_filter:
            query = query.where(Goal.status == status_filter)
            count_query = count_query.where(Goal.status == status_filter)

        total = (await self.session.execute(count_query)).scalar() or 0

        offset = (page - 1) * page_size
        query = query.order_by(desc(Goal.updated_at)).offset(offset).limit(page_size)

        goals = (await self.session.execute(query)).scalars().all()
        return PaginatedResponse.create(items=goals, total=int(total), page=page, page_size=page_size)

    async def update_goal(
        self,
        *,
        goal_id: str,
        org_id: str,
        actor_user_id: str | None,
        updates: dict[str, Any],
    ) -> Goal:
        goal = await self.get_goal(goal_id=goal_id, org_id=org_id)
        
        # Meta supervision: check if status change requires review
        old_status = goal.status
        new_status = updates.get("status")
        
        if new_status and new_status != old_status:
            requires_review = await self.meta_supervisor.requires_review_for_status_change(
                goal=goal,
                old_status=old_status,
                new_status=new_status,
            )
            
            if requires_review:
                # Perform Meta review (may raise ValueError if escalated)
                await self.meta_supervisor.review_status_change(
                    org_id=org_id,
                    goal=goal,
                    old_status=old_status,
                    new_status=new_status,
                )

        for field in (
            "title",
            "description",
            "status",
            "priority",
            "due_at",
            "confidence",
            "visibility_scope",
            "scope_id",
            "tags",
            "metadata",
        ):
            if field in updates and updates[field] is not None:
                if field == "metadata":
                    goal.extra_metadata = updates[field]
                else:
                    setattr(goal, field, updates[field])

        if updates.get("status") == "completed" and goal.completed_at is None:
            goal.completed_at = datetime.now(timezone.utc)

        goal.updated_at = datetime.now(timezone.utc)
        await self.session.flush()

        await self._log(
            org_id=org_id,
            goal_id=goal.id,
            node_id=None,
            actor_type="user" if actor_user_id else "system",
            actor_id=actor_user_id,
            action="update_goal",
            details={"updates": {k: v for k, v in updates.items() if v is not None}},
        )

        return goal

    async def add_node(
        self,
        *,
        org_id: str,
        goal_id: str,
        actor_user_id: str | None,
        payload: dict[str, Any],
    ) -> GoalNode:
        # Ensure goal exists / RLS applies.
        await self.get_goal(goal_id=goal_id, org_id=org_id)

        node = GoalNode(
            organization_id=org_id,
            goal_id=goal_id,
            parent_node_id=payload.get("parent_node_id"),
            node_type=payload["node_type"],
            title=payload["title"],
            description=payload.get("description"),
            status=payload.get("status", "todo"),
            priority=payload.get("priority", 0),
            assigned_to_user_id=payload.get("assigned_to_user_id"),
            assigned_to_team_id=payload.get("assigned_to_team_id"),
            ordering=payload.get("ordering", 0),
            expected_outputs=payload.get("expected_outputs"),
            success_criteria=payload.get("success_criteria"),
            blockers=payload.get("blockers"),
            confidence=payload.get("confidence", 0.5),
        )
        self.session.add(node)
        await self.session.flush()

        await self._log(
            org_id=org_id,
            goal_id=goal_id,
            node_id=node.id,
            actor_type="user" if actor_user_id else "system",
            actor_id=actor_user_id,
            action="add_node",
            details={"title": node.title, "node_type": node.node_type},
        )

        return node

    async def add_subgoal(
        self,
        *,
        org_id: str,
        goal_id: str,
        actor_user_id: str | None,
        node_type: str,
        title: str,
        description: str | None = None,
        success_criteria: list[str] | None = None,
        expected_outputs: dict[str, Any] | None = None,
        parent_node_id: str | None = None,
        status: str = "todo",
        priority: int = 0,
        confidence: float = 0.5,
    ) -> GoalNode:
        """Convenience method to add a subgoal node with keyword arguments."""
        payload = {
            "node_type": node_type,
            "title": title,
            "description": description,
            "success_criteria": success_criteria or [],
            "expected_outputs": expected_outputs or {},
            "parent_node_id": parent_node_id,
            "status": status,
            "priority": priority,
            "confidence": confidence,
        }
        return await self.add_node(
            org_id=org_id,
            goal_id=goal_id,
            actor_user_id=actor_user_id,
            payload=payload,
        )

    async def update_node_status(
        self,
        *,
        node_id: str,
        org_id: str,
        actor_user_id: str | None,
        status_value: str,
    ) -> GoalNode:
        result = await self.session.execute(
            select(GoalNode).where(and_(GoalNode.id == node_id, GoalNode.organization_id == org_id))
        )
        node = result.scalar_one_or_none()
        if not node:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Goal node not found")

        node.status = status_value
        node.updated_at = datetime.now(timezone.utc)
        if status_value == "done" and node.completed_at is None:
            node.completed_at = datetime.now(timezone.utc)

        await self.session.flush()

        await self._log(
            org_id=org_id,
            goal_id=node.goal_id,
            node_id=node.id,
            actor_type="user" if actor_user_id else "system",
            actor_id=actor_user_id,
            action="update_node_status",
            details={"status": status_value},
        )

        return node

    async def add_dependency(
        self,
        *,
        org_id: str,
        actor_user_id: str | None,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
    ) -> GoalEdge:
        edge = GoalEdge(
            organization_id=org_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            edge_type=edge_type,
        )
        self.session.add(edge)
        await self.session.flush()

        # We can only log if we can resolve goal id (RLS-safe).
        goal_id = None
        res = await self.session.execute(select(GoalNode.goal_id).where(GoalNode.id == from_node_id))
        goal_id = res.scalar_one_or_none()
        if goal_id:
            await self._log(
                org_id=org_id,
                goal_id=goal_id,
                node_id=from_node_id,
                actor_type="user" if actor_user_id else "system",
                actor_id=actor_user_id,
                action="add_dependency",
                details={"to_node_id": to_node_id, "edge_type": edge_type},
            )

        return edge

    async def link_memory(
        self,
        *,
        org_id: str,
        actor_user_id: str | None,
        goal_id: str,
        memory_id: str,
        link_type: str,
        confidence: float,
        node_id: str | None,
        linked_by: str = "user",
    ) -> GoalMemoryLink:
        # Ensure goal exists / RLS applies.
        goal = await self.get_goal(goal_id=goal_id, org_id=org_id)

        memory = await self.session.get(MemoryMetadata, memory_id)
        if not memory or memory.organization_id != org_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

        # Meta supervision: check if memory link requires review
        requires_review = await self.meta_supervisor.requires_review_for_memory_link(
            goal=goal,
            memory=memory,
            link_type=link_type,
        )

        if requires_review:
            # Perform Meta review (may raise ValueError if escalated)
            await self.meta_supervisor.review_memory_link(
                org_id=org_id,
                goal=goal,
                memory=memory,
                link_type=link_type,
            )

        values = {
            "organization_id": org_id,
            "goal_id": goal_id,
            "node_id": node_id,
            "memory_id": memory_id,
            "link_type": link_type,
            "linked_by": linked_by,
            "confidence": float(confidence),
        }

        stmt = insert(GoalMemoryLink).values(values)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_goal_memory_links_goal_memory",
            set_={
                # Keep original created_at for auditability; update mutable fields.
                "node_id": node_id,
                "link_type": link_type,
                "linked_by": linked_by,
                "confidence": float(confidence),
            },
        ).returning(GoalMemoryLink)

        # When RETURNING an ORM entity that already exists in the identity map,
        # SQLAlchemy may return the existing instance without refreshing attributes.
        # populate_existing ensures the returned instance reflects updated columns.
        res = await self.session.execute(stmt.execution_options(populate_existing=True))
        link = res.scalar_one()
        await self.session.flush()

        # Defensive: ensure updated values are visible to callers in this unit of work.
        await self.session.refresh(link)

        await self._log(
            org_id=org_id,
            goal_id=goal_id,
            node_id=node_id,
            actor_type="user" if actor_user_id else "system",
            actor_id=actor_user_id,
            action="add_link",
            details={"memory_id": memory_id, "link_type": link_type, "confidence": confidence},
        )

        return link

    async def update_node(
        self,
        *,
        node_id: str,
        org_id: str,
        actor_user_id: str | None,
        updates: dict[str, Any],
    ) -> GoalNode:
        result = await self.session.execute(
            select(GoalNode).where(and_(GoalNode.id == node_id, GoalNode.organization_id == org_id))
        )
        node = result.scalar_one_or_none()
        if not node:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Goal node not found")

        if updates.get("title") is not None:
            node.title = updates["title"]
        if updates.get("description") is not None:
            node.description = updates["description"]
        if updates.get("priority") is not None:
            node.priority = int(updates["priority"])
        if updates.get("ordering") is not None:
            node.ordering = int(updates["ordering"])

        if updates.get("status") is not None:
            node.status = updates["status"]
            if node.status == "done" and node.completed_at is None:
                node.completed_at = datetime.now(timezone.utc)

        node.updated_at = datetime.now(timezone.utc)
        await self.session.flush()

        await self._log(
            org_id=org_id,
            goal_id=node.goal_id,
            node_id=node.id,
            actor_type="user" if actor_user_id else "system",
            actor_id=actor_user_id,
            action="update_node",
            details={"updates": {k: v for k, v in updates.items() if v is not None}},
        )

        return node

    async def get_goal_detail_bundle(
        self,
        *,
        goal_id: str,
        org_id: str,
    ) -> tuple[Goal, list[GoalNode], list[GoalEdge], list[GoalMemoryLink], GoalProgress]:
        goal = await self.get_goal(goal_id=goal_id, org_id=org_id)

        nodes = (
            await self.session.execute(
                select(GoalNode).where(and_(GoalNode.goal_id == goal_id, GoalNode.organization_id == org_id))
            )
        ).scalars().all()

        edges = (
            await self.session.execute(
                select(GoalEdge).where(GoalEdge.organization_id == org_id)
            )
        ).scalars().all()
        # Keep only edges for nodes of this goal.
        node_ids = {n.id for n in nodes}
        edges = [e for e in edges if e.from_node_id in node_ids and e.to_node_id in node_ids]

        links = (
            await self.session.execute(
                select(GoalMemoryLink).where(and_(GoalMemoryLink.goal_id == goal_id, GoalMemoryLink.organization_id == org_id))
            )
        ).scalars().all()

        progress = compute_goal_progress(nodes=nodes, goal_confidence=float(goal.confidence or 0.5))
        return goal, nodes, edges, links, progress

    async def list_activity(
        self,
        *,
        goal_id: str,
        org_id: str,
        limit: int,
    ) -> list[GoalActivityLog]:
        # Ensure goal exists / RLS applies.
        await self.get_goal(goal_id=goal_id, org_id=org_id)

        result = await self.session.execute(
            select(GoalActivityLog)
            .where(and_(GoalActivityLog.goal_id == goal_id, GoalActivityLog.organization_id == org_id))
            .order_by(desc(GoalActivityLog.created_at))
            .limit(limit)
        )
        return result.scalars().all()

    async def _log(
        self,
        *,
        org_id: str,
        goal_id: str,
        node_id: str | None,
        actor_type: str,
        actor_id: str | None,
        action: str,
        details: dict[str, Any],
    ) -> None:
        entry = GoalActivityLog(
            organization_id=org_id,
            goal_id=goal_id,
            node_id=node_id,
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            details=details or {},
        )
        self.session.add(entry)
        await self.session.flush()
