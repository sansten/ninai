"""GoalNavigator: convenience queries over the GoalGraph.

This is intentionally lightweight; it should not run heavy planning/simulation.
"""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.goal import Goal, GoalEdge, GoalNode


class GoalNavigator:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_active_goals_for_user(self, *, org_id: str, user_id: str) -> list[Goal]:
        # Rely on RLS + basic creator filter.
        result = await self.session.execute(
            select(Goal)
            .where(and_(Goal.organization_id == org_id, Goal.status.in_(["proposed", "active", "blocked"])))
            .order_by(Goal.updated_at.desc())
        )
        goals = result.scalars().all()
        return [g for g in goals if g.created_by_user_id == user_id or g.visibility_scope != "personal"]

    async def get_goal_tree(self, *, org_id: str, goal_id: str) -> list[GoalNode]:
        result = await self.session.execute(
            select(GoalNode)
            .where(and_(GoalNode.organization_id == org_id, GoalNode.goal_id == goal_id))
            .order_by(GoalNode.ordering.asc(), GoalNode.created_at.asc())
        )
        return result.scalars().all()

    async def compute_blockers(self, *, org_id: str, goal_id: str) -> list[GoalNode]:
        """
        Compute blocked nodes using both status and edge-based blocking.
        
        A node is considered blocked if:
        1. Its status is explicitly "blocked", OR
        2. It depends on another node that is blocked/failed (via "blocks" or "depends_on" edges)
        """
        nodes = await self.get_goal_tree(org_id=org_id, goal_id=goal_id)
        edges = await self.list_edges(org_id=org_id, goal_id=goal_id)
        
        # Start with explicitly blocked nodes
        blocked_ids = {n.id for n in nodes if n.status == "blocked"}
        
        # Add edge-based blocking (nodes that depend on blocked/failed nodes)
        node_map = {n.id: n for n in nodes}
        for edge in edges:
            if edge.edge_type == "blocks":
                # from_node blocks to_node if from_node is blocked/failed
                from_node = node_map.get(edge.from_node_id)
                if from_node and from_node.status in ("blocked", "failed"):
                    blocked_ids.add(edge.to_node_id)
            elif edge.edge_type == "depends_on":
                # to_node depends on from_node, so from_node blocking affects to_node
                from_node = node_map.get(edge.from_node_id)
                if from_node and from_node.status in ("blocked", "failed"):
                    blocked_ids.add(edge.to_node_id)
        
        return [node_map[nid] for nid in blocked_ids if nid in node_map]

    async def list_edges(self, *, org_id: str, goal_id: str) -> list[GoalEdge]:
        # Fetch nodes first then filter edges.
        nodes = await self.get_goal_tree(org_id=org_id, goal_id=goal_id)
        node_ids = {n.id for n in nodes}
        result = await self.session.execute(select(GoalEdge).where(GoalEdge.organization_id == org_id))
        edges = result.scalars().all()
        return [e for e in edges if e.from_node_id in node_ids and e.to_node_id in node_ids]

    @staticmethod
    def as_tree(nodes: list[GoalNode]) -> dict[str | None, list[GoalNode]]:
        by_parent: dict[str | None, list[GoalNode]] = defaultdict(list)
        for node in nodes:
            by_parent[node.parent_node_id].append(node)
        for siblings in by_parent.values():
            siblings.sort(key=lambda n: (n.ordering or 0, n.created_at))
        return dict(by_parent)
