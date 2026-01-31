"""Unit tests for enhanced blocker detection."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.services.goal_navigator import GoalNavigator


@pytest.mark.asyncio
async def test_compute_blockers_finds_explicitly_blocked_nodes(db_session):
    """Test that compute_blockers finds nodes with status='blocked'."""
    
    # Mock nodes
    nodes = [
        SimpleNamespace(
            id="node1",
            status="todo",
            node_type="task",
            title="Task 1",
            created_at=datetime.now(timezone.utc),
        ),
        SimpleNamespace(
            id="node2",
            status="blocked",
            node_type="task",
            title="Task 2 (blocked)",
            created_at=datetime.now(timezone.utc),
        ),
        SimpleNamespace(
            id="node3",
            status="done",
            node_type="task",
            title="Task 3",
            created_at=datetime.now(timezone.utc),
        ),
    ]
    
    edges = []
    
    navigator = GoalNavigator(db_session)
    
    # Mock the database methods
    async def mock_get_tree(org_id, goal_id):
        return nodes
    
    async def mock_list_edges(org_id, goal_id):
        return edges
    
    navigator.get_goal_tree = mock_get_tree
    navigator.list_edges = mock_list_edges
    
    blockers = await navigator.compute_blockers(org_id="org1", goal_id="goal1")
    
    assert len(blockers) == 1
    assert blockers[0].id == "node2"


@pytest.mark.asyncio
async def test_compute_blockers_finds_dependency_based_blocks(db_session):
    """Test that compute_blockers finds nodes blocked by dependencies."""
    
    nodes = [
        SimpleNamespace(
            id="node1",
            status="failed",
            node_type="task",
            title="Failed prerequisite",
            created_at=datetime.now(timezone.utc),
        ),
        SimpleNamespace(
            id="node2",
            status="todo",
            node_type="task",
            title="Dependent task",
            created_at=datetime.now(timezone.utc),
        ),
    ]
    
    # node2 depends_on node1, so node2 should be blocked since node1 failed
    edges = [
        SimpleNamespace(
            from_node_id="node1",
            to_node_id="node2",
            edge_type="depends_on",
        )
    ]
    
    navigator = GoalNavigator(db_session)
    
    async def mock_get_tree(org_id, goal_id):
        return nodes
    
    async def mock_list_edges(org_id, goal_id):
        return edges
    
    navigator.get_goal_tree = mock_get_tree
    navigator.list_edges = mock_list_edges
    
    blockers = await navigator.compute_blockers(org_id="org1", goal_id="goal1")
    
    # node2 is blocked because it depends on failed node1
    # Only node2 should be returned (node1 is failed, not blocked)
    assert len(blockers) == 1
    assert blockers[0].id == "node2"


@pytest.mark.asyncio
async def test_compute_blockers_finds_blocks_edge_type(db_session):
    """Test that compute_blockers finds nodes blocked by 'blocks' edge type."""
    
    nodes = [
        SimpleNamespace(
            id="node1",
            status="blocked",
            node_type="task",
            title="Blocker",
            created_at=datetime.now(timezone.utc),
        ),
        SimpleNamespace(
            id="node2",
            status="todo",
            node_type="task",
            title="Blocked by node1",
            created_at=datetime.now(timezone.utc),
        ),
    ]
    
    # node1 blocks node2
    edges = [
        SimpleNamespace(
            from_node_id="node1",
            to_node_id="node2",
            edge_type="blocks",
        )
    ]
    
    navigator = GoalNavigator(db_session)
    
    async def mock_get_tree(org_id, goal_id):
        return nodes
    
    async def mock_list_edges(org_id, goal_id):
        return edges
    
    navigator.get_goal_tree = mock_get_tree
    navigator.list_edges = mock_list_edges
    
    blockers = await navigator.compute_blockers(org_id="org1", goal_id="goal1")
    
    assert len(blockers) == 2
    blocker_ids = {b.id for b in blockers}
    assert "node1" in blocker_ids
    assert "node2" in blocker_ids


@pytest.mark.asyncio
async def test_compute_blockers_no_false_positives(db_session):
    """Test that compute_blockers doesn't flag healthy nodes."""
    
    nodes = [
        SimpleNamespace(
            id="node1",
            status="done",
            node_type="task",
            title="Completed",
            created_at=datetime.now(timezone.utc),
        ),
        SimpleNamespace(
            id="node2",
            status="in_progress",
            node_type="task",
            title="In progress",
            created_at=datetime.now(timezone.utc),
        ),
        SimpleNamespace(
            id="node3",
            status="todo",
            node_type="task",
            title="Pending",
            created_at=datetime.now(timezone.utc),
        ),
    ]
    
    edges = [
        SimpleNamespace(
            from_node_id="node1",
            to_node_id="node2",
            edge_type="depends_on",
        )
    ]
    
    navigator = GoalNavigator(db_session)
    
    async def mock_get_tree(org_id, goal_id):
        return nodes
    
    async def mock_list_edges(org_id, goal_id):
        return edges
    
    navigator.get_goal_tree = mock_get_tree
    navigator.list_edges = mock_list_edges
    
    blockers = await navigator.compute_blockers(org_id="org1", goal_id="goal1")
    
    assert len(blockers) == 0  # No blocked nodes
