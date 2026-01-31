"""GoalGraph unit tests (no Postgres required)."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.goal_service import compute_goal_progress


def test_compute_goal_progress_empty():
    progress = compute_goal_progress(nodes=[], goal_confidence=0.7)
    assert progress.percent_complete == 0.0
    assert progress.completed_nodes == 0
    assert progress.total_nodes == 0
    assert progress.confidence == 0.7


def test_compute_goal_progress_counts_actionable_nodes_only():
    nodes = [
        SimpleNamespace(node_type="task", status="done"),
        SimpleNamespace(node_type="milestone", status="todo"),
        SimpleNamespace(node_type="subgoal", status="done"),
        # Non-actionable type should be ignored
        SimpleNamespace(node_type="note", status="done"),
    ]
    progress = compute_goal_progress(nodes=nodes, goal_confidence=0.9)
    assert progress.total_nodes == 3
    assert progress.completed_nodes == 2
    assert progress.percent_complete == 66.67


def test_compute_goal_progress_confidence_clamped():
    nodes = [SimpleNamespace(node_type="task", status="done")]
    assert compute_goal_progress(nodes=nodes, goal_confidence=-1.0).confidence == 0.0
    assert compute_goal_progress(nodes=nodes, goal_confidence=2.0).confidence == 1.0
