from __future__ import annotations

from app.agents.registry import get_write_time_agent_spec, list_write_time_agent_specs
from app.tasks.memory_pipeline import (
    build_memory_dag,
    build_memory_enrichment_canvas,
    build_write_time_execution_plan,
)


def _collect_queues(canvas) -> set[str]:
    queues: set[str] = set()
    for task in getattr(canvas, "tasks", []) or []:
        queue = (getattr(task, "options", {}) or {}).get("queue")
        if queue:
            queues.add(queue)
        if getattr(task, "tasks", None):
            queues.update(_collect_queues(task))
        body = getattr(task, "body", None)
        if body is not None:
            queues.update(_collect_queues(body))
    return queues


def test_write_time_registry_filters_by_tier():
    specs = list_write_time_agent_specs(trigger_type="memory_write", enabled_tiers={1})
    keys = {spec.key for spec in specs}

    assert "classification" in keys
    assert "entity_resolution" in keys
    assert "topic_modeling" not in keys
    assert "temporal_reasoning" not in keys


def test_get_write_time_agent_spec_supports_aliases_and_classes():
    assert get_write_time_agent_spec("entity_resolution").key == "entity_resolution"
    assert get_write_time_agent_spec("EntityResolutionAgent").key == "entity_resolution"
    assert get_write_time_agent_spec("graph").key == "graph_linking"


def test_build_write_time_execution_plan_expands_requested_dependencies():
    stages = build_write_time_execution_plan(
        trigger_type="memory_reenrich",
        requested_agents=["causal_reasoning"],
        enabled_tiers=None,
    )
    flattened = [spec.key for stage in stages for spec in stage]

    assert flattened == [
        "classification",
        "metadata",
        "semantic_normalization",
        "entity_resolution",
        "temporal_reasoning",
        "episodic_grouping",
        "causal_reasoning",
    ]


def test_build_write_time_execution_plan_orders_feedback_last():
    stages = build_write_time_execution_plan(
        trigger_type="memory_write",
        enabled_tiers={1, 2},
    )

    keys_by_stage = [[spec.key for spec in stage] for stage in stages]

    assert keys_by_stage[0] == ["classification"]
    assert keys_by_stage[1] == ["metadata"]
    assert "semantic_normalization" in keys_by_stage[2]
    assert "feedback_learning" in keys_by_stage[-1]

    stage_positions = {
        key: index
        for index, stage in enumerate(keys_by_stage)
        for key in stage
    }
    assert stage_positions["entity_resolution"] < stage_positions["graph_linking"]
    assert stage_positions["temporal_reasoning"] < stage_positions["episodic_grouping"]
    assert stage_positions["episodic_grouping"] < stage_positions["causal_reasoning"]


def test_build_memory_dag_routes_graph_and_reasoning_work_to_split_queues():
    dag = build_memory_dag(org_id="org", memory_id="mem")
    queues = _collect_queues(dag)

    assert "q.agent_graph" in queues
    assert "q.agent_reasoning" in queues
    assert "q.agent_topics" in queues
    assert "q.agent_patterns" in queues


def test_build_memory_enrichment_canvas_uses_reasoning_queue():
    canvas = build_memory_enrichment_canvas(
        agent_names=["entity_resolution", "causal_reasoning"],
        org_id="org",
        memory_id="mem",
    )
    queues = _collect_queues(canvas)

    assert "q.agent_enrich" in queues
    assert "q.agent_graph" in queues
    assert "q.agent_reasoning" in queues
