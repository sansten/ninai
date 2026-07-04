"""Tests for the OrchestrationBusAgent write-time execution spec.

OrchestrationBusAgent (Phase 29) was registered and had 65 passing unit
tests, but was never included in WRITE_TIME_AGENT_SPECS — the only spec list
the production Celery pipeline (app/tasks/memory_pipeline.py) dispatches —
so despite containing a working 23-agent dependency-aware composition engine
(ContextAmplifierAgent, SiloPropagationAgent, OrgAttentionAgent,
ProactiveMemoryPushAgent, ConflictDetectionAgent, GoalDecompositionAgent,
NarrativeSynthesisAgent, and more — see orchestration_bus_agent.py), none of
those agents ever ran outside their own test files.
"""
from __future__ import annotations

from app.agents.registry import (
    WRITE_TIME_AGENT_SPECS,
    get_write_time_agent_spec,
    list_write_time_agent_specs,
)


def _spec_by_key(key: str):
    for spec in WRITE_TIME_AGENT_SPECS:
        if spec.key == key:
            return spec
    return None


class TestOrchestrationBusSpecRegistered:
    def test_spec_exists(self):
        spec = _spec_by_key("orchestration_bus")
        assert spec is not None
        assert spec.class_name == "OrchestrationBusAgent"

    def test_resolvable_by_class_name_and_aliases(self):
        for identifier in ("OrchestrationBusAgent", "orchestration_bus", "orchestrationbus"):
            spec = get_write_time_agent_spec(identifier)
            assert spec is not None, f"could not resolve {identifier!r}"
            assert spec.key == "orchestration_bus"

    def test_runs_on_memory_write_and_reenrich(self):
        spec = _spec_by_key("orchestration_bus")
        assert "memory_write" in spec.trigger_types
        assert "memory_reenrich" in spec.trigger_types

    def test_is_tier_3_after_the_unconditional_tier_1_2_backbone(self):
        spec = _spec_by_key("orchestration_bus")
        assert spec.tier == 3
        # These three keys are unconditionally registered (not gated behind
        # an optional enterprise import), so depending on them guarantees a
        # real ordering constraint regardless of which optional agents are
        # available in a given deployment.
        assert set(spec.depends_on) == {"episodic_grouping", "pattern_detection", "logseq_export"}

    def test_included_in_default_memory_write_dispatch_plan(self):
        keys = {
            spec.key
            for spec in list_write_time_agent_specs(trigger_type="memory_write", enabled_tiers=None)
        }
        assert "orchestration_bus" in keys

    def test_runs_after_its_dependencies_in_topo_order(self):
        specs = list_write_time_agent_specs(trigger_type="memory_write", enabled_tiers=None)
        order = {spec.key: idx for idx, spec in enumerate(specs)}
        bus_idx = order["orchestration_bus"]
        for dep in ("episodic_grouping", "pattern_detection", "logseq_export"):
            assert order[dep] < bus_idx, f"{dep} must be ordered before orchestration_bus"
