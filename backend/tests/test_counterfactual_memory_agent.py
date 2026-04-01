"""Tests for Phase 51 - CounterfactualMemoryAgent."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.counterfactual_memory_agent import (
    CounterfactualMemoryAgent,
    bfs_affected_nodes,
    build_adjacency,
    intervention_magnitude,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


def _ctx(*, intervention=None, causal_graph=None, related_memories=None):
    return {
        "tenant": {"org_id": "org-1", "org_slug": "test"},
        "memory": {
            "content": "postmortem request",
            "enrichment": {
                "intervention": intervention or {},
                "causal_graph": causal_graph or [],
                "related_memories": related_memories or [],
            },
        },
        "runtime": {"job_id": "job-1"},
    }


def _result(outputs, status="success"):
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="CounterfactualMemoryAgent",
        agent_version="v1",
        memory_id="m0",
        status=status,
        confidence=0.6,
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
    )


# intervention_magnitude

def test_intervention_magnitude_boolean_is_one():
    assert intervention_magnitude({"from": False, "to": True}) == 1.0


def test_intervention_magnitude_numeric_formula():
    out = intervention_magnitude({"from": 2, "to": 10})
    assert out == pytest.approx(0.8)


def test_intervention_magnitude_numeric_negative_to_positive():
    out = intervention_magnitude({"from": -2, "to": 4})
    assert out == pytest.approx(1.5)


def test_intervention_magnitude_string_change_is_one():
    assert intervention_magnitude({"from": "low", "to": "high"}) == 1.0


def test_intervention_magnitude_same_string_is_zero():
    assert intervention_magnitude({"from": "low", "to": "low"}) == 0.0


def test_intervention_magnitude_missing_values_defaults_zero():
    assert intervention_magnitude({}) == 0.0


# graph helpers

def test_build_adjacency_supports_from_to_keys():
    adj = build_adjacency([{"from": "a", "to": "b", "weight": 0.7}])
    assert adj == {"a": [("b", 0.7)]}


def test_build_adjacency_supports_source_target_keys():
    adj = build_adjacency([{"source": "a", "target": "b", "edge_weight": 0.9}])
    assert adj == {"a": [("b", 0.9)]}


def test_build_adjacency_ignores_incomplete_edges():
    adj = build_adjacency([{"from": "a"}, {"to": "b"}])
    assert adj == {}


def test_bfs_affected_nodes_empty_start_returns_empty():
    nodes, weights = bfs_affected_nodes(start_node="", adjacency={})
    assert nodes == []
    assert weights == []


def test_bfs_single_edge_reaches_one_node():
    nodes, weights = bfs_affected_nodes(start_node="m0", adjacency={"m0": [("m1", 0.5)]})
    assert nodes == ["m1"]
    assert weights == [0.5]


def test_bfs_stops_at_depth_three():
    adjacency = {
        "m0": [("m1", 0.5)],
        "m1": [("m2", 0.5)],
        "m2": [("m3", 0.5)],
        "m3": [("m4", 0.5)],
    }
    nodes, _ = bfs_affected_nodes(start_node="m0", adjacency=adjacency, max_depth=3)
    assert "m4" not in nodes
    assert nodes == ["m1", "m2", "m3"]


def test_bfs_prevents_cycles():
    adjacency = {"m0": [("m1", 0.5)], "m1": [("m0", 0.4)]}
    nodes, _ = bfs_affected_nodes(start_node="m0", adjacency=adjacency)
    assert nodes == ["m1"]


# heuristic run

@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_empty_graph_has_no_affected_nodes_and_low_confidence():
    agent = CounterfactualMemoryAgent()
    result = await agent.run("m0", _ctx(intervention={"field": "severity", "from": "low", "to": "high"}))
    assert result.outputs["affected_nodes"] == []
    assert result.outputs["confidence"] <= 0.5


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_single_edge_graph_yields_one_affected_node():
    agent = CounterfactualMemoryAgent()
    graph = [{"from": "m0", "to": "m1", "weight": 0.8}]
    result = await agent.run("m0", _ctx(intervention={"field": "flag", "from": False, "to": True}, causal_graph=graph))
    assert result.outputs["affected_nodes"] == ["m1"]


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_probability_delta_uses_edge_weight_times_magnitude():
    agent = CounterfactualMemoryAgent()
    graph = [{"from": "m0", "to": "m1", "weight": 0.5}]
    result = await agent.run("m0", _ctx(intervention={"field": "latency", "from": 2, "to": 10}, causal_graph=graph))
    assert result.outputs["counterfactual_delta"]["probability_change"] == pytest.approx(0.4)


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_outcome_contains_field_and_values():
    agent = CounterfactualMemoryAgent()
    result = await agent.run(
        "m0",
        _ctx(intervention={"field": "severity", "from": "low", "to": "critical"}, causal_graph=[]),
    )
    text = result.outputs["counterfactual_outcome"]
    assert "severity" in text
    assert "low" in text
    assert "critical" in text


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_assumptions_non_empty():
    agent = CounterfactualMemoryAgent()
    result = await agent.run("m0", _ctx(intervention={"field": "x", "from": 1, "to": 2}))
    assert result.outputs["assumptions"]


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_confidence_clamped_to_point_nine():
    agent = CounterfactualMemoryAgent()
    graph = [{"from": f"m{i}", "to": f"m{i+1}", "weight": 0.5} for i in range(20)]
    result = await agent.run("m0", _ctx(intervention={"field": "f", "from": 0, "to": 1}, causal_graph=graph))
    assert result.outputs["confidence"] <= 0.9


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_severity_shift_uses_arrow():
    agent = CounterfactualMemoryAgent()
    result = await agent.run("m0", _ctx(intervention={"field": "severity", "from": "low", "to": "critical"}))
    assert result.outputs["counterfactual_delta"]["severity_shift"] == "low->critical"


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_result_status_success():
    agent = CounterfactualMemoryAgent()
    result = await agent.run("m0", _ctx(intervention={"field": "f", "from": 1, "to": 2}))
    assert result.status == "success"


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_trace_id_propagated():
    agent = CounterfactualMemoryAgent()
    ctx = _ctx(intervention={"field": "f", "from": 1, "to": 2})
    ctx["runtime"]["job_id"] = "trace-123"
    result = await agent.run("m0", ctx)
    assert result.trace_id == "trace-123"


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_related_memories_not_required_for_heuristic():
    agent = CounterfactualMemoryAgent()
    result = await agent.run("m0", _ctx(intervention={"field": "f", "from": 1, "to": 2}, related_memories=[{"id": "x"}]))
    assert isinstance(result.outputs["counterfactual_outcome"], str)


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "heuristic")
async def test_max_hops_three_enforced_in_run():
    agent = CounterfactualMemoryAgent()
    graph = [
        {"from": "m0", "to": "m1", "weight": 0.5},
        {"from": "m1", "to": "m2", "weight": 0.5},
        {"from": "m2", "to": "m3", "weight": 0.5},
        {"from": "m3", "to": "m4", "weight": 0.5},
    ]
    result = await agent.run("m0", _ctx(intervention={"field": "f", "from": 0, "to": 1}, causal_graph=graph))
    assert "m4" not in result.outputs["affected_nodes"]


# llm path

@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "llm")
async def test_llm_path_accepts_valid_json():
    agent = CounterfactualMemoryAgent()
    fake_client = AsyncMock()
    fake_client.complete_json = AsyncMock(
        return_value={
            "counterfactual_outcome": "mocked outcome",
            "affected_nodes": ["m1"],
            "confidence": 0.77,
            "counterfactual_delta": {"probability_change": 0.2, "severity_shift": "low->high"},
            "assumptions": ["x"],
        }
    )
    with patch("app.agents.counterfactual_memory_agent.create_ollama_client", return_value=fake_client):
        result = await agent.run("m0", _ctx(intervention={"field": "severity", "from": "low", "to": "high"}))

    assert result.outputs["counterfactual_outcome"] == "mocked outcome"
    assert result.outputs["rationale"] == "llm"


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "llm")
async def test_llm_invalid_json_falls_back_to_heuristic():
    agent = CounterfactualMemoryAgent()
    fake_client = AsyncMock()
    fake_client.complete_json = AsyncMock(return_value={"bad": "shape"})
    with patch("app.agents.counterfactual_memory_agent.create_ollama_client", return_value=fake_client):
        result = await agent.run("m0", _ctx(intervention={"field": "severity", "from": "low", "to": "high"}))

    assert result.outputs["rationale"] == "heuristic"
    assert "counterfactual_outcome" in result.outputs


@pytest.mark.asyncio
@patch("app.agents.counterfactual_memory_agent.settings.AGENT_STRATEGY", "llm")
async def test_llm_prompt_includes_intervention_and_graph():
    agent = CounterfactualMemoryAgent()
    fake_client = AsyncMock()
    fake_client.complete_json = AsyncMock(
        return_value={
            "counterfactual_outcome": "ok",
            "affected_nodes": [],
            "confidence": 0.6,
            "counterfactual_delta": {"probability_change": 0.0, "severity_shift": "a->b"},
            "assumptions": ["x"],
        }
    )
    with patch("app.agents.counterfactual_memory_agent.create_ollama_client", return_value=fake_client):
        await agent.run(
            "m0",
            _ctx(
                intervention={"field": "severity", "from": "low", "to": "high"},
                causal_graph=[{"from": "m0", "to": "m1", "weight": 0.8}],
            ),
        )

    prompt = fake_client.complete_json.call_args.kwargs["prompt"]
    assert "INTERVENTION" in prompt
    assert "CAUSAL_GRAPH" in prompt


# validate_outputs

def test_validate_outputs_passes_valid_payload():
    agent = CounterfactualMemoryAgent()
    out = {
        "counterfactual_outcome": "x",
        "affected_nodes": ["m1"],
        "confidence": 0.7,
        "counterfactual_delta": {"probability_change": 0.2, "severity_shift": "low->high"},
        "assumptions": ["a"],
    }
    agent.validate_outputs(_result(out))


def test_validate_outputs_requires_outcome_string():
    agent = CounterfactualMemoryAgent()
    with pytest.raises(ValueError, match="counterfactual_outcome"):
        agent.validate_outputs(_result({"counterfactual_outcome": 1, "affected_nodes": [], "confidence": 0.5, "counterfactual_delta": {}, "assumptions": []}))


def test_validate_outputs_requires_affected_nodes_list():
    agent = CounterfactualMemoryAgent()
    with pytest.raises(ValueError, match="affected_nodes"):
        agent.validate_outputs(_result({"counterfactual_outcome": "x", "affected_nodes": "bad", "confidence": 0.5, "counterfactual_delta": {}, "assumptions": []}))


def test_validate_outputs_requires_confidence_range():
    agent = CounterfactualMemoryAgent()
    with pytest.raises(ValueError, match="confidence"):
        agent.validate_outputs(_result({"counterfactual_outcome": "x", "affected_nodes": [], "confidence": 1.2, "counterfactual_delta": {}, "assumptions": []}))


def test_validate_outputs_requires_delta_dict():
    agent = CounterfactualMemoryAgent()
    with pytest.raises(ValueError, match="counterfactual_delta"):
        agent.validate_outputs(_result({"counterfactual_outcome": "x", "affected_nodes": [], "confidence": 0.5, "counterfactual_delta": [], "assumptions": []}))


def test_validate_outputs_requires_assumptions_list():
    agent = CounterfactualMemoryAgent()
    with pytest.raises(ValueError, match="assumptions"):
        agent.validate_outputs(_result({"counterfactual_outcome": "x", "affected_nodes": [], "confidence": 0.5, "counterfactual_delta": {}, "assumptions": "bad"}))


def test_validate_outputs_skips_non_success():
    agent = CounterfactualMemoryAgent()
    agent.validate_outputs(_result({"broken": "payload"}, status="failed"))


# registry / metadata

def test_registry_returns_counterfactual_agent_aliases():
    assert isinstance(get_agent("counterfactual"), CounterfactualMemoryAgent)
    assert isinstance(get_agent("counterfactual_memory"), CounterfactualMemoryAgent)


def test_agent_name_and_version():
    agent = CounterfactualMemoryAgent()
    assert agent.name == "CounterfactualMemoryAgent"
    assert agent.version == "v1"


def test_dependencies_include_causal_reasoning_agent():
    agent = CounterfactualMemoryAgent()
    assert "CausalReasoningAgent" in agent.dependencies()
