"""Tests for OrchestrationBusAgent — Phase 29."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.base import BaseAgent
from app.agents.orchestration_bus_agent import (
    OrchestrationBusAgent,
    _DEFAULT_AGENTS,
    _heuristic_agent_names,
    _parse_llm_agent_names,
    _topo_sort,
)
from app.agents.types import AgentContext, AgentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(agent_name: str, status: str = "success", outputs: dict | None = None) -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name=agent_name,
        agent_version="1.0",
        memory_id="mem-1",
        status=status,
        confidence=0.9,
        outputs=outputs or {"key": "val"},
        started_at=now,
        finished_at=now,
    )


def _stub_agent(name: str, deps: list[str] | None = None, status: str = "success", outputs: dict | None = None) -> BaseAgent:
    _deps = list(deps or [])
    _result = _make_result(name, status=status, outputs=outputs or {f"{name}_out": 1})

    class _Stub(BaseAgent):
        async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
            return _result

    _Stub.name = name
    _Stub.version = "1.0"
    stub = _Stub()
    stub.dependencies = lambda: _deps
    stub.run = AsyncMock(return_value=_result)
    return stub


def _context(
    content: str = "some memory",
    enrichment: dict | None = None,
    config: dict | None = None,
) -> AgentContext:
    return {
        "tenant": {"org_id": "org-1", "org_slug": "test"},
        "memory": {
            "id": "mem-1",
            "content": content,
            "enrichment": enrichment or {},
        },
        "config": config or {},
    }


# ---------------------------------------------------------------------------
# _topo_sort tests
# ---------------------------------------------------------------------------

class TestTopoSort:
    def test_empty(self):
        assert _topo_sort({}) == []

    def test_single_no_deps(self):
        a = _stub_agent("A")
        result = _topo_sort({"A": a})
        assert [ag.name for ag in result] == ["A"]

    def test_linear_chain(self):
        a = _stub_agent("A", deps=[])
        b = _stub_agent("B", deps=["A"])
        c = _stub_agent("C", deps=["B"])
        ordered = _topo_sort({"A": a, "B": b, "C": c})
        names = [ag.name for ag in ordered]
        assert names.index("A") < names.index("B") < names.index("C")

    def test_dep_satisfied_externally_not_raised(self):
        # B depends on X which is not in the dict — treated as satisfied
        b = _stub_agent("B", deps=["X"])
        ordered = _topo_sort({"B": b})
        assert [ag.name for ag in ordered] == ["B"]

    def test_diamond_dependency(self):
        a = _stub_agent("A", deps=[])
        b = _stub_agent("B", deps=["A"])
        c = _stub_agent("C", deps=["A"])
        d = _stub_agent("D", deps=["B", "C"])
        ordered = _topo_sort({"A": a, "B": b, "C": c, "D": d})
        names = [ag.name for ag in ordered]
        assert names.index("A") < names.index("B")
        assert names.index("A") < names.index("C")
        assert names.index("B") < names.index("D")
        assert names.index("C") < names.index("D")

    def test_cycle_raises(self):
        a = _stub_agent("A", deps=["B"])
        b = _stub_agent("B", deps=["A"])
        with pytest.raises(ValueError, match="cycle"):
            _topo_sort({"A": a, "B": b})

    def test_self_loop_raises(self):
        a = _stub_agent("A", deps=["A"])
        with pytest.raises(ValueError, match="cycle"):
            _topo_sort({"A": a})

    def test_two_independent_agents(self):
        a = _stub_agent("A")
        b = _stub_agent("B")
        ordered = _topo_sort({"A": a, "B": b})
        assert {ag.name for ag in ordered} == {"A", "B"}


# ---------------------------------------------------------------------------
# _heuristic_agent_names tests
# ---------------------------------------------------------------------------

class TestHeuristicAgentNames:
    def test_returns_list(self):
        result = _heuristic_agent_names("some content", {})
        assert isinstance(result, list)
        assert len(result) > 0

    def test_base_agents_always_included(self):
        result = _heuristic_agent_names("hello world", {})
        base = {"MetadataExtractionAgent", "SemanticNormalizationAgent", "EntityResolutionAgent", "ContextAmplifierAgent"}
        assert base.issubset(set(result))

    def test_conflict_content_triggers_conflict_pipeline(self):
        result = _heuristic_agent_names("there is a conflict between teams", {})
        assert "ConflictDetectionAgent" in result
        assert "CredibilityAgent" in result

    def test_goal_content_triggers_goal_pipeline(self):
        result = _heuristic_agent_names("our main goal is to ship the feature", {})
        assert "GoalDecompositionAgent" in result
        assert "PlaybookAgent" in result

    def test_temporal_content_triggers_temporal_pipeline(self):
        result = _heuristic_agent_names("deadline is next Friday", {})
        assert "TemporalReasoningAgent" in result
        assert "MemoryDecayAgent" in result

    def test_org_content_triggers_silo_pipeline(self):
        result = _heuristic_agent_names("cross team department org alignment", {})
        assert "SiloPropagationAgent" in result
        assert "OrgAttentionAgent" in result

    def test_no_duplicates(self):
        result = _heuristic_agent_names("goal conflict deadline team", {})
        assert len(result) == len(set(result))

    def test_synthesis_always_at_end(self):
        result = _heuristic_agent_names("plain text", {})
        assert "NarrativeSynthesisAgent" in result
        assert "UncertaintyReportingAgent" in result

    def test_empty_content(self):
        result = _heuristic_agent_names("", {})
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _parse_llm_agent_names tests
# ---------------------------------------------------------------------------

class TestParseLlmAgentNames:
    def test_valid_json(self):
        raw = '{"agents": ["EntityResolutionAgent", "CredibilityAgent"]}'
        result = _parse_llm_agent_names(raw, {"EntityResolutionAgent", "CredibilityAgent", "OtherAgent"})
        assert result == ["EntityResolutionAgent", "CredibilityAgent"]

    def test_filters_unknown_agents(self):
        raw = '{"agents": ["EntityResolutionAgent", "FakeAgent"]}'
        result = _parse_llm_agent_names(raw, {"EntityResolutionAgent"})
        assert result == ["EntityResolutionAgent"]

    def test_malformed_json_returns_empty(self):
        result = _parse_llm_agent_names("not json at all", {"EntityResolutionAgent"})
        assert result == []

    def test_missing_agents_key_returns_empty(self):
        result = _parse_llm_agent_names('{"something": []}', {"EntityResolutionAgent"})
        assert result == []

    def test_json_embedded_in_prose(self):
        raw = 'Sure! Here is my answer: {"agents": ["CredibilityAgent"]} Done.'
        result = _parse_llm_agent_names(raw, {"CredibilityAgent"})
        assert result == ["CredibilityAgent"]

    def test_empty_string(self):
        result = _parse_llm_agent_names("", {"EntityResolutionAgent"})
        assert result == []


# ---------------------------------------------------------------------------
# OrchestrationBusAgent — unit tests with mocked get_agent
# ---------------------------------------------------------------------------

def _make_bus() -> OrchestrationBusAgent:
    return OrchestrationBusAgent()


class TestOrchestrationBusAgentMeta:
    def test_name(self):
        assert OrchestrationBusAgent.name == "OrchestrationBusAgent"

    def test_version(self):
        assert OrchestrationBusAgent.version == "1.0"

    def test_dependencies_empty(self):
        assert _make_bus().dependencies() == []

    def test_should_run_true_with_memory_id(self):
        bus = _make_bus()
        assert bus.should_run("mem-1", _context()) is True

    def test_should_run_false_empty_memory_id(self):
        bus = _make_bus()
        assert bus.should_run("", _context()) is False


class TestOrchestrationBusAgentHeuristic:
    @pytest.mark.asyncio
    async def test_basic_run_returns_success(self):
        a = _stub_agent("A")
        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.side_effect = lambda name: a if name == "a" else None
            result = await bus.run("mem-1", _context(config={"agent_names": ["A"]}))
        assert result.status == "success"
        assert result.outputs["agent_order"] == ["A"]
        assert result.outputs["skipped_agents"] == []
        assert result.outputs["bus_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_outputs_propagated_to_downstream(self):
        a = _stub_agent("A", deps=[], outputs={"a_signal": "value_a"})
        b = _stub_agent("B", deps=["A"])

        async def check_b_context(memory_id, ctx):
            enrichment = ctx.get("memory", {}).get("enrichment", {})
            assert "a_signal" in enrichment, "A's output should be in enrichment for B"
            return _make_result("B", outputs={"b_signal": "value_b"})

        b.run = check_b_context

        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            def _get(name):
                return {"a": a, "b": b}.get(name)
            mock_get.side_effect = _get
            result = await bus.run("mem-1", _context(config={"agent_names": ["A", "B"]}))

        assert result.status == "success"
        assert "A" in result.outputs["agent_order"]
        assert "B" in result.outputs["agent_order"]

    @pytest.mark.asyncio
    async def test_failed_dep_causes_skip(self):
        a = _stub_agent("A", status="failed")
        b = _stub_agent("B", deps=["A"])

        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.side_effect = lambda name: {"a": a, "b": b}.get(name)
            result = await bus.run("mem-1", _context(config={"agent_names": ["A", "B"]}))

        assert result.status == "success"
        assert "B" in result.outputs["skipped_agents"]

    @pytest.mark.asyncio
    async def test_skipped_dep_causes_skip(self):
        a = _stub_agent("A")
        a.should_run = lambda mid, ctx: False  # A will be skipped
        b = _stub_agent("B", deps=["A"])

        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.side_effect = lambda name: {"a": a, "b": b}.get(name)
            result = await bus.run("mem-1", _context(config={"agent_names": ["A", "B"]}))

        assert "A" in result.outputs["skipped_agents"]
        assert "B" in result.outputs["skipped_agents"]

    @pytest.mark.asyncio
    async def test_should_run_false_skips_agent(self):
        a = _stub_agent("A")
        a.should_run = lambda mid, ctx: False

        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.side_effect = lambda name: a if name == "a" else None
            result = await bus.run("mem-1", _context(config={"agent_names": ["A"]}))

        assert "A" in result.outputs["skipped_agents"]

    @pytest.mark.asyncio
    async def test_agent_exception_marked_failed_not_raised(self):
        a = _stub_agent("A")
        a.run = AsyncMock(side_effect=RuntimeError("boom"))

        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.side_effect = lambda name: a if name == "a" else None
            result = await bus.run("mem-1", _context(config={"agent_names": ["A"]}))

        assert result.status == "success"  # bus itself succeeds
        plan = result.outputs["execution_plan"]
        a_entry = next(e for e in plan if e["agent_name"] == "A")
        assert a_entry["status"] == "failed"

    @pytest.mark.asyncio
    async def test_no_agents_resolved_returns_success_empty(self):
        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.return_value = None
            result = await bus.run("mem-1", _context(config={"agent_names": ["NonExistent"]}))

        assert result.status == "success"
        assert result.outputs["agent_order"] == []
        assert result.outputs["skipped_agents"] == []

    @pytest.mark.asyncio
    async def test_cycle_returns_failed_result(self):
        a = _stub_agent("A", deps=["B"])
        b = _stub_agent("B", deps=["A"])

        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.side_effect = lambda name: {"a": a, "b": b}.get(name)
            result = await bus.run("mem-1", _context(config={"agent_names": ["A", "B"]}))

        assert result.status == "failed"
        assert any("cycle" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_execution_plan_contains_latency(self):
        a = _stub_agent("A")
        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.side_effect = lambda name: a if name == "a" else None
            result = await bus.run("mem-1", _context(config={"agent_names": ["A"]}))

        plan = result.outputs["execution_plan"]
        assert len(plan) == 1
        assert isinstance(plan[0]["latency_ms"], float)

    @pytest.mark.asyncio
    async def test_bus_latency_ms_is_positive(self):
        a = _stub_agent("A")
        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.side_effect = lambda name: a if name == "a" else None
            result = await bus.run("mem-1", _context(config={"agent_names": ["A"]}))

        assert result.outputs["bus_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_topo_order_respected_in_agent_order(self):
        a = _stub_agent("A", deps=[])
        b = _stub_agent("B", deps=["A"])

        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.side_effect = lambda name: {"a": a, "b": b}.get(name)
            result = await bus.run("mem-1", _context(config={"agent_names": ["B", "A"]}))

        order = result.outputs["agent_order"]
        assert order.index("A") < order.index("B")

    async def test_agent_outputs_are_namespaced_alongside_flat_merge(self):
        """Regression: a flat dict.update() merge means a generic field name
        like "confidence" gets silently overwritten by whichever agent runs
        last — a downstream agent reading enrichment.get("confidence") has
        no way to know whose value it got. _agent_outputs preserves each
        agent's own outputs under its own name in the SAME shared enrichment
        dict later agents see, alongside (not instead of) the existing flat
        merge other agents already rely on for distinctively-named keys."""
        a = _stub_agent("A", deps=[], outputs={"confidence": 0.9, "a_only": 1})
        b = _stub_agent("B", deps=["A"], outputs={"confidence": 0.1, "b_only": 2})

        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.side_effect = lambda name: {"a": a, "b": b}.get(name)
            await bus.run("mem-1", _context(config={"agent_names": ["A", "B"]}))

        # B ran after A — inspect the enrichment dict the bus actually
        # handed to B's run() to confirm A's outputs are namespaced there.
        # (The dict is mutated in place as later agents run, so only
        # "_agent_outputs"[A] — which only A itself writes — is a reliable
        # post-hoc snapshot; the flat "confidence" key gets overwritten by
        # B's own output by the time the run completes, which is exactly
        # the pre-existing ambiguity this fix works around.)
        b_call_context = b.run.call_args.args[1]
        b_enrichment = b_call_context["memory"]["enrichment"]
        assert b_enrichment["_agent_outputs"]["A"]["confidence"] == 0.9
        assert b_enrichment["_agent_outputs"]["A"]["a_only"] == 1

    @pytest.mark.asyncio
    async def test_heuristic_strategy_default_used_when_no_agent_names(self):
        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.return_value = None
            result = await bus.run("mem-1", _context(config={"agent_strategy": "heuristic"}))
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_explicit_agent_names_override_heuristic(self):
        a = _stub_agent("A")
        bus = _make_bus()
        with patch("app.agents.registry.get_agent") as mock_get:
            mock_get.side_effect = lambda name: a if name == "a" else None
            result = await bus.run("mem-1", _context(config={"agent_names": ["A"], "agent_strategy": "heuristic"}))

        assert result.outputs["agent_order"] == ["A"]


# ---------------------------------------------------------------------------
# LLM path tests
# ---------------------------------------------------------------------------

class TestOrchestrationBusLLMPath:
    @pytest.mark.asyncio
    async def test_llm_strategy_uses_llm_agent_names(self):
        a = _stub_agent("EntityResolutionAgent")

        bus = _make_bus()
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value={
            "message": {"content": '{"agents": ["EntityResolutionAgent"]}'}
        })

        with patch("app.agents.orchestration_bus_agent.create_llm_client", return_value=mock_client):
            with patch("app.agents.registry.get_agent") as mock_get:
                mock_get.side_effect = lambda name: a if name == "entityresolutionagent" else None
                result = await bus.run(
                    "mem-1",
                    _context(config={"agent_strategy": "llm"}),
                )

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_llm_fallback_to_default_on_empty_response(self):
        bus = _make_bus()
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": "{}"}})

        with patch("app.agents.orchestration_bus_agent.create_llm_client", return_value=mock_client):
            with patch("app.agents.registry.get_agent") as mock_get:
                mock_get.return_value = None
                result = await bus.run("mem-1", _context(config={"agent_strategy": "llm"}))

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_llm_fallback_on_client_exception(self):
        bus = _make_bus()
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=ConnectionError("vllm down"))

        with patch("app.agents.orchestration_bus_agent.create_llm_client", return_value=mock_client):
            with patch("app.agents.registry.get_agent") as mock_get:
                mock_get.return_value = None
                result = await bus.run("mem-1", _context(config={"agent_strategy": "llm"}))

        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_llm_strategy_with_invalid_json_falls_back(self):
        bus = _make_bus()
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": "not json"}})

        with patch("app.agents.orchestration_bus_agent.create_llm_client", return_value=mock_client):
            with patch("app.agents.registry.get_agent") as mock_get:
                mock_get.return_value = None
                result = await bus.run("mem-1", _context(config={"agent_strategy": "llm"}))

        assert result.status == "success"


# ---------------------------------------------------------------------------
# validate_outputs tests
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _ok_result(self) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="OrchestrationBusAgent",
            agent_version="1.0",
            memory_id="mem-1",
            status="success",
            confidence=1.0,
            outputs={
                "execution_plan": [],
                "agent_order": [],
                "skipped_agents": [],
                "bus_latency_ms": 42.0,
            },
            started_at=now,
            finished_at=now,
        )

    def test_valid_outputs_no_raise(self):
        bus = _make_bus()
        bus.validate_outputs(self._ok_result())

    def test_missing_execution_plan_raises(self):
        bus = _make_bus()
        r = self._ok_result()
        del r.outputs["execution_plan"]
        with pytest.raises(ValueError, match="execution_plan"):
            bus.validate_outputs(r)

    def test_missing_agent_order_raises(self):
        bus = _make_bus()
        r = self._ok_result()
        del r.outputs["agent_order"]
        with pytest.raises(ValueError, match="agent_order"):
            bus.validate_outputs(r)

    def test_missing_skipped_agents_raises(self):
        bus = _make_bus()
        r = self._ok_result()
        del r.outputs["skipped_agents"]
        with pytest.raises(ValueError, match="skipped_agents"):
            bus.validate_outputs(r)

    def test_missing_bus_latency_raises(self):
        bus = _make_bus()
        r = self._ok_result()
        del r.outputs["bus_latency_ms"]
        with pytest.raises(ValueError, match="bus_latency_ms"):
            bus.validate_outputs(r)

    def test_failed_status_skips_validation(self):
        bus = _make_bus()
        now = datetime.now(timezone.utc)
        r = AgentResult(
            agent_name="OrchestrationBusAgent",
            agent_version="1.0",
            memory_id="mem-1",
            status="failed",
            confidence=0.0,
            outputs={},
            started_at=now,
            finished_at=now,
        )
        bus.validate_outputs(r)  # should not raise

    def test_wrong_type_execution_plan_raises(self):
        bus = _make_bus()
        r = self._ok_result()
        r.outputs["execution_plan"] = "not a list"
        with pytest.raises(ValueError, match="execution_plan"):
            bus.validate_outputs(r)


# ---------------------------------------------------------------------------
# Registry integration test
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_returns_bus_agent(self):
        from app.agents.registry import get_agent
        agent = get_agent("orchestration_bus")
        assert agent is not None
        assert isinstance(agent, OrchestrationBusAgent)

    def test_registry_alternate_names(self):
        from app.agents.registry import get_agent
        for alias in ("orchestrationbus", "orchestrationbusagent"):
            agent = get_agent(alias)
            assert agent is not None, f"alias '{alias}' should resolve"

    def test_registry_unknown_returns_none(self):
        from app.agents.registry import get_agent
        assert get_agent("does_not_exist_xyz") is None


# ---------------------------------------------------------------------------
# Default agent list sanity
# ---------------------------------------------------------------------------

class TestDefaultAgentList:
    def test_default_agents_is_list(self):
        assert isinstance(_DEFAULT_AGENTS, list)
        assert len(_DEFAULT_AGENTS) > 0

    def test_no_duplicates_in_default(self):
        assert len(_DEFAULT_AGENTS) == len(set(_DEFAULT_AGENTS))

    def test_key_agents_in_default(self):
        required = {
            "EntityResolutionAgent",
            "ConflictDetectionAgent",
            "CredibilityAgent",
            "UncertaintyReportingAgent",
        }
        assert required.issubset(set(_DEFAULT_AGENTS))
