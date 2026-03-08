"""Tests for Phase 4 — WorldModelAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.world_model_agent import (
    WorldModelAgent,
    build_entity_edges,
    build_graph_summary,
    build_world_nodes,
    compute_node_confidence,
    compute_state_version,
    detect_state_changes,
)
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CUSTOMER_ENTITY = {
    "original": "client",
    "canonical": "customer",
    "entity_type": "customer",
    "domains": ["sales", "finance", "support"],
    "match_type": "alias",
    "confidence": 0.85,
}

_REVENUE_ENTITY = {
    "original": "arr",
    "canonical": "revenue",
    "entity_type": "metric",
    "domains": ["sales", "finance"],
    "match_type": "alias",
    "confidence": 0.85,
}

_PROJECT_ENTITY = {
    "original": "initiative",
    "canonical": "project",
    "entity_type": "project",
    "domains": ["engineering", "product"],
    "match_type": "alias",
    "confidence": 0.65,
}

_CUSTOMER_LINK = {
    "canonical": "customer",
    "entity_type": "customer",
    "domains": ["sales", "finance", "support"],
    "silo_count": 3,
}

_REVENUE_LINK = {
    "canonical": "revenue",
    "entity_type": "metric",
    "domains": ["sales", "finance"],
    "silo_count": 2,
}


def _make_result(outputs: dict) -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="WorldModelAgent",
        agent_version="v1",
        memory_id="mem-1",
        status="success",
        confidence=outputs.get("confidence", 0.5),
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
    )


# ---------------------------------------------------------------------------
# TestComputeStateVersion
# ---------------------------------------------------------------------------


class TestComputeStateVersion:
    def test_base_version_one(self):
        entity = {"silo_count": 1, "confidence": 0.5}
        assert compute_state_version(entity) == 1

    def test_silo2_increments_version(self):
        entity = {"silo_count": 2, "confidence": 0.5}
        assert compute_state_version(entity) == 2

    def test_silo3_increments_twice(self):
        entity = {"silo_count": 3, "confidence": 0.5}
        assert compute_state_version(entity) == 3

    def test_high_confidence_increments(self):
        entity = {"silo_count": 1, "confidence": 0.85}
        assert compute_state_version(entity) == 2

    def test_max_version_four(self):
        entity = {"silo_count": 3, "confidence": 0.90}
        assert compute_state_version(entity) == 4

    def test_domains_fallback_for_silo_count(self):
        entity = {"domains": ["a", "b", "c"], "confidence": 0.5}
        assert compute_state_version(entity) == 3  # len(domains)=3 → +1+1

    def test_empty_entity_defaults(self):
        assert compute_state_version({}) == 1


# ---------------------------------------------------------------------------
# TestComputeNodeConfidence
# ---------------------------------------------------------------------------


class TestComputeNodeConfidence:
    def test_base_confidence_preserved(self):
        entity = {"confidence": 0.85, "entity_type": "concept"}
        conf = compute_node_confidence(entity)
        assert conf == pytest.approx(0.85 + 0.05 * 0.2, abs=0.001)

    def test_high_entity_type_weight(self):
        entity = {"confidence": 0.65, "entity_type": "person"}
        conf = compute_node_confidence(entity)
        assert conf == pytest.approx(0.65 + 0.20 * 0.2, abs=0.001)

    def test_customer_entity_type_weight(self):
        entity = {"confidence": 0.85, "entity_type": "customer"}
        conf = compute_node_confidence(entity)
        assert conf == pytest.approx(0.85 + 0.15 * 0.2, abs=0.001)

    def test_capped_at_one(self):
        entity = {"confidence": 1.0, "entity_type": "person"}
        conf = compute_node_confidence(entity)
        assert conf <= 1.0

    def test_unknown_entity_type_default_weight(self):
        entity = {"confidence": 0.5, "entity_type": "widget"}
        conf = compute_node_confidence(entity)
        assert conf == pytest.approx(0.5 + 0.05 * 0.2, abs=0.001)

    def test_missing_entity_type_defaults_concept(self):
        entity = {"confidence": 0.5}
        conf = compute_node_confidence(entity)
        assert conf == pytest.approx(0.5 + 0.05 * 0.2, abs=0.001)


# ---------------------------------------------------------------------------
# TestBuildWorldNodes
# ---------------------------------------------------------------------------


class TestBuildWorldNodes:
    def test_single_entity_builds_node(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY], [])
        assert len(nodes) == 1
        assert nodes[0]["entity"] == "customer"
        assert nodes[0]["entity_type"] == "customer"

    def test_deduplicates_by_canonical(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY, _CUSTOMER_ENTITY], [])
        assert len(nodes) == 1

    def test_cross_silo_link_enriches_silo_count(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY], [_CUSTOMER_LINK])
        assert nodes[0]["silo_count"] == 3

    def test_empty_inputs_returns_empty(self):
        nodes = build_world_nodes([], [])
        assert nodes == []

    def test_node_fields_complete(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY], [_CUSTOMER_LINK])
        node = nodes[0]
        assert "entity" in node
        assert "entity_type" in node
        assert "domains" in node
        assert "silo_count" in node
        assert "state_version" in node
        assert "node_confidence" in node

    def test_state_version_computed(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY], [_CUSTOMER_LINK])
        assert nodes[0]["state_version"] >= 1

    def test_silo_count_from_domains_fallback(self):
        entity = {**_PROJECT_ENTITY}  # domains: ["engineering", "product"], no link
        nodes = build_world_nodes([entity], [])
        assert nodes[0]["silo_count"] == 2  # len(domains)

    def test_two_entities_two_nodes(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY, _REVENUE_ENTITY], [])
        assert len(nodes) == 2
        entities = {n["entity"] for n in nodes}
        assert "customer" in entities
        assert "revenue" in entities

    def test_missing_canonical_skipped(self):
        bad = {"entity_type": "concept", "domains": []}
        nodes = build_world_nodes([bad], [])
        assert nodes == []


# ---------------------------------------------------------------------------
# TestBuildEntityEdges
# ---------------------------------------------------------------------------


class TestBuildEntityEdges:
    def test_two_nodes_one_edge(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY, _REVENUE_ENTITY], [_CUSTOMER_LINK, _REVENUE_LINK])
        edges = build_entity_edges(nodes)
        assert len(edges) == 1
        assert edges[0]["from_entity"] == "customer"
        assert edges[0]["to_entity"] == "revenue"

    def test_three_nodes_three_edges(self):
        nodes = build_world_nodes(
            [_CUSTOMER_ENTITY, _REVENUE_ENTITY, _PROJECT_ENTITY], []
        )
        edges = build_entity_edges(nodes)
        assert len(edges) == 3

    def test_single_node_no_edges(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY], [])
        edges = build_entity_edges(nodes)
        assert edges == []

    def test_edge_weight_is_average_confidence(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY, _REVENUE_ENTITY], [])
        edges = build_entity_edges(nodes)
        expected_weight = (nodes[0]["node_confidence"] + nodes[1]["node_confidence"]) / 2
        assert edges[0]["weight"] == pytest.approx(expected_weight, abs=0.001)

    def test_edge_fields_complete(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY, _REVENUE_ENTITY], [])
        edges = build_entity_edges(nodes)
        edge = edges[0]
        assert "from_entity" in edge
        assert "to_entity" in edge
        assert "relationship_type" in edge
        assert "weight" in edge
        assert edge["relationship_type"] == "co_occurrence"

    def test_capped_at_100(self):
        many_entities = [
            {
                "canonical": f"entity{i}",
                "entity_type": "concept",
                "domains": ["sales"],
                "confidence": 0.7,
            }
            for i in range(20)
        ]
        nodes = build_world_nodes(many_entities, [])
        edges = build_entity_edges(nodes)
        assert len(edges) <= 100

    def test_empty_nodes_no_edges(self):
        assert build_entity_edges([]) == []


# ---------------------------------------------------------------------------
# TestDetectStateChanges
# ---------------------------------------------------------------------------


class TestDetectStateChanges:
    def test_domain_expansion_detected(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY], [_CUSTOMER_LINK])
        changes = detect_state_changes(nodes)
        types = [c["change_type"] for c in changes]
        assert "domain_expansion" in types

    def test_single_silo_no_expansion(self):
        entity = {"canonical": "misc", "entity_type": "concept", "domains": ["sales"], "confidence": 0.5}
        nodes = build_world_nodes([entity], [])
        changes = detect_state_changes(nodes)
        expansion = [c for c in changes if c["change_type"] == "domain_expansion"]
        assert len(expansion) == 0

    def test_high_confidence_change_detected(self):
        # customer with silo_count=3 and confidence=0.85 → node_confidence > 0.90
        nodes = build_world_nodes([_CUSTOMER_ENTITY], [_CUSTOMER_LINK])
        changes = detect_state_changes(nodes)
        types = [c["change_type"] for c in changes]
        assert "high_confidence_node" in types

    def test_empty_nodes_no_changes(self):
        assert detect_state_changes([]) == []

    def test_change_fields_complete(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY], [_CUSTOMER_LINK])
        changes = detect_state_changes(nodes)
        for change in changes:
            assert "entity" in change
            assert "change_type" in change
            assert "description" in change


# ---------------------------------------------------------------------------
# TestBuildGraphSummary
# ---------------------------------------------------------------------------


class TestBuildGraphSummary:
    def test_counts_match_nodes_and_edges(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY, _REVENUE_ENTITY], [_CUSTOMER_LINK])
        edges = build_entity_edges(nodes)
        changes = detect_state_changes(nodes)
        summary = build_graph_summary(nodes, edges, changes)
        assert summary["total_nodes"] == len(nodes)
        assert summary["total_edges"] == len(edges)

    def test_cross_silo_count(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY, _PROJECT_ENTITY], [_CUSTOMER_LINK])
        edges = build_entity_edges(nodes)
        changes = detect_state_changes(nodes)
        summary = build_graph_summary(nodes, edges, changes)
        # customer has silo_count=3, project has silo_count=2 (len domains)
        assert summary["cross_silo_nodes"] == 2

    def test_expansion_events_counted(self):
        nodes = build_world_nodes([_CUSTOMER_ENTITY, _REVENUE_ENTITY], [_CUSTOMER_LINK, _REVENUE_LINK])
        edges = build_entity_edges(nodes)
        changes = detect_state_changes(nodes)
        summary = build_graph_summary(nodes, edges, changes)
        assert summary["expansion_events"] >= 1

    def test_empty_graph_summary(self):
        summary = build_graph_summary([], [], [])
        assert summary["total_nodes"] == 0
        assert summary["total_edges"] == 0
        assert summary["cross_silo_nodes"] == 0


# ---------------------------------------------------------------------------
# TestWorldModelHeuristic
# ---------------------------------------------------------------------------


@pytest.fixture
def agent():
    return WorldModelAgent()


def _ctx(
    *,
    enrichment: dict | None = None,
    content: str = "customer deal closed this quarter with revenue impact",
) -> dict:
    return {
        "memory": {"content": content, "enrichment": enrichment or {}},
        "runtime": {"job_id": "job-99"},
    }


@pytest.fixture
def mock_settings():
    with patch("app.agents.world_model_agent.settings") as m:
        m.WORLD_MODEL_STRATEGY = "heuristic"
        m.AGENT_STRATEGY = "heuristic"
        m.OLLAMA_BASE_URL = "http://localhost:11434"
        m.OLLAMA_MODEL = "llama3.1:8b"
        m.OLLAMA_TIMEOUT_SECONDS = 5.0
        m.OLLAMA_MAX_CONCURRENCY = 2
        yield m


class TestWorldModelHeuristic:
    @pytest.mark.asyncio
    async def test_basic_resolved_entities(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY, _REVENUE_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK],
            }
        )
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert len(result.outputs["world_nodes"]) == 2

    @pytest.mark.asyncio
    async def test_all_required_output_keys(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK],
            }
        )
        result = await agent.run("mem-1", ctx)
        for key in ("world_nodes", "entity_edges", "state_changes", "graph_summary", "confidence", "rationale"):
            assert key in result.outputs

    @pytest.mark.asyncio
    async def test_empty_enrichment_succeeds(self, mock_settings, agent):
        ctx = _ctx(enrichment={})
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["world_nodes"] == []
        assert result.outputs["entity_edges"] == []

    @pytest.mark.asyncio
    async def test_confidence_zero_entities(self, mock_settings, agent):
        ctx = _ctx(enrichment={"resolved_entities": [], "cross_silo_links": []})
        result = await agent.run("mem-1", ctx)
        assert result.outputs["confidence"] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_confidence_scales_with_cross_silo(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK],
            }
        )
        result = await agent.run("mem-1", ctx)
        assert result.outputs["confidence"] > 0.4

    @pytest.mark.asyncio
    async def test_rationale_heuristic(self, mock_settings, agent):
        ctx = _ctx(enrichment={"resolved_entities": [_CUSTOMER_ENTITY], "cross_silo_links": []})
        result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_set(self, mock_settings, agent):
        ctx = _ctx(enrichment={})
        result = await agent.run("mem-1", ctx)
        assert result.trace_id == "job-99"

    @pytest.mark.asyncio
    async def test_no_content_uses_heuristic(self, agent):
        with patch("app.agents.world_model_agent.settings") as m:
            m.WORLD_MODEL_STRATEGY = "llm"
            m.AGENT_STRATEGY = "llm"
            ctx = _ctx(
                enrichment={"resolved_entities": [_CUSTOMER_ENTITY], "cross_silo_links": []},
                content="",
            )
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_world_nodes_deduplicated(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY, _CUSTOMER_ENTITY],
                "cross_silo_links": [],
            }
        )
        result = await agent.run("mem-1", ctx)
        assert len(result.outputs["world_nodes"]) == 1

    @pytest.mark.asyncio
    async def test_entity_edges_between_pairs(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY, _REVENUE_ENTITY, _PROJECT_ENTITY],
                "cross_silo_links": [],
            }
        )
        result = await agent.run("mem-1", ctx)
        assert len(result.outputs["entity_edges"]) == 3

    @pytest.mark.asyncio
    async def test_state_changes_domain_expansion(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK],
            }
        )
        result = await agent.run("mem-1", ctx)
        types = [c["change_type"] for c in result.outputs["state_changes"]]
        assert "domain_expansion" in types

    @pytest.mark.asyncio
    async def test_graph_summary_totals_correct(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY, _REVENUE_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK, _REVENUE_LINK],
            }
        )
        result = await agent.run("mem-1", ctx)
        summary = result.outputs["graph_summary"]
        assert summary["total_nodes"] == len(result.outputs["world_nodes"])
        assert summary["total_edges"] == len(result.outputs["entity_edges"])

    @pytest.mark.asyncio
    async def test_missing_runtime_no_error(self, mock_settings, agent):
        ctx = {"memory": {"content": "test", "enrichment": {}}}
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.trace_id is None


# ---------------------------------------------------------------------------
# TestValidateOutputs
# ---------------------------------------------------------------------------


class TestValidateOutputs:
    def test_valid_outputs_passes(self):
        agent = WorldModelAgent()
        result = _make_result(
            {
                "world_nodes": [{"entity": "customer", "entity_type": "role"}],
                "entity_edges": [{"from_entity": "customer", "to_entity": "revenue", "relationship_type": "co_occurrence", "weight": 0.8}],
                "state_changes": [],
                "graph_summary": {"total_nodes": 1, "total_edges": 1, "cross_silo_nodes": 1, "high_confidence_nodes": 0, "expansion_events": 0},
                "confidence": 0.7,
                "rationale": "heuristic",
            }
        )
        agent.validate_outputs(result)  # no exception

    def test_world_nodes_not_list_raises(self):
        agent = WorldModelAgent()
        result = _make_result({"world_nodes": "bad", "entity_edges": [], "state_changes": [], "graph_summary": {}})
        with pytest.raises(ValueError, match="world_nodes must be a list"):
            agent.validate_outputs(result)

    def test_entity_edges_not_list_raises(self):
        agent = WorldModelAgent()
        result = _make_result({"world_nodes": [], "entity_edges": "bad", "state_changes": [], "graph_summary": {}})
        with pytest.raises(ValueError, match="entity_edges must be a list"):
            agent.validate_outputs(result)

    def test_state_changes_not_list_raises(self):
        agent = WorldModelAgent()
        result = _make_result({"world_nodes": [], "entity_edges": [], "state_changes": "bad", "graph_summary": {}})
        with pytest.raises(ValueError, match="state_changes must be a list"):
            agent.validate_outputs(result)

    def test_graph_summary_not_dict_raises(self):
        agent = WorldModelAgent()
        result = _make_result({"world_nodes": [], "entity_edges": [], "state_changes": [], "graph_summary": "bad"})
        with pytest.raises(ValueError, match="graph_summary must be a dict"):
            agent.validate_outputs(result)

    def test_world_node_missing_entity_raises(self):
        agent = WorldModelAgent()
        result = _make_result({
            "world_nodes": [{"entity_type": "role"}],
            "entity_edges": [], "state_changes": [], "graph_summary": {},
        })
        with pytest.raises(ValueError, match="entity and entity_type"):
            agent.validate_outputs(result)

    def test_entity_edge_missing_from_raises(self):
        agent = WorldModelAgent()
        result = _make_result({
            "world_nodes": [],
            "entity_edges": [{"to_entity": "revenue"}],
            "state_changes": [], "graph_summary": {},
        })
        with pytest.raises(ValueError, match="from_entity and to_entity"):
            agent.validate_outputs(result)

    def test_entity_edge_missing_to_raises(self):
        agent = WorldModelAgent()
        result = _make_result({
            "world_nodes": [],
            "entity_edges": [{"from_entity": "customer"}],
            "state_changes": [], "graph_summary": {},
        })
        with pytest.raises(ValueError, match="from_entity and to_entity"):
            agent.validate_outputs(result)

    def test_non_success_status_skips_validation(self):
        agent = WorldModelAgent()
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="WorldModelAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )
        agent.validate_outputs(result)  # should not raise


# ---------------------------------------------------------------------------
# TestWorldModelLLM
# ---------------------------------------------------------------------------


class TestWorldModelLLM:
    @pytest.mark.asyncio
    async def test_good_llm_response_used(self, agent):
        good_resp = {
            "world_nodes": [
                {"entity": "customer", "entity_type": "role", "domains": ["sales"], "silo_count": 2, "state_version": 2, "node_confidence": 0.85}
            ],
            "entity_edges": [],
            "state_changes": [{"entity": "customer", "change_type": "domain_expansion", "description": "customer across silos"}],
            "graph_summary": {"total_nodes": 1, "total_edges": 0, "cross_silo_nodes": 1, "high_confidence_nodes": 1, "expansion_events": 1},
            "confidence": 0.8,
            "rationale": "llm",
        }
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=good_resp)
        with patch("app.agents.world_model_agent.settings") as m, \
             patch("app.agents.world_model_agent.create_ollama_client", return_value=mock_client):
            m.WORLD_MODEL_STRATEGY = "llm"
            m.AGENT_STRATEGY = "llm"
            m.OLLAMA_BASE_URL = "http://localhost:11434"
            m.OLLAMA_MODEL = "llama3.1:8b"
            m.OLLAMA_TIMEOUT_SECONDS = 5.0
            m.OLLAMA_MAX_CONCURRENCY = 2
            ctx = _ctx(
                enrichment={
                    "resolved_entities": [_CUSTOMER_ENTITY],
                    "cross_silo_links": [_CUSTOMER_LINK],
                }
            )
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "llm"
        assert len(result.outputs["world_nodes"]) == 1

    @pytest.mark.asyncio
    async def test_bad_llm_response_falls_back(self, agent):
        with patch("app.agents.world_model_agent.settings") as m, \
             patch("app.agents.world_model_agent.create_ollama_client") as mock_factory:
            m.WORLD_MODEL_STRATEGY = "llm"
            m.AGENT_STRATEGY = "llm"
            m.OLLAMA_BASE_URL = "http://localhost:11434"
            m.OLLAMA_MODEL = "llama3.1:8b"
            m.OLLAMA_TIMEOUT_SECONDS = 5.0
            m.OLLAMA_MAX_CONCURRENCY = 2
            mock_client = AsyncMock()
            mock_client.complete_json = AsyncMock(return_value={"bad": "response"})
            mock_factory.return_value = mock_client
            ctx = _ctx(
                enrichment={
                    "resolved_entities": [_CUSTOMER_ENTITY],
                    "cross_silo_links": [_CUSTOMER_LINK],
                }
            )
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_missing_list_field_falls_back(self, agent):
        partial = {
            "world_nodes": [{"entity": "customer", "entity_type": "role"}],
            "entity_edges": "not-a-list",
            "state_changes": [],
        }
        with patch("app.agents.world_model_agent.settings") as m, \
             patch("app.agents.world_model_agent.create_ollama_client") as mock_factory:
            m.WORLD_MODEL_STRATEGY = "llm"
            m.AGENT_STRATEGY = "llm"
            m.OLLAMA_BASE_URL = "http://localhost:11434"
            m.OLLAMA_MODEL = "llama3.1:8b"
            m.OLLAMA_TIMEOUT_SECONDS = 5.0
            m.OLLAMA_MAX_CONCURRENCY = 2
            mock_client = AsyncMock()
            mock_client.complete_json = AsyncMock(return_value=partial)
            mock_factory.return_value = mock_client
            ctx = _ctx(
                enrichment={"resolved_entities": [_CUSTOMER_ENTITY], "cross_silo_links": []}
            )
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# TestAgentMetadata
# ---------------------------------------------------------------------------


class TestAgentMetadata:
    def test_name_and_version(self):
        agent = WorldModelAgent()
        assert agent.name == "WorldModelAgent"
        assert agent.version == "v1"

    def test_dependencies(self):
        agent = WorldModelAgent()
        assert agent.dependencies() == ["EntityResolutionAgent"]


# ---------------------------------------------------------------------------
# TestRegistry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_short_name(self):
        from app.agents.registry import get_agent
        assert isinstance(get_agent("world_model"), WorldModelAgent)

    def test_full_name(self):
        from app.agents.registry import get_agent
        assert isinstance(get_agent("WorldModelAgent"), WorldModelAgent)

    def test_camel_case(self):
        from app.agents.registry import get_agent
        assert isinstance(get_agent("worldmodel"), WorldModelAgent)
