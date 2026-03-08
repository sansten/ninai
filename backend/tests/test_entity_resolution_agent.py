"""Tests for EntityResolutionAgent and helpers — Phase 7."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.entity_resolution_agent import (
    EntityResolutionAgent,
    build_cross_silo_links,
    extract_entities_from_content,
    resolve_entity,
)
from app.agents.types import AgentContext, AgentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(
    content: str = "",
    relationships: list[dict] | None = None,
) -> AgentContext:
    enrichment = {}
    if relationships is not None:
        enrichment["semantic_relationships"] = relationships
    return {
        "tenant": {"org_id": "org-1", "org_slug": None},
        "actor": {"user_id": "user-1", "roles": []},
        "memory": {"id": "mem-1", "content": content, "enrichment": enrichment},
        "runtime": {"job_id": "trace-1", "attempt": 1},
    }


def _fake_now() -> datetime:
    return datetime(2026, 3, 8, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Unit: resolve_entity — exact matches
# ---------------------------------------------------------------------------

class TestResolveEntityExact:
    def test_canonical_name_exact(self):
        r = resolve_entity("customer")
        assert r is not None
        assert r["canonical"] == "customer"
        assert r["match_type"] == "exact"
        assert r["confidence"] == 1.0

    def test_canonical_name_case_insensitive(self):
        r = resolve_entity("Customer")
        assert r is not None
        assert r["canonical"] == "customer"
        assert r["match_type"] == "exact"

    def test_alias_match(self):
        r = resolve_entity("client")
        assert r is not None
        assert r["canonical"] == "customer"
        assert r["match_type"] == "alias"
        assert r["confidence"] == 0.85

    def test_alias_account(self):
        r = resolve_entity("account")
        assert r is not None
        assert r["canonical"] == "customer"

    def test_alias_arr(self):
        r = resolve_entity("arr")
        assert r is not None
        assert r["canonical"] == "revenue"
        assert r["match_type"] == "alias"

    def test_alias_nda(self):
        r = resolve_entity("nda")
        assert r is not None
        assert r["canonical"] == "contract"

    def test_alias_outage(self):
        r = resolve_entity("outage")
        assert r is not None
        assert r["canonical"] == "incident"

    def test_unresolved_returns_none(self):
        assert resolve_entity("xyzzy_not_in_ontology") is None

    def test_empty_string_returns_none(self):
        assert resolve_entity("") is None

    def test_whitespace_only_returns_none(self):
        assert resolve_entity("   ") is None


# ---------------------------------------------------------------------------
# Unit: resolve_entity — extra ontology injection
# ---------------------------------------------------------------------------

class TestResolveEntityExtraOntology:
    def test_extra_ontology_canonical(self):
        extra = [
            {
                "canonical": "acme corp",
                "entity_type": "organization",
                "aliases": ["acme", "acme inc"],
                "domains": ["sales", "finance"],
            }
        ]
        r = resolve_entity("Acme Corp", extra_ontology=extra)
        assert r is not None
        assert r["canonical"] == "acme corp"
        assert r["match_type"] == "exact"

    def test_extra_ontology_alias(self):
        extra = [
            {
                "canonical": "acme corp",
                "entity_type": "organization",
                "aliases": ["acme", "acme inc"],
                "domains": ["sales"],
            }
        ]
        r = resolve_entity("Acme", extra_ontology=extra)
        assert r is not None
        assert r["canonical"] == "acme corp"
        assert r["match_type"] == "alias"

    def test_extra_ontology_does_not_break_builtin(self):
        extra = [{"canonical": "widget", "entity_type": "product", "aliases": [], "domains": []}]
        r = resolve_entity("customer", extra_ontology=extra)
        assert r is not None
        assert r["canonical"] == "customer"

    def test_extra_ontology_unresolved_stays_none(self):
        extra = [{"canonical": "widget", "entity_type": "product", "aliases": [], "domains": []}]
        assert resolve_entity("completely_unknown", extra_ontology=extra) is None


# ---------------------------------------------------------------------------
# Unit: extract_entities_from_content
# ---------------------------------------------------------------------------

class TestExtractEntitiesFromContent:
    def test_extracts_capitalized_noun(self):
        entities = extract_entities_from_content("Acme Corp signed the NDA.")
        assert "Acme Corp" in entities or "Acme" in entities

    def test_empty_content_returns_empty(self):
        assert extract_entities_from_content("") == []

    def test_caps_at_twenty(self):
        text = " ".join(f"Entity{i} Corp" for i in range(30))
        assert len(extract_entities_from_content(text)) <= 20

    def test_deduplicates(self):
        text = "Acme Corp and Acme Corp are the same."
        entities = extract_entities_from_content(text)
        # "Acme Corp" should appear only once
        assert entities.count("Acme Corp") <= 1

    def test_ignores_lowercase_only(self):
        entities = extract_entities_from_content("no capitals here at all")
        assert entities == []


# ---------------------------------------------------------------------------
# Unit: build_cross_silo_links
# ---------------------------------------------------------------------------

class TestBuildCrossSimoLinks:
    def test_multi_domain_entity_flagged(self):
        resolved = [
            {
                "original": "client",
                "canonical": "customer",
                "entity_type": "role",
                "domains": ["sales", "finance", "support"],
                "match_type": "alias",
                "confidence": 0.85,
            }
        ]
        links = build_cross_silo_links(resolved)
        assert len(links) == 1
        assert links[0]["canonical"] == "customer"
        assert links[0]["silo_count"] == 3

    def test_single_domain_not_flagged(self):
        resolved = [
            {
                "original": "sprint",
                "canonical": "sprint",
                "entity_type": "time_period",
                "domains": ["product"],
                "match_type": "exact",
                "confidence": 1.0,
            }
        ]
        links = build_cross_silo_links(resolved)
        assert links == []

    def test_deduplicates_same_canonical(self):
        resolved = [
            {
                "original": "client",
                "canonical": "customer",
                "entity_type": "role",
                "domains": ["sales", "finance"],
                "match_type": "alias",
                "confidence": 0.85,
            },
            {
                "original": "account",
                "canonical": "customer",
                "entity_type": "role",
                "domains": ["sales", "finance"],
                "match_type": "alias",
                "confidence": 0.85,
            },
        ]
        links = build_cross_silo_links(resolved)
        assert len(links) == 1

    def test_empty_resolved_returns_empty(self):
        assert build_cross_silo_links([]) == []


# ---------------------------------------------------------------------------
# Agent: heuristic mode
# ---------------------------------------------------------------------------

class TestEntityResolutionAgentHeuristic:
    @pytest.fixture
    def agent(self) -> EntityResolutionAgent:
        return EntityResolutionAgent()

    @pytest.mark.asyncio
    async def test_resolves_customer_from_relationship(self, agent):
        rels = [{"type": "mentions_concept", "concept": "client"}]
        with patch("app.agents.entity_resolution_agent.settings") as mock_s:
            mock_s.ENTITY_RESOLUTION_STRATEGY = "heuristic"
            mock_s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _make_context("", relationships=rels))

        assert result.status == "success"
        canonicals = {e["canonical"] for e in result.outputs["resolved_entities"]}
        assert "customer" in canonicals

    @pytest.mark.asyncio
    async def test_resolves_arr_to_revenue(self, agent):
        rels = [{"type": "mentions_concept", "concept": "arr"}]
        with patch("app.agents.entity_resolution_agent.settings") as mock_s:
            mock_s.ENTITY_RESOLUTION_STRATEGY = "heuristic"
            mock_s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _make_context("", relationships=rels))

        canonicals = {e["canonical"] for e in result.outputs["resolved_entities"]}
        assert "revenue" in canonicals

    @pytest.mark.asyncio
    async def test_cross_silo_link_surfaced(self, agent):
        rels = [{"type": "mentions_concept", "concept": "client"}]
        with patch("app.agents.entity_resolution_agent.settings") as mock_s:
            mock_s.ENTITY_RESOLUTION_STRATEGY = "heuristic"
            mock_s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _make_context("", relationships=rels))

        links = result.outputs["cross_silo_links"]
        assert any(link["canonical"] == "customer" for link in links)

    @pytest.mark.asyncio
    async def test_empty_content_returns_success(self, agent):
        with patch("app.agents.entity_resolution_agent.settings") as mock_s:
            mock_s.ENTITY_RESOLUTION_STRATEGY = "heuristic"
            mock_s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _make_context(""))

        assert result.status == "success"
        assert result.outputs["resolved_entities"] == []
        assert result.outputs["unresolved_entities"] == []

    @pytest.mark.asyncio
    async def test_output_structure(self, agent):
        with patch("app.agents.entity_resolution_agent.settings") as mock_s:
            mock_s.ENTITY_RESOLUTION_STRATEGY = "heuristic"
            mock_s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _make_context("Some content mentioning a client."))

        assert "resolved_entities" in result.outputs
        assert "unresolved_entities" in result.outputs
        assert "cross_silo_links" in result.outputs
        assert "confidence" in result.outputs
        assert "rationale" in result.outputs
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_confidence_bounded(self, agent):
        rels = [
            {"type": "x", "concept": "client"},
            {"type": "x", "concept": "arr"},
            {"type": "x", "concept": "nda"},
        ]
        with patch("app.agents.entity_resolution_agent.settings") as mock_s:
            mock_s.ENTITY_RESOLUTION_STRATEGY = "heuristic"
            mock_s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _make_context("", relationships=rels))

        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_unknown_entity_in_unresolved(self, agent):
        rels = [{"type": "x", "concept": "ZorpInc"}]
        with patch("app.agents.entity_resolution_agent.settings") as mock_s:
            mock_s.ENTITY_RESOLUTION_STRATEGY = "heuristic"
            mock_s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _make_context("", relationships=rels))

        assert "ZorpInc" in result.outputs["unresolved_entities"]

    @pytest.mark.asyncio
    async def test_content_based_extraction(self, agent):
        content = "The Budget must be approved before the Deployment can proceed."
        with patch("app.agents.entity_resolution_agent.settings") as mock_s:
            mock_s.ENTITY_RESOLUTION_STRATEGY = "heuristic"
            mock_s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _make_context(content))

        canonicals = {e["canonical"] for e in result.outputs["resolved_entities"]}
        assert "budget" in canonicals or "deployment" in canonicals


# ---------------------------------------------------------------------------
# Agent: validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _make_result(self, outputs: dict) -> AgentResult:
        now = _fake_now()
        return AgentResult(
            agent_name="EntityResolutionAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="success",
            confidence=0.7,
            outputs=outputs,
            started_at=now,
            finished_at=now,
        )

    def test_valid_outputs(self):
        agent = EntityResolutionAgent()
        result = self._make_result({
            "resolved_entities": [
                {"canonical": "customer", "entity_type": "role", "domains": [], "original": "client",
                 "match_type": "alias", "confidence": 0.85}
            ],
            "unresolved_entities": [],
            "cross_silo_links": [],
        })
        agent.validate_outputs(result)  # no exception

    def test_resolved_entities_not_list_raises(self):
        agent = EntityResolutionAgent()
        result = self._make_result({
            "resolved_entities": "not_a_list",
            "unresolved_entities": [],
            "cross_silo_links": [],
        })
        with pytest.raises(ValueError, match="must be a list"):
            agent.validate_outputs(result)

    def test_cross_silo_links_not_list_raises(self):
        agent = EntityResolutionAgent()
        result = self._make_result({
            "resolved_entities": [],
            "unresolved_entities": [],
            "cross_silo_links": "bad",
        })
        with pytest.raises(ValueError, match="must be a list"):
            agent.validate_outputs(result)

    def test_missing_canonical_raises(self):
        agent = EntityResolutionAgent()
        result = self._make_result({
            "resolved_entities": [{"entity_type": "role"}],  # no canonical
            "unresolved_entities": [],
            "cross_silo_links": [],
        })
        with pytest.raises(ValueError, match="canonical"):
            agent.validate_outputs(result)

    def test_skipped_status_not_validated(self):
        agent = EntityResolutionAgent()
        now = _fake_now()
        result = AgentResult(
            agent_name="EntityResolutionAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="skipped",
            confidence=0.0,
            outputs={},
            started_at=now,
            finished_at=now,
        )
        agent.validate_outputs(result)  # no exception


# ---------------------------------------------------------------------------
# Agent: LLM mode
# ---------------------------------------------------------------------------

class TestEntityResolutionAgentLLM:
    @pytest.mark.asyncio
    async def test_llm_good_response_used(self):
        agent = EntityResolutionAgent()
        mock_resp = {
            "resolved_entities": [
                {
                    "original": "Acme Corp",
                    "canonical": "customer",
                    "entity_type": "role",
                    "domains": ["sales"],
                    "match_type": "alias",
                    "confidence": 0.9,
                }
            ],
            "unresolved_entities": [],
            "cross_silo_links": [],
            "confidence": 0.9,
            "rationale": "llm",
        }
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=mock_resp)

        with patch("app.agents.entity_resolution_agent.settings") as mock_s, \
             patch("app.agents.entity_resolution_agent.create_ollama_client", return_value=mock_client):
            mock_s.ENTITY_RESOLUTION_STRATEGY = None
            mock_s.AGENT_STRATEGY = "llm"
            mock_s.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_s.OLLAMA_MODEL = "llama3.1:8b"
            mock_s.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _make_context("Acme Corp renewed their contract."))

        assert result.confidence == 0.9
        assert result.outputs["resolved_entities"][0]["canonical"] == "customer"

    @pytest.mark.asyncio
    async def test_llm_bad_response_falls_back_to_heuristic(self):
        agent = EntityResolutionAgent()
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "response"})

        with patch("app.agents.entity_resolution_agent.settings") as mock_s, \
             patch("app.agents.entity_resolution_agent.create_ollama_client", return_value=mock_client):
            mock_s.ENTITY_RESOLUTION_STRATEGY = None
            mock_s.AGENT_STRATEGY = "llm"
            mock_s.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_s.OLLAMA_MODEL = "llama3.1:8b"
            mock_s.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_s.OLLAMA_MAX_CONCURRENCY = 2
            rels = [{"type": "x", "concept": "client"}]
            result = await agent.run("mem-1", _make_context("", relationships=rels))

        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_missing_list_field_falls_back(self):
        agent = EntityResolutionAgent()
        mock_client = AsyncMock()
        # missing cross_silo_links → invalid
        mock_client.complete_json = AsyncMock(return_value={
            "resolved_entities": [],
            "unresolved_entities": [],
            # cross_silo_links absent
            "confidence": 0.8,
        })

        with patch("app.agents.entity_resolution_agent.settings") as mock_s, \
             patch("app.agents.entity_resolution_agent.create_ollama_client", return_value=mock_client):
            mock_s.ENTITY_RESOLUTION_STRATEGY = None
            mock_s.AGENT_STRATEGY = "llm"
            mock_s.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_s.OLLAMA_MODEL = "llama3.1:8b"
            mock_s.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _make_context("Budget review meeting."))

        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# Agent: metadata
# ---------------------------------------------------------------------------

class TestAgentMetadata:
    def test_name_and_version(self):
        agent = EntityResolutionAgent()
        assert agent.name == "EntityResolutionAgent"
        assert agent.version == "v1"

    def test_dependencies(self):
        agent = EntityResolutionAgent()
        assert "SemanticNormalizationAgent" in agent.dependencies()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered_by_short_name(self):
        from app.agents.registry import get_agent
        agent = get_agent("entity_resolution")
        assert isinstance(agent, EntityResolutionAgent)

    def test_registered_by_full_name(self):
        from app.agents.registry import get_agent
        agent = get_agent("EntityResolutionAgent")
        assert isinstance(agent, EntityResolutionAgent)

    def test_registered_by_camel_case(self):
        from app.agents.registry import get_agent
        agent = get_agent("entityresolution")
        assert isinstance(agent, EntityResolutionAgent)
