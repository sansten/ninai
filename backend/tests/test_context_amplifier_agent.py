"""Tests for Phase 8 — ContextAmplifierAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.context_amplifier_agent import (
    ContextAmplifierAgent,
    amplify_context,
    get_relevant_domains,
)
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CUSTOMER_ENTITY = {
    "original": "client",
    "canonical": "customer",
    "entity_type": "role",
    "domains": ["sales", "finance", "support"],
    "match_type": "alias",
    "confidence": 0.85,
}

_BUDGET_ENTITY = {
    "original": "budget",
    "canonical": "budget",
    "entity_type": "metric",
    "domains": ["finance", "operations", "engineering"],
    "match_type": "exact",
    "confidence": 1.0,
}

_CUSTOMER_LINK = {
    "canonical": "customer",
    "entity_type": "role",
    "domains": ["sales", "finance", "support"],
    "silo_count": 3,
}

_BUDGET_LINK = {
    "canonical": "budget",
    "entity_type": "metric",
    "domains": ["finance", "operations", "engineering"],
    "silo_count": 3,
}


def _make_result(outputs: dict) -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="ContextAmplifierAgent",
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
# TestGetRelevantDomains
# ---------------------------------------------------------------------------

class TestGetRelevantDomains:
    def test_engineer_role(self):
        domains = get_relevant_domains(["engineer"])
        assert "engineering" in domains
        assert "operations" in domains
        assert "product" in domains

    def test_cfo_role(self):
        domains = get_relevant_domains(["cfo"])
        assert "finance" in domains
        assert "sales" in domains
        assert "hr" in domains
        assert "legal" in domains

    def test_multiple_roles_union(self):
        domains = get_relevant_domains(["engineer", "sales"])
        assert "engineering" in domains
        assert "sales" in domains
        assert "finance" in domains  # from sales role

    def test_unknown_role_returns_all(self):
        domains = get_relevant_domains(["unknown_role"])
        assert len(domains) >= 9  # all domains

    def test_empty_roles_returns_all(self):
        domains = get_relevant_domains([])
        assert len(domains) >= 9

    def test_case_insensitive(self):
        lower = get_relevant_domains(["engineer"])
        upper = get_relevant_domains(["Engineer"])
        assert set(lower) == set(upper)

    def test_sorted_output(self):
        domains = get_relevant_domains(["cfo"])
        assert domains == sorted(domains)


# ---------------------------------------------------------------------------
# TestAmplifyContext
# ---------------------------------------------------------------------------

class TestAmplifyContext:
    def test_cross_silo_entity_emits_foreign_domain_signals(self):
        bundle, amplified, touched = amplify_context(
            resolved_entities=[_CUSTOMER_ENTITY],
            cross_silo_links=[_CUSTOMER_LINK],
            primary_domain="support",
            relevant_domains=["support", "sales", "finance"],
        )
        domains_in_bundle = {item["domain"] for item in bundle if item["source"] == "cross_silo"}
        assert "sales" in domains_in_bundle
        assert "finance" in domains_in_bundle
        # primary domain not in cross-silo signals
        assert "support" not in domains_in_bundle

    def test_primary_domain_included_as_resolved_signal(self):
        bundle, _, _ = amplify_context(
            resolved_entities=[_CUSTOMER_ENTITY],
            cross_silo_links=[_CUSTOMER_LINK],
            primary_domain="support",
            relevant_domains=["support", "sales", "finance"],
        )
        resolved_signals = [item for item in bundle if item["source"] == "resolved"]
        assert len(resolved_signals) >= 1
        assert resolved_signals[0]["domain"] == "support"

    def test_entity_outside_relevant_domains_skipped(self):
        bundle, amplified, _ = amplify_context(
            resolved_entities=[_CUSTOMER_ENTITY],
            cross_silo_links=[_CUSTOMER_LINK],
            primary_domain="hr",
            relevant_domains=["hr"],  # customer doesn't span hr
        )
        cross_signals = [item for item in bundle if item["source"] == "cross_silo"]
        assert len(cross_signals) == 0
        assert "customer" not in amplified

    def test_amplified_entities_populated(self):
        _, amplified, _ = amplify_context(
            resolved_entities=[_CUSTOMER_ENTITY],
            cross_silo_links=[_CUSTOMER_LINK],
            primary_domain="support",
            relevant_domains=["support", "sales", "finance"],
        )
        assert "customer" in amplified

    def test_domains_touched_populated(self):
        _, _, touched = amplify_context(
            resolved_entities=[_CUSTOMER_ENTITY],
            cross_silo_links=[_CUSTOMER_LINK],
            primary_domain="support",
            relevant_domains=["support", "sales", "finance"],
        )
        assert "sales" in touched
        assert "finance" in touched
        assert "support" in touched

    def test_empty_inputs_returns_empty(self):
        bundle, amplified, touched = amplify_context(
            resolved_entities=[],
            cross_silo_links=[],
            primary_domain="general",
            relevant_domains=["engineering"],
        )
        assert bundle == []
        assert amplified == []
        assert touched == []

    def test_bundle_capped_at_50(self):
        # 20 entities each spanning 4 domains → 80+ signals → capped
        entities = [
            {
                "original": f"entity{i}",
                "canonical": f"entity{i}",
                "entity_type": "concept",
                "domains": ["engineering", "sales", "finance", "hr"],
                "match_type": "exact",
                "confidence": 1.0,
            }
            for i in range(20)
        ]
        links = [
            {
                "canonical": f"entity{i}",
                "entity_type": "concept",
                "domains": ["engineering", "sales", "finance", "hr"],
                "silo_count": 4,
            }
            for i in range(20)
        ]
        bundle, _, _ = amplify_context(
            resolved_entities=entities,
            cross_silo_links=links,
            primary_domain="engineering",
            relevant_domains=["engineering", "sales", "finance", "hr"],
        )
        assert len(bundle) <= 50

    def test_silo_count_preserved_in_bundle(self):
        bundle, _, _ = amplify_context(
            resolved_entities=[_CUSTOMER_ENTITY],
            cross_silo_links=[_CUSTOMER_LINK],
            primary_domain="support",
            relevant_domains=["support", "sales", "finance"],
        )
        cross_signals = [item for item in bundle if item["source"] == "cross_silo"]
        assert all(item["silo_count"] == 3 for item in cross_signals)

    def test_signal_text_format(self):
        bundle, _, _ = amplify_context(
            resolved_entities=[_CUSTOMER_ENTITY],
            cross_silo_links=[_CUSTOMER_LINK],
            primary_domain="support",
            relevant_domains=["support", "sales"],
        )
        cross_signals = [item for item in bundle if item["source"] == "cross_silo"]
        assert len(cross_signals) >= 1
        assert "customer" in cross_signals[0]["signal"]
        assert "sales" in cross_signals[0]["signal"]

    def test_multiple_entities_multiple_domains(self):
        bundle, amplified, touched = amplify_context(
            resolved_entities=[_CUSTOMER_ENTITY, _BUDGET_ENTITY],
            cross_silo_links=[_CUSTOMER_LINK, _BUDGET_LINK],
            primary_domain="finance",
            relevant_domains=["finance", "sales", "engineering", "operations"],
        )
        assert "customer" in amplified
        assert "budget" in amplified
        assert "sales" in touched


# ---------------------------------------------------------------------------
# TestContextAmplifierHeuristic
# ---------------------------------------------------------------------------

class TestContextAmplifierHeuristic:
    @pytest.fixture
    def agent(self):
        return ContextAmplifierAgent()

    def _ctx(self, enrichment=None, roles=None, content="Some content"):
        return {
            "memory": {
                "content": content,
                "enrichment": enrichment or {},
            },
            "actor": {"roles": roles or []},
        }

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    async def test_resolves_cross_silo_from_enrichment(self, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "heuristic"
        ctx = self._ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK],
                "business_domain": "support",
            },
            roles=["support"],
        )
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        bundle = result.outputs["context_bundle"]
        assert any(item["entity"] == "customer" for item in bundle)

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    async def test_role_scoped_filtering(self, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "heuristic"
        ctx = self._ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK],
                "business_domain": "support",
            },
            roles=["sre"],  # sre cares about engineering/operations � no overlap with customer's sales/finance/support
        )
        result = await agent.run("mem-1", ctx)
        cross_signals = [
            item for item in result.outputs["context_bundle"]
            if item["source"] == "cross_silo"
        ]
        # customer spans sales/finance/support — none in hr's relevant domains
        assert len(cross_signals) == 0

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    async def test_empty_enrichment_returns_empty_bundle(self, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "heuristic"
        ctx = self._ctx(enrichment={}, content="")
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["context_bundle"] == []
        assert result.outputs["amplified_entities"] == []

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    async def test_output_has_all_required_keys(self, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "heuristic"
        ctx = self._ctx()
        result = await agent.run("mem-1", ctx)
        for key in ("context_bundle", "amplified_entities", "domains_touched", "confidence", "rationale"):
            assert key in result.outputs

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    async def test_confidence_in_bounds(self, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "heuristic"
        ctx = self._ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY, _BUDGET_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK, _BUDGET_LINK],
                "business_domain": "finance",
            },
            roles=["cfo"],
        )
        result = await agent.run("mem-1", ctx)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    async def test_primary_domain_from_enrichment(self, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "heuristic"
        ctx = self._ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK],
                "business_domain": "sales",
            },
            roles=["sales"],
        )
        result = await agent.run("mem-1", ctx)
        # Cross-silo signals should not include sales (primary domain)
        cross_signals = [
            item for item in result.outputs["context_bundle"]
            if item["source"] == "cross_silo"
        ]
        assert all(item["domain"] != "sales" for item in cross_signals)

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    async def test_actor_roles_consumed_from_context(self, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "heuristic"
        ctx = self._ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK],
                "business_domain": "support",
            },
            roles=["sales"],  # sales cares about finance too
        )
        result = await agent.run("mem-1", ctx)
        touched = result.outputs["domains_touched"]
        assert "finance" in touched

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    async def test_no_roles_surfaces_all_domains(self, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "heuristic"
        ctx = self._ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK],
                "business_domain": "support",
            },
            roles=[],
        )
        result = await agent.run("mem-1", ctx)
        # With no roles, all domains are relevant — finance and sales should appear
        touched = result.outputs["domains_touched"]
        assert "sales" in touched
        assert "finance" in touched

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    async def test_rationale_is_heuristic(self, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "heuristic"
        ctx = self._ctx()
        result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    async def test_trace_id_from_context(self, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "heuristic"
        ctx = self._ctx()
        ctx["runtime"] = {"job_id": "job-42"}
        result = await agent.run("mem-1", ctx)
        assert result.trace_id == "job-42"

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    async def test_no_content_uses_heuristic(self, mock_settings, agent):
        # strategy=llm but no content → falls back to heuristic
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "llm"
        mock_settings.AGENT_STRATEGY = "llm"
        ctx = self._ctx(
            enrichment={
                "resolved_entities": [_CUSTOMER_ENTITY],
                "cross_silo_links": [_CUSTOMER_LINK],
                "business_domain": "support",
            },
            roles=["support"],
            content="",  # empty content
        )
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# TestValidateOutputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    @pytest.fixture
    def agent(self):
        return ContextAmplifierAgent()

    def test_valid_outputs_pass(self, agent):
        result = _make_result(
            {
                "context_bundle": [{"entity": "customer", "domain": "finance"}],
                "amplified_entities": ["customer"],
                "domains_touched": ["finance"],
                "confidence": 0.7,
                "rationale": "heuristic",
            }
        )
        agent.validate_outputs(result)  # should not raise

    def test_context_bundle_not_list_raises(self, agent):
        result = _make_result(
            {
                "context_bundle": "not-a-list",
                "amplified_entities": [],
                "domains_touched": [],
                "confidence": 0.5,
            }
        )
        with pytest.raises(ValueError, match="context_bundle"):
            agent.validate_outputs(result)

    def test_amplified_entities_not_list_raises(self, agent):
        result = _make_result(
            {
                "context_bundle": [],
                "amplified_entities": "not-a-list",
                "domains_touched": [],
                "confidence": 0.5,
            }
        )
        with pytest.raises(ValueError, match="amplified_entities"):
            agent.validate_outputs(result)

    def test_bundle_item_missing_entity_raises(self, agent):
        result = _make_result(
            {
                "context_bundle": [{"domain": "finance"}],  # missing entity
                "amplified_entities": [],
                "domains_touched": [],
                "confidence": 0.5,
            }
        )
        with pytest.raises(ValueError, match="entity"):
            agent.validate_outputs(result)

    def test_skipped_status_not_validated(self, agent):
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="ContextAmplifierAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="skipped",
            confidence=0.5,
            outputs={"context_bundle": "bad"},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )
        agent.validate_outputs(result)  # should not raise


# ---------------------------------------------------------------------------
# TestContextAmplifierLLM
# ---------------------------------------------------------------------------

class TestContextAmplifierLLM:
    @pytest.fixture
    def agent(self):
        return ContextAmplifierAgent()

    def _ctx(self, enrichment=None, roles=None):
        return {
            "memory": {
                "content": "Support ticket: client is unhappy, budget exceeded.",
                "enrichment": enrichment
                or {
                    "resolved_entities": [_CUSTOMER_ENTITY],
                    "cross_silo_links": [_CUSTOMER_LINK],
                    "business_domain": "support",
                },
            },
            "actor": {"roles": roles or ["support"]},
        }

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    @patch("app.agents.context_amplifier_agent.create_ollama_client")
    async def test_good_llm_response_used(self, mock_client_factory, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "llm"
        mock_settings.AGENT_STRATEGY = "llm"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.1:8b"
        mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
        mock_settings.OLLAMA_MAX_CONCURRENCY = 2

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "context_bundle": [{"entity": "customer", "domain": "finance", "signal": "x", "original": "client", "entity_type": "role", "source": "cross_silo", "silo_count": 3}],
                "amplified_entities": ["customer"],
                "domains_touched": ["finance"],
                "confidence": 0.75,
                "rationale": "llm",
            }
        )
        mock_client_factory.return_value = mock_client

        result = await agent.run("mem-1", self._ctx())
        assert result.outputs["rationale"] == "llm"
        assert result.outputs["amplified_entities"] == ["customer"]

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    @patch("app.agents.context_amplifier_agent.create_ollama_client")
    async def test_bad_llm_response_falls_back(self, mock_client_factory, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "llm"
        mock_settings.AGENT_STRATEGY = "llm"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.1:8b"
        mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
        mock_settings.OLLAMA_MAX_CONCURRENCY = 2

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value="not a dict")
        mock_client_factory.return_value = mock_client

        result = await agent.run("mem-1", self._ctx())
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    @patch("app.agents.context_amplifier_agent.settings")
    @patch("app.agents.context_amplifier_agent.create_ollama_client")
    async def test_missing_list_field_falls_back(self, mock_client_factory, mock_settings, agent):
        mock_settings.CONTEXT_AMPLIFIER_STRATEGY = "llm"
        mock_settings.AGENT_STRATEGY = "llm"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.1:8b"
        mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
        mock_settings.OLLAMA_MAX_CONCURRENCY = 2

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "context_bundle": [],
                # missing amplified_entities and domains_touched
                "confidence": 0.6,
            }
        )
        mock_client_factory.return_value = mock_client

        result = await agent.run("mem-1", self._ctx())
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# TestAgentMetadata
# ---------------------------------------------------------------------------

class TestAgentMetadata:
    def test_name_and_version(self):
        agent = ContextAmplifierAgent()
        assert agent.name == "ContextAmplifierAgent"
        assert agent.version == "v1"

    def test_dependencies(self):
        agent = ContextAmplifierAgent()
        deps = agent.dependencies()
        assert "EntityResolutionAgent" in deps


# ---------------------------------------------------------------------------
# TestRegistry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_short_name(self):
        from app.agents.registry import get_agent
        agent = get_agent("context_amplifier")
        assert isinstance(agent, ContextAmplifierAgent)

    def test_full_name(self):
        from app.agents.registry import get_agent
        agent = get_agent("contextamplifieragent")
        assert isinstance(agent, ContextAmplifierAgent)

    def test_camel_case(self):
        from app.agents.registry import get_agent
        agent = get_agent("contextamplifier")
        assert isinstance(agent, ContextAmplifierAgent)
