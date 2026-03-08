"""Tests for SemanticNormalizationAgent and helpers — Phase 6."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.semantic_normalization_agent import (
    SemanticNormalizationAgent,
    _classify_domain,
    _classify_intent,
    _extract_relationships,
)
from app.agents.types import AgentContext, AgentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(content: str = "") -> AgentContext:
    return {
        "tenant": {"org_id": "org-1", "org_slug": None},
        "actor": {"user_id": "user-1", "roles": []},
        "memory": {"id": "mem-1", "content": content},
        "runtime": {"job_id": "trace-1", "attempt": 1},
    }


def _fake_now() -> datetime:
    return datetime(2026, 3, 8, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Unit: _classify_intent
# ---------------------------------------------------------------------------

class TestClassifyIntent:
    def test_document_decision(self):
        assert _classify_intent("We have decided to migrate to Postgres.") == "document_decision"

    def test_report_issue(self):
        assert _classify_intent("There is a bug in the auth API.") == "report_issue"

    def test_share_learning(self):
        assert _classify_intent("Key insight from the retrospective: deploy cadence needs improvement.") == "share_learning"

    def test_request_action(self):
        assert _classify_intent("Please follow up with the vendor by EOD.") == "request_action"

    def test_record_status(self):
        assert _classify_intent("Deployment completed successfully.") == "record_status"

    def test_capture_context(self):
        assert _classify_intent("FYI: background on the contract negotiation.") == "capture_context"

    def test_general_fallback(self):
        assert _classify_intent("xyz qrs abc") == "capture_context"


# ---------------------------------------------------------------------------
# Unit: _classify_domain
# ---------------------------------------------------------------------------

class TestClassifyDomain:
    def test_engineering(self):
        assert _classify_domain("Critical bug in the deploy pipeline, stack trace attached.") == "engineering"

    def test_sales(self):
        assert _classify_domain("Deal closed. Revenue impact $200k ARR, added to CRM.") == "sales"

    def test_finance(self):
        assert _classify_domain("The invoice for Q1 is overdue. Budget variance noted.") == "finance"

    def test_support(self):
        assert _classify_domain("Customer ticket escalated. SLA breach imminent.") == "support"

    def test_hr(self):
        assert _classify_domain("New employee onboarding checklist for Q2 hiring.") == "hr"

    def test_legal(self):
        assert _classify_domain("NDA signed. Contract under legal review for compliance.") == "legal"

    def test_product(self):
        assert _classify_domain("Roadmap updated. Sprint backlog reviewed with user stories.") == "product"

    def test_marketing(self):
        assert _classify_domain("Campaign launched. SEO and conversion rates up 15%.") == "marketing"

    def test_operations(self):
        assert _classify_domain("Runbook updated. On-call alert monitoring improved.") == "operations"

    def test_general_fallback(self):
        assert _classify_domain("Nothing specific here.") == "general"

    def test_picks_highest_score(self):
        # engineering has more hits than support
        text = "Deploy bug incident stack trace API error code error error"
        assert _classify_domain(text) == "engineering"


# ---------------------------------------------------------------------------
# Unit: _extract_relationships
# ---------------------------------------------------------------------------

class TestExtractRelationships:
    def test_extracts_mention(self):
        rels = _extract_relationships("@alice reviewed the PR.")
        assert any(r["type"] == "mentions_person" and r["concept"] == "alice" for r in rels)

    def test_extracts_system(self):
        rels = _extract_relationships("Filed in Jira and linked to GitHub.")
        types = {r["concept"].lower() for r in rels}
        assert "jira" in types
        assert "github" in types

    def test_deduplicates(self):
        rels = _extract_relationships("@bob @bob asked about GitHub GitHub.")
        concepts = [r["concept"].lower() for r in rels]
        assert concepts.count("bob") == 1
        assert concepts.count("github") == 1

    def test_empty_content(self):
        assert _extract_relationships("") == []

    def test_caps_at_ten(self):
        text = " ".join(f"@user{i}" for i in range(20))
        rels = _extract_relationships(text)
        assert len(rels) <= 10


# ---------------------------------------------------------------------------
# Agent: heuristic mode
# ---------------------------------------------------------------------------

class TestSemanticNormalizationAgentHeuristic:
    @pytest.fixture
    def agent(self) -> SemanticNormalizationAgent:
        return SemanticNormalizationAgent()

    @pytest.mark.asyncio
    async def test_engineering_content(self, agent):
        content = "Bug found in the deploy pipeline. Stack trace attached. Error in API layer."
        with patch("app.agents.semantic_normalization_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            mock_settings.SEMANTIC_NORMALIZATION_STRATEGY = None
            result = await agent.run("mem-1", _make_context(content))

        assert result.status == "success"
        assert result.outputs["business_domain"] == "engineering"
        assert result.outputs["intent"] == "report_issue"
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_sales_content(self, agent):
        content = "Deal closed with Acme Corp. $500k ARR added to CRM pipeline."
        with patch("app.agents.semantic_normalization_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            mock_settings.SEMANTIC_NORMALIZATION_STRATEGY = None
            result = await agent.run("mem-1", _make_context(content))

        assert result.status == "success"
        assert result.outputs["business_domain"] == "sales"

    @pytest.mark.asyncio
    async def test_decision_intent(self, agent):
        content = "We have decided to move to a microservices architecture."
        with patch("app.agents.semantic_normalization_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            mock_settings.SEMANTIC_NORMALIZATION_STRATEGY = None
            result = await agent.run("mem-1", _make_context(content))

        assert result.outputs["intent"] == "document_decision"

    @pytest.mark.asyncio
    async def test_empty_content_returns_general(self, agent):
        with patch("app.agents.semantic_normalization_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            mock_settings.SEMANTIC_NORMALIZATION_STRATEGY = None
            result = await agent.run("mem-1", _make_context(""))

        assert result.status == "success"
        assert result.outputs["business_domain"] == "general"

    @pytest.mark.asyncio
    async def test_relationships_extracted(self, agent):
        content = "@alice opened a ticket in Jira about the GitHub Actions build."
        with patch("app.agents.semantic_normalization_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            mock_settings.SEMANTIC_NORMALIZATION_STRATEGY = None
            result = await agent.run("mem-1", _make_context(content))

        rels = result.outputs["semantic_relationships"]
        concepts = {r["concept"].lower() for r in rels}
        assert "alice" in concepts
        assert "jira" in concepts
        assert "github" in concepts

    @pytest.mark.asyncio
    async def test_outputs_structure(self, agent):
        with patch("app.agents.semantic_normalization_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            mock_settings.SEMANTIC_NORMALIZATION_STRATEGY = None
            result = await agent.run("mem-1", _make_context("Some content"))

        assert "intent" in result.outputs
        assert "business_domain" in result.outputs
        assert "semantic_relationships" in result.outputs
        assert "confidence" in result.outputs
        assert "rationale" in result.outputs
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# Agent: validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _make_result(self, outputs: dict) -> AgentResult:
        now = _fake_now()
        return AgentResult(
            agent_name="SemanticNormalizationAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="success",
            confidence=0.7,
            outputs=outputs,
            started_at=now,
            finished_at=now,
        )

    def test_valid_outputs(self):
        agent = SemanticNormalizationAgent()
        result = self._make_result({
            "intent": "report_issue",
            "business_domain": "engineering",
            "semantic_relationships": [],
        })
        agent.validate_outputs(result)  # no exception

    def test_invalid_intent_raises(self):
        agent = SemanticNormalizationAgent()
        result = self._make_result({
            "intent": "invalid_intent",
            "business_domain": "engineering",
            "semantic_relationships": [],
        })
        with pytest.raises(ValueError, match="invalid intent"):
            agent.validate_outputs(result)

    def test_invalid_domain_raises(self):
        agent = SemanticNormalizationAgent()
        result = self._make_result({
            "intent": "report_issue",
            "business_domain": "unknown_dept",
            "semantic_relationships": [],
        })
        with pytest.raises(ValueError, match="invalid business_domain"):
            agent.validate_outputs(result)

    def test_invalid_relationships_type_raises(self):
        agent = SemanticNormalizationAgent()
        result = self._make_result({
            "intent": "report_issue",
            "business_domain": "engineering",
            "semantic_relationships": "not_a_list",
        })
        with pytest.raises(ValueError, match="must be a list"):
            agent.validate_outputs(result)

    def test_skipped_status_not_validated(self):
        agent = SemanticNormalizationAgent()
        now = _fake_now()
        result = AgentResult(
            agent_name="SemanticNormalizationAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="skipped",
            confidence=0.0,
            outputs={},
            started_at=now,
            finished_at=now,
        )
        agent.validate_outputs(result)  # no exception for skipped


# ---------------------------------------------------------------------------
# Agent: LLM mode — fallback to heuristic on bad response
# ---------------------------------------------------------------------------

class TestSemanticNormalizationAgentLLM:
    @pytest.mark.asyncio
    async def test_llm_good_response_used(self):
        agent = SemanticNormalizationAgent()
        mock_resp = {
            "intent": "share_learning",
            "business_domain": "product",
            "semantic_relationships": [{"type": "references_system", "concept": "jira"}],
            "confidence": 0.9,
            "rationale": "llm",
        }
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=mock_resp)

        with patch("app.agents.semantic_normalization_agent.settings") as mock_settings, \
             patch("app.agents.semantic_normalization_agent.create_ollama_client", return_value=mock_client):
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.SEMANTIC_NORMALIZATION_STRATEGY = None
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _make_context("Sprint retro findings for product roadmap."))

        assert result.outputs["intent"] == "share_learning"
        assert result.outputs["business_domain"] == "product"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_llm_bad_response_falls_back_to_heuristic(self):
        agent = SemanticNormalizationAgent()
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "response"})

        with patch("app.agents.semantic_normalization_agent.settings") as mock_settings, \
             patch("app.agents.semantic_normalization_agent.create_ollama_client", return_value=mock_client):
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.SEMANTIC_NORMALIZATION_STRATEGY = None
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _make_context("Budget invoice expense report."))

        # Falls back to heuristic — should detect finance domain
        assert result.status == "success"
        assert result.outputs["business_domain"] == "finance"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_invalid_intent_falls_back(self):
        agent = SemanticNormalizationAgent()
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={
            "intent": "not_valid",  # invalid intent
            "business_domain": "engineering",
            "semantic_relationships": [],
            "confidence": 0.8,
        })

        with patch("app.agents.semantic_normalization_agent.settings") as mock_settings, \
             patch("app.agents.semantic_normalization_agent.create_ollama_client", return_value=mock_client):
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.SEMANTIC_NORMALIZATION_STRATEGY = None
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _make_context("Bug fix deployed."))

        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# Agent: metadata
# ---------------------------------------------------------------------------

class TestAgentMetadata:
    def test_name_and_version(self):
        agent = SemanticNormalizationAgent()
        assert agent.name == "SemanticNormalizationAgent"
        assert agent.version == "v1"

    def test_dependencies(self):
        agent = SemanticNormalizationAgent()
        assert "MetadataExtractionAgent" in agent.dependencies()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered_by_name(self):
        from app.agents.registry import get_agent
        agent = get_agent("semantic_normalization")
        assert isinstance(agent, SemanticNormalizationAgent)

    def test_registered_by_full_name(self):
        from app.agents.registry import get_agent
        agent = get_agent("SemanticNormalizationAgent")
        assert isinstance(agent, SemanticNormalizationAgent)

    def test_registered_by_camel_case(self):
        from app.agents.registry import get_agent
        agent = get_agent("semanticnormalization")
        assert isinstance(agent, SemanticNormalizationAgent)
