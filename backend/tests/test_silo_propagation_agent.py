"""Tests for Phase 9 — SiloPropagationAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.silo_propagation_agent import (
    SiloPropagationAgent,
    propagate_signals,
    score_signal,
)
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CROSS_SILO_ITEM = {
    "entity": "customer",
    "original": "client",
    "entity_type": "customer",
    "domain": "sales",
    "signal": "customer (customer) is also active in sales",
    "source": "cross_silo",
    "silo_count": 3,
}

_CROSS_SILO_ITEM_2 = {
    "entity": "budget",
    "original": "budget",
    "entity_type": "metric",
    "domain": "finance",
    "signal": "budget (metric) is also active in finance",
    "source": "cross_silo",
    "silo_count": 2,
}

_RESOLVED_ITEM = {
    "entity": "customer",
    "original": "client",
    "entity_type": "customer",
    "domain": "support",
    "signal": "customer resolved from 'client' (confidence 0.85)",
    "source": "resolved",
    "silo_count": 3,
}


def _make_result(outputs: dict) -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="SiloPropagationAgent",
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
# TestScoreSignal
# ---------------------------------------------------------------------------


class TestScoreSignal:
    def test_cross_silo_silo3_customer(self):
        score = score_signal(_CROSS_SILO_ITEM)  # cross_silo +0.10, silo>=2 +0.10, silo>=3 +0.10, customer +0.15
        assert score == pytest.approx(0.4 + 0.10 + 0.10 + 0.10 + 0.15, abs=0.001)

    def test_resolved_item_lower_score(self):
        score = score_signal(_RESOLVED_ITEM)  # resolved → no +0.10, silo>=2 +0.10, silo>=3 +0.10, customer +0.15
        assert score == pytest.approx(0.4 + 0.10 + 0.10 + 0.15, abs=0.001)

    def test_silo_count_1_no_bonus(self):
        item = {"source": "cross_silo", "silo_count": 1, "entity_type": "concept"}
        score = score_signal(item)
        assert score == pytest.approx(0.4 + 0.10 + 0.05, abs=0.001)

    def test_silo_count_2_one_bonus(self):
        item = {"source": "cross_silo", "silo_count": 2, "entity_type": "concept"}
        score = score_signal(item)
        assert score == pytest.approx(0.4 + 0.10 + 0.10 + 0.05, abs=0.001)

    def test_person_entity_type_weight(self):
        item = {"source": "cross_silo", "silo_count": 1, "entity_type": "person"}
        score = score_signal(item)
        assert score == pytest.approx(0.4 + 0.10 + 0.20, abs=0.001)

    def test_org_entity_type_weight(self):
        item = {"source": "cross_silo", "silo_count": 1, "entity_type": "org"}
        score = score_signal(item)
        assert score == pytest.approx(0.4 + 0.10 + 0.20, abs=0.001)

    def test_unknown_entity_type_default_weight(self):
        item = {"source": "cross_silo", "silo_count": 1, "entity_type": "widget"}
        score = score_signal(item)
        assert score == pytest.approx(0.4 + 0.10 + 0.05, abs=0.001)

    def test_capped_at_1(self):
        item = {"source": "cross_silo", "silo_count": 10, "entity_type": "person"}
        score = score_signal(item)
        assert score <= 1.0

    def test_missing_fields_default(self):
        score = score_signal({})
        assert 0.0 <= score <= 1.0

    def test_entity_type_case_insensitive(self):
        item = {"source": "cross_silo", "silo_count": 1, "entity_type": "PERSON"}
        score = score_signal(item)
        assert score == pytest.approx(0.4 + 0.10 + 0.20, abs=0.001)


# ---------------------------------------------------------------------------
# TestPropagateSignals
# ---------------------------------------------------------------------------


class TestPropagateSignals:
    def test_cross_silo_above_threshold_included(self):
        signals, domains = propagate_signals(
            context_bundle=[_CROSS_SILO_ITEM],
            primary_domain="support",
            threshold=0.5,
        )
        assert len(signals) == 1
        assert signals[0]["target_domain"] == "sales"
        assert signals[0]["source_domain"] == "support"

    def test_resolved_item_excluded(self):
        signals, domains = propagate_signals(
            context_bundle=[_RESOLVED_ITEM],
            primary_domain="support",
            threshold=0.5,
        )
        assert len(signals) == 0

    def test_below_threshold_excluded(self):
        # Force a very low score item by using resolved item (won't pass cross_silo check anyway)
        # Use a cross_silo item with silo_count=1 and concept type → 0.40+0.10+0.05 = 0.55 — above 0.5
        # To get below threshold, use a higher threshold
        item = {
            "entity": "misc",
            "entity_type": "concept",
            "domain": "engineering",
            "signal": "misc active in engineering",
            "source": "cross_silo",
            "silo_count": 1,
        }
        signals, _ = propagate_signals(
            context_bundle=[item],
            primary_domain="support",
            threshold=0.99,
        )
        assert len(signals) == 0

    def test_target_domains_collected(self):
        signals, domains = propagate_signals(
            context_bundle=[_CROSS_SILO_ITEM, _CROSS_SILO_ITEM_2],
            primary_domain="support",
            threshold=0.5,
        )
        assert "sales" in domains
        assert "finance" in domains
        assert domains == sorted(domains)

    def test_output_fields_complete(self):
        signals, _ = propagate_signals(
            context_bundle=[_CROSS_SILO_ITEM],
            primary_domain="support",
            threshold=0.5,
        )
        item = signals[0]
        assert "entity" in item
        assert "original" in item
        assert "entity_type" in item
        assert "signal" in item
        assert "source_domain" in item
        assert "target_domain" in item
        assert "relevance_score" in item
        assert "silo_count" in item

    def test_empty_bundle(self):
        signals, domains = propagate_signals(
            context_bundle=[],
            primary_domain="support",
            threshold=0.5,
        )
        assert signals == []
        assert domains == []

    def test_mixed_bundle(self):
        signals, _ = propagate_signals(
            context_bundle=[_CROSS_SILO_ITEM, _RESOLVED_ITEM],
            primary_domain="support",
            threshold=0.5,
        )
        assert all(s["source_domain"] == "support" for s in signals)
        # only cross_silo propagated
        assert len(signals) == 1

    def test_capped_at_50(self):
        items = [
            {
                "entity": f"entity{i}",
                "original": f"entity{i}",
                "entity_type": "concept",
                "domain": f"domain{i}",
                "signal": f"signal{i}",
                "source": "cross_silo",
                "silo_count": 3,
            }
            for i in range(60)
        ]
        signals, _ = propagate_signals(
            context_bundle=items,
            primary_domain="support",
            threshold=0.5,
        )
        assert len(signals) == 50

    def test_relevance_score_in_output(self):
        signals, _ = propagate_signals(
            context_bundle=[_CROSS_SILO_ITEM],
            primary_domain="support",
            threshold=0.5,
        )
        assert signals[0]["relevance_score"] >= 0.5

    def test_zero_threshold_includes_all_cross_silo(self):
        signals, _ = propagate_signals(
            context_bundle=[_CROSS_SILO_ITEM, _CROSS_SILO_ITEM_2, _RESOLVED_ITEM],
            primary_domain="support",
            threshold=0.0,
        )
        assert len(signals) == 2  # only cross_silo items


# ---------------------------------------------------------------------------
# TestSiloPropagationHeuristic
# ---------------------------------------------------------------------------


@pytest.fixture
def agent():
    return SiloPropagationAgent()


def _ctx(
    *,
    enrichment: dict | None = None,
    content: str = "memory content about customer deals",
) -> dict:
    return {
        "memory": {"content": content, "enrichment": enrichment or {}},
        "runtime": {"job_id": "job-42"},
    }


@pytest.fixture
def mock_settings():
    with patch("app.agents.silo_propagation_agent.settings") as m:
        m.SILO_PROPAGATION_STRATEGY = "heuristic"
        m.SILO_PROPAGATION_THRESHOLD = 0.5
        m.VLLM_BASE_URL = "http://localhost:11434"
        m.VLLM_MODEL = "llama3.1:8b"
        m.VLLM_TIMEOUT_SECONDS = 5.0
        m.VLLM_MAX_CONCURRENCY = 2
        m.AGENT_STRATEGY = "heuristic"
        yield m


class TestSiloPropagationHeuristic:
    @pytest.mark.asyncio
    async def test_propagates_cross_silo_signals(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "context_bundle": [_CROSS_SILO_ITEM],
                "business_domain": "support",
            }
        )
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert len(result.outputs["propagation_signals"]) == 1
        assert result.outputs["propagation_signals"][0]["target_domain"] == "sales"

    @pytest.mark.asyncio
    async def test_resolved_signals_not_propagated(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "context_bundle": [_RESOLVED_ITEM],
                "business_domain": "support",
            }
        )
        result = await agent.run("mem-1", ctx)
        assert result.outputs["signal_count"] == 0

    @pytest.mark.asyncio
    async def test_empty_bundle(self, mock_settings, agent):
        ctx = _ctx(enrichment={"context_bundle": [], "business_domain": "support"})
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["propagation_signals"] == []
        assert result.outputs["target_domains"] == []

    @pytest.mark.asyncio
    async def test_all_required_output_keys(self, mock_settings, agent):
        ctx = _ctx(enrichment={"context_bundle": [_CROSS_SILO_ITEM], "business_domain": "support"})
        result = await agent.run("mem-1", ctx)
        for key in ("propagation_signals", "target_domains", "signal_count", "threshold_used", "confidence", "rationale"):
            assert key in result.outputs

    @pytest.mark.asyncio
    async def test_confidence_zero_cross_silo(self, mock_settings, agent):
        ctx = _ctx(enrichment={"context_bundle": [], "business_domain": "support"})
        result = await agent.run("mem-1", ctx)
        assert result.outputs["confidence"] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_confidence_scales_with_propagation(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "context_bundle": [_CROSS_SILO_ITEM, _CROSS_SILO_ITEM_2],
                "business_domain": "support",
            }
        )
        result = await agent.run("mem-1", ctx)
        assert 0.4 < result.outputs["confidence"] <= 0.9

    @pytest.mark.asyncio
    async def test_primary_domain_from_enrichment(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "context_bundle": [_CROSS_SILO_ITEM],
                "business_domain": "engineering",
            }
        )
        result = await agent.run("mem-1", ctx)
        for sig in result.outputs["propagation_signals"]:
            assert sig["source_domain"] == "engineering"

    @pytest.mark.asyncio
    async def test_rationale_heuristic(self, mock_settings, agent):
        ctx = _ctx(enrichment={"context_bundle": [_CROSS_SILO_ITEM], "business_domain": "support"})
        result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_threshold_used_in_output(self, mock_settings, agent):
        mock_settings.SILO_PROPAGATION_THRESHOLD = 0.6
        ctx = _ctx(enrichment={"context_bundle": [_CROSS_SILO_ITEM], "business_domain": "support"})
        result = await agent.run("mem-1", ctx)
        assert result.outputs["threshold_used"] == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_no_content_uses_heuristic(self, agent):
        with patch("app.agents.silo_propagation_agent.settings") as m:
            m.SILO_PROPAGATION_STRATEGY = "llm"
            m.SILO_PROPAGATION_THRESHOLD = 0.5
            m.AGENT_STRATEGY = "llm"
            ctx = _ctx(
                enrichment={"context_bundle": [_CROSS_SILO_ITEM], "business_domain": "support"},
                content="",
            )
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_set(self, mock_settings, agent):
        ctx = _ctx(enrichment={"context_bundle": [], "business_domain": "support"})
        result = await agent.run("mem-1", ctx)
        assert result.trace_id == "job-42"

    @pytest.mark.asyncio
    async def test_signal_count_matches_list(self, mock_settings, agent):
        ctx = _ctx(
            enrichment={
                "context_bundle": [_CROSS_SILO_ITEM, _CROSS_SILO_ITEM_2],
                "business_domain": "support",
            }
        )
        result = await agent.run("mem-1", ctx)
        assert result.outputs["signal_count"] == len(result.outputs["propagation_signals"])

    @pytest.mark.asyncio
    async def test_missing_enrichment_defaults(self, mock_settings, agent):
        ctx = _ctx(enrichment={})
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["propagation_signals"] == []


# ---------------------------------------------------------------------------
# TestValidateOutputs
# ---------------------------------------------------------------------------


class TestValidateOutputs:
    def test_valid_outputs_passes(self):
        agent = SiloPropagationAgent()
        result = _make_result(
            {
                "propagation_signals": [
                    {"entity": "customer", "target_domain": "sales", "relevance_score": 0.75}
                ],
                "target_domains": ["sales"],
                "signal_count": 1,
                "threshold_used": 0.5,
                "confidence": 0.7,
                "rationale": "heuristic",
            }
        )
        agent.validate_outputs(result)  # no exception

    def test_propagation_signals_not_list_raises(self):
        agent = SiloPropagationAgent()
        result = _make_result({"propagation_signals": "bad", "target_domains": []})
        with pytest.raises(ValueError, match="propagation_signals must be a list"):
            agent.validate_outputs(result)

    def test_target_domains_not_list_raises(self):
        agent = SiloPropagationAgent()
        result = _make_result({"propagation_signals": [], "target_domains": "bad"})
        with pytest.raises(ValueError, match="target_domains must be a list"):
            agent.validate_outputs(result)

    def test_missing_entity_in_signal_raises(self):
        agent = SiloPropagationAgent()
        result = _make_result(
            {
                "propagation_signals": [{"target_domain": "sales"}],
                "target_domains": ["sales"],
            }
        )
        with pytest.raises(ValueError, match="entity and target_domain"):
            agent.validate_outputs(result)

    def test_missing_target_domain_in_signal_raises(self):
        agent = SiloPropagationAgent()
        result = _make_result(
            {
                "propagation_signals": [{"entity": "customer"}],
                "target_domains": [],
            }
        )
        with pytest.raises(ValueError, match="entity and target_domain"):
            agent.validate_outputs(result)

    def test_non_success_status_skips_validation(self):
        agent = SiloPropagationAgent()
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="SiloPropagationAgent",
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
# TestSiloPropagationLLM
# ---------------------------------------------------------------------------


class TestSiloPropagationLLM:
    @pytest.mark.asyncio
    async def test_good_llm_response_used(self, agent):
        good_resp = {
            "propagation_signals": [
                {
                    "entity": "customer",
                    "original": "client",
                    "entity_type": "customer",
                    "signal": "customer active in sales",
                    "source_domain": "support",
                    "target_domain": "sales",
                    "relevance_score": 0.75,
                    "silo_count": 3,
                }
            ],
            "target_domains": ["sales"],
            "signal_count": 1,
            "threshold_used": 0.5,
            "confidence": 0.8,
            "rationale": "llm",
        }
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=good_resp)
        with patch("app.agents.silo_propagation_agent.settings") as m, \
             patch("app.agents.silo_propagation_agent.create_llm_client", return_value=mock_client):
            m.SILO_PROPAGATION_STRATEGY = "llm"
            m.SILO_PROPAGATION_THRESHOLD = 0.5
            m.AGENT_STRATEGY = "llm"
            m.VLLM_BASE_URL = "http://localhost:11434"
            m.VLLM_MODEL = "llama3.1:8b"
            m.VLLM_TIMEOUT_SECONDS = 5.0
            m.VLLM_MAX_CONCURRENCY = 2
            ctx = _ctx(
                enrichment={"context_bundle": [_CROSS_SILO_ITEM], "business_domain": "support"}
            )
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "llm"
        assert len(result.outputs["propagation_signals"]) == 1

    @pytest.mark.asyncio
    async def test_bad_llm_response_falls_back(self, agent):
        with patch("app.agents.silo_propagation_agent.settings") as m, \
             patch("app.agents.silo_propagation_agent.create_llm_client") as mock_factory:
            m.SILO_PROPAGATION_STRATEGY = "llm"
            m.SILO_PROPAGATION_THRESHOLD = 0.5
            m.AGENT_STRATEGY = "llm"
            m.VLLM_BASE_URL = "http://localhost:11434"
            m.VLLM_MODEL = "llama3.1:8b"
            m.VLLM_TIMEOUT_SECONDS = 5.0
            m.VLLM_MAX_CONCURRENCY = 2
            mock_client = AsyncMock()
            mock_client.complete_json = AsyncMock(return_value={"bad": "response"})
            mock_factory.return_value = mock_client
            ctx = _ctx(
                enrichment={"context_bundle": [_CROSS_SILO_ITEM], "business_domain": "support"}
            )
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_missing_list_field_falls_back(self, agent):
        partial = {
            "propagation_signals": [{"entity": "x", "target_domain": "sales"}],
            "target_domains": "not-a-list",  # wrong type
        }
        with patch("app.agents.silo_propagation_agent.settings") as m, \
             patch("app.agents.silo_propagation_agent.create_llm_client") as mock_factory:
            m.SILO_PROPAGATION_STRATEGY = "llm"
            m.SILO_PROPAGATION_THRESHOLD = 0.5
            m.AGENT_STRATEGY = "llm"
            m.VLLM_BASE_URL = "http://localhost:11434"
            m.VLLM_MODEL = "llama3.1:8b"
            m.VLLM_TIMEOUT_SECONDS = 5.0
            m.VLLM_MAX_CONCURRENCY = 2
            mock_client = AsyncMock()
            mock_client.complete_json = AsyncMock(return_value=partial)
            mock_factory.return_value = mock_client
            ctx = _ctx(
                enrichment={"context_bundle": [_CROSS_SILO_ITEM], "business_domain": "support"}
            )
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# TestAgentMetadata
# ---------------------------------------------------------------------------


class TestAgentMetadata:
    def test_name_and_version(self):
        agent = SiloPropagationAgent()
        assert agent.name == "SiloPropagationAgent"
        assert agent.version == "v1"

    def test_dependencies(self):
        agent = SiloPropagationAgent()
        assert agent.dependencies() == ["ContextAmplifierAgent"]


# ---------------------------------------------------------------------------
# TestRegistry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_short_name(self):
        from app.agents.registry import get_agent
        assert isinstance(get_agent("silo_propagation"), SiloPropagationAgent)

    def test_full_name(self):
        from app.agents.registry import get_agent
        assert isinstance(get_agent("SiloPropagationAgent"), SiloPropagationAgent)

    def test_camel_case(self):
        from app.agents.registry import get_agent
        assert isinstance(get_agent("silopropagation"), SiloPropagationAgent)
