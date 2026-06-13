"""Tests for Phase 10 — OrgAttentionAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.org_attention_agent import (
    OrgAttentionAgent,
    build_team_topic_map,
    detect_overlaps,
    score_overlap,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SIG_SALES_SUPPORT = {
    "entity": "acme_corp",
    "original": "Acme",
    "entity_type": "customer",
    "source_domain": "sales",
    "target_domain": "support",
    "relevance_score": 0.75,
    "silo_count": 2,
}

_SIG_ENG_FINANCE = {
    "entity": "budget",
    "original": "budget",
    "entity_type": "metric",
    "source_domain": "engineering",
    "target_domain": "finance",
    "relevance_score": 0.65,
    "silo_count": 2,
}

_SIG_SALES_ENG = {
    "entity": "acme_corp",
    "original": "Acme Corp",
    "entity_type": "customer",
    "source_domain": "sales",
    "target_domain": "engineering",
    "relevance_score": 0.80,
    "silo_count": 3,
}


def _make_result(outputs: dict) -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="OrgAttentionAgent",
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


def _ctx(
    *,
    propagation_signals: list | None = None,
    business_domain: str = "sales",
    entities: list | None = None,
    content: str = "Acme Corp budget review",
) -> dict:
    enrichment: dict = {
        "propagation_signals": propagation_signals or [],
        "business_domain": business_domain,
        "entities": entities or [],
    }
    return {
        "memory": {"content": content, "enrichment": enrichment},
        "runtime": {"job_id": "job-10"},
    }


@pytest.fixture
def agent():
    return OrgAttentionAgent()


@pytest.fixture
def mock_settings():
    with patch("app.agents.org_attention_agent.settings") as m:
        m.ORG_ATTENTION_STRATEGY = "heuristic"
        m.ORG_ATTENTION_THRESHOLD = 0.5
        m.VLLM_BASE_URL = "http://localhost:11434"
        m.VLLM_MODEL = "llama3.1:8b"
        m.VLLM_TIMEOUT_SECONDS = 5.0
        m.VLLM_MAX_CONCURRENCY = 2
        m.AGENT_STRATEGY = "heuristic"
        yield m


# ---------------------------------------------------------------------------
# TestBuildTeamTopicMap
# ---------------------------------------------------------------------------


class TestBuildTeamTopicMap:
    def test_empty_signals_no_entities(self):
        result = build_team_topic_map(
            propagation_signals=[], primary_domain="sales", entities=[]
        )
        assert result == {}

    def test_single_signal_two_teams(self):
        result = build_team_topic_map(
            propagation_signals=[_SIG_SALES_SUPPORT],
            primary_domain="sales",
            entities=[],
        )
        assert "acme_corp" in result
        assert result["acme_corp"]["teams"] == {"sales", "support"}

    def test_entities_add_primary_domain(self):
        result = build_team_topic_map(
            propagation_signals=[],
            primary_domain="engineering",
            entities=["project_x"],
        )
        assert "project_x" in result
        assert "engineering" in result["project_x"]["teams"]

    def test_two_signals_same_entity_merges_teams(self):
        result = build_team_topic_map(
            propagation_signals=[_SIG_SALES_SUPPORT, _SIG_SALES_ENG],
            primary_domain="sales",
            entities=[],
        )
        assert result["acme_corp"]["teams"] == {"sales", "support", "engineering"}

    def test_different_entities_separate_entries(self):
        result = build_team_topic_map(
            propagation_signals=[_SIG_SALES_SUPPORT, _SIG_ENG_FINANCE],
            primary_domain="sales",
            entities=[],
        )
        assert "acme_corp" in result
        assert "budget" in result

    def test_entity_type_preserved(self):
        result = build_team_topic_map(
            propagation_signals=[_SIG_SALES_SUPPORT],
            primary_domain="sales",
            entities=[],
        )
        assert result["acme_corp"]["entity_type"] == "customer"


# ---------------------------------------------------------------------------
# TestScoreOverlap
# ---------------------------------------------------------------------------


class TestScoreOverlap:
    def test_two_teams_customer(self):
        # base 0.40 + 1 extra * 0.10 + customer 0.15 = 0.65
        score = score_overlap(team_count=2, entity_type="customer")
        assert score == pytest.approx(0.65, abs=0.001)

    def test_three_teams_metric(self):
        # base 0.40 + 2 extra * 0.10 + metric 0.08 = 0.68
        score = score_overlap(team_count=3, entity_type="metric")
        assert score == pytest.approx(0.68, abs=0.001)

    def test_four_teams_capped_extra(self):
        # base 0.40 + min(3,3)*0.10 + concept 0.05 = 0.75
        score = score_overlap(team_count=4, entity_type="concept")
        assert score == pytest.approx(0.75, abs=0.001)

    def test_five_teams_same_as_four(self):
        score4 = score_overlap(team_count=4, entity_type="org")
        score5 = score_overlap(team_count=5, entity_type="org")
        assert score4 == score5

    def test_unknown_entity_type_default_weight(self):
        # base 0.40 + 1 extra * 0.10 + unknown 0.05 = 0.55
        score = score_overlap(team_count=2, entity_type="unknown_type")
        assert score == pytest.approx(0.55, abs=0.001)

    def test_project_type(self):
        # base 0.40 + 1 extra * 0.10 + project 0.12 = 0.62
        score = score_overlap(team_count=2, entity_type="project")
        assert score == pytest.approx(0.62, abs=0.001)


# ---------------------------------------------------------------------------
# TestDetectOverlaps
# ---------------------------------------------------------------------------


class TestDetectOverlaps:
    def test_empty_map_no_alerts(self):
        alerts = detect_overlaps(topic_map={}, threshold=0.5)
        assert alerts == []

    def test_single_team_no_alert(self):
        topic_map = {"acme": {"teams": {"sales"}, "entity_type": "customer"}}
        alerts = detect_overlaps(topic_map=topic_map, threshold=0.5)
        assert alerts == []

    def test_two_teams_above_threshold(self):
        topic_map = {"acme": {"teams": {"sales", "support"}, "entity_type": "customer"}}
        alerts = detect_overlaps(topic_map=topic_map, threshold=0.5)
        assert len(alerts) == 1
        assert alerts[0]["topic"] == "acme"
        assert set(alerts[0]["teams"]) == {"sales", "support"}
        assert alerts[0]["team_count"] == 2
        assert alerts[0]["alert_type"] == "parallel_work"

    def test_below_threshold_not_alerted(self):
        topic_map = {"concept_x": {"teams": {"a", "b"}, "entity_type": "unknown_type"}}
        alerts = detect_overlaps(topic_map=topic_map, threshold=0.9)
        assert alerts == []

    def test_multiple_topics_only_qualifying(self):
        topic_map = {
            "acme": {"teams": {"sales", "support"}, "entity_type": "customer"},
            "lone": {"teams": {"engineering"}, "entity_type": "system"},
        }
        alerts = detect_overlaps(topic_map=topic_map, threshold=0.5)
        assert len(alerts) == 1
        assert alerts[0]["topic"] == "acme"

    def test_overlap_score_in_alert(self):
        # budget, 2 teams (finance, sales): 0.40 + 0.10 + 0.08 = 0.58
        topic_map = {"budget": {"teams": {"finance", "sales"}, "entity_type": "metric"}}
        alerts = detect_overlaps(topic_map=topic_map, threshold=0.5)
        assert alerts[0]["overlap_score"] == pytest.approx(0.58, abs=0.001)

    def test_capped_at_50_alerts(self):
        topic_map = {
            f"topic_{i}": {"teams": {"a", "b"}, "entity_type": "customer"}
            for i in range(60)
        }
        alerts = detect_overlaps(topic_map=topic_map, threshold=0.0)
        assert len(alerts) == 50


# ---------------------------------------------------------------------------
# TestPhase10Heuristic
# ---------------------------------------------------------------------------


class TestPhase10Heuristic:
    @pytest.mark.asyncio
    async def test_empty_context_returns_defaults(self, agent, mock_settings):
        ctx = _ctx(propagation_signals=[], entities=[], content="")
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["overlap_count"] == 0
        assert result.outputs["active_topics"] == []
        assert result.outputs["attention_signals"] == []
        assert result.outputs["overlap_alerts"] == []

    @pytest.mark.asyncio
    async def test_single_team_no_overlap(self, agent, mock_settings):
        ctx = _ctx(
            propagation_signals=[],
            entities=["project_x"],
            business_domain="engineering",
            content="",
        )
        result = await agent.run("mem-2", ctx)
        assert result.outputs["overlap_count"] == 0
        assert "project_x" in result.outputs["active_topics"]

    @pytest.mark.asyncio
    async def test_two_teams_same_entity_overlap(self, agent, mock_settings):
        ctx = _ctx(propagation_signals=[_SIG_SALES_SUPPORT], business_domain="sales")
        result = await agent.run("mem-3", ctx)
        assert result.outputs["overlap_count"] == 1
        alert = result.outputs["overlap_alerts"][0]
        assert alert["topic"] == "acme_corp"
        assert set(alert["teams"]) == {"sales", "support"}

    @pytest.mark.asyncio
    async def test_three_teams_overlap(self, agent, mock_settings):
        ctx = _ctx(
            propagation_signals=[_SIG_SALES_SUPPORT, _SIG_SALES_ENG],
            business_domain="sales",
        )
        result = await agent.run("mem-4", ctx)
        assert result.outputs["overlap_count"] == 1
        alert = result.outputs["overlap_alerts"][0]
        assert alert["team_count"] == 3
        assert set(alert["teams"]) == {"sales", "support", "engineering"}

    @pytest.mark.asyncio
    async def test_active_topics_only_primary_domain(self, agent, mock_settings):
        ctx = _ctx(propagation_signals=[_SIG_SALES_SUPPORT], business_domain="support")
        result = await agent.run("mem-5", ctx)
        assert "acme_corp" in result.outputs["active_topics"]

    @pytest.mark.asyncio
    async def test_active_topics_from_entities(self, agent, mock_settings):
        ctx = _ctx(
            propagation_signals=[],
            entities=["internal_tool", "roadmap"],
            business_domain="engineering",
            content="",
        )
        result = await agent.run("mem-6", ctx)
        assert "internal_tool" in result.outputs["active_topics"]
        assert "roadmap" in result.outputs["active_topics"]

    @pytest.mark.asyncio
    async def test_attention_signals_structure(self, agent, mock_settings):
        ctx = _ctx(propagation_signals=[_SIG_SALES_SUPPORT], business_domain="sales")
        result = await agent.run("mem-7", ctx)
        signals = result.outputs["attention_signals"]
        assert len(signals) > 0
        for sig in signals:
            assert "team" in sig
            assert "topic" in sig
            assert "entity_type" in sig
            assert "relevance" in sig

    @pytest.mark.asyncio
    async def test_attention_signals_relevance_from_propagation(self, agent, mock_settings):
        ctx = _ctx(propagation_signals=[_SIG_SALES_SUPPORT], business_domain="sales")
        result = await agent.run("mem-8", ctx)
        sales_sig = next(
            (
                s
                for s in result.outputs["attention_signals"]
                if s["team"] == "sales" and s["topic"] == "acme_corp"
            ),
            None,
        )
        assert sales_sig is not None
        assert sales_sig["relevance"] == pytest.approx(0.75, abs=0.001)

    @pytest.mark.asyncio
    async def test_alert_type_is_parallel_work(self, agent, mock_settings):
        ctx = _ctx(propagation_signals=[_SIG_SALES_SUPPORT], business_domain="sales")
        result = await agent.run("mem-9", ctx)
        assert result.outputs["overlap_alerts"][0]["alert_type"] == "parallel_work"

    @pytest.mark.asyncio
    async def test_confidence_zero_topics(self, agent, mock_settings):
        ctx = _ctx(propagation_signals=[], entities=[], content="")
        result = await agent.run("mem-10", ctx)
        assert result.outputs["confidence"] == pytest.approx(0.4, abs=0.001)

    @pytest.mark.asyncio
    async def test_confidence_with_overlaps(self, agent, mock_settings):
        ctx = _ctx(propagation_signals=[_SIG_SALES_SUPPORT], business_domain="sales")
        result = await agent.run("mem-11", ctx)
        assert result.outputs["confidence"] > 0.4

    @pytest.mark.asyncio
    async def test_rationale_is_heuristic(self, agent, mock_settings):
        ctx = _ctx(content="")
        result = await agent.run("mem-12", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_multiple_entities_multiple_overlaps(self, agent, mock_settings):
        ctx = _ctx(
            propagation_signals=[_SIG_SALES_SUPPORT, _SIG_ENG_FINANCE],
            business_domain="sales",
        )
        result = await agent.run("mem-13", ctx)
        assert result.outputs["overlap_count"] == 2

    @pytest.mark.asyncio
    async def test_trace_id_in_result(self, agent, mock_settings):
        ctx = _ctx()
        result = await agent.run("mem-14", ctx)
        assert result.trace_id == "job-10"

    @pytest.mark.asyncio
    async def test_memory_id_in_result(self, agent, mock_settings):
        ctx = _ctx()
        result = await agent.run("mem-99", ctx)
        assert result.memory_id == "mem-99"


# ---------------------------------------------------------------------------
# TestValidateOutputs
# ---------------------------------------------------------------------------


class TestValidateOutputs:
    def test_valid_outputs_passes(self):
        result = _make_result(
            {
                "attention_signals": [
                    {"team": "sales", "topic": "acme", "entity_type": "customer", "relevance": 0.7}
                ],
                "overlap_alerts": [
                    {
                        "topic": "acme",
                        "teams": ["sales", "support"],
                        "team_count": 2,
                        "overlap_score": 0.65,
                        "entity_type": "customer",
                        "alert_type": "parallel_work",
                    }
                ],
                "active_topics": ["acme"],
                "overlap_count": 1,
                "confidence": 0.7,
                "rationale": "heuristic",
            }
        )
        OrgAttentionAgent().validate_outputs(result)  # no exception

    def test_non_list_attention_signals_raises(self):
        result = _make_result(
            {
                "attention_signals": "bad",
                "overlap_alerts": [],
                "active_topics": [],
                "overlap_count": 0,
                "confidence": 0.4,
                "rationale": "heuristic",
            }
        )
        with pytest.raises(ValueError, match="attention_signals"):
            OrgAttentionAgent().validate_outputs(result)

    def test_non_list_overlap_alerts_raises(self):
        result = _make_result(
            {
                "attention_signals": [],
                "overlap_alerts": "bad",
                "active_topics": [],
                "overlap_count": 0,
                "confidence": 0.4,
                "rationale": "heuristic",
            }
        )
        with pytest.raises(ValueError, match="overlap_alerts"):
            OrgAttentionAgent().validate_outputs(result)

    def test_alert_missing_topic_raises(self):
        result = _make_result(
            {
                "attention_signals": [],
                "overlap_alerts": [{"teams": ["a", "b"]}],
                "active_topics": [],
                "overlap_count": 0,
                "confidence": 0.4,
                "rationale": "heuristic",
            }
        )
        with pytest.raises(ValueError, match="topic"):
            OrgAttentionAgent().validate_outputs(result)

    def test_alert_missing_teams_raises(self):
        result = _make_result(
            {
                "attention_signals": [],
                "overlap_alerts": [{"topic": "x"}],
                "active_topics": [],
                "overlap_count": 0,
                "confidence": 0.4,
                "rationale": "heuristic",
            }
        )
        with pytest.raises(ValueError, match="teams"):
            OrgAttentionAgent().validate_outputs(result)

    def test_non_success_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="OrgAttentionAgent",
            agent_version="v1",
            memory_id="m",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )
        OrgAttentionAgent().validate_outputs(result)  # no exception


# ---------------------------------------------------------------------------
# TestPhase10LLM
# ---------------------------------------------------------------------------


class TestPhase10LLM:
    @pytest.mark.asyncio
    async def test_llm_good_response(self, agent):
        good_resp = {
            "attention_signals": [
                {"team": "sales", "topic": "acme", "entity_type": "customer", "relevance": 0.8}
            ],
            "overlap_alerts": [
                {
                    "topic": "acme",
                    "teams": ["sales", "support"],
                    "team_count": 2,
                    "overlap_score": 0.65,
                    "entity_type": "customer",
                    "alert_type": "parallel_work",
                }
            ],
            "active_topics": ["acme"],
            "overlap_count": 1,
            "confidence": 0.8,
            "rationale": "llm",
        }
        with patch("app.agents.org_attention_agent.settings") as ms:
            ms.ORG_ATTENTION_STRATEGY = "llm"
            ms.ORG_ATTENTION_THRESHOLD = 0.5
            ms.VLLM_BASE_URL = "http://localhost:11434"
            ms.VLLM_MODEL = "llama3.1:8b"
            ms.VLLM_TIMEOUT_SECONDS = 5.0
            ms.VLLM_MAX_CONCURRENCY = 2
            ms.AGENT_STRATEGY = "llm"
            with patch("app.agents.org_attention_agent.create_llm_client") as mc:
                client_mock = AsyncMock()
                client_mock.complete_json = AsyncMock(return_value=good_resp)
                mc.return_value = client_mock
                ctx = _ctx(propagation_signals=[_SIG_SALES_SUPPORT])
                result = await agent.run("mem-llm-1", ctx)
        assert result.outputs["rationale"] == "llm"
        assert result.outputs["overlap_count"] == 1

    @pytest.mark.asyncio
    async def test_llm_bad_response_falls_back_to_heuristic(self, agent):
        with patch("app.agents.org_attention_agent.settings") as ms:
            ms.ORG_ATTENTION_STRATEGY = "llm"
            ms.ORG_ATTENTION_THRESHOLD = 0.5
            ms.VLLM_BASE_URL = "http://localhost:11434"
            ms.VLLM_MODEL = "llama3.1:8b"
            ms.VLLM_TIMEOUT_SECONDS = 5.0
            ms.VLLM_MAX_CONCURRENCY = 2
            ms.AGENT_STRATEGY = "llm"
            with patch("app.agents.org_attention_agent.create_llm_client") as mc:
                client_mock = AsyncMock()
                client_mock.complete_json = AsyncMock(return_value={"bad": "response"})
                mc.return_value = client_mock
                ctx = _ctx(propagation_signals=[_SIG_SALES_SUPPORT])
                result = await agent.run("mem-llm-2", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_empty_content_forces_heuristic(self, agent):
        with patch("app.agents.org_attention_agent.settings") as ms:
            ms.ORG_ATTENTION_STRATEGY = "llm"
            ms.ORG_ATTENTION_THRESHOLD = 0.5
            ms.AGENT_STRATEGY = "llm"
            ctx = _ctx(content="")
            result = await agent.run("mem-llm-3", ctx)
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# TestAgentMetadata
# ---------------------------------------------------------------------------


class TestAgentMetadata:
    def test_name(self):
        assert OrgAttentionAgent().name == "OrgAttentionAgent"

    def test_dependencies(self):
        assert OrgAttentionAgent().dependencies() == ["SiloPropagationAgent"]

    def test_version(self):
        assert OrgAttentionAgent().version == "v1"


# ---------------------------------------------------------------------------
# TestRegistry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_short_name(self):
        agent = get_agent("org_attention")
        assert isinstance(agent, OrgAttentionAgent)

    def test_camel_case(self):
        agent = get_agent("orgattentionagent")
        assert isinstance(agent, OrgAttentionAgent)

    def test_full_name(self):
        agent = get_agent("OrgAttentionAgent")
        assert isinstance(agent, OrgAttentionAgent)
