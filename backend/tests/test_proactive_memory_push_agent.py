"""Tests for Phase 11 — ProactiveMemoryPushAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.proactive_memory_push_agent import (
    ProactiveMemoryPushAgent,
    build_push_candidates,
    detect_triggers,
    score_push_candidate,
    _delivery_hint,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_ALERT_ACME = {
    "topic": "acme_corp",
    "entity_type": "customer",
    "teams": ["sales", "support"],
    "team_count": 2,
    "overlap_score": 0.70,
    "alert_type": "parallel_work",
}

_ALERT_BUDGET = {
    "topic": "budget",
    "entity_type": "metric",
    "teams": ["engineering", "finance"],
    "team_count": 2,
    "overlap_score": 0.65,
    "alert_type": "parallel_work",
}

_ALERT_ACME_ENG = {
    "topic": "acme_corp",
    "entity_type": "customer",
    "teams": ["sales", "engineering"],
    "team_count": 2,
    "overlap_score": 0.72,
    "alert_type": "parallel_work",
}

_SIG_SALES_ACME = {
    "team": "sales",
    "topic": "acme_corp",
    "entity_type": "customer",
    "relevance": 0.80,
}

_SIG_SUPPORT_ACME = {
    "team": "support",
    "topic": "acme_corp",
    "entity_type": "customer",
    "relevance": 0.60,
}

_SIG_ENG_BUDGET = {
    "team": "engineering",
    "topic": "budget",
    "entity_type": "metric",
    "relevance": 0.50,
}


def _make_result(outputs: dict) -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="ProactiveMemoryPushAgent",
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
    attention_signals: list | None = None,
    overlap_alerts: list | None = None,
    content: str = "New project kickoff for Acme Corp Q3",
) -> dict:
    enrichment: dict = {
        "attention_signals": attention_signals or [],
        "overlap_alerts": overlap_alerts or [],
    }
    return {
        "tenant": {"org_id": "org-1", "org_slug": "test"},
        "memory": {
            "id": "mem-1",
            "content": content,
            "enrichment": enrichment,
        },
        "runtime": {"job_id": "job-1"},
    }


# ---------------------------------------------------------------------------
# TestDetectTriggers
# ---------------------------------------------------------------------------

class TestDetectTriggers:
    def test_project_start_kickoff(self):
        result = detect_triggers(content="Team kickoff meeting scheduled for Monday")
        assert "project_start" in result

    def test_project_start_new_project(self):
        result = detect_triggers(content="We are starting a new project for Acme Corp")
        assert "project_start" in result

    def test_project_start_initiated(self):
        result = detect_triggers(content="The migration was initiated last week")
        assert "project_start" in result

    def test_onboarding_new_hire(self):
        result = detect_triggers(content="New hire Sarah joins the engineering team")
        assert "onboarding" in result

    def test_onboarding_just_joined(self):
        result = detect_triggers(content="Alex just joined as a backend engineer")
        assert "onboarding" in result

    def test_both_triggers(self):
        result = detect_triggers(
            content="New project kickoff and we are onboarding two new employees"
        )
        assert "project_start" in result
        assert "onboarding" in result

    def test_empty_content(self):
        assert detect_triggers(content="") == []

    def test_no_matching_keywords(self):
        result = detect_triggers(content="Quarterly revenue report for Q3 2024")
        assert result == []

    def test_case_insensitive(self):
        result = detect_triggers(content="KICKOFF MEETING TOMORROW")
        assert "project_start" in result

    def test_onboarding_keyword(self):
        result = detect_triggers(content="Onboarding session scheduled for new member")
        assert "onboarding" in result


# ---------------------------------------------------------------------------
# TestScorePushCandidate
# ---------------------------------------------------------------------------

class TestScorePushCandidate:
    def test_overlap_alert_base(self):
        score = score_push_candidate(trigger_type="overlap_alert", relevance=0.0)
        assert score == pytest.approx(0.70, abs=0.001)

    def test_project_start_base(self):
        score = score_push_candidate(trigger_type="project_start", relevance=0.0)
        assert score == pytest.approx(0.65, abs=0.001)

    def test_onboarding_base(self):
        score = score_push_candidate(trigger_type="onboarding", relevance=0.0)
        assert score == pytest.approx(0.60, abs=0.001)

    def test_recurring_pattern_base(self):
        score = score_push_candidate(trigger_type="recurring_pattern", relevance=0.0)
        assert score == pytest.approx(0.75, abs=0.001)

    def test_relevance_boost(self):
        score = score_push_candidate(trigger_type="overlap_alert", relevance=1.0)
        assert score == pytest.approx(0.80, abs=0.001)

    def test_unknown_trigger_uses_default_weight(self):
        score = score_push_candidate(trigger_type="unknown_trigger", relevance=0.0)
        assert score == pytest.approx(0.55, abs=0.001)

    def test_score_capped_at_one(self):
        score = score_push_candidate(trigger_type="recurring_pattern", relevance=10.0)
        assert score == 1.0

    def test_empty_trigger_uses_default(self):
        score = score_push_candidate(trigger_type="", relevance=0.0)
        assert score == pytest.approx(0.55, abs=0.001)


# ---------------------------------------------------------------------------
# TestDeliveryHint
# ---------------------------------------------------------------------------

class TestDeliveryHint:
    def test_above_075_immediate(self):
        assert _delivery_hint(0.80) == "immediate_notification"

    def test_exactly_075_immediate(self):
        assert _delivery_hint(0.75) == "immediate_notification"

    def test_above_050_next_query(self):
        assert _delivery_hint(0.65) == "next_query_injection"

    def test_exactly_050_next_query(self):
        assert _delivery_hint(0.50) == "next_query_injection"

    def test_below_050_webhook(self):
        assert _delivery_hint(0.40) == "webhook"


# ---------------------------------------------------------------------------
# TestBuildPushCandidates
# ---------------------------------------------------------------------------

class TestBuildPushCandidates:
    def test_empty_inputs_returns_empty(self):
        result = build_push_candidates(
            triggers_detected=[],
            overlap_alerts=[],
            attention_signals=[],
            threshold=0.5,
            recurring_topics=set(),
        )
        assert result == []

    def test_overlap_alert_generates_per_team(self):
        result = build_push_candidates(
            triggers_detected=["overlap_alert"],
            overlap_alerts=[_ALERT_ACME],
            attention_signals=[],
            threshold=0.5,
            recurring_topics=set(),
        )
        teams = {c["recipient_team"] for c in result}
        assert "sales" in teams
        assert "support" in teams

    def test_recurring_pattern_overrides_overlap_alert(self):
        result = build_push_candidates(
            triggers_detected=["overlap_alert", "recurring_pattern"],
            overlap_alerts=[_ALERT_ACME],
            attention_signals=[],
            threshold=0.5,
            recurring_topics={"acme_corp"},
        )
        for c in result:
            if c["topic"] == "acme_corp":
                assert c["trigger_type"] == "recurring_pattern"

    def test_threshold_filters_low_urgency(self):
        # With threshold=0.9 only recurring_pattern+high_relevance passes
        result = build_push_candidates(
            triggers_detected=["overlap_alert"],
            overlap_alerts=[_ALERT_ACME],
            attention_signals=[],
            threshold=0.90,
            recurring_topics=set(),
        )
        assert result == []

    def test_content_trigger_uses_attention_signals(self):
        result = build_push_candidates(
            triggers_detected=["project_start"],
            overlap_alerts=[],
            attention_signals=[_SIG_SALES_ACME],
            threshold=0.5,
            recurring_topics=set(),
        )
        assert len(result) == 1
        assert result[0]["recipient_team"] == "sales"
        assert result[0]["trigger_type"] == "project_start"

    def test_sorted_by_urgency_descending(self):
        result = build_push_candidates(
            triggers_detected=["overlap_alert"],
            overlap_alerts=[_ALERT_ACME, _ALERT_BUDGET],
            attention_signals=[_SIG_SALES_ACME],
            threshold=0.5,
            recurring_topics=set(),
        )
        urgencies = [c["urgency_score"] for c in result]
        assert urgencies == sorted(urgencies, reverse=True)

    def test_deduplication_same_team_topic_trigger(self):
        # Two alerts with same topic — candidates should not duplicate
        result = build_push_candidates(
            triggers_detected=["overlap_alert"],
            overlap_alerts=[_ALERT_ACME, _ALERT_ACME],
            attention_signals=[],
            threshold=0.5,
            recurring_topics=set(),
        )
        keys = [(c["recipient_team"], c["topic"], c["trigger_type"]) for c in result]
        assert len(keys) == len(set(keys))

    def test_relevance_boost_from_attention_signal(self):
        result = build_push_candidates(
            triggers_detected=["overlap_alert"],
            overlap_alerts=[_ALERT_ACME],
            attention_signals=[_SIG_SALES_ACME],  # relevance=0.80
            threshold=0.5,
            recurring_topics=set(),
        )
        sales_cand = next(c for c in result if c["recipient_team"] == "sales")
        # overlap_alert base=0.70 + 0.80*0.10=0.08 → 0.78
        assert sales_cand["urgency_score"] == pytest.approx(0.78, abs=0.001)

    def test_candidates_capped_at_50(self):
        # Generate 60 unique (team, topic) combos via alerts
        alerts = [
            {
                "topic": f"topic_{i}",
                "entity_type": "concept",
                "teams": ["sales", "support"],
                "team_count": 2,
            }
            for i in range(30)
        ]
        result = build_push_candidates(
            triggers_detected=["overlap_alert"],
            overlap_alerts=alerts,
            attention_signals=[],
            threshold=0.5,
            recurring_topics=set(),
        )
        assert len(result) <= 50


# ---------------------------------------------------------------------------
# TestPhase11Heuristic
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return ProactiveMemoryPushAgent()


@pytest.fixture
def mock_settings():
    with patch("app.agents.proactive_memory_push_agent.settings") as m:
        m.PROACTIVE_PUSH_THRESHOLD = None
        m.PROACTIVE_PUSH_STRATEGY = "heuristic"
        m.AGENT_STRATEGY = "heuristic"
        yield m


class TestPhase11Heuristic:
    @pytest.mark.asyncio
    async def test_no_signals_no_content_empty_candidates(self, agent, mock_settings):
        result = await agent.run("mem-1", _ctx(content="plain revenue note"))
        assert result.outputs["push_candidates"] == []
        assert result.outputs["push_count"] == 0

    @pytest.mark.asyncio
    async def test_overlap_alerts_trigger_detected(self, agent, mock_settings):
        ctx = _ctx(content="plain note", overlap_alerts=[_ALERT_ACME])
        result = await agent.run("mem-1", ctx)
        assert "overlap_alert" in result.outputs["triggers_detected"]

    @pytest.mark.asyncio
    async def test_project_start_content_trigger(self, agent, mock_settings):
        ctx = _ctx(content="New project kickoff for Q3", overlap_alerts=[])
        result = await agent.run("mem-1", ctx)
        # No attention_signals so no content-trigger candidates, but trigger detected
        assert "project_start" in result.outputs["triggers_detected"]

    @pytest.mark.asyncio
    async def test_onboarding_content_trigger(self, agent, mock_settings):
        ctx = _ctx(content="Onboarding new hire starting Monday", overlap_alerts=[])
        result = await agent.run("mem-1", ctx)
        assert "onboarding" in result.outputs["triggers_detected"]

    @pytest.mark.asyncio
    async def test_recurring_pattern_multiple_alerts_same_topic(
        self, agent, mock_settings
    ):
        ctx = _ctx(
            content="budget review",
            overlap_alerts=[_ALERT_ACME, _ALERT_ACME_ENG],
            attention_signals=[],
        )
        result = await agent.run("mem-1", ctx)
        assert "recurring_pattern" in result.outputs["triggers_detected"]
        topics = [c["topic"] for c in result.outputs["push_candidates"]]
        assert "acme_corp" in topics

    @pytest.mark.asyncio
    async def test_confidence_zero_pushes_is_04(self, agent, mock_settings):
        ctx = _ctx(content="plain note")
        result = await agent.run("mem-1", ctx)
        assert result.outputs["confidence"] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_confidence_scales_with_push_count(self, agent, mock_settings):
        ctx = _ctx(
            content="plain note",
            overlap_alerts=[_ALERT_ACME, _ALERT_BUDGET],
            attention_signals=[],
        )
        result = await agent.run("mem-1", ctx)
        assert result.outputs["confidence"] > 0.4

    @pytest.mark.asyncio
    async def test_confidence_capped_at_09(self, agent, mock_settings):
        alerts = [
            {
                "topic": f"topic_{i}",
                "entity_type": "concept",
                "teams": ["sales", "support"],
                "team_count": 2,
            }
            for i in range(10)
        ]
        ctx = _ctx(content="plain note", overlap_alerts=alerts, attention_signals=[])
        result = await agent.run("mem-1", ctx)
        assert result.outputs["confidence"] <= 0.9

    @pytest.mark.asyncio
    async def test_rationale_is_heuristic(self, agent, mock_settings):
        result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_heuristic_when_content_empty(self, agent):
        with patch("app.agents.proactive_memory_push_agent.settings") as m:
            m.PROACTIVE_PUSH_THRESHOLD = None
            m.PROACTIVE_PUSH_STRATEGY = "llm"
            m.AGENT_STRATEGY = "llm"
            ctx = _ctx(content="", overlap_alerts=[_ALERT_ACME])
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_attention_signals_boost_urgency(self, agent, mock_settings):
        ctx = _ctx(
            content="plain note",
            overlap_alerts=[_ALERT_ACME],
            attention_signals=[_SIG_SALES_ACME],
        )
        result = await agent.run("mem-1", ctx)
        sales_cand = next(
            (c for c in result.outputs["push_candidates"] if c["recipient_team"] == "sales"),
            None,
        )
        assert sales_cand is not None
        assert sales_cand["urgency_score"] > 0.70

    @pytest.mark.asyncio
    async def test_combined_triggers_project_start_and_overlap(
        self, agent, mock_settings
    ):
        ctx = _ctx(
            content="New project kickoff for Acme Corp",
            overlap_alerts=[_ALERT_ACME],
            attention_signals=[_SIG_SALES_ACME],
        )
        result = await agent.run("mem-1", ctx)
        triggers = result.outputs["triggers_detected"]
        assert "project_start" in triggers
        assert "overlap_alert" in triggers

    @pytest.mark.asyncio
    async def test_push_candidates_have_delivery_hint(self, agent, mock_settings):
        ctx = _ctx(
            content="plain note",
            overlap_alerts=[_ALERT_ACME],
            attention_signals=[],
        )
        result = await agent.run("mem-1", ctx)
        for cand in result.outputs["push_candidates"]:
            assert "delivery_hint" in cand
            assert cand["delivery_hint"] in (
                "immediate_notification", "next_query_injection", "webhook"
            )

    @pytest.mark.asyncio
    async def test_push_count_matches_candidates_length(self, agent, mock_settings):
        ctx = _ctx(
            content="plain note",
            overlap_alerts=[_ALERT_ACME, _ALERT_BUDGET],
            attention_signals=[],
        )
        result = await agent.run("mem-1", ctx)
        assert result.outputs["push_count"] == len(result.outputs["push_candidates"])

    @pytest.mark.asyncio
    async def test_threshold_config_filters_candidates(self):
        with patch("app.agents.proactive_memory_push_agent.settings") as m:
            m.PROACTIVE_PUSH_THRESHOLD = 0.90
            m.PROACTIVE_PUSH_STRATEGY = "heuristic"
            m.AGENT_STRATEGY = "heuristic"
            agent = ProactiveMemoryPushAgent()
            ctx = _ctx(
                content="plain note",
                overlap_alerts=[_ALERT_ACME],
                attention_signals=[],
            )
            result = await agent.run("mem-1", ctx)
        # overlap_alert base=0.70 < 0.90 threshold → filtered out
        assert result.outputs["push_count"] == 0


# ---------------------------------------------------------------------------
# TestValidateOutputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def test_valid_result_passes(self):
        result = _make_result(
            {
                "push_candidates": [
                    {
                        "recipient_team": "sales",
                        "topic": "acme",
                        "trigger_type": "overlap_alert",
                        "urgency_score": 0.70,
                        "delivery_hint": "next_query_injection",
                    }
                ],
                "triggers_detected": ["overlap_alert"],
                "push_count": 1,
                "confidence": 0.6,
                "rationale": "heuristic",
            }
        )
        agent = ProactiveMemoryPushAgent()
        agent.validate_outputs(result)  # no exception

    def test_non_list_push_candidates_raises(self):
        result = _make_result({"push_candidates": "bad", "triggers_detected": []})
        agent = ProactiveMemoryPushAgent()
        with pytest.raises(ValueError, match="push_candidates must be a list"):
            agent.validate_outputs(result)

    def test_non_list_triggers_detected_raises(self):
        result = _make_result({"push_candidates": [], "triggers_detected": "bad"})
        agent = ProactiveMemoryPushAgent()
        with pytest.raises(ValueError, match="triggers_detected must be a list"):
            agent.validate_outputs(result)

    def test_non_dict_candidate_raises(self):
        result = _make_result(
            {"push_candidates": ["not_a_dict"], "triggers_detected": []}
        )
        agent = ProactiveMemoryPushAgent()
        with pytest.raises(ValueError, match="each push_candidates item must be a dict"):
            agent.validate_outputs(result)

    def test_missing_recipient_team_raises(self):
        result = _make_result(
            {
                "push_candidates": [{"trigger_type": "overlap_alert"}],
                "triggers_detected": [],
            }
        )
        agent = ProactiveMemoryPushAgent()
        with pytest.raises(ValueError, match="recipient_team"):
            agent.validate_outputs(result)

    def test_missing_trigger_type_raises(self):
        result = _make_result(
            {
                "push_candidates": [{"recipient_team": "sales"}],
                "triggers_detected": [],
            }
        )
        agent = ProactiveMemoryPushAgent()
        with pytest.raises(ValueError, match="trigger_type"):
            agent.validate_outputs(result)

    def test_skipped_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="ProactiveMemoryPushAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="skipped",
            confidence=0.5,
            outputs={"push_candidates": "bad"},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )
        agent = ProactiveMemoryPushAgent()
        agent.validate_outputs(result)  # no exception for non-success


# ---------------------------------------------------------------------------
# TestPhase11LLM
# ---------------------------------------------------------------------------

class TestPhase11LLM:
    @pytest.mark.asyncio
    async def test_llm_strategy_calls_ollama(self):
        with patch("app.agents.proactive_memory_push_agent.settings") as m:
            m.PROACTIVE_PUSH_THRESHOLD = None
            m.PROACTIVE_PUSH_STRATEGY = "llm"
            m.AGENT_STRATEGY = "llm"
            m.OLLAMA_BASE_URL = "http://localhost:11434"
            m.OLLAMA_MODEL = "llama3.1:8b"
            m.OLLAMA_TIMEOUT_SECONDS = 5.0
            m.OLLAMA_MAX_CONCURRENCY = 2
            mock_resp = {
                "push_candidates": [],
                "triggers_detected": ["overlap_alert"],
                "push_count": 0,
                "confidence": 0.5,
                "rationale": "llm test",
            }
            with patch(
                "app.agents.proactive_memory_push_agent.create_ollama_client"
            ) as mock_create:
                mock_client = AsyncMock()
                mock_client.complete_json = AsyncMock(return_value=mock_resp)
                mock_create.return_value = mock_client

                agent = ProactiveMemoryPushAgent()
                result = await agent.run(
                    "mem-1",
                    _ctx(
                        content="Q3 kickoff meeting scheduled",
                        overlap_alerts=[_ALERT_ACME],
                    ),
                )
        assert result.outputs["triggers_detected"] == ["overlap_alert"]
        mock_client.complete_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_llm_response_falls_back_to_heuristic(self):
        with patch("app.agents.proactive_memory_push_agent.settings") as m:
            m.PROACTIVE_PUSH_THRESHOLD = None
            m.PROACTIVE_PUSH_STRATEGY = "llm"
            m.AGENT_STRATEGY = "llm"
            m.OLLAMA_BASE_URL = "http://localhost:11434"
            m.OLLAMA_MODEL = "llama3.1:8b"
            m.OLLAMA_TIMEOUT_SECONDS = 5.0
            m.OLLAMA_MAX_CONCURRENCY = 2
            with patch(
                "app.agents.proactive_memory_push_agent.create_ollama_client"
            ) as mock_create:
                mock_client = AsyncMock()
                mock_client.complete_json = AsyncMock(return_value={"bad": "response"})
                mock_create.return_value = mock_client

                agent = ProactiveMemoryPushAgent()
                result = await agent.run(
                    "mem-1",
                    _ctx(
                        content="Q3 kickoff meeting",
                        overlap_alerts=[_ALERT_ACME],
                    ),
                )
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_none_llm_response_falls_back_to_heuristic(self):
        with patch("app.agents.proactive_memory_push_agent.settings") as m:
            m.PROACTIVE_PUSH_THRESHOLD = None
            m.PROACTIVE_PUSH_STRATEGY = "llm"
            m.AGENT_STRATEGY = "llm"
            m.OLLAMA_BASE_URL = "http://localhost:11434"
            m.OLLAMA_MODEL = "llama3.1:8b"
            m.OLLAMA_TIMEOUT_SECONDS = 5.0
            m.OLLAMA_MAX_CONCURRENCY = 2
            with patch(
                "app.agents.proactive_memory_push_agent.create_ollama_client"
            ) as mock_create:
                mock_client = AsyncMock()
                mock_client.complete_json = AsyncMock(return_value=None)
                mock_create.return_value = mock_client

                agent = ProactiveMemoryPushAgent()
                result = await agent.run(
                    "mem-1",
                    _ctx(
                        content="Q3 kickoff meeting",
                        overlap_alerts=[_ALERT_ACME],
                    ),
                )
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# TestAgentMetadata
# ---------------------------------------------------------------------------

class TestAgentMetadata:
    def test_name(self):
        assert ProactiveMemoryPushAgent.name == "ProactiveMemoryPushAgent"

    def test_version(self):
        assert ProactiveMemoryPushAgent.version == "v1"

    def test_dependencies_includes_org_attention(self):
        agent = ProactiveMemoryPushAgent()
        assert "OrgAttentionAgent" in agent.dependencies()


# ---------------------------------------------------------------------------
# TestRegistry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_agent_proactive_push(self):
        agent = get_agent("proactive_push")
        assert isinstance(agent, ProactiveMemoryPushAgent)

    def test_get_agent_proactivememorypush(self):
        agent = get_agent("proactivememorypush")
        assert isinstance(agent, ProactiveMemoryPushAgent)

    def test_get_agent_case_insensitive_full_name(self):
        agent = get_agent("ProactiveMemoryPushAgent")
        assert isinstance(agent, ProactiveMemoryPushAgent)
