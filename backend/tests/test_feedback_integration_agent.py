"""Tests for FeedbackIntegrationAgent — Phase 24."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.feedback_integration_agent import (
    FeedbackIntegrationAgent,
    apply_credibility_correction,
    apply_resolution_correction,
    apply_playbook_correction,
    compute_feedback_confidence,
    run_heuristic,
    _ema,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx(enrichment: dict | None = None, human_verdict: dict | None = None) -> dict:
    return {
        "memory": {
            "enrichment": enrichment or {},
            "human_verdict": human_verdict,
        },
        "runtime": {"job_id": "trace-24"},
    }


def _full_enrichment() -> dict:
    return {
        "credibility_score": 0.60,
        "source_tier": "internal",
        "resolution_rate": 0.50,
        "escalation_targets": ["eng-lead"],
        "matched_playbook_id": "pb-001",
        "playbook_confidence": 0.75,
        "uncertainty_level": "medium",
        "review_recommended": True,
    }


def _approve_verdict() -> dict:
    return {
        "verdict": "approve",
        "credibility_override": None,
        "resolution_confirmed": True,
        "playbook_correct": None,
        "reviewer_id": "reviewer-alice",
    }


def _reject_verdict() -> dict:
    return {
        "verdict": "reject",
        "credibility_override": None,
        "resolution_confirmed": False,
        "playbook_correct": False,
        "reviewer_id": "reviewer-bob",
    }


# ---------------------------------------------------------------------------
# TestEma
# ---------------------------------------------------------------------------

class TestEma:
    def test_blends_toward_new_signal(self):
        result = _ema(0.6, 0.9)
        assert 0.6 < result < 0.9

    def test_alpha_zero_keeps_current(self):
        assert _ema(0.6, 0.9, alpha=0.0) == pytest.approx(0.6)

    def test_alpha_one_equals_new_signal(self):
        assert _ema(0.6, 0.9, alpha=1.0) == pytest.approx(0.9)

    def test_default_alpha_25(self):
        expected = round(0.75 * 0.6 + 0.25 * 0.9, 4)
        assert _ema(0.6, 0.9) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestApplyCredibilityCorrection
# ---------------------------------------------------------------------------

class TestApplyCredibilityCorrection:
    def test_approve_verdict_boosts_credibility(self):
        adj, delta = apply_credibility_correction({"credibility_score": 0.60}, {"verdict": "approve"})
        assert adj is not None
        assert adj > 0.60
        assert delta > 0

    def test_reject_verdict_lowers_credibility(self):
        adj, delta = apply_credibility_correction({"credibility_score": 0.60}, {"verdict": "reject"})
        assert adj is not None
        assert adj < 0.60
        assert delta < 0

    def test_explicit_override_used_over_verdict(self):
        adj, _ = apply_credibility_correction(
            {"credibility_score": 0.50},
            {"verdict": "reject", "credibility_override": 0.95},
        )
        assert adj is not None
        assert adj > 0.50  # override wins over reject verdict

    def test_partial_verdict_no_override_returns_none(self):
        adj, delta = apply_credibility_correction({"credibility_score": 0.50}, {"verdict": "partial"})
        assert adj is None
        assert delta == 0.0

    def test_no_verdict_no_override_returns_none(self):
        adj, delta = apply_credibility_correction({"credibility_score": 0.50}, {})
        assert adj is None
        assert delta == 0.0

    def test_missing_current_defaults_to_0_5(self):
        adj, _ = apply_credibility_correction({}, {"verdict": "approve"})
        assert adj is not None

    def test_override_clamped_to_1(self):
        adj, _ = apply_credibility_correction({"credibility_score": 0.8}, {"credibility_override": 99.0})
        assert adj is not None
        assert adj <= 1.0

    def test_override_clamped_to_0(self):
        adj, _ = apply_credibility_correction({"credibility_score": 0.5}, {"credibility_override": -5.0})
        assert adj is not None
        assert adj >= 0.0

    def test_delta_sign_matches_direction(self):
        adj, delta = apply_credibility_correction({"credibility_score": 0.50}, {"verdict": "approve"})
        assert delta > 0  # approve → upward


# ---------------------------------------------------------------------------
# TestApplyResolutionCorrection
# ---------------------------------------------------------------------------

class TestApplyResolutionCorrection:
    def test_confirmed_true_targets_1(self):
        delta = apply_resolution_correction({"resolution_rate": 0.50}, {"resolution_confirmed": True})
        assert delta is not None
        assert delta > 0  # moving toward 1.0

    def test_confirmed_false_targets_0(self):
        delta = apply_resolution_correction({"resolution_rate": 0.80}, {"resolution_confirmed": False})
        assert delta is not None
        assert delta < 0  # moving toward 0.0

    def test_none_returns_none(self):
        assert apply_resolution_correction({"resolution_rate": 0.5}, {}) is None

    def test_already_fully_resolved_returns_zero_delta(self):
        delta = apply_resolution_correction({"resolution_rate": 1.0}, {"resolution_confirmed": True})
        assert delta == pytest.approx(0.0)

    def test_missing_current_defaults_to_0(self):
        delta = apply_resolution_correction({}, {"resolution_confirmed": True})
        assert delta == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestApplyPlaybookCorrection
# ---------------------------------------------------------------------------

class TestApplyPlaybookCorrection:
    def test_playbook_wrong_reduces_confidence(self):
        delta = apply_playbook_correction(
            {"playbook_confidence": 0.75}, {"playbook_correct": False}
        )
        assert delta is not None
        assert delta == pytest.approx(-0.10)

    def test_playbook_correct_true_no_change(self):
        assert apply_playbook_correction({"playbook_confidence": 0.75}, {"playbook_correct": True}) is None

    def test_playbook_correct_none_no_change(self):
        assert apply_playbook_correction({"playbook_confidence": 0.75}, {}) is None

    def test_floors_at_zero(self):
        delta = apply_playbook_correction(
            {"playbook_confidence": 0.05}, {"playbook_correct": False}
        )
        assert delta is not None
        assert 0.05 + delta >= 0.0

    def test_missing_playbook_confidence_returns_none(self):
        assert apply_playbook_correction({}, {"playbook_correct": False}) is None


# ---------------------------------------------------------------------------
# TestComputeFeedbackConfidence
# ---------------------------------------------------------------------------

class TestComputeFeedbackConfidence:
    def test_reviewer_id_boosts(self):
        base = compute_feedback_confidence(["credibility_score"], None, "approve")
        boosted = compute_feedback_confidence(["credibility_score"], "alice", "approve")
        assert boosted > base

    def test_no_corrections_penalises(self):
        conf = compute_feedback_confidence([], None, None)
        assert conf < 0.65

    def test_multiple_corrections_boosts(self):
        one = compute_feedback_confidence(["credibility_score"], None, "approve")
        two = compute_feedback_confidence(["credibility_score", "resolution_rate"], None, "approve")
        assert two > one

    def test_clear_verdict_boosts(self):
        partial = compute_feedback_confidence(["f"], None, "partial")
        clear = compute_feedback_confidence(["f"], None, "approve")
        assert clear > partial

    def test_always_in_range(self):
        for fields, reviewer, verdict in [
            ([], None, None),
            (["a", "b", "c"], "bob", "approve"),
            (["a"], "alice", "reject"),
        ]:
            conf = compute_feedback_confidence(fields, reviewer, verdict)
            assert 0.30 <= conf <= 0.90


# ---------------------------------------------------------------------------
# TestRunHeuristic
# ---------------------------------------------------------------------------

class TestRunHeuristic:
    def test_no_verdict_returns_not_applied(self):
        result = run_heuristic({}, None)
        assert result["feedback_applied"] is False
        assert result["corrected_fields"] == []
        assert result["correction_delta"] == {}

    def test_empty_verdict_dict_returns_not_applied(self):
        result = run_heuristic({}, {})
        assert result["feedback_applied"] is False

    def test_approve_applies_credibility(self):
        result = run_heuristic(_full_enrichment(), _approve_verdict())
        assert result["feedback_applied"] is True
        assert "credibility_score" in result["corrected_fields"]

    def test_approve_with_resolution_applies_both(self):
        result = run_heuristic(_full_enrichment(), _approve_verdict())
        assert "resolution_rate" in result["corrected_fields"]

    def test_reject_lowers_credibility(self):
        result = run_heuristic(_full_enrichment(), _reject_verdict())
        assert result["correction_delta"].get("credibility_score", 0) < 0

    def test_reject_with_wrong_playbook(self):
        result = run_heuristic(_full_enrichment(), _reject_verdict())
        assert "playbook_confidence" in result["corrected_fields"]
        assert result["correction_delta"]["playbook_confidence"] == pytest.approx(-0.10)

    def test_rationale_is_heuristic(self):
        assert run_heuristic({}, None)["rationale"] == "heuristic"

    def test_adjusted_credibility_in_range(self):
        result = run_heuristic(_full_enrichment(), _approve_verdict())
        adj = result["adjusted_credibility"]
        if adj is not None:
            assert 0.0 <= adj <= 1.0

    def test_correction_delta_types(self):
        result = run_heuristic(_full_enrichment(), _approve_verdict())
        for v in result["correction_delta"].values():
            assert isinstance(v, float)

    def test_partial_verdict_with_explicit_override(self):
        verdict = {"verdict": "partial", "credibility_override": 0.85, "reviewer_id": "admin"}
        result = run_heuristic({"credibility_score": 0.50}, verdict)
        assert result["feedback_applied"] is True
        assert "credibility_score" in result["corrected_fields"]


# ---------------------------------------------------------------------------
# TestFeedbackIntegrationAgentHeuristic
# ---------------------------------------------------------------------------

class TestFeedbackIntegrationAgentHeuristic:
    @pytest.mark.asyncio
    async def test_runs_heuristic_strategy(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment(), _approve_verdict()))
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.trace_id == "trace-24"

    @pytest.mark.asyncio
    async def test_no_trace_id_when_missing(self):
        agent = FeedbackIntegrationAgent()
        ctx = {"memory": {"enrichment": {}}, "runtime": {}}
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.trace_id is None

    @pytest.mark.asyncio
    async def test_memory_id_preserved(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-xyz", _ctx())
        assert result.memory_id == "mem-xyz"

    @pytest.mark.asyncio
    async def test_timestamps_set(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.finished_at >= result.started_at

    @pytest.mark.asyncio
    async def test_no_verdict_not_applied(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment(), None))
        assert result.outputs["feedback_applied"] is False

    @pytest.mark.asyncio
    async def test_approve_applies_corrections(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment(), _approve_verdict()))
        assert result.outputs["feedback_applied"] is True
        assert len(result.outputs["corrected_fields"]) >= 1

    @pytest.mark.asyncio
    async def test_corrected_fields_is_list(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert isinstance(result.outputs["corrected_fields"], list)

    @pytest.mark.asyncio
    async def test_correction_delta_is_dict(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment(), _approve_verdict()))
        assert isinstance(result.outputs["correction_delta"], dict)

    @pytest.mark.asyncio
    async def test_feedback_confidence_in_range(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment(), _approve_verdict()))
        assert 0.30 <= result.outputs["feedback_confidence"] <= 0.90

    @pytest.mark.asyncio
    async def test_agent_name(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.agent_name == "FeedbackIntegrationAgent"

    @pytest.mark.asyncio
    async def test_agent_strategy_fallback(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = None
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_reject_lowers_credibility(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment(), _reject_verdict()))
        delta = result.outputs["correction_delta"].get("credibility_score", 0)
        assert delta < 0

    @pytest.mark.asyncio
    async def test_warnings_empty(self):
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.warnings == []


# ---------------------------------------------------------------------------
# TestFeedbackIntegrationAgentLLM
# ---------------------------------------------------------------------------

def _valid_llm_response() -> dict:
    return {
        "feedback_applied": True,
        "corrected_fields": ["credibility_score"],
        "correction_delta": {"credibility_score": 0.075},
        "adjusted_credibility": 0.6750,
        "feedback_confidence": 0.72,
        "rationale": "LLM applied EMA correction.",
    }


class TestFeedbackIntegrationAgentLLM:
    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = FeedbackIntegrationAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=_valid_llm_response())
        with patch("app.agents.feedback_integration_agent.settings") as s, \
             patch("app.agents.feedback_integration_agent.create_ollama_client", return_value=mock_client):
            s.FEEDBACK_INTEGRATION_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "llama3.1:8b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx(_full_enrichment(), _approve_verdict()))
        assert result.status == "success"
        assert result.outputs["feedback_applied"] is True

    @pytest.mark.asyncio
    async def test_llm_none_response_falls_back(self):
        agent = FeedbackIntegrationAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=None)
        with patch("app.agents.feedback_integration_agent.settings") as s, \
             patch("app.agents.feedback_integration_agent.create_ollama_client", return_value=mock_client):
            s.FEEDBACK_INTEGRATION_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "llama3.1:8b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_missing_feedback_applied_falls_back(self):
        agent = FeedbackIntegrationAgent()
        bad = {k: v for k, v in _valid_llm_response().items() if k != "feedback_applied"}
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=bad)
        with patch("app.agents.feedback_integration_agent.settings") as s, \
             patch("app.agents.feedback_integration_agent.create_ollama_client", return_value=mock_client):
            s.FEEDBACK_INTEGRATION_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "llama3.1:8b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_wrong_corrected_fields_type_falls_back(self):
        agent = FeedbackIntegrationAgent()
        bad = dict(_valid_llm_response(), corrected_fields="not-a-list")
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=bad)
        with patch("app.agents.feedback_integration_agent.settings") as s, \
             patch("app.agents.feedback_integration_agent.create_ollama_client", return_value=mock_client):
            s.FEEDBACK_INTEGRATION_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "llama3.1:8b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_wrong_correction_delta_type_falls_back(self):
        agent = FeedbackIntegrationAgent()
        bad = dict(_valid_llm_response(), correction_delta="not-a-dict")
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=bad)
        with patch("app.agents.feedback_integration_agent.settings") as s, \
             patch("app.agents.feedback_integration_agent.create_ollama_client", return_value=mock_client):
            s.FEEDBACK_INTEGRATION_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "llama3.1:8b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_confidence_used(self):
        agent = FeedbackIntegrationAgent()
        resp = dict(_valid_llm_response(), feedback_confidence=0.88)
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=resp)
        with patch("app.agents.feedback_integration_agent.settings") as s, \
             patch("app.agents.feedback_integration_agent.create_ollama_client", return_value=mock_client):
            s.FEEDBACK_INTEGRATION_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "llama3.1:8b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.confidence == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_llm_non_dict_response_falls_back(self):
        agent = FeedbackIntegrationAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=[1, 2, 3])
        with patch("app.agents.feedback_integration_agent.settings") as s, \
             patch("app.agents.feedback_integration_agent.create_ollama_client", return_value=mock_client):
            s.FEEDBACK_INTEGRATION_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "llama3.1:8b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# TestFeedbackIntegrationAgentValidateOutputs
# ---------------------------------------------------------------------------

class TestFeedbackIntegrationAgentValidateOutputs:
    def _agent(self):
        return FeedbackIntegrationAgent()

    def _result(self, outputs: dict) -> AgentResult:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="FeedbackIntegrationAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="success",
            confidence=0.70,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )

    def _valid(self) -> dict:
        return {
            "feedback_applied": False,
            "corrected_fields": [],
            "correction_delta": {},
            "adjusted_credibility": None,
            "feedback_confidence": 0.60,
            "rationale": "heuristic",
        }

    def test_valid_outputs_pass(self):
        self._agent().validate_outputs(self._result(self._valid()))

    def test_feedback_applied_not_bool_raises(self):
        with pytest.raises(ValueError, match="feedback_applied"):
            self._agent().validate_outputs(self._result(dict(self._valid(), feedback_applied="yes")))

    def test_corrected_fields_not_list_raises(self):
        with pytest.raises(ValueError, match="corrected_fields"):
            self._agent().validate_outputs(self._result(dict(self._valid(), corrected_fields="oops")))

    def test_correction_delta_not_dict_raises(self):
        with pytest.raises(ValueError, match="correction_delta"):
            self._agent().validate_outputs(self._result(dict(self._valid(), correction_delta=[1, 2])))

    def test_adjusted_credibility_non_numeric_raises(self):
        with pytest.raises(ValueError, match="adjusted_credibility"):
            self._agent().validate_outputs(self._result(dict(self._valid(), adjusted_credibility="high")))

    def test_adjusted_credibility_out_of_range_raises(self):
        with pytest.raises(ValueError, match="adjusted_credibility"):
            self._agent().validate_outputs(self._result(dict(self._valid(), adjusted_credibility=1.5)))

    def test_adjusted_credibility_none_passes(self):
        self._agent().validate_outputs(self._result(dict(self._valid(), adjusted_credibility=None)))

    def test_adjusted_credibility_float_in_range_passes(self):
        self._agent().validate_outputs(self._result(dict(self._valid(), adjusted_credibility=0.75)))

    def test_feedback_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="feedback_confidence"):
            self._agent().validate_outputs(self._result(dict(self._valid(), feedback_confidence=1.5)))

    def test_failed_status_skips_validation(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="FeedbackIntegrationAgent",
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
        self._agent().validate_outputs(result)  # must not raise


# ---------------------------------------------------------------------------
# TestFeedbackIntegrationAgentMetadata
# ---------------------------------------------------------------------------

class TestFeedbackIntegrationAgentMetadata:
    def test_name(self):
        assert FeedbackIntegrationAgent.name == "FeedbackIntegrationAgent"

    def test_version(self):
        assert FeedbackIntegrationAgent.version == "v1"

    def test_dependencies_include_uncertainty(self):
        assert "UncertaintyReportingAgent" in FeedbackIntegrationAgent().dependencies()

    def test_dependencies_include_credibility(self):
        assert "CredibilityAgent" in FeedbackIntegrationAgent().dependencies()

    def test_dependencies_is_list(self):
        assert isinstance(FeedbackIntegrationAgent().dependencies(), list)


# ---------------------------------------------------------------------------
# TestFeedbackIntegrationRegistry
# ---------------------------------------------------------------------------

class TestFeedbackIntegrationRegistry:
    def test_feedback_integration(self):
        assert isinstance(get_agent("feedback_integration"), FeedbackIntegrationAgent)

    def test_feedbackintegration(self):
        assert isinstance(get_agent("feedbackintegration"), FeedbackIntegrationAgent)

    def test_feedbackintegrationagent(self):
        assert isinstance(get_agent("feedbackintegrationagent"), FeedbackIntegrationAgent)

    def test_feedback_agent(self):
        assert isinstance(get_agent("feedback_agent"), FeedbackIntegrationAgent)

    def test_unknown_returns_none(self):
        assert get_agent("no_such_agent_xyz") is None


# ---------------------------------------------------------------------------
# TestFeedbackIntegrationIntegration
# ---------------------------------------------------------------------------

class TestFeedbackIntegrationIntegration:
    @pytest.mark.asyncio
    async def test_approve_full_pipeline(self):
        """Approve verdict adjusts credibility up and confirms resolution."""
        enrichment = {
            "credibility_score": 0.55,
            "resolution_rate": 0.40,
            "matched_playbook_id": "pb-001",
            "playbook_confidence": 0.70,
        }
        verdict = {
            "verdict": "approve",
            "resolution_confirmed": True,
            "reviewer_id": "alice",
        }
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-full", _ctx(enrichment, verdict))
        assert result.outputs["feedback_applied"] is True
        assert result.outputs["correction_delta"]["credibility_score"] > 0
        assert result.outputs["correction_delta"]["resolution_rate"] > 0

    @pytest.mark.asyncio
    async def test_reject_full_pipeline(self):
        """Reject verdict lowers credibility, marks conflict open, penalises playbook."""
        enrichment = {
            "credibility_score": 0.70,
            "resolution_rate": 0.80,
            "playbook_confidence": 0.75,
        }
        verdict = {
            "verdict": "reject",
            "resolution_confirmed": False,
            "playbook_correct": False,
            "reviewer_id": "bob",
        }
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-full", _ctx(enrichment, verdict))
        assert result.outputs["feedback_applied"] is True
        assert result.outputs["correction_delta"]["credibility_score"] < 0
        assert result.outputs["correction_delta"]["resolution_rate"] < 0
        assert result.outputs["correction_delta"]["playbook_confidence"] == pytest.approx(-0.10)

    @pytest.mark.asyncio
    async def test_partial_with_explicit_credibility_override(self):
        enrichment = {"credibility_score": 0.50}
        verdict = {"verdict": "partial", "credibility_override": 0.85, "reviewer_id": "carol"}
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(enrichment, verdict))
        assert result.outputs["feedback_applied"] is True
        adj = result.outputs["adjusted_credibility"]
        assert adj is not None and adj > 0.50

    @pytest.mark.asyncio
    async def test_no_verdict_outputs_json_serialisable(self):
        import json
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx({}, None))
        json.dumps(result.outputs)  # must not raise

    @pytest.mark.asyncio
    async def test_repeated_approvals_compound(self):
        """Two approve rounds push credibility higher than one."""
        enrichment = {"credibility_score": 0.50}
        verdict = {"verdict": "approve"}
        agent = FeedbackIntegrationAgent()
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            r1 = await agent.run("mem-1", _ctx(enrichment, verdict))

        adj1 = r1.outputs["adjusted_credibility"]
        enrichment2 = {"credibility_score": adj1}
        with patch("app.agents.feedback_integration_agent.settings") as s:
            s.FEEDBACK_INTEGRATION_STRATEGY = "heuristic"
            r2 = await agent.run("mem-1", _ctx(enrichment2, verdict))

        assert r2.outputs["adjusted_credibility"] > adj1
