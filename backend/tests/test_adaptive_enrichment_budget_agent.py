"""Tests for AdaptiveEnrichmentBudgetAgent — Phase 35."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.adaptive_enrichment_budget_agent import (
    AdaptiveEnrichmentBudgetAgent,
    _CORE_AGENTS,
    _IMPORTANCE_HIGH,
    _IMPORTANCE_LOW,
    _MIN_FEEDBACK_CONF,
    _MIN_VALUE_TO_SKIP,
    compute_importance_score,
    compute_budget_confidence,
    score_agent_value,
    should_skip_agent,
    run_heuristic,
)
from app.agents.registry import get_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mk_enrichment(
    *,
    credibility_score: float | None = 0.7,
    uncertainty_level: str = "low",
    anomaly_detected: bool = False,
    feedback_applied: bool = False,
    feedback_confidence: float = 0.0,
    correction_delta: dict | None = None,
    agent_order: list[str] | None = None,
) -> dict:
    return {
        "credibility_score": credibility_score,
        "uncertainty_level": uncertainty_level,
        "anomaly_detected": anomaly_detected,
        "feedback_applied": feedback_applied,
        "feedback_confidence": feedback_confidence,
        "correction_delta": correction_delta or {},
        "agent_order": agent_order,
    }


def _mk_readiness(
    *,
    is_ready: bool = True,
    active: bool = True,
    params: dict | None = None,
) -> dict:
    return {"is_ready": is_ready, "active": active, "params": params or {}}


def _mk_context(
    enrichment: dict | None = None,
    feature_readiness: dict | None = None,
    candidate_agents: list[str] | None = None,
    strategy: str = "heuristic",
) -> dict:
    return {
        "memory": {"enrichment": enrichment or {}},
        "feature_readiness": feature_readiness or {},
        "candidate_agents": candidate_agents or [],
        "runtime": {"job_id": "test-job"},
    }


_ALL_CANDIDATES = [
    "MetadataExtractionAgent",
    "EntityResolutionAgent",
    "CredibilityAgent",
    "UncertaintyReportingAgent",
    "PlaybookAgent",
    "AnomalyDetectionAgent",
    "MemoryDecayAgent",
    "FeedbackIntegrationAgent",
]


# ---------------------------------------------------------------------------
# compute_importance_score
# ---------------------------------------------------------------------------

class TestComputeImportanceScore:
    def test_baseline_no_signals(self):
        score = compute_importance_score({})
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.30, abs=0.01)

    def test_high_credibility_boosts_score(self):
        score = compute_importance_score({"credibility_score": 1.0})
        assert score > 0.50

    def test_high_uncertainty_adds(self):
        score_high = compute_importance_score({"uncertainty_level": "high"})
        score_low = compute_importance_score({"uncertainty_level": "low"})
        assert score_high > score_low

    def test_medium_uncertainty_adds_less_than_high(self):
        s_high = compute_importance_score({"uncertainty_level": "high"})
        s_med = compute_importance_score({"uncertainty_level": "medium"})
        assert s_high > s_med

    def test_anomaly_boosts_score(self):
        without = compute_importance_score({"anomaly_detected": False})
        with_anomaly = compute_importance_score({"anomaly_detected": True})
        assert with_anomaly > without

    def test_all_signals_caps_at_one(self):
        score = compute_importance_score({
            "credibility_score": 1.0,
            "uncertainty_level": "high",
            "anomaly_detected": True,
        })
        assert score <= 1.0

    def test_none_credibility_treated_as_zero(self):
        score = compute_importance_score({"credibility_score": None})
        assert score == pytest.approx(0.30, abs=0.01)

    def test_unknown_uncertainty_no_bonus(self):
        s1 = compute_importance_score({"uncertainty_level": "unknown"})
        s2 = compute_importance_score({})
        assert s1 == s2


# ---------------------------------------------------------------------------
# score_agent_value
# ---------------------------------------------------------------------------

class TestScoreAgentValue:
    def test_returns_zero_below_confidence_threshold(self):
        delta = {"credibility": 0.5}
        val = score_agent_value("CredibilityAgent", delta, _MIN_FEEDBACK_CONF - 0.01)
        assert val == 0.0

    def test_matches_agent_name_fragment(self):
        delta = {"credibility_score": 0.3}
        val = score_agent_value("CredibilityAgent", delta, 0.80)
        assert val == pytest.approx(0.3, abs=0.001)

    def test_sums_multiple_matching_fields(self):
        delta = {"playbook_confidence": 0.2, "playbook_match": 0.1}
        val = score_agent_value("PlaybookAgent", delta, 0.80)
        assert val == pytest.approx(0.3, abs=0.001)

    def test_no_matching_fields_returns_zero(self):
        delta = {"credibility_score": 0.5}
        val = score_agent_value("PlaybookAgent", delta, 0.80)
        assert val == 0.0

    def test_empty_delta_returns_zero(self):
        val = score_agent_value("AnyAgent", {}, 0.80)
        assert val == 0.0

    def test_absolute_value_of_negative_delta(self):
        delta = {"anomaly_score": -0.4}
        val = score_agent_value("AnomalyDetectionAgent", delta, 0.80)
        assert val == pytest.approx(0.4, abs=0.001)


# ---------------------------------------------------------------------------
# should_skip_agent
# ---------------------------------------------------------------------------

class TestShouldSkipAgent:
    def test_core_agent_never_skipped(self):
        for agent in _CORE_AGENTS:
            skip, reason = should_skip_agent(agent, 0.0, 0.0, None)
            assert not skip
            assert reason == "core"

    def test_inactive_agent_skipped(self):
        skip, reason = should_skip_agent(
            "PlaybookAgent", 0.5, 0.0, {"is_ready": True, "active": False}
        )
        assert skip
        assert reason == "inactive"

    def test_not_ready_agent_skipped(self):
        skip, reason = should_skip_agent(
            "PlaybookAgent", 0.5, 0.0, {"is_ready": False, "active": True}
        )
        assert skip
        assert reason == "not_ready"

    def test_high_importance_includes_all_ready(self):
        skip, reason = should_skip_agent(
            "PlaybookAgent", _IMPORTANCE_HIGH, 0.0, {"is_ready": True, "active": True}
        )
        assert not skip
        assert reason == "high_importance"

    def test_low_importance_low_value_skipped(self):
        skip, reason = should_skip_agent(
            "MemoryDecayAgent",
            _IMPORTANCE_LOW - 0.01,
            _MIN_VALUE_TO_SKIP - 0.001,
            None,
        )
        assert skip
        assert reason == "low_importance_low_value"

    def test_zero_value_medium_importance_skipped(self):
        # Between LOW and HIGH with zero value → skip
        importance = (_IMPORTANCE_LOW + _IMPORTANCE_HIGH) / 2
        skip, reason = should_skip_agent("MemoryDecayAgent", importance, 0.0, None)
        assert skip
        assert reason == "zero_value"

    def test_non_zero_value_medium_importance_included(self):
        importance = (_IMPORTANCE_LOW + _IMPORTANCE_HIGH) / 2
        skip, _ = should_skip_agent("PlaybookAgent", importance, 0.20, None)
        assert not skip

    def test_none_readiness_uses_defaults(self):
        # None readiness → skip only by importance/value logic
        skip, _ = should_skip_agent(
            "MemoryDecayAgent", _IMPORTANCE_HIGH, 0.5, None
        )
        assert not skip

    def test_active_and_ready_respected(self):
        skip, reason = should_skip_agent(
            "PlaybookAgent",
            _IMPORTANCE_LOW - 0.01,
            0.0,
            {"is_ready": True, "active": True},
        )
        assert skip


# ---------------------------------------------------------------------------
# compute_budget_confidence
# ---------------------------------------------------------------------------

class TestComputeBudgetConfidence:
    def test_baseline_no_signals(self):
        c = compute_budget_confidence(
            has_feedback=False,
            feedback_confidence=0.0,
            has_readiness=False,
            importance=0.3,
            skip_count=0,
            total=0,
        )
        assert 0.0 <= c <= 1.0

    def test_feedback_boosts_confidence(self):
        c_no = compute_budget_confidence(
            has_feedback=False,
            feedback_confidence=0.0,
            has_readiness=False,
            importance=0.3,
            skip_count=0,
            total=5,
        )
        c_yes = compute_budget_confidence(
            has_feedback=True,
            feedback_confidence=0.80,
            has_readiness=False,
            importance=0.3,
            skip_count=0,
            total=5,
        )
        assert c_yes > c_no

    def test_readiness_boosts_confidence(self):
        c_no = compute_budget_confidence(
            has_feedback=False,
            feedback_confidence=0.0,
            has_readiness=False,
            importance=0.3,
            skip_count=0,
            total=5,
        )
        c_yes = compute_budget_confidence(
            has_feedback=False,
            feedback_confidence=0.0,
            has_readiness=True,
            importance=0.3,
            skip_count=0,
            total=5,
        )
        assert c_yes > c_no

    def test_high_skip_ratio_reduces_confidence(self):
        c_low_skip = compute_budget_confidence(
            has_feedback=True,
            feedback_confidence=0.80,
            has_readiness=True,
            importance=0.5,
            skip_count=1,
            total=10,
        )
        c_high_skip = compute_budget_confidence(
            has_feedback=True,
            feedback_confidence=0.80,
            has_readiness=True,
            importance=0.5,
            skip_count=9,
            total=10,
        )
        assert c_high_skip < c_low_skip

    def test_caps_at_max(self):
        c = compute_budget_confidence(
            has_feedback=True,
            feedback_confidence=1.0,
            has_readiness=True,
            importance=1.0,
            skip_count=0,
            total=10,
        )
        assert c <= 0.90

    def test_zero_total_no_crash(self):
        c = compute_budget_confidence(
            has_feedback=False,
            feedback_confidence=0.0,
            has_readiness=False,
            importance=0.3,
            skip_count=0,
            total=0,
        )
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# run_heuristic
# ---------------------------------------------------------------------------

class TestRunHeuristic:
    def test_empty_candidates(self):
        out = run_heuristic({}, {}, [])
        assert out["agent_budget"] == []
        assert out["skipped_agents"] == []
        assert out["budget_reason"] == {}
        assert 0.0 <= out["budget_confidence"] <= 1.0
        assert out["rationale"] == "heuristic"

    def test_core_agents_always_in_budget(self):
        candidates = list(_CORE_AGENTS)
        out = run_heuristic({}, {}, candidates)
        for agent in _CORE_AGENTS:
            assert agent in out["agent_budget"]
        assert out["skipped_agents"] == []

    def test_low_importance_low_value_skips_non_core(self):
        enrichment = _mk_enrichment(
            credibility_score=0.0,
            uncertainty_level="low",
            anomaly_detected=False,
            feedback_confidence=0.0,
        )
        candidates = ["MemoryDecayAgent", "PlaybookAgent"]
        out = run_heuristic(enrichment, {}, candidates)
        assert "MemoryDecayAgent" in out["skipped_agents"]
        assert "PlaybookAgent" in out["skipped_agents"]

    def test_high_importance_includes_non_core(self):
        enrichment = _mk_enrichment(
            credibility_score=1.0,
            uncertainty_level="high",
            anomaly_detected=True,
        )
        candidates = ["PlaybookAgent", "MemoryDecayAgent"]
        out = run_heuristic(enrichment, {}, candidates)
        assert "PlaybookAgent" in out["agent_budget"]
        assert "MemoryDecayAgent" in out["agent_budget"]

    def test_inactive_agent_skipped_regardless_of_importance(self):
        enrichment = _mk_enrichment(credibility_score=1.0, uncertainty_level="high")
        readiness = {"PlaybookAgent": _mk_readiness(active=False)}
        out = run_heuristic(enrichment, readiness, ["PlaybookAgent"])
        assert "PlaybookAgent" in out["skipped_agents"]
        assert out["budget_reason"]["PlaybookAgent"] == "inactive"

    def test_not_ready_agent_skipped(self):
        readiness = {"AnomalyDetectionAgent": _mk_readiness(is_ready=False)}
        out = run_heuristic({}, readiness, ["AnomalyDetectionAgent"])
        assert "AnomalyDetectionAgent" in out["skipped_agents"]
        assert out["budget_reason"]["AnomalyDetectionAgent"] == "not_ready"

    def test_feedback_value_keeps_agent_in_budget(self):
        enrichment = _mk_enrichment(
            credibility_score=0.4,
            uncertainty_level="medium",
            feedback_applied=True,
            feedback_confidence=0.80,
            correction_delta={"playbook_match": 0.30},
        )
        out = run_heuristic(enrichment, {}, ["PlaybookAgent"])
        assert "PlaybookAgent" in out["agent_budget"]

    def test_budget_reason_populated_for_all(self):
        candidates = _ALL_CANDIDATES
        out = run_heuristic(_mk_enrichment(), {}, candidates)
        for agent in candidates:
            assert agent in out["budget_reason"]

    def test_rationale_always_heuristic(self):
        out = run_heuristic({}, {}, ["PlaybookAgent"])
        assert out["rationale"] == "heuristic"

    def test_budget_confidence_in_range(self):
        out = run_heuristic(_mk_enrichment(), {}, _ALL_CANDIDATES)
        assert 0.0 <= out["budget_confidence"] <= 1.0

    def test_agent_order_from_enrichment(self):
        enrichment = _mk_enrichment(
            agent_order=["EntityResolutionAgent", "PlaybookAgent"]
        )
        # run_heuristic uses candidate_agents param, not agent_order directly
        out = run_heuristic(enrichment, {}, ["EntityResolutionAgent", "PlaybookAgent"])
        assert "EntityResolutionAgent" in out["agent_budget"]  # core


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _make_result(self, **overrides):
        from app.agents.types import AgentResult
        from datetime import datetime, timezone
        base = {
            "agent_name": "AdaptiveEnrichmentBudgetAgent",
            "agent_version": "v1",
            "memory_id": "mem-1",
            "status": "success",
            "confidence": 0.60,
            "outputs": {
                "agent_budget": ["EntityResolutionAgent"],
                "skipped_agents": ["PlaybookAgent"],
                "budget_reason": {
                    "EntityResolutionAgent": "core",
                    "PlaybookAgent": "zero_value",
                },
                "budget_confidence": 0.60,
                "rationale": "heuristic",
            },
            "warnings": [],
            "errors": [],
            "started_at": datetime.now(timezone.utc),
            "finished_at": datetime.now(timezone.utc),
            "trace_id": None,
        }
        base.update(overrides)
        return AgentResult(**base)

    def test_valid_outputs_passes(self):
        agent = AdaptiveEnrichmentBudgetAgent()
        agent.validate_outputs(self._make_result())  # no exception

    def test_skipped_status_passes(self):
        agent = AdaptiveEnrichmentBudgetAgent()
        result = self._make_result(status="skipped", outputs={})
        agent.validate_outputs(result)  # no exception

    def test_agent_budget_not_list_raises(self):
        agent = AdaptiveEnrichmentBudgetAgent()
        out = self._make_result()
        out.outputs["agent_budget"] = "not_a_list"
        with pytest.raises(ValueError, match="agent_budget"):
            agent.validate_outputs(out)

    def test_skipped_agents_not_list_raises(self):
        agent = AdaptiveEnrichmentBudgetAgent()
        out = self._make_result()
        out.outputs["skipped_agents"] = 123
        with pytest.raises(ValueError, match="skipped_agents"):
            agent.validate_outputs(out)

    def test_budget_reason_not_dict_raises(self):
        agent = AdaptiveEnrichmentBudgetAgent()
        out = self._make_result()
        out.outputs["budget_reason"] = []
        with pytest.raises(ValueError, match="budget_reason"):
            agent.validate_outputs(out)

    def test_budget_confidence_out_of_range_raises(self):
        agent = AdaptiveEnrichmentBudgetAgent()
        out = self._make_result()
        out.outputs["budget_confidence"] = 1.5
        with pytest.raises(ValueError, match="budget_confidence"):
            agent.validate_outputs(out)

    def test_budget_confidence_negative_raises(self):
        agent = AdaptiveEnrichmentBudgetAgent()
        out = self._make_result()
        out.outputs["budget_confidence"] = -0.1
        with pytest.raises(ValueError, match="budget_confidence"):
            agent.validate_outputs(out)


# ---------------------------------------------------------------------------
# async run — heuristic path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunHeuristicPath:
    async def _run(self, enrichment=None, readiness=None, candidates=None, strategy="heuristic"):
        agent = AdaptiveEnrichmentBudgetAgent()
        ctx = _mk_context(
            enrichment=enrichment or _mk_enrichment(),
            feature_readiness=readiness or {},
            candidate_agents=_ALL_CANDIDATES if candidates is None else candidates,
        )
        with patch("app.agents.adaptive_enrichment_budget_agent.settings") as ms:
            ms.ENRICHMENT_BUDGET_STRATEGY = strategy
            ms.AGENT_STRATEGY = strategy
            result = await agent.run("mem-1", ctx)
        return result

    async def test_returns_success(self):
        result = await self._run()
        assert result.status == "success"
        assert result.agent_name == "AdaptiveEnrichmentBudgetAgent"

    async def test_core_always_in_budget(self):
        candidates = list(_CORE_AGENTS) + ["PlaybookAgent"]
        result = await self._run(candidates=candidates)
        for agent in _CORE_AGENTS:
            if agent in candidates:
                assert agent in result.outputs["agent_budget"]

    async def test_empty_candidates_no_crash(self):
        result = await self._run(candidates=[])
        assert result.outputs["agent_budget"] == []
        assert result.outputs["skipped_agents"] == []

    async def test_high_importance_includes_non_core(self):
        enrichment = _mk_enrichment(
            credibility_score=1.0, uncertainty_level="high", anomaly_detected=True
        )
        result = await self._run(enrichment=enrichment, candidates=["PlaybookAgent"])
        assert "PlaybookAgent" in result.outputs["agent_budget"]

    async def test_low_importance_skips_non_core(self):
        enrichment = _mk_enrichment(credibility_score=0.0, uncertainty_level="low")
        result = await self._run(
            enrichment=enrichment, candidates=["MemoryDecayAgent"]
        )
        assert "MemoryDecayAgent" in result.outputs["skipped_agents"]

    async def test_inactive_readiness_skips(self):
        readiness = {"PlaybookAgent": _mk_readiness(active=False)}
        enrichment = _mk_enrichment(credibility_score=1.0, uncertainty_level="high")
        result = await self._run(
            enrichment=enrichment, readiness=readiness, candidates=["PlaybookAgent"]
        )
        assert "PlaybookAgent" in result.outputs["skipped_agents"]

    async def test_confidence_in_range(self):
        result = await self._run()
        assert 0.0 <= result.outputs["budget_confidence"] <= 1.0

    async def test_rationale_is_heuristic(self):
        result = await self._run()
        assert result.outputs["rationale"] == "heuristic"

    async def test_candidate_agents_from_context(self):
        result = await self._run(candidates=["EntityResolutionAgent"])
        assert "EntityResolutionAgent" in result.outputs["agent_budget"]

    async def test_candidate_agents_from_enrichment_agent_order(self):
        enrichment = _mk_enrichment(agent_order=["CredibilityAgent"])
        agent = AdaptiveEnrichmentBudgetAgent()
        ctx = {
            "memory": {"enrichment": enrichment},
            "feature_readiness": {},
            # no candidate_agents key
            "runtime": {"job_id": "x"},
        }
        with patch("app.agents.adaptive_enrichment_budget_agent.settings") as ms:
            ms.ENRICHMENT_BUDGET_STRATEGY = "heuristic"
            ms.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-2", ctx)
        assert "CredibilityAgent" in result.outputs["agent_budget"]

    async def test_feedback_value_retains_agent(self):
        enrichment = _mk_enrichment(
            credibility_score=0.4,
            uncertainty_level="medium",
            feedback_applied=True,
            feedback_confidence=0.80,
            correction_delta={"playbook_confidence": 0.35},
        )
        result = await self._run(enrichment=enrichment, candidates=["PlaybookAgent"])
        assert "PlaybookAgent" in result.outputs["agent_budget"]

    async def test_trace_id_propagated(self):
        result = await self._run()
        assert result.trace_id == "test-job"

    async def test_budget_reason_complete(self):
        candidates = ["EntityResolutionAgent", "PlaybookAgent"]
        result = await self._run(candidates=candidates)
        for a in candidates:
            assert a in result.outputs["budget_reason"]


# ---------------------------------------------------------------------------
# async run — LLM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunLLMPath:
    def _make_llm_response(self, candidates=None):
        candidates = candidates or ["EntityResolutionAgent"]
        return {
            "agent_budget": candidates,
            "skipped_agents": [],
            "budget_reason": {c: "llm_approved" for c in candidates},
            "budget_confidence": 0.75,
            "rationale": "llm",
        }

    async def _run_llm(self, llm_resp=None, candidates=None):
        agent = AdaptiveEnrichmentBudgetAgent()
        effective = candidates or ["EntityResolutionAgent"]
        ctx = _mk_context(
            enrichment=_mk_enrichment(),
            candidate_agents=effective,
        )
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=llm_resp)
        with patch("app.agents.adaptive_enrichment_budget_agent.settings") as ms, \
             patch("app.agents.adaptive_enrichment_budget_agent.create_llm_client",
                   return_value=mock_client):
            ms.ENRICHMENT_BUDGET_STRATEGY = "llm"
            ms.AGENT_STRATEGY = "llm"
            ms.VLLM_BASE_URL = "http://localhost:11434"
            ms.VLLM_MODEL = "llama3.1:8b"
            ms.VLLM_TIMEOUT_SECONDS = 5.0
            ms.VLLM_MAX_CONCURRENCY = 2
            result = await agent.run("mem-llm", ctx)
        return result

    async def test_llm_valid_response_used(self):
        resp = self._make_llm_response(["EntityResolutionAgent"])
        result = await self._run_llm(llm_resp=resp, candidates=["EntityResolutionAgent"])
        assert result.status == "success"
        assert result.outputs["rationale"] == "llm"
        assert "EntityResolutionAgent" in result.outputs["agent_budget"]

    async def test_llm_bad_response_falls_back_to_heuristic(self):
        result = await self._run_llm(llm_resp={"bad": "data"})
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    async def test_llm_none_response_falls_back(self):
        result = await self._run_llm(llm_resp=None)
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    async def test_llm_missing_agent_budget_falls_back(self):
        result = await self._run_llm(
            llm_resp={"skipped_agents": [], "budget_reason": {}}
        )
        assert result.outputs["rationale"] == "heuristic"

    async def test_llm_missing_budget_reason_falls_back(self):
        result = await self._run_llm(
            llm_resp={"agent_budget": [], "skipped_agents": []}
        )
        assert result.outputs["rationale"] == "heuristic"

    async def test_fallback_via_agent_strategy(self):
        agent = AdaptiveEnrichmentBudgetAgent()
        ctx = _mk_context(candidate_agents=["PlaybookAgent"])
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=None)
        with patch("app.agents.adaptive_enrichment_budget_agent.settings") as ms, \
             patch("app.agents.adaptive_enrichment_budget_agent.create_llm_client",
                   return_value=mock_client):
            ms.ENRICHMENT_BUDGET_STRATEGY = None
            ms.AGENT_STRATEGY = "llm"
            ms.VLLM_BASE_URL = "http://localhost:11434"
            ms.VLLM_MODEL = "llama3.1:8b"
            ms.VLLM_TIMEOUT_SECONDS = 5.0
            ms.VLLM_MAX_CONCURRENCY = 2
            result = await agent.run("mem-3", ctx)
        assert result.status == "success"


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

class TestAgentMetadata:
    def test_name(self):
        assert AdaptiveEnrichmentBudgetAgent.name == "AdaptiveEnrichmentBudgetAgent"

    def test_version(self):
        assert AdaptiveEnrichmentBudgetAgent.version == "v1"

    def test_dependencies(self):
        deps = AdaptiveEnrichmentBudgetAgent().dependencies()
        assert "FeedbackIntegrationAgent" in deps
        assert "OrchestrationBusAgent" in deps


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_lookup_snake_case(self):
        agent = get_agent("adaptive_enrichment_budget")
        assert isinstance(agent, AdaptiveEnrichmentBudgetAgent)

    def test_lookup_camel_case(self):
        agent = get_agent("AdaptiveEnrichmentBudgetAgent")
        assert isinstance(agent, AdaptiveEnrichmentBudgetAgent)

    def test_lookup_short(self):
        agent = get_agent("enrichment_budget")
        assert isinstance(agent, AdaptiveEnrichmentBudgetAgent)

    def test_unknown_returns_none(self):
        assert get_agent("does_not_exist_phase35") is None
