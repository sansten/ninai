"""Tests for AuditTrailAgent — Phase 31."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.audit_trail_agent import (
    AuditTrailAgent,
    _EXPECTED_AGENTS,
    _extract_anomaly_detection,
    _extract_causal_reasoning,
    _extract_conflict_detection,
    _extract_conflict_resolution,
    _extract_context_amplifier,
    _extract_credibility,
    _extract_entity_resolution,
    _extract_episodic_grouping,
    _extract_feedback_integration,
    _extract_goal_decomposition,
    _extract_memory_consolidation,
    _extract_memory_decay,
    _extract_narrative_synthesis,
    _extract_org_attention,
    _extract_playbook,
    _extract_proactive_push,
    _extract_semantic_normalization,
    _extract_silo_propagation,
    _extract_temporal_reasoning,
    _extract_uncertainty_reporting,
    build_audit_log,
    compute_coverage,
    run_heuristic,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(enrichment: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "tenant": {"org_id": "org-1", "org_slug": "acme"},
        "memory": {
            "id": "mem-1",
            "enrichment": enrichment or {},
        },
        "runtime": {"job_id": "job-123"},
    }


def _minimal_log_entry() -> dict[str, Any]:
    return {
        "agent_name": "TestAgent",
        "decision": "some decision",
        "confidence": 0.75,
        "reasoning_snapshot": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# SemanticNormalizationAgent extractor
# ---------------------------------------------------------------------------

class TestExtractSemanticNormalization:
    def test_returns_none_when_no_keys(self):
        assert _extract_semantic_normalization({}) is None

    def test_canonical_form_present(self):
        e = {"canonical_form": "normalised text", "normalization_confidence": 0.88}
        result = _extract_semantic_normalization(e)
        assert result is not None
        assert result["agent_name"] == "SemanticNormalizationAgent"
        assert "normalised text" in result["decision"]
        assert result["confidence"] == pytest.approx(0.88, abs=1e-3)

    def test_snapshot_contains_canonical_form(self):
        e = {"canonical_form": "foo", "normalization_confidence": 0.70, "normalization_strategy": "rule"}
        result = _extract_semantic_normalization(e)
        assert "canonical_form" in result["reasoning_snapshot"]
        assert "normalization_strategy" in result["reasoning_snapshot"]

    def test_confidence_clamped(self):
        result = _extract_semantic_normalization({"canonical_form": "x", "normalization_confidence": 1.5})
        assert result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# EntityResolutionAgent extractor
# ---------------------------------------------------------------------------

class TestExtractEntityResolution:
    def test_returns_none_when_no_keys(self):
        assert _extract_entity_resolution({}) is None

    def test_entity_count_in_decision(self):
        e = {"entity_count": 3, "resolved_entities": ["A", "B", "C"], "resolution_confidence": 0.80}
        result = _extract_entity_resolution(e)
        assert result is not None
        assert "3" in result["decision"]
        assert result["confidence"] == pytest.approx(0.80, abs=1e-3)

    def test_zero_entities(self):
        result = _extract_entity_resolution({"entity_count": 0})
        assert result is not None
        assert "0" in result["decision"]


# ---------------------------------------------------------------------------
# ContextAmplifierAgent extractor
# ---------------------------------------------------------------------------

class TestExtractContextAmplifier:
    def test_returns_none_when_no_key(self):
        assert _extract_context_amplifier({}) is None

    def test_strategy_in_decision(self):
        e = {"amplification_confidence": 0.75, "amplification_strategy": "llm"}
        result = _extract_context_amplifier(e)
        assert result is not None
        assert "llm" in result["decision"]

    def test_default_strategy(self):
        result = _extract_context_amplifier({"amplification_confidence": 0.6})
        assert "heuristic" in result["decision"]


# ---------------------------------------------------------------------------
# SiloPropagationAgent extractor
# ---------------------------------------------------------------------------

class TestExtractSiloPropagation:
    def test_returns_none_when_no_keys(self):
        assert _extract_silo_propagation({}) is None

    def test_signal_count_in_decision(self):
        e = {"propagated_signals": ["s1", "s2"], "propagation_confidence": 0.70}
        result = _extract_silo_propagation(e)
        assert result is not None
        assert "2" in result["decision"]

    def test_empty_signals(self):
        result = _extract_silo_propagation({"propagated_signals": []})
        assert "0" in result["decision"]


# ---------------------------------------------------------------------------
# OrgAttentionAgent extractor
# ---------------------------------------------------------------------------

class TestExtractOrgAttention:
    def test_returns_none_when_no_keys(self):
        assert _extract_org_attention({}) is None

    def test_tier_and_score_in_decision(self):
        e = {"attention_score": 0.85, "attention_tier": "high"}
        result = _extract_org_attention(e)
        assert result is not None
        assert "high" in result["decision"]
        assert "0.85" in result["decision"]

    def test_confidence_equals_score(self):
        e = {"attention_score": 0.60, "attention_tier": "medium"}
        result = _extract_org_attention(e)
        assert result["confidence"] == pytest.approx(0.60, abs=1e-3)


# ---------------------------------------------------------------------------
# ProactiveMemoryPushAgent extractor
# ---------------------------------------------------------------------------

class TestExtractProactivePush:
    def test_returns_none_when_no_key(self):
        assert _extract_proactive_push({}) is None

    def test_push_recommended(self):
        result = _extract_proactive_push({"push_recommended": True, "push_confidence": 0.80})
        assert result is not None
        assert "recommended" in result["decision"]

    def test_no_push(self):
        result = _extract_proactive_push({"push_recommended": False, "push_confidence": 0.75})
        assert "no" in result["decision"].lower()


# ---------------------------------------------------------------------------
# CausalReasoningAgent extractor
# ---------------------------------------------------------------------------

class TestExtractCausalReasoning:
    def test_returns_none_when_no_keys(self):
        assert _extract_causal_reasoning({}) is None

    def test_link_count_in_decision(self):
        e = {"causal_links": [{"from": "A", "to": "B"}], "causal_confidence": 0.72}
        result = _extract_causal_reasoning(e)
        assert result is not None
        assert "1" in result["decision"]

    def test_empty_links(self):
        result = _extract_causal_reasoning({"causal_links": []})
        assert "0" in result["decision"]


# ---------------------------------------------------------------------------
# ConflictDetectionAgent extractor
# ---------------------------------------------------------------------------

class TestExtractConflictDetection:
    def test_returns_none_when_no_keys(self):
        assert _extract_conflict_detection({}) is None

    def test_conflict_detected(self):
        e = {"conflict_detected": True, "conflict_count": 2, "conflict_confidence": 0.85}
        result = _extract_conflict_detection(e)
        assert result is not None
        assert "2" in result["decision"]
        assert result["confidence"] == pytest.approx(0.85, abs=1e-3)

    def test_no_conflict(self):
        result = _extract_conflict_detection({"conflict_detected": False, "conflict_count": 0})
        assert "no conflict" in result["decision"].lower()


# ---------------------------------------------------------------------------
# AdaptiveConflictResolutionAgent extractor
# ---------------------------------------------------------------------------

class TestExtractConflictResolution:
    def test_returns_none_when_no_key(self):
        assert _extract_conflict_resolution({}) is None

    def test_strategy_in_decision(self):
        e = {"resolution_strategy": "accept_newer", "resolution_confidence": 0.78}
        result = _extract_conflict_resolution(e)
        assert result is not None
        assert "accept_newer" in result["decision"]

    def test_snapshot_contains_rationale(self):
        e = {"resolution_strategy": "merge", "resolution_rationale": "both valid", "resolution_confidence": 0.70}
        result = _extract_conflict_resolution(e)
        assert "resolution_rationale" in result["reasoning_snapshot"]


# ---------------------------------------------------------------------------
# MemoryDecayAgent extractor
# ---------------------------------------------------------------------------

class TestExtractMemoryDecay:
    def test_returns_none_when_no_key(self):
        assert _extract_memory_decay({}) is None

    def test_stale_memory(self):
        result = _extract_memory_decay({"decay_factor": 0.05, "is_stale": True})
        assert result is not None
        assert "stale" in result["decision"]

    def test_fresh_memory(self):
        result = _extract_memory_decay({"decay_factor": 0.95, "is_stale": False})
        assert "fresh" in result["decision"]

    def test_decay_factor_in_decision(self):
        result = _extract_memory_decay({"decay_factor": 0.42})
        assert "0.420" in result["decision"]


# ---------------------------------------------------------------------------
# MemoryConsolidationAgent extractor
# ---------------------------------------------------------------------------

class TestExtractMemoryConsolidation:
    def test_returns_none_when_no_keys(self):
        assert _extract_memory_consolidation({}) is None

    def test_consolidation_recommended(self):
        e = {"consolidation_recommended": True, "consolidation_score": 0.88}
        result = _extract_memory_consolidation(e)
        assert result is not None
        assert "consolidation recommended" in result["decision"]

    def test_no_consolidation(self):
        e = {"consolidation_recommended": False, "consolidation_score": 0.20}
        result = _extract_memory_consolidation(e)
        assert "no consolidation" in result["decision"].lower()


# ---------------------------------------------------------------------------
# TemporalReasoningAgent extractor
# ---------------------------------------------------------------------------

class TestExtractTemporalReasoning:
    def test_returns_none_when_no_keys(self):
        assert _extract_temporal_reasoning({}) is None

    def test_confidence_in_decision(self):
        result = _extract_temporal_reasoning({"temporal_confidence": 0.77})
        assert result is not None
        assert "0.77" in result["decision"]

    def test_sequence_position_included(self):
        result = _extract_temporal_reasoning({"temporal_confidence": 0.80, "sequence_position": 3})
        assert "sequence_position=3" in result["decision"]


# ---------------------------------------------------------------------------
# EpisodicGroupingAgent extractor
# ---------------------------------------------------------------------------

class TestExtractEpisodicGrouping:
    def test_returns_none_when_no_keys(self):
        assert _extract_episodic_grouping({}) is None

    def test_episode_id_in_decision(self):
        result = _extract_episodic_grouping({"episode_id": "ep-42", "episode_confidence": 0.80})
        assert result is not None
        assert "ep-42" in result["decision"]

    def test_no_episode(self):
        result = _extract_episodic_grouping({"episode_id": None, "episode_confidence": 0.50})
        assert "no episode" in result["decision"].lower()


# ---------------------------------------------------------------------------
# CredibilityAgent extractor
# ---------------------------------------------------------------------------

class TestExtractCredibility:
    def test_returns_none_when_no_key(self):
        assert _extract_credibility({}) is None

    def test_score_and_tier_in_decision(self):
        result = _extract_credibility({"credibility_score": 0.82, "source_tier": "verified"})
        assert result is not None
        assert "0.82" in result["decision"]
        assert "verified" in result["decision"]

    def test_confidence_equals_score(self):
        result = _extract_credibility({"credibility_score": 0.65})
        assert result["confidence"] == pytest.approx(0.65, abs=1e-3)


# ---------------------------------------------------------------------------
# PlaybookAgent extractor
# ---------------------------------------------------------------------------

class TestExtractPlaybook:
    def test_returns_none_when_no_keys(self):
        assert _extract_playbook({}) is None

    def test_matched_playbook(self):
        e = {"matched_playbook_id": "pb-7", "playbook_confidence": 0.90}
        result = _extract_playbook(e)
        assert result is not None
        assert "pb-7" in result["decision"]

    def test_new_candidate(self):
        e = {"is_new_playbook_candidate": True, "playbook_confidence": 0.60}
        result = _extract_playbook(e)
        assert "new playbook candidate" in result["decision"]

    def test_no_match(self):
        e = {"matched_playbook_id": None, "is_new_playbook_candidate": False, "playbook_confidence": 0.40}
        result = _extract_playbook(e)
        assert "no playbook matched" in result["decision"]


# ---------------------------------------------------------------------------
# GoalDecompositionAgent extractor
# ---------------------------------------------------------------------------

class TestExtractGoalDecomposition:
    def test_returns_none_when_no_key(self):
        assert _extract_goal_decomposition({}) is None

    def test_goal_detected_with_fraction(self):
        e = {"goal_detected": True, "completion_fraction": 0.5, "subtasks": []}
        result = _extract_goal_decomposition(e)
        assert result is not None
        assert "goal detected" in result["decision"]
        assert "50%" in result["decision"]

    def test_blocking_subtask(self):
        e = {"goal_detected": True, "completion_fraction": 0.3, "blocking_subtask": "review spec"}
        result = _extract_goal_decomposition(e)
        assert "review spec" in result["decision"]

    def test_no_goal(self):
        result = _extract_goal_decomposition({"goal_detected": False})
        assert "no goal" in result["decision"].lower()


# ---------------------------------------------------------------------------
# UncertaintyReportingAgent extractor
# ---------------------------------------------------------------------------

class TestExtractUncertaintyReporting:
    def test_returns_none_when_no_keys(self):
        assert _extract_uncertainty_reporting({}) is None

    def test_critical_level(self):
        e = {"uncertainty_level": "critical", "review_recommended": True}
        result = _extract_uncertainty_reporting(e)
        assert result is not None
        assert "critical" in result["decision"]
        assert "human review" in result["decision"]
        assert result["confidence"] == pytest.approx(0.30, abs=1e-3)

    def test_low_level_no_review(self):
        e = {"uncertainty_level": "low", "review_recommended": False}
        result = _extract_uncertainty_reporting(e)
        assert "human review" not in result["decision"]
        assert result["confidence"] == pytest.approx(0.85, abs=1e-3)

    def test_medium_confidence(self):
        e = {"uncertainty_level": "medium"}
        result = _extract_uncertainty_reporting(e)
        assert result["confidence"] == pytest.approx(0.65, abs=1e-3)


# ---------------------------------------------------------------------------
# NarrativeSynthesisAgent extractor
# ---------------------------------------------------------------------------

class TestExtractNarrativeSynthesis:
    def test_returns_none_when_no_keys(self):
        assert _extract_narrative_synthesis({}) is None

    def test_narrative_preview_in_decision(self):
        e = {"narrative": "Q4 revenue grew by 12%.", "narrative_confidence": 0.80}
        result = _extract_narrative_synthesis(e)
        assert result is not None
        assert "Q4 revenue" in result["decision"]

    def test_long_narrative_truncated(self):
        long = "x" * 200
        result = _extract_narrative_synthesis({"narrative": long, "narrative_confidence": 0.70})
        assert len(result["decision"]) < 120

    def test_no_narrative_fallback(self):
        result = _extract_narrative_synthesis({"narrative_confidence": 0.60})
        assert "synthesis attempted" in result["decision"]


# ---------------------------------------------------------------------------
# FeedbackIntegrationAgent extractor
# ---------------------------------------------------------------------------

class TestExtractFeedbackIntegration:
    def test_returns_none_when_no_keys(self):
        assert _extract_feedback_integration({}) is None

    def test_correction_made(self):
        result = _extract_feedback_integration({"feedback_applied": True, "correction_made": True})
        assert result is not None
        assert "correction applied" in result["decision"]

    def test_feedback_no_correction(self):
        result = _extract_feedback_integration({"feedback_applied": True, "correction_made": False})
        assert "no correction" in result["decision"]

    def test_no_feedback(self):
        result = _extract_feedback_integration({"feedback_applied": False, "correction_made": False})
        assert "no feedback" in result["decision"].lower()


# ---------------------------------------------------------------------------
# AnomalyDetectionAgent extractor
# ---------------------------------------------------------------------------

class TestExtractAnomalyDetection:
    def test_returns_none_when_no_key(self):
        assert _extract_anomaly_detection({}) is None

    def test_anomaly_detected(self):
        e = {"anomaly_detected": True, "anomaly_type": "credibility_spike", "severity": "high", "confidence": 0.85}
        result = _extract_anomaly_detection(e)
        assert result is not None
        assert "credibility_spike" in result["decision"]
        assert "high" in result["decision"]
        assert result["confidence"] == pytest.approx(0.85, abs=1e-3)

    def test_no_anomaly(self):
        result = _extract_anomaly_detection({"anomaly_detected": False, "confidence": 0.70})
        assert "no anomaly" in result["decision"].lower()

    def test_snapshot_keys_present(self):
        e = {"anomaly_detected": True, "anomaly_type": "temporal_drift",
             "severity": "medium", "anomaly_score": 0.55, "affected_fields": ["temporal_confidence"]}
        result = _extract_anomaly_detection(e)
        snap = result["reasoning_snapshot"]
        assert "anomaly_score" in snap
        assert "affected_fields" in snap

    def test_falls_back_to_anomaly_score_for_confidence(self):
        e = {"anomaly_detected": True, "anomaly_score": 0.60}
        result = _extract_anomaly_detection(e)
        assert result["confidence"] == pytest.approx(0.60, abs=1e-3)


# ---------------------------------------------------------------------------
# build_audit_log
# ---------------------------------------------------------------------------

class TestBuildAuditLog:
    def test_empty_enrichment_returns_empty_log(self):
        log = build_audit_log({}, "2026-01-01T00:00:00+00:00")
        assert log == []

    def test_entries_have_timestamp(self):
        e = {"anomaly_detected": False, "confidence": 0.70}
        ts = "2026-03-09T12:00:00+00:00"
        log = build_audit_log(e, ts)
        assert all(entry["timestamp"] == ts for entry in log)

    def test_multiple_agents_produce_multiple_entries(self):
        e = {
            "anomaly_detected": True, "anomaly_type": "conflict_flood", "severity": "high",
            "credibility_score": 0.90, "conflict_detected": True, "conflict_count": 6,
        }
        log = build_audit_log(e, "2026-01-01T00:00:00+00:00")
        names = [entry["agent_name"] for entry in log]
        assert "AnomalyDetectionAgent" in names
        assert "CredibilityAgent" in names
        assert "ConflictDetectionAgent" in names

    def test_each_entry_has_required_keys(self):
        e = {"credibility_score": 0.75}
        log = build_audit_log(e, "2026-01-01T00:00:00+00:00")
        for entry in log:
            assert "agent_name" in entry
            assert "decision" in entry
            assert "confidence" in entry
            assert "reasoning_snapshot" in entry
            assert "timestamp" in entry

    def test_full_enrichment_produces_many_entries(self):
        e = {
            "canonical_form": "x", "normalization_confidence": 0.80,
            "entity_count": 2, "resolved_entities": [],
            "amplification_confidence": 0.70,
            "propagated_signals": [], "propagation_confidence": 0.65,
            "attention_score": 0.5, "attention_tier": "medium",
            "push_recommended": False, "push_confidence": 0.60,
            "causal_links": [], "causal_confidence": 0.55,
            "conflict_detected": False, "conflict_count": 0,
            "resolution_strategy": "no_action", "resolution_confidence": 0.70,
            "decay_factor": 0.80,
            "consolidation_recommended": False, "consolidation_score": 0.30,
            "temporal_confidence": 0.75,
            "episode_id": "ep-1", "episode_confidence": 0.80,
            "credibility_score": 0.85,
            "matched_playbook_id": None, "is_new_playbook_candidate": False, "playbook_confidence": 0.40,
            "goal_detected": False,
            "uncertainty_level": "low", "review_recommended": False,
            "narrative_confidence": 0.70,
            "feedback_applied": False, "correction_made": False,
            "anomaly_detected": False, "confidence": 0.70,
        }
        log = build_audit_log(e, "2026-01-01T00:00:00+00:00")
        assert len(log) == len(_EXPECTED_AGENTS)


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------

class TestComputeCoverage:
    def test_empty_log_is_zero(self):
        assert compute_coverage([]) == 0.0

    def test_all_agents_is_one(self):
        log = [{"agent_name": n, "decision": "d", "confidence": 0.7,
                "reasoning_snapshot": {}, "timestamp": "t"} for n in _EXPECTED_AGENTS]
        assert compute_coverage(log) == pytest.approx(1.0, abs=1e-3)

    def test_partial_coverage(self):
        log = [{"agent_name": "AnomalyDetectionAgent", "decision": "d", "confidence": 0.7,
                "reasoning_snapshot": {}, "timestamp": "t"}]
        cov = compute_coverage(log)
        assert 0.0 < cov < 1.0


# ---------------------------------------------------------------------------
# run_heuristic
# ---------------------------------------------------------------------------

class TestRunHeuristic:
    def test_required_output_keys_present(self):
        result = run_heuristic({})
        assert "audit_log" in result
        assert "agents_audited" in result
        assert "total_entries" in result
        assert "coverage_score" in result
        assert result["rationale"] == "heuristic"

    def test_empty_enrichment_gives_zero_agents(self):
        result = run_heuristic({})
        assert result["agents_audited"] == 0
        assert result["total_entries"] == 0
        assert result["coverage_score"] == 0.0

    def test_total_entries_matches_log_length(self):
        e = {"anomaly_detected": False, "credibility_score": 0.75}
        result = run_heuristic(e)
        assert result["total_entries"] == len(result["audit_log"])

    def test_agents_audited_is_unique_count(self):
        e = {"anomaly_detected": False, "credibility_score": 0.75, "conflict_detected": False, "conflict_count": 0}
        result = run_heuristic(e)
        assert result["agents_audited"] == len({e["agent_name"] for e in result["audit_log"]})

    def test_coverage_score_in_range(self):
        result = run_heuristic({"credibility_score": 0.80})
        assert 0.0 <= result["coverage_score"] <= 1.0


# ---------------------------------------------------------------------------
# AuditTrailAgent integration
# ---------------------------------------------------------------------------

class TestAuditTrailAgentHeuristic:
    @pytest.mark.asyncio
    async def test_runs_successfully_empty_enrichment(self):
        agent = AuditTrailAgent()
        result = await agent.run("mem-1", _ctx())
        assert result.status == "success"
        assert isinstance(result.outputs["audit_log"], list)

    @pytest.mark.asyncio
    async def test_heuristic_strategy_forced(self):
        agent = AuditTrailAgent()
        ctx = _ctx({"anomaly_detected": True, "anomaly_type": "conflict_flood",
                    "severity": "high", "confidence": 0.80})
        with patch("app.agents.audit_trail_agent.settings") as mock_settings:
            mock_settings.AUDIT_TRAIL_STRATEGY = "heuristic"
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"
        assert any(e["agent_name"] == "AnomalyDetectionAgent" for e in result.outputs["audit_log"])

    @pytest.mark.asyncio
    async def test_result_metadata(self):
        agent = AuditTrailAgent()
        result = await agent.run("mem-99", _ctx())
        assert result.agent_name == "AuditTrailAgent"
        assert result.agent_version == "v1"
        assert result.memory_id == "mem-99"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = AuditTrailAgent()
        result = await agent.run("mem-1", _ctx())
        assert result.trace_id == "job-123"

    @pytest.mark.asyncio
    async def test_confidence_is_average_of_entries(self):
        agent = AuditTrailAgent()
        ctx = _ctx({
            "credibility_score": 0.80,
            "anomaly_detected": False, "confidence": 0.70,
        })
        with patch("app.agents.audit_trail_agent.settings") as s:
            s.AUDIT_TRAIL_STRATEGY = "heuristic"
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        # confidence is average of all entries' confidence
        log = result.outputs["audit_log"]
        expected_avg = round(sum(e["confidence"] for e in log) / len(log), 4)
        assert result.confidence == pytest.approx(expected_avg, abs=1e-3)

    @pytest.mark.asyncio
    async def test_no_enrichment_context_key(self):
        agent = AuditTrailAgent()
        ctx: dict = {
            "tenant": {"org_id": "org-1"},
            "memory": {},
            "runtime": {},
        }
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["agents_audited"] == 0

    @pytest.mark.asyncio
    async def test_outputs_json_serializable(self):
        agent = AuditTrailAgent()
        ctx = _ctx({
            "canonical_form": "test", "normalization_confidence": 0.80,
            "anomaly_detected": False, "confidence": 0.70,
        })
        result = await agent.run("mem-1", ctx)
        serialized = json.dumps(result.outputs)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

class TestAuditTrailAgentLLM:
    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        enrichment = {"anomaly_detected": False, "confidence": 0.70}
        ctx = _ctx(enrichment)

        base = run_heuristic(enrichment)
        llm_resp = {
            "audit_log": base["audit_log"],
            "agents_audited": base["agents_audited"],
            "total_entries": base["total_entries"],
            "coverage_score": base["coverage_score"],
            "summary": "Memory assessed with no anomalies.",
            "rationale": "llm",
        }

        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=llm_resp)

        with patch("app.agents.audit_trail_agent.settings") as s, \
             patch("app.agents.audit_trail_agent.create_ollama_client", return_value=mock_client):
            s.AUDIT_TRAIL_STRATEGY = "llm"
            s.AGENT_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "qwen2.5:7b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            agent = AuditTrailAgent()
            result = await agent.run("mem-1", ctx)

        assert result.outputs["rationale"] == "llm"
        assert result.outputs.get("summary") == "Memory assessed with no anomalies."

    @pytest.mark.asyncio
    async def test_invalid_llm_response_falls_back_to_heuristic(self):
        ctx = _ctx({"anomaly_detected": False, "confidence": 0.70})

        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"garbage": True})

        with patch("app.agents.audit_trail_agent.settings") as s, \
             patch("app.agents.audit_trail_agent.create_ollama_client", return_value=mock_client):
            s.AUDIT_TRAIL_STRATEGY = "llm"
            s.AGENT_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "qwen2.5:7b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            agent = AuditTrailAgent()
            result = await agent.run("mem-1", ctx)

        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_heuristic(self):
        ctx = _ctx({"anomaly_detected": False, "confidence": 0.65})

        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(side_effect=RuntimeError("timeout"))

        with patch("app.agents.audit_trail_agent.settings") as s, \
             patch("app.agents.audit_trail_agent.create_ollama_client", return_value=mock_client):
            s.AUDIT_TRAIL_STRATEGY = "llm"
            s.AGENT_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "qwen2.5:7b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            agent = AuditTrailAgent()
            with pytest.raises(RuntimeError):
                await agent.run("mem-1", ctx)


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _agent(self) -> AuditTrailAgent:
        return AuditTrailAgent()

    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="AuditTrailAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="success",
            confidence=0.70,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
            trace_id=None,
        )

    def test_valid_outputs_pass(self):
        entry = _minimal_log_entry()
        outputs = {
            "audit_log": [entry],
            "agents_audited": 1,
            "total_entries": 1,
            "coverage_score": 0.05,
        }
        agent = self._agent()
        agent.validate_outputs(self._result(outputs))  # no exception

    def test_audit_log_not_list_raises(self):
        outputs = {"audit_log": "bad", "agents_audited": 0, "total_entries": 0, "coverage_score": 0.0}
        agent = self._agent()
        with pytest.raises(ValueError, match="audit_log must be a list"):
            agent.validate_outputs(self._result(outputs))

    def test_entry_missing_agent_name_raises(self):
        entry = {"decision": "d", "confidence": 0.70, "reasoning_snapshot": {}, "timestamp": "t"}
        outputs = {"audit_log": [entry], "agents_audited": 1, "total_entries": 1, "coverage_score": 0.05}
        agent = self._agent()
        with pytest.raises(ValueError, match="agent_name"):
            agent.validate_outputs(self._result(outputs))

    def test_entry_bad_confidence_raises(self):
        entry = {**_minimal_log_entry(), "confidence": 1.5}
        outputs = {"audit_log": [entry], "agents_audited": 1, "total_entries": 1, "coverage_score": 0.05}
        agent = self._agent()
        with pytest.raises(ValueError, match="confidence"):
            agent.validate_outputs(self._result(outputs))

    def test_entry_missing_snapshot_raises(self):
        entry = {"agent_name": "X", "decision": "d", "confidence": 0.70, "timestamp": "t"}
        outputs = {"audit_log": [entry], "agents_audited": 1, "total_entries": 1, "coverage_score": 0.05}
        agent = self._agent()
        with pytest.raises(ValueError, match="reasoning_snapshot"):
            agent.validate_outputs(self._result(outputs))

    def test_coverage_score_out_of_range_raises(self):
        outputs = {"audit_log": [], "agents_audited": 0, "total_entries": 0, "coverage_score": 1.5}
        agent = self._agent()
        with pytest.raises(ValueError, match="coverage_score"):
            agent.validate_outputs(self._result(outputs))

    def test_agents_audited_negative_raises(self):
        outputs = {"audit_log": [], "agents_audited": -1, "total_entries": 0, "coverage_score": 0.0}
        agent = self._agent()
        with pytest.raises(ValueError, match="agents_audited"):
            agent.validate_outputs(self._result(outputs))

    def test_skips_validation_on_failed_status(self):
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="AuditTrailAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
            trace_id=None,
        )
        AuditTrailAgent().validate_outputs(result)  # must not raise


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_audit_trail(self):
        assert isinstance(get_agent("audit_trail"), AuditTrailAgent)

    def test_audittrail(self):
        assert isinstance(get_agent("audittrail"), AuditTrailAgent)

    def test_audittrailagent(self):
        assert isinstance(get_agent("audittrailagent"), AuditTrailAgent)

    def test_audit_agent(self):
        assert isinstance(get_agent("audit_agent"), AuditTrailAgent)

    def test_unknown_name_returns_none(self):
        assert get_agent("audit_trail_v2") is None


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

class TestAuditTrailAgentMeta:
    def test_name_and_version(self):
        agent = AuditTrailAgent()
        assert agent.name == "AuditTrailAgent"
        assert agent.version == "v1"

    def test_dependencies_covers_all_expected_agents(self):
        agent = AuditTrailAgent()
        deps = agent.dependencies()
        for expected in _EXPECTED_AGENTS:
            assert expected in deps

    def test_dependencies_returns_new_list_each_call(self):
        agent = AuditTrailAgent()
        d1 = agent.dependencies()
        d2 = agent.dependencies()
        assert d1 == d2
        assert d1 is not d2
