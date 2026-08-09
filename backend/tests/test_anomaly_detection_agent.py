"""Tests for AnomalyDetectionAgent — Phase 25."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.anomaly_detection_agent import (
    AnomalyDetectionAgent,
    detect_credibility_spike,
    detect_uncertainty_surge,
    detect_conflict_flood,
    detect_temporal_drift,
    detect_stale_revival,
    pick_primary_anomaly_type,
    classify_severity,
    compute_anomaly_confidence,
    run_heuristic,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx(enrichment: dict | None = None) -> dict:
    return {
        "memory": {"enrichment": enrichment or {}},
        "runtime": {"job_id": "trace-25"},
    }


def _clean_enrichment() -> dict:
    return {
        "credibility_score": 0.85,
        "uncertainty_score": 0.15,
        "uncertainty_level": "low",
        "conflict_count": 0,
        "temporal_confidence": 0.80,
        "decay_factor": 0.70,
        "is_stale": False,
    }


def _crisis_enrichment() -> dict:
    return {
        "credibility_score": 0.10,
        "uncertainty_score": 0.92,
        "uncertainty_level": "critical",
        "conflict_count": 8,
        "temporal_confidence": 0.05,
        "decay_factor": 0.05,
        "is_stale": False,
    }


# ---------------------------------------------------------------------------
# TestDetectCredibilitySpike
# ---------------------------------------------------------------------------

class TestDetectCredibilitySpike:
    def test_below_threshold_detected(self):
        detected, contrib = detect_credibility_spike({"credibility_score": 0.10})
        assert detected is True
        assert contrib > 0

    def test_above_threshold_not_detected(self):
        detected, contrib = detect_credibility_spike({"credibility_score": 0.80})
        assert detected is False
        assert contrib == 0.0

    def test_exactly_at_threshold_not_detected(self):
        detected, _ = detect_credibility_spike({"credibility_score": 0.25})
        assert detected is False

    def test_missing_field_not_detected(self):
        detected, _ = detect_credibility_spike({})
        assert detected is False

    def test_non_numeric_not_detected(self):
        detected, _ = detect_credibility_spike({"credibility_score": "bad"})
        assert detected is False

    def test_contribution_in_range(self):
        _, contrib = detect_credibility_spike({"credibility_score": 0.05})
        assert 0.0 < contrib <= 0.50


# ---------------------------------------------------------------------------
# TestDetectUncertaintySurge
# ---------------------------------------------------------------------------

class TestDetectUncertaintySurge:
    def test_score_above_threshold_detected(self):
        detected, contrib = detect_uncertainty_surge({"uncertainty_score": 0.90})
        assert detected is True
        assert contrib > 0

    def test_score_below_threshold_not_detected(self):
        detected, _ = detect_uncertainty_surge({"uncertainty_score": 0.50})
        assert detected is False

    def test_critical_level_fallback_detected(self):
        detected, contrib = detect_uncertainty_surge({"uncertainty_level": "critical"})
        assert detected is True
        assert contrib > 0

    def test_low_level_no_score_not_detected(self):
        detected, _ = detect_uncertainty_surge({"uncertainty_level": "low"})
        assert detected is False

    def test_missing_fields_not_detected(self):
        detected, _ = detect_uncertainty_surge({})
        assert detected is False

    def test_contribution_in_range(self):
        _, contrib = detect_uncertainty_surge({"uncertainty_score": 0.95})
        assert 0.0 < contrib <= 0.50


# ---------------------------------------------------------------------------
# TestDetectConflictFlood
# ---------------------------------------------------------------------------

class TestDetectConflictFlood:
    def test_count_at_threshold_detected(self):
        detected, contrib = detect_conflict_flood({"conflict_count": 5})
        assert detected is True
        assert contrib > 0

    def test_count_below_threshold_not_detected(self):
        detected, _ = detect_conflict_flood({"conflict_count": 4})
        assert detected is False

    def test_count_zero_not_detected(self):
        detected, _ = detect_conflict_flood({"conflict_count": 0})
        assert detected is False

    def test_uses_conflicts_list_when_no_count(self):
        conflicts = [{"entity": f"E{i}"} for i in range(6)]
        detected, _ = detect_conflict_flood({"conflicts": conflicts})
        assert detected is True

    def test_missing_fields_not_detected(self):
        detected, _ = detect_conflict_flood({})
        assert detected is False

    def test_contribution_capped_at_0_5(self):
        _, contrib = detect_conflict_flood({"conflict_count": 100})
        assert contrib <= 0.50


# ---------------------------------------------------------------------------
# TestDetectTemporalDrift
# ---------------------------------------------------------------------------

class TestDetectTemporalDrift:
    def test_below_threshold_detected(self):
        detected, contrib = detect_temporal_drift({"temporal_confidence": 0.10})
        assert detected is True
        assert contrib > 0

    def test_above_threshold_not_detected(self):
        detected, _ = detect_temporal_drift({"temporal_confidence": 0.80})
        assert detected is False

    def test_missing_field_not_detected(self):
        detected, _ = detect_temporal_drift({})
        assert detected is False

    def test_non_numeric_not_detected(self):
        detected, _ = detect_temporal_drift({"temporal_confidence": "na"})
        assert detected is False

    def test_contribution_in_range(self):
        _, contrib = detect_temporal_drift({"temporal_confidence": 0.05})
        assert 0.0 < contrib <= 0.40


# ---------------------------------------------------------------------------
# TestDetectStaleRevival
# ---------------------------------------------------------------------------

class TestDetectStaleRevival:
    def test_low_decay_not_stale_detected(self):
        detected, contrib = detect_stale_revival({"decay_factor": 0.05, "is_stale": False})
        assert detected is True
        assert contrib > 0

    def test_low_decay_already_stale_not_detected(self):
        detected, _ = detect_stale_revival({"decay_factor": 0.05, "is_stale": True})
        assert detected is False

    def test_normal_decay_not_detected(self):
        detected, _ = detect_stale_revival({"decay_factor": 0.60})
        assert detected is False

    def test_missing_decay_not_detected(self):
        detected, _ = detect_stale_revival({})
        assert detected is False

    def test_contribution_in_range(self):
        _, contrib = detect_stale_revival({"decay_factor": 0.02, "is_stale": False})
        assert 0.0 < contrib <= 0.35


# ---------------------------------------------------------------------------
# TestPickPrimaryAnomalyType
# ---------------------------------------------------------------------------

class TestPickPrimaryAnomalyType:
    def test_highest_contribution_wins(self):
        detections = {
            "credibility_spike": (True, 0.30),
            "conflict_flood": (True, 0.45),
            "temporal_drift": (False, 0.0),
        }
        assert pick_primary_anomaly_type(detections) == "conflict_flood"

    def test_none_when_all_false(self):
        detections = {
            "credibility_spike": (False, 0.0),
            "conflict_flood": (False, 0.0),
        }
        assert pick_primary_anomaly_type(detections) is None

    def test_single_detection(self):
        detections = {"credibility_spike": (True, 0.35)}
        assert pick_primary_anomaly_type(detections) == "credibility_spike"


# ---------------------------------------------------------------------------
# TestClassifySeverity
# ---------------------------------------------------------------------------

class TestClassifySeverity:
    def test_zero_returns_none(self):
        assert classify_severity(0.0) is None

    def test_below_medium_is_low(self):
        assert classify_severity(0.20) == "low"

    def test_medium_range(self):
        assert classify_severity(0.55) == "medium"

    def test_high_range(self):
        assert classify_severity(0.80) == "high"

    def test_boundary_medium(self):
        assert classify_severity(0.40) == "medium"

    def test_boundary_high(self):
        assert classify_severity(0.70) == "high"


# ---------------------------------------------------------------------------
# TestComputeAnomalyConfidence
# ---------------------------------------------------------------------------

class TestComputeAnomalyConfidence:
    def test_no_anomaly_high_confidence(self):
        conf = compute_anomaly_confidence(False, 0, 0.0)
        assert conf >= 0.65

    def test_multiple_affected_fields_boosts(self):
        single = compute_anomaly_confidence(True, 1, 0.35)
        multi = compute_anomaly_confidence(True, 3, 0.35)
        assert multi > single

    def test_high_score_boosts(self):
        low = compute_anomaly_confidence(True, 1, 0.30)
        high = compute_anomaly_confidence(True, 1, 0.75)
        assert high > low

    def test_always_in_range(self):
        for detected, count, score in [
            (False, 0, 0.0),
            (True, 5, 0.95),
            (True, 1, 0.20),
        ]:
            conf = compute_anomaly_confidence(detected, count, score)
            assert 0.40 <= conf <= 0.90


# ---------------------------------------------------------------------------
# TestRunHeuristic
# ---------------------------------------------------------------------------

class TestRunHeuristic:
    def test_clean_enrichment_no_anomaly(self):
        result = run_heuristic(_clean_enrichment())
        assert result["anomaly_detected"] is False
        assert result["anomaly_type"] is None
        assert result["severity"] is None
        assert result["anomaly_score"] == 0.0

    def test_crisis_enrichment_anomaly_detected(self):
        result = run_heuristic(_crisis_enrichment())
        assert result["anomaly_detected"] is True
        assert result["anomaly_type"] is not None
        assert result["severity"] == "high"

    def test_rationale_is_heuristic(self):
        assert run_heuristic({})["rationale"] == "heuristic"

    def test_required_keys_present(self):
        result = run_heuristic({})
        for k in ("anomaly_detected", "anomaly_type", "severity", "affected_fields",
                   "anomaly_score", "confidence", "rationale"):
            assert k in result

    def test_affected_fields_is_list(self):
        assert isinstance(run_heuristic(_clean_enrichment())["affected_fields"], list)

    def test_anomaly_score_in_range(self):
        score = run_heuristic(_crisis_enrichment())["anomaly_score"]
        assert 0.0 <= score <= 1.0

    def test_single_spike_is_low_or_medium(self):
        result = run_heuristic({"credibility_score": 0.20})
        assert result["anomaly_detected"] is True
        assert result["severity"] in {"low", "medium"}

    def test_empty_enrichment_no_anomaly(self):
        result = run_heuristic({})
        assert result["anomaly_detected"] is False


# ---------------------------------------------------------------------------
# TestAnomalyDetectionAgentHeuristic
# ---------------------------------------------------------------------------

class TestAnomalyDetectionAgentHeuristic:
    @pytest.mark.asyncio
    async def test_heuristic_strategy(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_clean_enrichment()))
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_clean_memory_no_anomaly(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_clean_enrichment()))
        assert result.outputs["anomaly_detected"] is False

    @pytest.mark.asyncio
    async def test_crisis_memory_anomaly(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_crisis_enrichment()))
        assert result.outputs["anomaly_detected"] is True
        assert result.outputs["severity"] == "high"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.trace_id == "trace-25"

    @pytest.mark.asyncio
    async def test_memory_id_preserved(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-abc", _ctx())
        assert result.memory_id == "mem-abc"

    @pytest.mark.asyncio
    async def test_timestamps_set(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.finished_at >= result.started_at

    @pytest.mark.asyncio
    async def test_affected_fields_list(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_crisis_enrichment()))
        assert isinstance(result.outputs["affected_fields"], list)

    @pytest.mark.asyncio
    async def test_anomaly_score_in_range(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_crisis_enrichment()))
        assert 0.0 <= result.outputs["anomaly_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_no_trace_when_absent(self):
        agent = AnomalyDetectionAgent()
        ctx = {"memory": {"enrichment": {}}, "runtime": {}}
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.trace_id is None

    @pytest.mark.asyncio
    async def test_agent_strategy_fallback(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = None
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_warnings_empty(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.warnings == []


# ---------------------------------------------------------------------------
# TestAnomalyDetectionAgentLLM
# ---------------------------------------------------------------------------

def _valid_llm_response() -> dict:
    return {
        "anomaly_detected": True,
        "anomaly_type": "credibility_spike",
        "severity": "medium",
        "affected_fields": ["credibility_score"],
        "anomaly_score": 0.42,
        "confidence": 0.75,
        "rationale": "LLM detected low credibility.",
    }


class TestAnomalyDetectionAgentLLM:
    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = AnomalyDetectionAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=_valid_llm_response())
        with patch("app.agents.anomaly_detection_agent.settings") as s, \
             patch("app.agents.anomaly_detection_agent.create_llm_client", return_value=mock_client):
            s.ANOMALY_DETECTION_STRATEGY = "llm"
            s.VLLM_BASE_URL = "http://localhost:11434"
            s.VLLM_MODEL = "llama3.1:8b"
            s.VLLM_TIMEOUT_SECONDS = 5.0
            s.VLLM_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.status == "success"
        assert result.outputs["anomaly_type"] == "credibility_spike"

    @pytest.mark.asyncio
    async def test_llm_none_falls_back(self):
        agent = AnomalyDetectionAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=None)
        with patch("app.agents.anomaly_detection_agent.settings") as s, \
             patch("app.agents.anomaly_detection_agent.create_llm_client", return_value=mock_client):
            s.ANOMALY_DETECTION_STRATEGY = "llm"
            s.VLLM_BASE_URL = "http://localhost:11434"
            s.VLLM_MODEL = "llama3.1:8b"
            s.VLLM_TIMEOUT_SECONDS = 5.0
            s.VLLM_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_invalid_anomaly_type_falls_back(self):
        agent = AnomalyDetectionAgent()
        bad = dict(_valid_llm_response(), anomaly_type="unknown_type")
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=bad)
        with patch("app.agents.anomaly_detection_agent.settings") as s, \
             patch("app.agents.anomaly_detection_agent.create_llm_client", return_value=mock_client):
            s.ANOMALY_DETECTION_STRATEGY = "llm"
            s.VLLM_BASE_URL = "http://localhost:11434"
            s.VLLM_MODEL = "llama3.1:8b"
            s.VLLM_TIMEOUT_SECONDS = 5.0
            s.VLLM_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_wrong_affected_fields_type_falls_back(self):
        agent = AnomalyDetectionAgent()
        bad = dict(_valid_llm_response(), affected_fields="not-a-list")
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=bad)
        with patch("app.agents.anomaly_detection_agent.settings") as s, \
             patch("app.agents.anomaly_detection_agent.create_llm_client", return_value=mock_client):
            s.ANOMALY_DETECTION_STRATEGY = "llm"
            s.VLLM_BASE_URL = "http://localhost:11434"
            s.VLLM_MODEL = "llama3.1:8b"
            s.VLLM_TIMEOUT_SECONDS = 5.0
            s.VLLM_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_null_anomaly_type_accepted(self):
        agent = AnomalyDetectionAgent()
        resp = dict(_valid_llm_response(), anomaly_type=None, anomaly_detected=False)
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=resp)
        with patch("app.agents.anomaly_detection_agent.settings") as s, \
             patch("app.agents.anomaly_detection_agent.create_llm_client", return_value=mock_client):
            s.ANOMALY_DETECTION_STRATEGY = "llm"
            s.VLLM_BASE_URL = "http://localhost:11434"
            s.VLLM_MODEL = "llama3.1:8b"
            s.VLLM_TIMEOUT_SECONDS = 5.0
            s.VLLM_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["anomaly_type"] is None


# ---------------------------------------------------------------------------
# TestAnomalyDetectionAgentValidateOutputs
# ---------------------------------------------------------------------------

class TestAnomalyDetectionAgentValidateOutputs:
    def _agent(self):
        return AnomalyDetectionAgent()

    def _result(self, outputs: dict) -> AgentResult:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="AnomalyDetectionAgent",
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
            "anomaly_detected": False,
            "anomaly_type": None,
            "severity": None,
            "affected_fields": [],
            "anomaly_score": 0.0,
            "confidence": 0.70,
            "rationale": "heuristic",
        }

    def test_valid_passes(self):
        self._agent().validate_outputs(self._result(self._valid()))

    def test_anomaly_detected_not_bool_raises(self):
        with pytest.raises(ValueError, match="anomaly_detected"):
            self._agent().validate_outputs(self._result(dict(self._valid(), anomaly_detected="yes")))

    def test_invalid_anomaly_type_raises(self):
        with pytest.raises(ValueError, match="anomaly_type"):
            self._agent().validate_outputs(self._result(dict(self._valid(), anomaly_type="weird")))

    def test_valid_anomaly_types_pass(self):
        for atype in ("credibility_spike", "uncertainty_surge", "conflict_flood",
                       "temporal_drift", "stale_revival", None):
            self._agent().validate_outputs(self._result(dict(self._valid(), anomaly_type=atype)))

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            self._agent().validate_outputs(self._result(dict(self._valid(), severity="critical")))

    def test_affected_fields_not_list_raises(self):
        with pytest.raises(ValueError, match="affected_fields"):
            self._agent().validate_outputs(self._result(dict(self._valid(), affected_fields="oops")))

    def test_anomaly_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="anomaly_score"):
            self._agent().validate_outputs(self._result(dict(self._valid(), anomaly_score=1.5)))

    def test_failed_status_skips_validation(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="AnomalyDetectionAgent",
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
        self._agent().validate_outputs(result)


# ---------------------------------------------------------------------------
# TestAnomalyDetectionAgentMetadata
# ---------------------------------------------------------------------------

class TestAnomalyDetectionAgentMetadata:
    def test_name(self):
        assert AnomalyDetectionAgent.name == "AnomalyDetectionAgent"

    def test_version(self):
        assert AnomalyDetectionAgent.version == "v1"

    def test_dependencies_include_credibility(self):
        assert "CredibilityAgent" in AnomalyDetectionAgent().dependencies()

    def test_dependencies_include_uncertainty(self):
        assert "UncertaintyReportingAgent" in AnomalyDetectionAgent().dependencies()

    def test_dependencies_is_list(self):
        assert isinstance(AnomalyDetectionAgent().dependencies(), list)


# ---------------------------------------------------------------------------
# TestAnomalyDetectionRegistry
# ---------------------------------------------------------------------------

class TestAnomalyDetectionRegistry:
    def test_anomaly_detection(self):
        assert isinstance(get_agent("anomaly_detection"), AnomalyDetectionAgent)

    def test_anomalydetection(self):
        assert isinstance(get_agent("anomalydetection"), AnomalyDetectionAgent)

    def test_anomalydetectionagent(self):
        assert isinstance(get_agent("anomalydetectionagent"), AnomalyDetectionAgent)

    def test_anomaly_agent(self):
        assert isinstance(get_agent("anomaly_agent"), AnomalyDetectionAgent)

    def test_unknown_still_none(self):
        assert get_agent("no_such_xyz") is None


# ---------------------------------------------------------------------------
# TestAnomalyDetectionIntegration
# ---------------------------------------------------------------------------

class TestAnomalyDetectionIntegration:
    @pytest.mark.asyncio
    async def test_clean_memory_json_serialisable(self):
        import json
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_clean_enrichment()))
        json.dumps(result.outputs)

    @pytest.mark.asyncio
    async def test_single_credibility_spike(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx({"credibility_score": 0.10}))
        assert result.outputs["anomaly_detected"] is True
        assert result.outputs["anomaly_type"] == "credibility_spike"
        assert "credibility_score" in result.outputs["affected_fields"]

    @pytest.mark.asyncio
    async def test_conflict_flood_detection(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx({"conflict_count": 7}))
        assert result.outputs["anomaly_detected"] is True
        assert result.outputs["anomaly_type"] == "conflict_flood"

    @pytest.mark.asyncio
    async def test_multiple_anomalies_highest_score_wins(self):
        agent = AnomalyDetectionAgent()
        enrichment = {
            "credibility_score": 0.15,
            "uncertainty_score": 0.95,
        }
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(enrichment))
        assert result.outputs["anomaly_detected"] is True
        assert len(result.outputs["affected_fields"]) >= 2

    @pytest.mark.asyncio
    async def test_stale_revival_detection(self):
        agent = AnomalyDetectionAgent()
        enrichment = {"decay_factor": 0.03, "is_stale": False}
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(enrichment))
        assert result.outputs["anomaly_detected"] is True
        assert result.outputs["anomaly_type"] == "stale_revival"

    @pytest.mark.asyncio
    async def test_full_crisis_high_severity(self):
        agent = AnomalyDetectionAgent()
        with patch("app.agents.anomaly_detection_agent.settings") as s:
            s.ANOMALY_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_crisis_enrichment()))
        assert result.outputs["severity"] == "high"
        assert result.outputs["anomaly_score"] >= 0.70
