"""Tests for Phase 22 — UncertaintyReportingAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.uncertainty_reporting_agent import (
    UncertaintyReportingAgent,
    collect_unresolved_conflicts,
    collect_low_confidence_fields,
    compute_uncertainty_score,
    classify_uncertainty_level,
    compute_agent_confidence,
    run_heuristic,
    _is_float,
    _LOW_CONFIDENCE_THRESHOLD,
    _REVIEW_SCORE_THRESHOLD,
    _STALE_DECAY_THRESHOLD,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


_NOW = datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(
    *,
    conflicts: list | None = None,
    high_severity_conflicts: list | None = None,
    resolution_rate: float | None = None,
    escalation_targets: list | None = None,
    credibility_score: float | None = None,
    credibility_flags: list | None = None,
    decay_factor: float | None = None,
    is_stale: bool | None = None,
    playbook_confidence: float | None = None,
    temporal_confidence: float | None = None,
    causal_confidence: float | None = None,
    blocking_subtask: str | None = None,
    job_id: str | None = None,
) -> dict:
    enrichment: dict = {}
    if conflicts is not None:
        enrichment["conflicts"] = conflicts
    if high_severity_conflicts is not None:
        enrichment["high_severity_conflicts"] = high_severity_conflicts
    if resolution_rate is not None:
        enrichment["resolution_rate"] = resolution_rate
    if escalation_targets is not None:
        enrichment["escalation_targets"] = escalation_targets
    if credibility_score is not None:
        enrichment["credibility_score"] = credibility_score
    if credibility_flags is not None:
        enrichment["credibility_flags"] = credibility_flags
    if decay_factor is not None:
        enrichment["decay_factor"] = decay_factor
    if is_stale is not None:
        enrichment["is_stale"] = is_stale
    if playbook_confidence is not None:
        enrichment["playbook_confidence"] = playbook_confidence
    if temporal_confidence is not None:
        enrichment["temporal_confidence"] = temporal_confidence
    if causal_confidence is not None:
        enrichment["causal_confidence"] = causal_confidence
    if blocking_subtask is not None:
        enrichment["blocking_subtask"] = blocking_subtask

    return {
        "memory": {"content": "some memory content", "enrichment": enrichment},
        "runtime": {"job_id": job_id} if job_id else {},
    }


def _make_result(outputs: dict, status: str = "success") -> AgentResult:
    return AgentResult(
        agent_name="UncertaintyReportingAgent",
        agent_version="v1",
        memory_id="mem-1",
        status=status,
        confidence=outputs.get("confidence", 0.65),
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=_NOW,
        finished_at=_NOW,
        trace_id=None,
    )


def _conflict(entity: str = "ProjectX", severity: str = "high", conflict_type: str = "state_mismatch") -> dict:
    return {"entity": entity, "severity": severity, "conflict_type": conflict_type}


# ---------------------------------------------------------------------------
# _is_float
# ---------------------------------------------------------------------------

class TestIsFloat:
    def test_int(self):
        assert _is_float(1) is True

    def test_float(self):
        assert _is_float(0.5) is True

    def test_string_float(self):
        assert _is_float("0.75") is True

    def test_string_invalid(self):
        assert _is_float("abc") is False

    def test_none(self):
        assert _is_float(None) is False

    def test_list(self):
        assert _is_float([]) is False

    def test_zero(self):
        assert _is_float(0) is True

    def test_negative(self):
        assert _is_float(-1.5) is True


# ---------------------------------------------------------------------------
# collect_unresolved_conflicts
# ---------------------------------------------------------------------------

class TestCollectUnresolvedConflicts:
    def test_no_conflicts_returns_empty(self):
        assert collect_unresolved_conflicts({}) == []

    def test_empty_conflict_list(self):
        assert collect_unresolved_conflicts({"conflicts": []}) == []

    def test_resolution_rate_1_returns_empty(self):
        enrichment = {
            "conflicts": [_conflict()],
            "resolution_rate": 1.0,
        }
        assert collect_unresolved_conflicts(enrichment) == []

    def test_high_severity_kept(self):
        enrichment = {
            "conflicts": [_conflict(severity="high")],
            "resolution_rate": 0.5,
        }
        result = collect_unresolved_conflicts(enrichment)
        assert len(result) == 1
        assert result[0]["severity"] == "high"

    def test_low_severity_below_half_resolution_kept(self):
        enrichment = {
            "conflicts": [_conflict(severity="low")],
            "resolution_rate": 0.3,
        }
        result = collect_unresolved_conflicts(enrichment)
        assert len(result) == 1

    def test_low_severity_above_half_resolution_skipped(self):
        enrichment = {
            "conflicts": [_conflict(entity="B", severity="low")],
            "resolution_rate": 0.6,
        }
        result = collect_unresolved_conflicts(enrichment)
        assert len(result) == 0

    def test_escalated_entity_kept(self):
        enrichment = {
            "conflicts": [_conflict(entity="CustomerDB", severity="medium")],
            "resolution_rate": 0.8,
            "escalation_targets": ["CustomerDB"],
        }
        result = collect_unresolved_conflicts(enrichment)
        assert len(result) == 1
        assert result[0]["entity"] == "CustomerDB"

    def test_invalid_conflict_items_skipped(self):
        enrichment = {
            "conflicts": [None, "bad", 42, _conflict()],
            "resolution_rate": 0.0,
        }
        result = collect_unresolved_conflicts(enrichment)
        assert all(isinstance(c, dict) for c in result)

    def test_multiple_conflicts_mixed_severity(self):
        enrichment = {
            "conflicts": [
                _conflict(entity="A", severity="high"),
                _conflict(entity="B", severity="low"),
                _conflict(entity="C", severity="medium"),
            ],
            "resolution_rate": 0.6,
        }
        result = collect_unresolved_conflicts(enrichment)
        entities = [c["entity"] for c in result]
        assert "A" in entities
        # B (low) not escalated + resolution_rate=0.6 -> skipped
        assert "B" not in entities

    def test_output_fields_present(self):
        enrichment = {"conflicts": [_conflict()], "resolution_rate": 0.0}
        result = collect_unresolved_conflicts(enrichment)
        assert "entity" in result[0]
        assert "severity" in result[0]
        assert "conflict_type" in result[0]

    def test_no_resolution_rate_defaults_zero(self):
        # resolution_rate missing → defaults to 0.0 → keep high severity
        enrichment = {"conflicts": [_conflict(severity="high")]}
        result = collect_unresolved_conflicts(enrichment)
        assert len(result) == 1

    def test_none_conflicts_returns_empty(self):
        assert collect_unresolved_conflicts({"conflicts": None}) == []


# ---------------------------------------------------------------------------
# collect_low_confidence_fields
# ---------------------------------------------------------------------------

class TestCollectLowConfidenceFields:
    def test_empty_enrichment(self):
        assert collect_low_confidence_fields({}) == []

    def test_credibility_score_low(self):
        fields = collect_low_confidence_fields({"credibility_score": 0.40})
        assert "credibility_score" in fields

    def test_credibility_score_high_not_flagged(self):
        fields = collect_low_confidence_fields({"credibility_score": 0.80})
        assert "credibility_score" not in fields

    def test_credibility_score_at_threshold_not_flagged(self):
        fields = collect_low_confidence_fields({"credibility_score": _LOW_CONFIDENCE_THRESHOLD})
        assert "credibility_score" not in fields

    def test_playbook_confidence_low(self):
        fields = collect_low_confidence_fields({"playbook_confidence": 0.30})
        assert "playbook_confidence" in fields

    def test_playbook_confidence_high(self):
        fields = collect_low_confidence_fields({"playbook_confidence": 0.90})
        assert "playbook_confidence" not in fields

    def test_temporal_confidence_low(self):
        fields = collect_low_confidence_fields({"temporal_confidence": 0.20})
        assert "temporal_confidence" in fields

    def test_causal_confidence_low(self):
        fields = collect_low_confidence_fields({"causal_confidence": 0.40})
        assert "causal_confidence" in fields

    def test_decay_factor_stale(self):
        fields = collect_low_confidence_fields({"decay_factor": 0.10})
        assert "decay_factor" in fields

    def test_decay_factor_healthy(self):
        fields = collect_low_confidence_fields({"decay_factor": 0.80})
        assert "decay_factor" not in fields

    def test_blocking_subtask_flags_goal_completion(self):
        fields = collect_low_confidence_fields({"blocking_subtask": "deploy service"})
        assert "goal_completion" in fields

    def test_no_blocking_subtask(self):
        fields = collect_low_confidence_fields({"blocking_subtask": None})
        assert "goal_completion" not in fields

    def test_multiple_low_fields(self):
        enrichment = {
            "credibility_score": 0.20,
            "playbook_confidence": 0.10,
            "temporal_confidence": 0.30,
        }
        fields = collect_low_confidence_fields(enrichment)
        assert len(fields) >= 3

    def test_non_numeric_values_ignored(self):
        enrichment = {"credibility_score": "not-a-number"}
        fields = collect_low_confidence_fields(enrichment)
        assert "credibility_score" not in fields


# ---------------------------------------------------------------------------
# compute_uncertainty_score
# ---------------------------------------------------------------------------

class TestComputeUncertaintyScore:
    def test_all_zero_signals(self):
        score = compute_uncertainty_score(
            unresolved_count=0,
            low_confidence_count=0,
            high_severity_count=0,
            credibility_score=None,
            is_stale=False,
        )
        assert score == 0.0

    def test_single_unresolved_conflict(self):
        score = compute_uncertainty_score(
            unresolved_count=1,
            low_confidence_count=0,
            high_severity_count=0,
            credibility_score=None,
            is_stale=False,
        )
        assert score > 0.0

    def test_unresolved_conflicts_capped_at_30(self):
        score_3 = compute_uncertainty_score(
            unresolved_count=3,
            low_confidence_count=0,
            high_severity_count=0,
            credibility_score=None,
            is_stale=False,
        )
        score_100 = compute_uncertainty_score(
            unresolved_count=100,
            low_confidence_count=0,
            high_severity_count=0,
            credibility_score=None,
            is_stale=False,
        )
        assert score_3 == score_100  # both capped at 0.30

    def test_high_severity_adds_weight(self):
        score_no_high = compute_uncertainty_score(
            unresolved_count=0,
            low_confidence_count=0,
            high_severity_count=0,
            credibility_score=None,
            is_stale=False,
        )
        score_with_high = compute_uncertainty_score(
            unresolved_count=0,
            low_confidence_count=0,
            high_severity_count=2,
            credibility_score=None,
            is_stale=False,
        )
        assert score_with_high > score_no_high

    def test_poor_credibility_increases_score(self):
        score_good = compute_uncertainty_score(
            unresolved_count=0,
            low_confidence_count=0,
            high_severity_count=0,
            credibility_score=0.90,
            is_stale=False,
        )
        score_bad = compute_uncertainty_score(
            unresolved_count=0,
            low_confidence_count=0,
            high_severity_count=0,
            credibility_score=0.20,
            is_stale=False,
        )
        assert score_bad > score_good

    def test_staleness_adds_score(self):
        score_fresh = compute_uncertainty_score(
            unresolved_count=0,
            low_confidence_count=0,
            high_severity_count=0,
            credibility_score=None,
            is_stale=False,
        )
        score_stale = compute_uncertainty_score(
            unresolved_count=0,
            low_confidence_count=0,
            high_severity_count=0,
            credibility_score=None,
            is_stale=True,
        )
        assert score_stale > score_fresh

    def test_score_capped_at_1(self):
        score = compute_uncertainty_score(
            unresolved_count=10,
            low_confidence_count=10,
            high_severity_count=10,
            credibility_score=0.10,
            is_stale=True,
        )
        assert score <= 1.0

    def test_score_is_float(self):
        score = compute_uncertainty_score(
            unresolved_count=2,
            low_confidence_count=2,
            high_severity_count=1,
            credibility_score=0.50,
            is_stale=False,
        )
        assert isinstance(score, float)

    def test_none_credibility_not_penalty(self):
        score_none = compute_uncertainty_score(
            unresolved_count=0,
            low_confidence_count=0,
            high_severity_count=0,
            credibility_score=None,
            is_stale=False,
        )
        assert score_none == 0.0


# ---------------------------------------------------------------------------
# classify_uncertainty_level
# ---------------------------------------------------------------------------

class TestClassifyUncertaintyLevel:
    def test_score_zero_is_low(self):
        assert classify_uncertainty_level(0.0) == "low"

    def test_score_just_above_low_threshold(self):
        assert classify_uncertainty_level(0.30) == "medium"

    def test_score_medium_range(self):
        assert classify_uncertainty_level(0.45) == "medium"

    def test_score_high_range(self):
        assert classify_uncertainty_level(0.60) == "high"

    def test_score_critical(self):
        assert classify_uncertainty_level(0.85) == "critical"

    def test_score_1_is_critical(self):
        assert classify_uncertainty_level(1.0) == "critical"

    def test_score_just_below_high(self):
        level = classify_uncertainty_level(0.54)
        assert level == "medium"

    def test_score_at_high_threshold(self):
        level = classify_uncertainty_level(0.55)
        assert level == "high"

    def test_score_at_critical_threshold(self):
        level = classify_uncertainty_level(0.80)
        assert level == "critical"

    def test_returns_string(self):
        assert isinstance(classify_uncertainty_level(0.5), str)


# ---------------------------------------------------------------------------
# compute_agent_confidence
# ---------------------------------------------------------------------------

class TestComputeAgentConfidence:
    def test_critical_level_high_base(self):
        conf = compute_agent_confidence(
            uncertainty_level="critical",
            unresolved_count=3,
            low_confidence_count=2,
        )
        assert conf >= 0.80

    def test_low_level_lower_base(self):
        conf = compute_agent_confidence(
            uncertainty_level="low",
            unresolved_count=0,
            low_confidence_count=0,
        )
        assert conf < 0.70

    def test_many_signals_boosts_confidence(self):
        conf_few = compute_agent_confidence(
            uncertainty_level="medium",
            unresolved_count=0,
            low_confidence_count=0,
        )
        conf_many = compute_agent_confidence(
            uncertainty_level="medium",
            unresolved_count=3,
            low_confidence_count=2,
        )
        assert conf_many >= conf_few

    def test_result_in_range(self):
        conf = compute_agent_confidence(
            uncertainty_level="high",
            unresolved_count=2,
            low_confidence_count=1,
        )
        assert 0.40 <= conf <= 0.90

    def test_returns_float(self):
        conf = compute_agent_confidence(
            uncertainty_level="low",
            unresolved_count=0,
            low_confidence_count=0,
        )
        assert isinstance(conf, float)


# ---------------------------------------------------------------------------
# run_heuristic
# ---------------------------------------------------------------------------

class TestRunHeuristic:
    def test_clean_enrichment_low_uncertainty(self):
        result = run_heuristic({
            "credibility_score": 0.90,
            "decay_factor": 0.80,
            "conflicts": [],
            "resolution_rate": 1.0,
        })
        assert result["uncertainty_level"] == "low"
        assert result["review_recommended"] is False
        assert result["unresolved_conflicts"] == []
        assert result["rationale"] == "heuristic"

    def test_high_severity_conflicts_raise_level(self):
        result = run_heuristic({
            "conflicts": [_conflict(severity="high"), _conflict(entity="Y", severity="high")],
            "high_severity_conflicts": [_conflict(severity="high"), _conflict(entity="Y", severity="high")],
            "resolution_rate": 0.0,
        })
        # 2 unresolved + 2 high-severity → score 0.40 → medium or above
        assert result["uncertainty_level"] in {"medium", "high", "critical"}
        assert result["uncertainty_score"] > 0.0

    def test_stale_memory_contributes(self):
        result = run_heuristic({"is_stale": True})
        assert result["uncertainty_score"] > 0.0

    def test_multiple_low_confidence_fields(self):
        result = run_heuristic({
            "credibility_score": 0.20,
            "playbook_confidence": 0.10,
            "temporal_confidence": 0.30,
            "causal_confidence": 0.40,
        })
        assert len(result["low_confidence_fields"]) >= 3

    def test_blocking_subtask_appears_in_low_fields(self):
        result = run_heuristic({"blocking_subtask": "deploy the service"})
        assert "goal_completion" in result["low_confidence_fields"]

    def test_output_keys_present(self):
        result = run_heuristic({})
        for key in ("uncertainty_level", "unresolved_conflicts", "low_confidence_fields",
                    "review_recommended", "uncertainty_score", "confidence", "rationale"):
            assert key in result

    def test_uncertainty_score_in_range(self):
        result = run_heuristic({
            "conflicts": [_conflict()],
            "high_severity_conflicts": [_conflict()],
            "credibility_score": 0.30,
        })
        assert 0.0 <= result["uncertainty_score"] <= 1.0

    def test_fully_resolved_no_conflicts_reported(self):
        result = run_heuristic({
            "conflicts": [_conflict(severity="high")],
            "resolution_rate": 1.0,
        })
        assert result["unresolved_conflicts"] == []

    def test_review_not_recommended_for_clean_memory(self):
        result = run_heuristic({
            "credibility_score": 0.95,
            "decay_factor": 0.95,
        })
        assert result["review_recommended"] is False

    def test_critical_level_when_everything_wrong(self):
        result = run_heuristic({
            "conflicts": [_conflict(), _conflict(entity="B"), _conflict(entity="C")],
            "high_severity_conflicts": [_conflict(), _conflict(entity="B"), _conflict(entity="C")],
            "resolution_rate": 0.0,
            "credibility_score": 0.10,
            "decay_factor": 0.05,
            "is_stale": True,
            "playbook_confidence": 0.10,
            "temporal_confidence": 0.10,
            "causal_confidence": 0.10,
            "blocking_subtask": "blocked step",
        })
        assert result["uncertainty_level"] == "critical"
        assert result["review_recommended"] is True


# ---------------------------------------------------------------------------
# UncertaintyReportingAgent — heuristic path
# ---------------------------------------------------------------------------

class TestUncertaintyReportingAgentHeuristic:
    @pytest.mark.asyncio
    async def test_returns_success_clean_input(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(credibility_score=0.90, decay_factor=0.80)
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_high_conflicts_triggers_review(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(
            conflicts=[_conflict(severity="high"), _conflict(entity="B", severity="high")],
            high_severity_conflicts=[_conflict(), _conflict(entity="B")],
            resolution_rate=0.0,
        )
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.outputs["review_recommended"] is True

    @pytest.mark.asyncio
    async def test_no_enrichment(self):
        agent = UncertaintyReportingAgent()
        ctx = {"memory": {"content": "something", "enrichment": {}}, "runtime": {}}
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["uncertainty_level"] == "low"

    @pytest.mark.asyncio
    async def test_missing_memory_key(self):
        agent = UncertaintyReportingAgent()
        ctx: dict = {}
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(job_id="job-123")
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.trace_id == "job-123"

    @pytest.mark.asyncio
    async def test_timestamps_set(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context()
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.started_at <= result.finished_at

    @pytest.mark.asyncio
    async def test_confidence_in_range(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(credibility_score=0.40, playbook_confidence=0.30)
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_uncertainty_score_in_range(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(
            conflicts=[_conflict()],
            resolution_rate=0.0,
            credibility_score=0.30,
        )
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        score = result.outputs["uncertainty_score"]
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_low_confidence_fields_list(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(credibility_score=0.20, temporal_confidence=0.10)
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert "credibility_score" in result.outputs["low_confidence_fields"]
        assert "temporal_confidence" in result.outputs["low_confidence_fields"]

    @pytest.mark.asyncio
    async def test_agent_strategy_fallback(self):
        """AGENT_STRATEGY=heuristic should be respected."""
        agent = UncertaintyReportingAgent()
        ctx = _make_context()
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_blocking_subtask_in_low_fields(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(blocking_subtask="pending approval")
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert "goal_completion" in result.outputs["low_confidence_fields"]

    @pytest.mark.asyncio
    async def test_stale_memory(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(is_stale=True)
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.outputs["uncertainty_score"] > 0.0

    @pytest.mark.asyncio
    async def test_decay_factor_stale_threshold(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(decay_factor=_STALE_DECAY_THRESHOLD - 0.05)
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings:
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert "decay_factor" in result.outputs["low_confidence_fields"]


# ---------------------------------------------------------------------------
# UncertaintyReportingAgent — LLM path
# ---------------------------------------------------------------------------

class TestUncertaintyReportingAgentLLM:
    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(conflicts=[_conflict()], resolution_rate=0.5)
        llm_response = {
            "uncertainty_level": "high",
            "unresolved_conflicts": [{"entity": "ProjectX", "severity": "high", "conflict_type": "state_mismatch"}],
            "low_confidence_fields": ["credibility_score"],
            "review_recommended": True,
            "uncertainty_score": 0.70,
            "confidence": 0.82,
            "rationale": "high conflict load with low resolution",
        }
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_response)
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings, \
             patch("app.agents.uncertainty_reporting_agent.create_ollama_client", return_value=mock_client):
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)
        assert result.outputs["uncertainty_level"] == "high"
        assert result.outputs["review_recommended"] is True

    @pytest.mark.asyncio
    async def test_invalid_llm_response_falls_back(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context()
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "data"})
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings, \
             patch("app.agents.uncertainty_reporting_agent.create_ollama_client", return_value=mock_client):
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_none_response_falls_back(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context()
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=None)
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings, \
             patch("app.agents.uncertainty_reporting_agent.create_ollama_client", return_value=mock_client):
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_wrong_level_falls_back(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context()
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={
            "uncertainty_level": "extreme",  # invalid
            "unresolved_conflicts": [],
            "review_recommended": True,
        })
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings, \
             patch("app.agents.uncertainty_reporting_agent.create_ollama_client", return_value=mock_client):
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_missing_review_recommended_falls_back(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context()
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={
            "uncertainty_level": "medium",
            "unresolved_conflicts": [],
            # review_recommended missing
        })
        with patch("app.agents.uncertainty_reporting_agent.settings") as mock_settings, \
             patch("app.agents.uncertainty_reporting_agent.create_ollama_client", return_value=mock_client):
            mock_settings.UNCERTAINTY_REPORTING_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _agent(self) -> UncertaintyReportingAgent:
        return UncertaintyReportingAgent()

    def _good(self) -> dict:
        return {
            "uncertainty_level": "medium",
            "unresolved_conflicts": [],
            "low_confidence_fields": [],
            "review_recommended": False,
            "uncertainty_score": 0.40,
            "confidence": 0.70,
            "rationale": "heuristic",
        }

    def test_valid_outputs_pass(self):
        agent = self._agent()
        result = _make_result(self._good())
        agent.validate_outputs(result)  # no exception

    def test_invalid_level_raises(self):
        agent = self._agent()
        outputs = {**self._good(), "uncertainty_level": "extreme"}
        result = _make_result(outputs)
        with pytest.raises(ValueError, match="uncertainty_level"):
            agent.validate_outputs(result)

    def test_unresolved_conflicts_not_list_raises(self):
        agent = self._agent()
        outputs = {**self._good(), "unresolved_conflicts": "not-a-list"}
        result = _make_result(outputs)
        with pytest.raises(ValueError, match="unresolved_conflicts"):
            agent.validate_outputs(result)

    def test_low_confidence_fields_not_list_raises(self):
        agent = self._agent()
        outputs = {**self._good(), "low_confidence_fields": None}
        result = _make_result(outputs)
        with pytest.raises(ValueError, match="low_confidence_fields"):
            agent.validate_outputs(result)

    def test_review_recommended_not_bool_raises(self):
        agent = self._agent()
        outputs = {**self._good(), "review_recommended": "yes"}
        result = _make_result(outputs)
        with pytest.raises(ValueError, match="review_recommended"):
            agent.validate_outputs(result)

    def test_uncertainty_score_out_of_range_raises(self):
        agent = self._agent()
        outputs = {**self._good(), "uncertainty_score": 1.5}
        result = _make_result(outputs)
        with pytest.raises(ValueError, match="uncertainty_score"):
            agent.validate_outputs(result)

    def test_uncertainty_score_negative_raises(self):
        agent = self._agent()
        outputs = {**self._good(), "uncertainty_score": -0.1}
        result = _make_result(outputs)
        with pytest.raises(ValueError, match="uncertainty_score"):
            agent.validate_outputs(result)

    def test_failed_status_skips_validation(self):
        agent = self._agent()
        result = _make_result({}, status="failed")
        agent.validate_outputs(result)  # no exception

    def test_all_valid_levels_pass(self):
        agent = self._agent()
        for level in ("low", "medium", "high", "critical"):
            outputs = {**self._good(), "uncertainty_level": level}
            result = _make_result(outputs)
            agent.validate_outputs(result)  # no exception


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

class TestAgentMetadata:
    def test_name(self):
        assert UncertaintyReportingAgent().name == "UncertaintyReportingAgent"

    def test_version(self):
        assert UncertaintyReportingAgent().version == "v1"

    def test_dependencies(self):
        deps = UncertaintyReportingAgent().dependencies()
        assert "ConflictDetectionAgent" in deps
        assert "CredibilityAgent" in deps
        assert "GoalDecompositionAgent" in deps


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_uncertainty_reporting_alias(self):
        assert isinstance(get_agent("uncertainty_reporting"), UncertaintyReportingAgent)

    def test_uncertaintyreportingagent_alias(self):
        assert isinstance(get_agent("uncertaintyreportingagent"), UncertaintyReportingAgent)

    def test_uncertainty_agent_alias(self):
        assert isinstance(get_agent("uncertainty_agent"), UncertaintyReportingAgent)

    def test_case_insensitive(self):
        assert isinstance(get_agent("UncertaintyReportingAgent"), UncertaintyReportingAgent)

    def test_unknown_returns_none(self):
        assert get_agent("phase22") is None


# ---------------------------------------------------------------------------
# Integration — end-to-end heuristic scenarios
# ---------------------------------------------------------------------------

class TestIntegration:
    @pytest.mark.asyncio
    async def test_fully_clean_memory_low(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(
            credibility_score=0.92,
            decay_factor=0.88,
            playbook_confidence=0.80,
            temporal_confidence=0.85,
            causal_confidence=0.78,
            conflicts=[],
            resolution_rate=1.0,
        )
        with patch("app.agents.uncertainty_reporting_agent.settings") as s:
            s.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-clean", ctx)
        assert result.outputs["uncertainty_level"] == "low"
        assert result.outputs["review_recommended"] is False
        assert result.outputs["unresolved_conflicts"] == []

    @pytest.mark.asyncio
    async def test_mixed_signals_medium(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(
            credibility_score=0.30,  # below threshold → triggers penalty
            decay_factor=0.20,       # below stale threshold → stale flag
            conflicts=[_conflict(severity="low")],
            resolution_rate=0.3,
        )
        with patch("app.agents.uncertainty_reporting_agent.settings") as s:
            s.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-mixed", ctx)
        level = result.outputs["uncertainty_level"]
        assert level in {"medium", "high"}

    @pytest.mark.asyncio
    async def test_crisis_memory_critical(self):
        agent = UncertaintyReportingAgent()
        ctx = _make_context(
            credibility_score=0.15,
            decay_factor=0.05,
            is_stale=True,
            conflicts=[_conflict(), _conflict(entity="B"), _conflict(entity="C")],
            high_severity_conflicts=[_conflict(), _conflict(entity="B"), _conflict(entity="C")],
            resolution_rate=0.0,
            playbook_confidence=0.10,
            temporal_confidence=0.10,
            causal_confidence=0.10,
            blocking_subtask="cannot proceed",
        )
        with patch("app.agents.uncertainty_reporting_agent.settings") as s:
            s.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-crisis", ctx)
        assert result.outputs["uncertainty_level"] == "critical"
        assert result.outputs["review_recommended"] is True
        assert len(result.outputs["unresolved_conflicts"]) > 0

    @pytest.mark.asyncio
    async def test_outputs_serializable(self):
        import json
        agent = UncertaintyReportingAgent()
        ctx = _make_context(
            conflicts=[_conflict()],
            high_severity_conflicts=[_conflict()],
            credibility_score=0.30,
            resolution_rate=0.0,
        )
        with patch("app.agents.uncertainty_reporting_agent.settings") as s:
            s.UNCERTAINTY_REPORTING_STRATEGY = "heuristic"
            result = await agent.run("mem-serial", ctx)
        # should not raise
        json.dumps(result.outputs)
