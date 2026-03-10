"""Tests for PlaybookExecutionTrackerAgent — Phase 33."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.playbook_execution_tracker_agent import (
    PlaybookExecutionTrackerAgent,
    compute_step_coverage,
    compute_outcome_score,
    find_deviation_steps,
    compute_agent_confidence,
    run_heuristic,
)
from app.agents.registry import get_agent


# ---------------------------------------------------------------------------
# compute_step_coverage
# ---------------------------------------------------------------------------

class TestComputeStepCoverage:
    def test_empty_executed_returns_zero(self):
        assert compute_step_coverage([], 5, None) == 0.0

    def test_full_coverage_with_total_steps(self):
        assert compute_step_coverage([1, 2, 3, 4, 5], 5, None) == 1.0

    def test_partial_coverage_with_total_steps(self):
        assert compute_step_coverage([1, 2, 3], 5, None) == 0.6

    def test_steps_beyond_total_are_not_counted(self):
        assert compute_step_coverage([1, 2, 6, 7], 5, None) == 0.4

    def test_deduplication(self):
        assert compute_step_coverage([1, 1, 2, 2, 3], 5, None) == 0.6

    def test_fallback_to_step_position(self):
        result = compute_step_coverage([1, 2], total_steps=None, step_position=4)
        assert 0.0 < result <= 1.0

    def test_fallback_no_total_no_position(self):
        result = compute_step_coverage([1, 2, 3], total_steps=None, step_position=None)
        assert 0.0 <= result <= 1.0

    def test_single_step_full_coverage(self):
        assert compute_step_coverage([1], 1, None) == 1.0

    def test_zero_total_steps_uses_fallback(self):
        result = compute_step_coverage([1, 2], total_steps=0, step_position=None)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# find_deviation_steps
# ---------------------------------------------------------------------------

class TestFindDeviationSteps:
    def test_all_steps_executed_no_deviations(self):
        assert find_deviation_steps([1, 2, 3], 3, None) == []

    def test_missing_middle_step(self):
        result = find_deviation_steps([1, 3], 3, None)
        assert result == [2]

    def test_missing_first_step(self):
        result = find_deviation_steps([2, 3], 3, None)
        assert result == [1]

    def test_no_total_no_position_empty(self):
        assert find_deviation_steps([1, 2], None, None) == []

    def test_fallback_to_step_position(self):
        result = find_deviation_steps([1, 3], total_steps=None, step_position=3)
        assert result == [2]

    def test_empty_executed_all_deviate(self):
        result = find_deviation_steps([], 3, None)
        assert result == [1, 2, 3]

    def test_extra_executed_steps_ignored(self):
        # executed steps beyond total are fine — only missing ones matter
        result = find_deviation_steps([1, 2, 3, 5], 3, None)
        assert result == []

    def test_total_takes_precedence_over_position(self):
        result = find_deviation_steps([1, 2], total_steps=4, step_position=2)
        assert result == [3, 4]


# ---------------------------------------------------------------------------
# compute_outcome_score
# ---------------------------------------------------------------------------

class TestComputeOutcomeScore:
    def test_success_high_coverage(self):
        score = compute_outcome_score("success", 1.0)
        assert score == 1.0

    def test_success_zero_coverage(self):
        score = compute_outcome_score("success", 0.0)
        assert score == pytest.approx(0.85)

    def test_partial_outcome(self):
        score = compute_outcome_score("partial", 0.5)
        assert 0.4 < score < 0.8

    def test_failure_outcome_low_score(self):
        score = compute_outcome_score("failure", 0.0)
        assert score == pytest.approx(0.10)

    def test_none_outcome_equals_coverage(self):
        score = compute_outcome_score(None, 0.6)
        assert score == pytest.approx(0.6)

    def test_unknown_outcome_treated_as_none(self):
        score = compute_outcome_score("unknown", 0.4)
        # "unknown" falls through to None branch
        assert score == pytest.approx(0.4)

    def test_score_within_bounds(self):
        for outcome in ("success", "partial", "failure", None):
            for cov in (0.0, 0.5, 1.0):
                s = compute_outcome_score(outcome, cov)
                assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# compute_agent_confidence
# ---------------------------------------------------------------------------

class TestComputeAgentConfidence:
    def test_base_confidence_no_signals(self):
        conf = compute_agent_confidence(False, None, 0.0, False)
        assert conf == pytest.approx(0.50)

    def test_full_signals_high_confidence(self):
        conf = compute_agent_confidence(True, "success", 1.0, True)
        assert conf >= 0.80

    def test_capped_at_0_90(self):
        conf = compute_agent_confidence(True, "success", 1.0, True)
        assert conf <= 0.90

    def test_matched_playbook_adds_confidence(self):
        base = compute_agent_confidence(False, None, 0.0, False)
        with_match = compute_agent_confidence(True, None, 0.0, False)
        assert with_match > base

    def test_feedback_applied_adds_confidence(self):
        base = compute_agent_confidence(False, None, 0.0, False)
        with_feedback = compute_agent_confidence(False, None, 0.0, True)
        assert with_feedback > base


# ---------------------------------------------------------------------------
# run_heuristic
# ---------------------------------------------------------------------------

class TestRunHeuristic:
    def _enrichment(self, **kwargs) -> dict:
        return {
            "matched_playbook_id": "pb-001",
            "step_position": 2,
            "playbook_confidence": 0.75,
            "feedback_applied": False,
            **kwargs,
        }

    def _trace(self, **kwargs) -> dict:
        return {
            "executed_steps": [1, 2, 3],
            "outcome": "success",
            "total_steps": 3,
            "session_id": "sess-1",
            **kwargs,
        }

    def test_followed_success(self):
        result = run_heuristic(self._enrichment(), self._trace())
        assert result["playbook_followed"] is True
        assert result["outcome_score"] >= 0.85
        assert result["deviation_steps"] == []
        assert result["tracked_playbook_id"] == "pb-001"

    def test_not_followed_no_executed_steps(self):
        result = run_heuristic(self._enrichment(), self._trace(executed_steps=[]))
        assert result["playbook_followed"] is False

    def test_not_followed_low_coverage(self):
        result = run_heuristic(self._enrichment(), self._trace(executed_steps=[1], total_steps=5))
        assert result["playbook_followed"] is False

    def test_not_followed_no_matched_playbook(self):
        enrichment = self._enrichment(matched_playbook_id=None)
        result = run_heuristic(enrichment, self._trace())
        assert result["playbook_followed"] is False
        assert result["tracked_playbook_id"] is None

    def test_deviation_steps_detected(self):
        result = run_heuristic(self._enrichment(), self._trace(executed_steps=[1, 3], total_steps=3))
        assert 2 in result["deviation_steps"]

    def test_outcome_failure_low_score(self):
        result = run_heuristic(
            self._enrichment(),
            self._trace(executed_steps=[1], outcome="failure", total_steps=5),
        )
        assert result["outcome_score"] < 0.3

    def test_outcome_partial(self):
        result = run_heuristic(
            self._enrichment(),
            self._trace(executed_steps=[1, 2], outcome="partial", total_steps=4),
        )
        assert 0.4 < result["outcome_score"] < 0.8

    def test_no_execution_trace(self):
        result = run_heuristic(self._enrichment(), None)
        assert result["playbook_followed"] is False
        assert result["outcome_score"] == 0.0
        # step_position=2 with no executed steps means steps [1,2] deviate
        assert isinstance(result["deviation_steps"], list)

    def test_empty_enrichment(self):
        result = run_heuristic({}, None)
        assert result["playbook_followed"] is False
        assert result["tracked_playbook_id"] is None

    def test_rationale_is_heuristic(self):
        result = run_heuristic(self._enrichment(), self._trace())
        assert result["rationale"] == "heuristic"

    def test_confidence_in_bounds(self):
        result = run_heuristic(self._enrichment(), self._trace())
        assert 0.0 <= result["confidence"] <= 1.0

    def test_invalid_outcome_treated_gracefully(self):
        result = run_heuristic(
            self._enrichment(),
            self._trace(outcome="catastrophic_meltdown"),
        )
        assert 0.0 <= result["outcome_score"] <= 1.0

    def test_non_integer_executed_steps_skipped(self):
        result = run_heuristic(
            self._enrichment(),
            {"executed_steps": ["one", "two"], "outcome": "success", "total_steps": 3},
        )
        assert result["playbook_followed"] is False

    def test_low_playbook_confidence_not_followed(self):
        enrichment = self._enrichment(playbook_confidence=0.10)
        result = run_heuristic(enrichment, self._trace())
        assert result["playbook_followed"] is False

    def test_feedback_applied_boosts_confidence(self):
        without = run_heuristic(self._enrichment(feedback_applied=False), self._trace())
        with_fb = run_heuristic(self._enrichment(feedback_applied=True), self._trace())
        assert with_fb["confidence"] >= without["confidence"]


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _agent(self) -> PlaybookExecutionTrackerAgent:
        return PlaybookExecutionTrackerAgent()

    def _good_result(self, **kwargs):
        from app.agents.types import AgentResult
        from datetime import datetime, timezone
        defaults = {
            "playbook_followed": True,
            "outcome_score": 0.9,
            "deviation_steps": [],
            "tracked_playbook_id": "pb-1",
            "confidence": 0.75,
            "rationale": "heuristic",
        }
        defaults.update(kwargs)
        return AgentResult(
            agent_name="PlaybookExecutionTrackerAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=defaults["confidence"],
            outputs=defaults,
            warnings=[],
            errors=[],
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

    def test_valid_passes(self):
        self._agent().validate_outputs(self._good_result())

    def test_invalid_playbook_followed(self):
        with pytest.raises(ValueError, match="playbook_followed"):
            self._agent().validate_outputs(self._good_result(playbook_followed="yes"))

    def test_invalid_outcome_score_type(self):
        with pytest.raises(ValueError, match="outcome_score"):
            self._agent().validate_outputs(self._good_result(outcome_score="high"))

    def test_outcome_score_out_of_range(self):
        with pytest.raises(ValueError, match="outcome_score"):
            self._agent().validate_outputs(self._good_result(outcome_score=1.5))

    def test_invalid_deviation_steps_type(self):
        with pytest.raises(ValueError, match="deviation_steps"):
            self._agent().validate_outputs(self._good_result(deviation_steps="none"))

    def test_invalid_tracked_playbook_id_type(self):
        with pytest.raises(ValueError, match="tracked_playbook_id"):
            self._agent().validate_outputs(self._good_result(tracked_playbook_id=123))

    def test_none_tracked_id_ok(self):
        self._agent().validate_outputs(self._good_result(tracked_playbook_id=None))

    def test_invalid_confidence_type(self):
        with pytest.raises(ValueError, match="confidence"):
            self._agent().validate_outputs(self._good_result(confidence="high"))

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="confidence"):
            self._agent().validate_outputs(self._good_result(confidence=1.1))

    def test_skips_on_non_success_status(self):
        from app.agents.types import AgentResult
        from datetime import datetime, timezone
        result = AgentResult(
            agent_name="PlaybookExecutionTrackerAgent",
            agent_version="v1",
            memory_id="m1",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=["boom"],
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )
        self._agent().validate_outputs(result)  # should not raise


# ---------------------------------------------------------------------------
# Agent.run — heuristic path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAgentRunHeuristic:
    def _context(self, enrichment: dict | None = None, trace: dict | None = None) -> dict:
        return {
            "memory": {
                "content": "Deployed the fix following the runbook.",
                "enrichment": enrichment or {
                    "matched_playbook_id": "pb-007",
                    "step_position": 3,
                    "playbook_confidence": 0.65,
                },
                "execution_trace": trace or {
                    "executed_steps": [1, 2, 3],
                    "outcome": "success",
                    "total_steps": 3,
                },
            },
            "runtime": {"job_id": "job-abc"},
        }

    async def test_run_heuristic_returns_success(self):
        agent = PlaybookExecutionTrackerAgent()
        with patch("app.agents.playbook_execution_tracker_agent.settings") as mock_settings:
            mock_settings.PLAYBOOK_TRACKER_STRATEGY = "heuristic"
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", self._context())
        assert result.status == "success"
        assert result.outputs["playbook_followed"] is True
        assert result.outputs["outcome_score"] >= 0.85

    async def test_run_heuristic_no_trace(self):
        agent = PlaybookExecutionTrackerAgent()
        ctx = self._context(trace=None)
        ctx["memory"]["execution_trace"] = None
        with patch("app.agents.playbook_execution_tracker_agent.settings") as mock_settings:
            mock_settings.PLAYBOOK_TRACKER_STRATEGY = "heuristic"
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-2", ctx)
        assert result.status == "success"
        assert result.outputs["playbook_followed"] is False

    async def test_run_sets_trace_id(self):
        agent = PlaybookExecutionTrackerAgent()
        with patch("app.agents.playbook_execution_tracker_agent.settings") as mock_settings:
            mock_settings.PLAYBOOK_TRACKER_STRATEGY = "heuristic"
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-3", self._context())
        assert result.trace_id == "job-abc"

    async def test_run_empty_context(self):
        agent = PlaybookExecutionTrackerAgent()
        with patch("app.agents.playbook_execution_tracker_agent.settings") as mock_settings:
            mock_settings.PLAYBOOK_TRACKER_STRATEGY = "heuristic"
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-4", {})
        assert result.status == "success"
        assert result.outputs["playbook_followed"] is False

    async def test_deviation_steps_in_output(self):
        agent = PlaybookExecutionTrackerAgent()
        ctx = self._context(
            trace={"executed_steps": [1, 3], "outcome": "partial", "total_steps": 3}
        )
        with patch("app.agents.playbook_execution_tracker_agent.settings") as mock_settings:
            mock_settings.PLAYBOOK_TRACKER_STRATEGY = "heuristic"
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-5", ctx)
        assert 2 in result.outputs["deviation_steps"]


# ---------------------------------------------------------------------------
# Agent.run — LLM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAgentRunLLM:
    def _context(self) -> dict:
        return {
            "memory": {
                "content": "Completed incident response using playbook.",
                "enrichment": {
                    "matched_playbook_id": "pb-llm",
                    "step_position": 2,
                    "playbook_confidence": 0.70,
                },
                "execution_trace": {
                    "executed_steps": [1, 2],
                    "outcome": "success",
                    "total_steps": 2,
                },
            },
            "runtime": {"job_id": "job-llm"},
        }

    async def test_llm_valid_response_used(self):
        agent = PlaybookExecutionTrackerAgent()
        llm_response = {
            "playbook_followed": True,
            "outcome_score": 0.92,
            "deviation_steps": [],
            "tracked_playbook_id": "pb-llm",
            "confidence": 0.80,
            "rationale": "All steps executed in order.",
        }
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=llm_response)

        with patch("app.agents.playbook_execution_tracker_agent.settings") as mock_settings, \
             patch("app.agents.playbook_execution_tracker_agent.create_ollama_client",
                   return_value=mock_client):
            mock_settings.PLAYBOOK_TRACKER_STRATEGY = "llm"
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-llm", self._context())

        assert result.status == "success"
        assert result.outputs["playbook_followed"] is True
        assert result.outputs["rationale"] == "All steps executed in order."

    async def test_llm_invalid_response_falls_back_to_heuristic(self):
        agent = PlaybookExecutionTrackerAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"garbage": True})

        with patch("app.agents.playbook_execution_tracker_agent.settings") as mock_settings, \
             patch("app.agents.playbook_execution_tracker_agent.create_ollama_client",
                   return_value=mock_client):
            mock_settings.PLAYBOOK_TRACKER_STRATEGY = "llm"
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-llm-fallback", self._context())

        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    async def test_llm_missing_deviation_steps_falls_back(self):
        agent = PlaybookExecutionTrackerAgent()
        # Response has playbook_followed but no deviation_steps list
        llm_response = {
            "playbook_followed": True,
            "outcome_score": 0.9,
            "confidence": 0.8,
        }
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=llm_response)

        with patch("app.agents.playbook_execution_tracker_agent.settings") as mock_settings, \
             patch("app.agents.playbook_execution_tracker_agent.create_ollama_client",
                   return_value=mock_client):
            mock_settings.PLAYBOOK_TRACKER_STRATEGY = "llm"
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-llm-fb2", self._context())

        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered_by_canonical_name(self):
        agent = get_agent("playbookexecutiontracker")
        assert agent is not None
        assert isinstance(agent, PlaybookExecutionTrackerAgent)

    def test_registered_by_snake_name(self):
        agent = get_agent("playbook_execution_tracker")
        assert agent is not None

    def test_registered_by_full_agent_name(self):
        agent = get_agent("playbookexecutiontrackeragent")
        assert agent is not None

    def test_unknown_name_returns_none(self):
        assert get_agent("not_a_real_agent_xyz") is None


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

class TestAgentMetadata:
    def test_name(self):
        assert PlaybookExecutionTrackerAgent().name == "PlaybookExecutionTrackerAgent"

    def test_version(self):
        assert PlaybookExecutionTrackerAgent().version == "v1"

    def test_dependencies(self):
        deps = PlaybookExecutionTrackerAgent().dependencies()
        assert "PlaybookAgent" in deps
        assert "FeedbackIntegrationAgent" in deps
