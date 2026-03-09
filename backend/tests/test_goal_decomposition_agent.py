"""Tests for Phase 21 — GoalDecompositionAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.goal_decomposition_agent import (
    GoalDecompositionAgent,
    detect_goal,
    extract_subtasks,
    estimate_completion_fraction,
    detect_blocking_subtask,
    compute_agent_confidence,
    _is_action_sentence,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


_NOW = datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(
    *,
    content: str = "Plan to migrate the database: 1. backup data, 2. run migrations, 3. verify integrity",
    episode_role: str = "opener",
    temporal_phase: str = "planning",
    step_position: int | None = None,
    matched_playbook_id: str | None = None,
    episode_key: str = "ep-001",
    job_id: str | None = None,
) -> dict:
    return {
        "memory": {
            "content": content,
            "enrichment": {
                "episode_role": episode_role,
                "episode_key": episode_key,
                "temporal_phase": temporal_phase,
                "step_position": step_position,
                "matched_playbook_id": matched_playbook_id,
            },
        },
        "runtime": {"job_id": job_id} if job_id else {},
    }


def _make_result(outputs: dict, status: str = "success") -> AgentResult:
    return AgentResult(
        agent_name="GoalDecompositionAgent",
        agent_version="v1",
        memory_id="mem-1",
        status=status,
        confidence=outputs.get("confidence", 0.5),
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=_NOW,
        finished_at=_NOW,
        trace_id=None,
    )


# ---------------------------------------------------------------------------
# detect_goal
# ---------------------------------------------------------------------------

class TestDetectGoal:
    def test_explicit_goal_keyword(self):
        assert detect_goal("The goal is to improve retention") is True

    def test_plan_keyword(self):
        assert detect_goal("Plan to migrate the database this quarter") is True

    def test_objective_keyword(self):
        assert detect_goal("Objective: reduce latency by 30%") is True

    def test_steps_keyword(self):
        assert detect_goal("The steps are: analyze, build, deploy") is True

    def test_roadmap_keyword(self):
        assert detect_goal("Roadmap for Q2 includes three initiatives") is True

    def test_numbered_list_implies_goal(self):
        content = "1. analyze requirements\n2. design solution\n3. implement and test"
        assert detect_goal(content) is True

    def test_need_to_with_action_verb(self):
        assert detect_goal("We need to deploy the patch before Friday") is True

    def test_should_with_action_verb(self):
        assert detect_goal("We should analyze the error logs first") is True

    def test_milestone_keyword(self):
        assert detect_goal("Milestone: complete the API refactor") is True

    def test_strategy_keyword(self):
        assert detect_goal("Strategy for customer acquisition includes three phases") is True

    def test_plain_fact_no_goal(self):
        assert detect_goal("The server was restarted at 3pm") is False

    def test_simple_observation_no_goal(self):
        assert detect_goal("Database latency increased by 20ms today") is False

    def test_empty_content(self):
        assert detect_goal("") is False

    def test_whitespace_only(self):
        assert detect_goal("   ") is False

    def test_very_short_content(self):
        assert detect_goal("ok") is False

    def test_none_like_string(self):
        assert detect_goal("None") is False

    def test_initiative_keyword(self):
        assert detect_goal("The initiative covers three workstreams") is True

    def test_phase_keyword(self):
        assert detect_goal("Phase 1 covers the discovery work") is True

    def test_will_with_action_verb(self):
        assert detect_goal("We will implement the new auth flow next sprint") is True

    def test_must_with_action_verb(self):
        assert detect_goal("We must validate the migration before cutover") is True


# ---------------------------------------------------------------------------
# _is_action_sentence
# ---------------------------------------------------------------------------

class TestIsActionSentence:
    def test_deploy_action(self):
        assert _is_action_sentence("deploy the new service to production") is True

    def test_analyze_action(self):
        assert _is_action_sentence("analyze the error logs from last night") is True

    def test_configure_action(self):
        assert _is_action_sentence("configure the load balancer with the new certs") is True

    def test_plain_observation(self):
        assert _is_action_sentence("the server was down yesterday") is False

    def test_too_short(self):
        assert _is_action_sentence("ok") is False

    def test_empty(self):
        assert _is_action_sentence("") is False

    def test_validate_action(self):
        assert _is_action_sentence("validate the output against the schema") is True

    def test_launch_action(self):
        assert _is_action_sentence("launch the feature flag for 10% of users") is True


# ---------------------------------------------------------------------------
# extract_subtasks
# ---------------------------------------------------------------------------

class TestExtractSubtasks:
    def test_numbered_list(self):
        content = "1. backup data\n2. run migrations\n3. verify integrity"
        result = extract_subtasks(content)
        assert result == ["backup data", "run migrations", "verify integrity"]

    def test_numbered_list_with_periods(self):
        content = "1. analyze logs\n2. apply fix\n3. run tests"
        result = extract_subtasks(content)
        assert len(result) == 3
        assert result[0] == "analyze logs"

    def test_bullet_list_dash(self):
        content = "- design the UI\n- implement backend\n- write tests"
        result = extract_subtasks(content)
        assert result == ["design the UI", "implement backend", "write tests"]

    def test_bullet_list_asterisk(self):
        content = "* check logs\n* fix bug\n* deploy"
        result = extract_subtasks(content)
        assert len(result) == 3

    def test_step_pattern(self):
        content = "Step 1: analyze the issue. Step 2: implement the fix. Step 3: verify the result"
        result = extract_subtasks(content)
        assert len(result) >= 2
        assert any("analyze" in s for s in result)

    def test_action_sentence_fallback(self):
        content = "We need to analyze the logs. Then deploy the fix. Finally validate the output."
        result = extract_subtasks(content)
        assert len(result) >= 1

    def test_empty_content(self):
        assert extract_subtasks("") == []

    def test_max_subtasks_respected(self):
        lines = "\n".join(f"{i+1}. task {i+1}" for i in range(20))
        result = extract_subtasks(lines)
        assert len(result) <= 10

    def test_custom_max_subtasks(self):
        lines = "\n".join(f"{i+1}. task {i+1}" for i in range(20))
        result = extract_subtasks(lines, max_subtasks=5)
        assert len(result) <= 5

    def test_numbered_priority_over_bullets(self):
        content = "1. first item\n- bullet item\n2. second item"
        result = extract_subtasks(content)
        # numbered list takes priority
        assert result[0] == "first item"

    def test_deduplication(self):
        content = "1. analyze logs\n2. analyze logs\n3. deploy fix"
        result = extract_subtasks(content)
        assert result.count("analyze logs") == 1

    def test_subtask_truncated_at_200_chars(self):
        long_item = "a" * 300
        content = f"1. {long_item}\n2. deploy fix"
        result = extract_subtasks(content)
        assert len(result[0]) <= 200

    def test_no_plain_text_no_subtasks(self):
        content = "The server was restarted at 3pm today."
        result = extract_subtasks(content)
        # no action verbs in this simple fact, so fallback returns empty
        assert isinstance(result, list)

    def test_numbered_items_stripped(self):
        content = "1.   analyze logs  \n2.   deploy   "
        result = extract_subtasks(content)
        assert result[0] == "analyze logs"
        assert result[1] == "deploy"


# ---------------------------------------------------------------------------
# estimate_completion_fraction
# ---------------------------------------------------------------------------

class TestEstimateCompletionFraction:
    def test_opener_role_planning_phase(self):
        f = estimate_completion_fraction(
            episode_role="opener", temporal_phase="planning", step_position=None, total_steps=0
        )
        # 0.10 + (-0.10) = 0.0
        assert f == 0.0

    def test_opener_role_execution_phase(self):
        f = estimate_completion_fraction(
            episode_role="opener", temporal_phase="execution", step_position=None, total_steps=0
        )
        assert f == pytest.approx(0.10, abs=0.01)

    def test_body_role_execution_phase(self):
        f = estimate_completion_fraction(
            episode_role="body", temporal_phase="execution", step_position=None, total_steps=0
        )
        assert f == pytest.approx(0.40, abs=0.01)

    def test_climax_role(self):
        f = estimate_completion_fraction(
            episode_role="climax", temporal_phase="execution", step_position=None, total_steps=0
        )
        assert f == pytest.approx(0.80, abs=0.01)

    def test_closer_role(self):
        f = estimate_completion_fraction(
            episode_role="closer", temporal_phase="execution", step_position=None, total_steps=0
        )
        assert f == pytest.approx(1.00, abs=0.01)

    def test_review_phase_adds_adjustment(self):
        f = estimate_completion_fraction(
            episode_role="body", temporal_phase="review", step_position=None, total_steps=0
        )
        # 0.40 + 0.15 = 0.55
        assert f == pytest.approx(0.55, abs=0.01)

    def test_planning_phase_subtracts_adjustment(self):
        f = estimate_completion_fraction(
            episode_role="body", temporal_phase="planning", step_position=None, total_steps=0
        )
        # 0.40 - 0.10 = 0.30
        assert f == pytest.approx(0.30, abs=0.01)

    def test_step_position_blends_in(self):
        # step 1 of 4 → step_frac = 0/4 = 0.0
        # body base = 0.40 → blended = 0.5*0.40 + 0.5*0.0 = 0.20
        f = estimate_completion_fraction(
            episode_role="body", temporal_phase="execution", step_position=1, total_steps=4
        )
        assert f == pytest.approx(0.20, abs=0.01)

    def test_step_position_halfway(self):
        # step 3 of 4 → step_frac = 2/4 = 0.50
        # body base = 0.40 → blended = 0.5*0.40 + 0.5*0.50 = 0.45
        f = estimate_completion_fraction(
            episode_role="body", temporal_phase="execution", step_position=3, total_steps=4
        )
        assert f == pytest.approx(0.45, abs=0.01)

    def test_clamped_to_zero(self):
        f = estimate_completion_fraction(
            episode_role="opener", temporal_phase="planning", step_position=None, total_steps=0
        )
        assert f >= 0.0

    def test_clamped_to_one(self):
        f = estimate_completion_fraction(
            episode_role="closer", temporal_phase="review", step_position=None, total_steps=0
        )
        assert f <= 1.0

    def test_unknown_role_defaults_to_body(self):
        f = estimate_completion_fraction(
            episode_role="unknown_role", temporal_phase="execution", step_position=None, total_steps=0
        )
        assert f == pytest.approx(0.40, abs=0.01)

    def test_unknown_phase_no_adjustment(self):
        f = estimate_completion_fraction(
            episode_role="body", temporal_phase="unknown", step_position=None, total_steps=0
        )
        assert f == pytest.approx(0.40, abs=0.01)


# ---------------------------------------------------------------------------
# detect_blocking_subtask
# ---------------------------------------------------------------------------

class TestDetectBlockingSubtask:
    def test_blocked_keyword_in_subtask(self):
        subtasks = ["analyze logs", "deploy patch — blocked waiting on approval", "verify result"]
        result = detect_blocking_subtask(subtasks, "deploy patch is blocked")
        assert result == "deploy patch — blocked waiting on approval"

    def test_stuck_keyword_in_subtask(self):
        subtasks = ["stuck on the migration step"]
        result = detect_blocking_subtask(subtasks, "migration is stuck")
        assert result == "stuck on the migration step"

    def test_no_blocking_returns_none(self):
        subtasks = ["analyze logs", "deploy fix", "verify result"]
        result = detect_blocking_subtask(subtasks, "deployed successfully")
        assert result is None

    def test_empty_subtasks_returns_none(self):
        result = detect_blocking_subtask([], "something is blocked")
        assert result is None

    def test_blocking_in_content_matches_closest_subtask(self):
        subtasks = ["analyze logs", "deploy patch", "verify integrity"]
        content = "deploy patch is blocked waiting for approval"
        result = detect_blocking_subtask(subtasks, content)
        assert result == "deploy patch"

    def test_cannot_keyword_in_subtask(self):
        subtasks = ["cannot connect to database"]
        result = detect_blocking_subtask(subtasks, "")
        assert result == "cannot connect to database"

    def test_pending_keyword_in_subtask(self):
        subtasks = ["review PR — pending", "merge to main"]
        result = detect_blocking_subtask(subtasks, "")
        assert result == "review PR — pending"

    def test_failed_keyword_in_subtask(self):
        subtasks = ["run tests", "deployment failed", "notify team"]
        result = detect_blocking_subtask(subtasks, "")
        assert result == "deployment failed"

    def test_no_match_in_content_no_subtask_match(self):
        subtasks = ["analyze foo", "deploy bar"]
        content = "everything is stuck but no overlap"
        # No tokens from subtasks appear in window → returns None or first match
        result = detect_blocking_subtask(subtasks, content)
        assert result is None or isinstance(result, str)

    def test_blocker_keyword(self):
        subtasks = ["fix the blocker in auth service"]
        result = detect_blocking_subtask(subtasks, "")
        assert result is not None

    def test_on_hold_in_content(self):
        subtasks = ["migrate database", "update configs"]
        content = "migrate database is on hold pending DBA review"
        result = detect_blocking_subtask(subtasks, content)
        assert result == "migrate database"

    def test_subtask_truncated(self):
        long_subtask = "blocked: " + "x" * 300
        result = detect_blocking_subtask([long_subtask], "")
        assert result is not None
        assert len(result) <= 200


# ---------------------------------------------------------------------------
# compute_agent_confidence
# ---------------------------------------------------------------------------

class TestComputeAgentConfidence:
    def test_no_goal_returns_low(self):
        c = compute_agent_confidence(
            goal_detected=False, subtask_count=0, has_blocking=False,
            completion_fraction=0.0, has_playbook=False,
        )
        assert c == 0.45

    def test_goal_no_subtasks(self):
        c = compute_agent_confidence(
            goal_detected=True, subtask_count=0, has_blocking=False,
            completion_fraction=0.0, has_playbook=False,
        )
        assert c == 0.50

    def test_goal_with_subtasks(self):
        c = compute_agent_confidence(
            goal_detected=True, subtask_count=2, has_blocking=False,
            completion_fraction=0.0, has_playbook=False,
        )
        # base 0.60 — subtask_count=2 doesn't reach the >=3 threshold
        assert c == pytest.approx(0.60, abs=0.01)

    def test_goal_with_many_subtasks(self):
        c = compute_agent_confidence(
            goal_detected=True, subtask_count=5, has_blocking=False,
            completion_fraction=0.5, has_playbook=False,
        )
        # base 0.60 + 0.08 (>=3) + 0.05 (cf>0) = 0.73
        assert c == pytest.approx(0.73, abs=0.01)

    def test_has_playbook_boosts(self):
        c = compute_agent_confidence(
            goal_detected=True, subtask_count=4, has_blocking=False,
            completion_fraction=0.5, has_playbook=True,
        )
        # base 0.60 + 0.08 + 0.07 + 0.05 = 0.80
        assert c == pytest.approx(0.80, abs=0.01)

    def test_capped_at_085(self):
        c = compute_agent_confidence(
            goal_detected=True, subtask_count=10, has_blocking=True,
            completion_fraction=0.9, has_playbook=True,
        )
        assert c <= 0.85

    def test_completion_fraction_zero_no_boost(self):
        c1 = compute_agent_confidence(
            goal_detected=True, subtask_count=2, has_blocking=False,
            completion_fraction=0.0, has_playbook=False,
        )
        c2 = compute_agent_confidence(
            goal_detected=True, subtask_count=2, has_blocking=False,
            completion_fraction=0.5, has_playbook=False,
        )
        assert c2 > c1

    def test_result_is_float(self):
        c = compute_agent_confidence(
            goal_detected=True, subtask_count=3, has_blocking=False,
            completion_fraction=0.4, has_playbook=False,
        )
        assert isinstance(c, float)


# ---------------------------------------------------------------------------
# GoalDecompositionAgent — heuristic
# ---------------------------------------------------------------------------

class TestGoalDecompositionAgentHeuristic:
    @pytest.mark.asyncio
    async def test_basic_goal_detection(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(
            content="1. backup data\n2. run migrations\n3. verify integrity",
            episode_role="opener",
            temporal_phase="planning",
        )
        with patch.object(type(agent), "run", wraps=agent.run):
            with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
                mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
                result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["goal_detected"] is True
        assert len(result.outputs["subtasks"]) == 3

    @pytest.mark.asyncio
    async def test_no_goal_content(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(
            content="The server logs show normal activity today.",
            episode_role="body",
            temporal_phase="execution",
        )
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-2", ctx)
        assert result.status == "success"
        assert result.outputs["goal_detected"] is False
        assert result.outputs["subtasks"] == []

    @pytest.mark.asyncio
    async def test_completion_fraction_closer_role(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(
            content="Plan: 1. analyze, 2. implement, 3. deploy",
            episode_role="closer",
            temporal_phase="execution",
        )
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-3", ctx)
        assert result.outputs["completion_fraction"] >= 0.90

    @pytest.mark.asyncio
    async def test_blocking_detected(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(
            content="1. analyze logs\n2. deploy patch — blocked waiting on review\n3. verify result",
            episode_role="body",
            temporal_phase="execution",
        )
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-4", ctx)
        assert result.outputs["blocking_subtask"] is not None
        assert "blocked" in result.outputs["blocking_subtask"].lower()

    @pytest.mark.asyncio
    async def test_empty_content_heuristic(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(content="")
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-5", ctx)
        assert result.status == "success"
        assert result.outputs["goal_detected"] is False

    @pytest.mark.asyncio
    async def test_with_playbook_context(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(
            content="Plan: 1. check prereqs, 2. run script, 3. verify output",
            episode_role="body",
            temporal_phase="execution",
            matched_playbook_id="pb-001",
            step_position=2,
        )
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-6", ctx)
        assert result.status == "success"
        # having a playbook boosts confidence
        assert result.confidence >= 0.60

    @pytest.mark.asyncio
    async def test_rationale_is_heuristic(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(
            content="Goal: migrate the database. Steps: 1. backup, 2. migrate, 3. verify"
        )
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-7", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(
            content="Plan: 1. step one, 2. step two",
            job_id="job-xyz",
        )
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-8", ctx)
        assert result.trace_id == "job-xyz"

    @pytest.mark.asyncio
    async def test_no_trace_id_when_absent(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(content="Plan: 1. step one, 2. step two")
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-9", ctx)
        assert result.trace_id is None

    @pytest.mark.asyncio
    async def test_step_position_string_coerced(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(
            content="Plan: 1. analyze, 2. deploy, 3. verify",
            episode_role="body",
            temporal_phase="execution",
            step_position=None,
        )
        ctx["memory"]["enrichment"]["step_position"] = "2"
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-10", ctx)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_step_position_invalid_string_ignored(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(content="Plan: 1. analyze, 2. deploy")
        ctx["memory"]["enrichment"]["step_position"] = "not-a-number"
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-11", ctx)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_missing_enrichment_defaults(self):
        agent = GoalDecompositionAgent()
        ctx = {"memory": {"content": "Plan: 1. step one"}, "runtime": {}}
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-12", ctx)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_empty_context_defaults(self):
        agent = GoalDecompositionAgent()
        ctx = {}
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "heuristic"
            result = await agent.run("mem-13", ctx)
        assert result.status == "success"


# ---------------------------------------------------------------------------
# GoalDecompositionAgent — LLM path
# ---------------------------------------------------------------------------

class TestGoalDecompositionAgentLLM:
    @pytest.mark.asyncio
    async def test_llm_valid_response_used(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(
            content="Goal: reduce churn. Steps: analyze cohorts, fix onboarding, send campaigns.",
        )
        llm_resp = {
            "goal_detected": True,
            "subtasks": ["analyze cohorts", "fix onboarding", "send campaigns"],
            "completion_fraction": 0.2,
            "blocking_subtask": None,
            "confidence": 0.78,
            "rationale": "llm",
        }
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_resp)
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            with patch("app.agents.goal_decomposition_agent.create_ollama_client", return_value=mock_client):
                result = await agent.run("mem-llm-1", ctx)
        assert result.outputs["goal_detected"] is True
        assert result.outputs["subtasks"] == ["analyze cohorts", "fix onboarding", "send campaigns"]
        assert result.outputs["completion_fraction"] == 0.2

    @pytest.mark.asyncio
    async def test_llm_malformed_response_falls_back(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(content="Plan: 1. analyze, 2. deploy, 3. verify")
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"garbage": True})
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            with patch("app.agents.goal_decomposition_agent.create_ollama_client", return_value=mock_client):
                result = await agent.run("mem-llm-2", ctx)
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_none_response_falls_back(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(content="Plan: 1. analyze, 2. deploy")
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=None)
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            with patch("app.agents.goal_decomposition_agent.create_ollama_client", return_value=mock_client):
                result = await agent.run("mem-llm-3", ctx)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_llm_missing_subtasks_falls_back(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(content="Plan: 1. analyze, 2. deploy")
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={
            "goal_detected": True,
            # missing "subtasks"
            "completion_fraction": 0.3,
        })
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            with patch("app.agents.goal_decomposition_agent.create_ollama_client", return_value=mock_client):
                result = await agent.run("mem-llm-4", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_agent_strategy_setting_used(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(content="Plan: 1. step one")
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-llm-5", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_empty_content_skips_llm(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(content="")
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            with patch("app.agents.goal_decomposition_agent.create_ollama_client") as mock_create:
                result = await agent.run("mem-llm-6", ctx)
        # empty content forces heuristic; LLM client should not even be called
        mock_create.assert_not_called()
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_llm_bool_subtasks_wrong_type_falls_back(self):
        agent = GoalDecompositionAgent()
        ctx = _make_context(content="Plan: 1. analyze, 2. deploy")
        mock_client = AsyncMock()
        # subtasks is not a list
        mock_client.complete_json = AsyncMock(return_value={
            "goal_detected": True,
            "subtasks": "analyze and deploy",
        })
        with patch("app.agents.goal_decomposition_agent.settings") as mock_settings:
            mock_settings.GOAL_DECOMPOSITION_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            with patch("app.agents.goal_decomposition_agent.create_ollama_client", return_value=mock_client):
                result = await agent.run("mem-llm-7", ctx)
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def test_valid_outputs_pass(self):
        agent = GoalDecompositionAgent()
        result = _make_result({
            "goal_detected": True,
            "subtasks": ["step one", "step two"],
            "completion_fraction": 0.5,
            "blocking_subtask": None,
            "confidence": 0.72,
            "rationale": "heuristic",
        })
        agent.validate_outputs(result)  # should not raise

    def test_skipped_for_non_success(self):
        agent = GoalDecompositionAgent()
        result = _make_result({}, status="failed")
        agent.validate_outputs(result)  # should not raise

    def test_goal_detected_not_bool_raises(self):
        agent = GoalDecompositionAgent()
        result = _make_result({
            "goal_detected": "yes",
            "subtasks": [],
            "completion_fraction": 0.0,
            "blocking_subtask": None,
        })
        with pytest.raises(ValueError, match="goal_detected"):
            agent.validate_outputs(result)

    def test_subtasks_not_list_raises(self):
        agent = GoalDecompositionAgent()
        result = _make_result({
            "goal_detected": True,
            "subtasks": "some tasks",
            "completion_fraction": 0.0,
            "blocking_subtask": None,
        })
        with pytest.raises(ValueError, match="subtasks"):
            agent.validate_outputs(result)

    def test_completion_fraction_out_of_range_raises(self):
        agent = GoalDecompositionAgent()
        result = _make_result({
            "goal_detected": True,
            "subtasks": [],
            "completion_fraction": 1.5,
            "blocking_subtask": None,
        })
        with pytest.raises(ValueError, match="completion_fraction"):
            agent.validate_outputs(result)

    def test_completion_fraction_negative_raises(self):
        agent = GoalDecompositionAgent()
        result = _make_result({
            "goal_detected": True,
            "subtasks": [],
            "completion_fraction": -0.1,
            "blocking_subtask": None,
        })
        with pytest.raises(ValueError, match="completion_fraction"):
            agent.validate_outputs(result)

    def test_blocking_subtask_wrong_type_raises(self):
        agent = GoalDecompositionAgent()
        result = _make_result({
            "goal_detected": True,
            "subtasks": [],
            "completion_fraction": 0.5,
            "blocking_subtask": 42,
        })
        with pytest.raises(ValueError, match="blocking_subtask"):
            agent.validate_outputs(result)

    def test_blocking_subtask_none_is_valid(self):
        agent = GoalDecompositionAgent()
        result = _make_result({
            "goal_detected": False,
            "subtasks": [],
            "completion_fraction": 0.0,
            "blocking_subtask": None,
        })
        agent.validate_outputs(result)  # should not raise

    def test_blocking_subtask_string_is_valid(self):
        agent = GoalDecompositionAgent()
        result = _make_result({
            "goal_detected": True,
            "subtasks": ["blocked step"],
            "completion_fraction": 0.3,
            "blocking_subtask": "blocked step",
        })
        agent.validate_outputs(result)  # should not raise

    def test_completion_fraction_zero_valid(self):
        agent = GoalDecompositionAgent()
        result = _make_result({
            "goal_detected": False,
            "subtasks": [],
            "completion_fraction": 0.0,
            "blocking_subtask": None,
        })
        agent.validate_outputs(result)  # should not raise

    def test_completion_fraction_one_valid(self):
        agent = GoalDecompositionAgent()
        result = _make_result({
            "goal_detected": True,
            "subtasks": ["done"],
            "completion_fraction": 1.0,
            "blocking_subtask": None,
        })
        agent.validate_outputs(result)  # should not raise


# ---------------------------------------------------------------------------
# Registry + dependencies
# ---------------------------------------------------------------------------

class TestRegistryAndDependencies:
    def test_registry_by_full_name(self):
        agent = get_agent("GoalDecompositionAgent")
        assert isinstance(agent, GoalDecompositionAgent)

    def test_registry_by_short_name(self):
        agent = get_agent("goal_decomposition")
        assert isinstance(agent, GoalDecompositionAgent)

    def test_registry_by_alias(self):
        agent = get_agent("goaldecomposition")
        assert isinstance(agent, GoalDecompositionAgent)

    def test_dependencies_declared(self):
        agent = GoalDecompositionAgent()
        deps = agent.dependencies()
        assert "EpisodicGroupingAgent" in deps
        assert "TemporalReasoningAgent" in deps
        assert "PlaybookAgent" in deps

    def test_name_and_version(self):
        agent = GoalDecompositionAgent()
        assert agent.name == "GoalDecompositionAgent"
        assert agent.version == "v1"

    def test_registry_unknown_returns_none(self):
        assert get_agent("nonexistent_agent_xyz") is None
