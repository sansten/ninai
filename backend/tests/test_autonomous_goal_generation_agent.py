"""Tests for AutonomousGoalGenerationAgent — Phase 37."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.agents.autonomous_goal_generation_agent import (
    AutonomousGoalGenerationAgent,
    _VALID_MOTIVATION_TYPES,
    _LOW_RELIABILITY_THRESHOLD,
    _MAX_GOALS,
    _DEFAULT_PRIORITY,
    _build_curiosity_goals,
    _build_prediction_goals,
    _build_improvement_goals,
    _rank_goals,
    _dominant_motivation,
    _aggregate_priority,
    _parse_llm_output,
    _safe_default,
    run_heuristic_async,
)
from app.agents.registry import get_agent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _gap(
    gap_type: str = "missing_fact",
    domain: str = "finance",
    description: str = "Missing Q2 revenue data",
    confidence: float = 0.75,
    related: list | None = None,
) -> dict:
    return {
        "gap_type": gap_type,
        "domain": domain,
        "description": description,
        "confidence_in_gap": confidence,
        "related_memories": related or [],
        "suggested_learning_approach": "search",
    }


def _activity(domain: str = "contracts", count: int = 20) -> dict:
    return {"domain": domain, "event_count": count}


def _tool_stats(success_rate: float = 0.4, sample_size: int = 10) -> dict:
    return {"success_rate_30d": success_rate, "sample_size_30d": sample_size}


def _mk_context(
    knowledge_gaps: list | None = None,
    tool_reliability: dict | None = None,
    activity_domains: list | None = None,
    org_id: str = "org-1",
    job_id: str | None = "job-001",
) -> dict:
    enrichment: dict = {}
    if knowledge_gaps is not None:
        enrichment["knowledge_gaps"] = knowledge_gaps
    if tool_reliability is not None:
        enrichment["tool_reliability"] = tool_reliability
    if activity_domains is not None:
        enrichment["activity_domains"] = activity_domains
    return {
        "tenant": {"org_id": org_id, "org_slug": "test"},
        "memory": {"enrichment": enrichment},
        "runtime": {"job_id": job_id},
    }


# ---------------------------------------------------------------------------
# _build_curiosity_goals
# ---------------------------------------------------------------------------

class TestBuildCuriosityGoals:
    def test_empty_gaps(self):
        assert _build_curiosity_goals([]) == []

    def test_single_gap(self):
        gaps = [_gap()]
        goals = _build_curiosity_goals(gaps)
        assert len(goals) == 1
        g = goals[0]
        assert g["motivation_type"] == "curiosity"
        assert "finance" in g["title"]
        assert 0.0 <= g["priority_score"] <= 1.0
        assert 0.0 <= g["confidence"] <= 1.0
        assert isinstance(g["trigger_ids"], list)

    def test_multiple_gaps(self):
        gaps = [_gap(), _gap(domain="hr"), _gap(domain="it")]
        goals = _build_curiosity_goals(gaps)
        assert len(goals) == 3
        assert all(g["motivation_type"] == "curiosity" for g in goals)

    def test_related_memories_propagated(self):
        gaps = [_gap(related=["mem-1", "mem-2"])]
        goals = _build_curiosity_goals(gaps)
        assert goals[0]["trigger_ids"] == ["mem-1", "mem-2"]

    def test_high_confidence_gap_high_priority(self):
        low = _build_curiosity_goals([_gap(confidence=0.3)])
        high = _build_curiosity_goals([_gap(confidence=0.9)])
        assert high[0]["priority_score"] > low[0]["priority_score"]

    def test_missing_description_fallback(self):
        gap = _gap(description="")
        goals = _build_curiosity_goals([gap])
        assert goals[0]["description"] != ""

    def test_gap_type_in_title(self):
        goals = _build_curiosity_goals([_gap(gap_type="contradiction")])
        assert "contradiction" in goals[0]["title"]

    def test_metadata_contains_domain(self):
        goals = _build_curiosity_goals([_gap(domain="sales")])
        assert goals[0]["metadata"]["domain"] == "sales"

    def test_suggested_approach_in_metadata(self):
        goals = _build_curiosity_goals([_gap()])
        assert goals[0]["metadata"]["suggested_approach"] == "search"

    def test_id_is_string(self):
        goals = _build_curiosity_goals([_gap()])
        assert isinstance(goals[0]["id"], str)
        assert len(goals[0]["id"]) > 0

    def test_priority_clamped(self):
        goals = _build_curiosity_goals([_gap(confidence=2.0)])
        assert goals[0]["priority_score"] <= 1.0

    def test_none_related_becomes_empty_list(self):
        gap = _gap()
        gap["related_memories"] = None
        goals = _build_curiosity_goals([gap])
        assert goals[0]["trigger_ids"] == []


# ---------------------------------------------------------------------------
# _build_prediction_goals
# ---------------------------------------------------------------------------

class TestBuildPredictionGoals:
    def test_empty(self):
        assert _build_prediction_goals([]) == []

    def test_single_domain(self):
        goals = _build_prediction_goals([_activity("contracts", 10)])
        assert len(goals) == 1
        assert goals[0]["motivation_type"] == "prediction"
        assert "contracts" in goals[0]["title"]

    def test_multiple_domains_capped_at_5(self):
        domains = [_activity(f"domain{i}", 10) for i in range(8)]
        goals = _build_prediction_goals(domains)
        assert len(goals) <= 5

    def test_higher_count_higher_priority(self):
        low = _build_prediction_goals([_activity("xyzlow", 1), _activity("xyzhigh", 100)])
        xyzlow_goal = next(g for g in low if "xyzlow" in g["title"])
        xyzhigh_goal = next(g for g in low if "xyzhigh" in g["title"])
        assert xyzhigh_goal["priority_score"] > xyzlow_goal["priority_score"]

    def test_zero_events_skipped(self):
        goals = _build_prediction_goals([_activity("a", 0)])
        assert goals == []

    def test_total_zero_guard(self):
        # all counts are 0
        goals = _build_prediction_goals([{"domain": "x", "event_count": 0}])
        assert goals == []

    def test_confidence_within_range(self):
        goals = _build_prediction_goals([_activity("d", 50)])
        assert 0.0 <= goals[0]["confidence"] <= 1.0

    def test_metadata_has_event_count(self):
        goals = _build_prediction_goals([_activity("hr", 15)])
        assert goals[0]["metadata"]["event_count"] == 15

    def test_trigger_ids_empty(self):
        goals = _build_prediction_goals([_activity("it", 5)])
        assert goals[0]["trigger_ids"] == []


# ---------------------------------------------------------------------------
# _build_improvement_goals
# ---------------------------------------------------------------------------

class TestBuildImprovementGoals:
    def test_empty(self):
        assert _build_improvement_goals({}) == []

    def test_high_reliability_skipped(self):
        goals = _build_improvement_goals({"search": _tool_stats(0.9, 10)})
        assert goals == []

    def test_low_reliability_produces_goal(self):
        goals = _build_improvement_goals({"search": _tool_stats(0.4, 10)})
        assert len(goals) == 1
        assert goals[0]["motivation_type"] == "self_improvement"
        assert "search" in goals[0]["title"]

    def test_insufficient_sample_skipped(self):
        goals = _build_improvement_goals({"search": _tool_stats(0.1, 2)})
        assert goals == []

    def test_sorted_by_urgency(self):
        goals = _build_improvement_goals({
            "a": _tool_stats(0.5, 10),
            "b": _tool_stats(0.2, 10),
        })
        assert goals[0]["priority_score"] >= goals[1]["priority_score"]

    def test_priority_equals_1_minus_success_rate(self):
        goals = _build_improvement_goals({"tool_x": _tool_stats(0.3, 10)})
        assert abs(goals[0]["priority_score"] - 0.7) < 0.01

    def test_target_rate_capped_at_1(self):
        goals = _build_improvement_goals({"tool_x": _tool_stats(0.9 - 0.001, 10)})
        # success_rate just below threshold — just in: target = min(1.0, 0.899 + 0.25) <= 1.0
        # Actually 0.599 < 0.60 → qualifies
        goals2 = _build_improvement_goals({"tool_x": _tool_stats(0.1, 10)})
        assert goals2[0]["metadata"]["target_success_rate"] <= 1.0

    def test_confidence_increases_with_sample_size(self):
        small = _build_improvement_goals({"t": _tool_stats(0.4, 5)})
        large = _build_improvement_goals({"t": _tool_stats(0.4, 80)})
        assert large[0]["confidence"] > small[0]["confidence"]

    def test_metadata_has_tool_name(self):
        goals = _build_improvement_goals({"deploy": _tool_stats(0.3, 10)})
        assert goals[0]["metadata"]["tool_name"] == "deploy"

    def test_at_threshold_skipped(self):
        goals = _build_improvement_goals({"t": _tool_stats(_LOW_RELIABILITY_THRESHOLD, 10)})
        assert goals == []


# ---------------------------------------------------------------------------
# _rank_goals
# ---------------------------------------------------------------------------

class TestRankGoals:
    def test_empty(self):
        assert _rank_goals([]) == []

    def test_sorted_descending(self):
        goals = [
            {"priority_score": 0.3, "id": "a"},
            {"priority_score": 0.9, "id": "b"},
            {"priority_score": 0.6, "id": "c"},
        ]
        ranked = _rank_goals(goals)
        scores = [g["priority_score"] for g in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_capped_at_max_goals(self):
        goals = [{"priority_score": float(i) / 20, "id": str(i)} for i in range(25)]
        assert len(_rank_goals(goals)) <= _MAX_GOALS

    def test_returns_top_n(self):
        goals = [{"priority_score": float(i) / 20, "id": str(i)} for i in range(15)]
        ranked = _rank_goals(goals)
        assert len(ranked) == _MAX_GOALS


# ---------------------------------------------------------------------------
# _dominant_motivation
# ---------------------------------------------------------------------------

class TestDominantMotivation:
    def test_empty_returns_curiosity(self):
        assert _dominant_motivation([]) == "curiosity"

    def test_single_type(self):
        goals = [{"motivation_type": "prediction", "priority_score": 0.8}]
        assert _dominant_motivation(goals) == "prediction"

    def test_highest_average_wins(self):
        goals = [
            {"motivation_type": "curiosity", "priority_score": 0.2},
            {"motivation_type": "self_improvement", "priority_score": 0.9},
            {"motivation_type": "self_improvement", "priority_score": 0.8},
        ]
        assert _dominant_motivation(goals) == "self_improvement"

    def test_tie_deterministic(self):
        goals = [
            {"motivation_type": "curiosity", "priority_score": 0.5},
            {"motivation_type": "prediction", "priority_score": 0.5},
        ]
        result = _dominant_motivation(goals)
        assert result in _VALID_MOTIVATION_TYPES


# ---------------------------------------------------------------------------
# _aggregate_priority
# ---------------------------------------------------------------------------

class TestAggregatePriority:
    def test_empty(self):
        assert _aggregate_priority([]) == 0.0

    def test_single(self):
        goals = [{"priority_score": 0.8}]
        result = _aggregate_priority(goals)
        assert 0.0 <= result <= 1.0

    def test_decays_with_rank(self):
        top = [{"priority_score": 0.9}, {"priority_score": 0.1}]
        bottom = [{"priority_score": 0.1}, {"priority_score": 0.9}]
        assert _aggregate_priority(top) > _aggregate_priority(bottom)

    def test_clamped_to_1(self):
        goals = [{"priority_score": 1.0} for _ in range(5)]
        assert _aggregate_priority(goals) <= 1.0


# ---------------------------------------------------------------------------
# _parse_llm_output
# ---------------------------------------------------------------------------

class TestParseLlmOutput:
    def _valid_resp(self) -> dict:
        return {
            "generated_goals": [
                {
                    "id": "abc-123",
                    "title": "Investigate gap in finance",
                    "description": "Missing data",
                    "motivation_type": "curiosity",
                    "priority_score": 0.7,
                    "trigger_ids": [],
                    "confidence": 0.8,
                    "metadata": {},
                }
            ],
            "motivation_type": "curiosity",
            "priority_score": 0.7,
        }

    def test_valid(self):
        result = _parse_llm_output(self._valid_resp())
        assert result is not None
        assert result["motivation_type"] == "curiosity"
        assert result["rationale"] == "llm"
        assert len(result["generated_goals"]) == 1

    def test_not_dict(self):
        assert _parse_llm_output("bad") is None

    def test_missing_goals_list(self):
        resp = self._valid_resp()
        del resp["generated_goals"]
        assert _parse_llm_output(resp) is None

    def test_goals_not_list(self):
        resp = self._valid_resp()
        resp["generated_goals"] = "not a list"
        assert _parse_llm_output(resp) is None

    def test_invalid_motivation_type(self):
        resp = self._valid_resp()
        resp["motivation_type"] = "unknown"
        assert _parse_llm_output(resp) is None

    def test_invalid_priority_score(self):
        resp = self._valid_resp()
        resp["priority_score"] = 1.5
        assert _parse_llm_output(resp) is None

    def test_goal_missing_title_skipped(self):
        resp = self._valid_resp()
        resp["generated_goals"].append({"description": "no title"})
        result = _parse_llm_output(resp)
        assert result is not None
        assert len(result["generated_goals"]) == 1

    def test_empty_goals_returns_none(self):
        resp = self._valid_resp()
        resp["generated_goals"] = [{"description": "no title"}]
        assert _parse_llm_output(resp) is None

    def test_goal_id_fallback(self):
        resp = self._valid_resp()
        del resp["generated_goals"][0]["id"]
        result = _parse_llm_output(resp)
        assert result is not None
        assert isinstance(result["generated_goals"][0]["id"], str)

    def test_goal_count_set(self):
        result = _parse_llm_output(self._valid_resp())
        assert result["goal_count"] == 1


# ---------------------------------------------------------------------------
# _safe_default
# ---------------------------------------------------------------------------

class TestSafeDefault:
    def test_structure(self):
        d = _safe_default()
        assert d["generated_goals"] == []
        assert d["motivation_type"] == "curiosity"
        assert d["priority_score"] == 0.0
        assert d["goal_count"] == 0
        assert d["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# run_heuristic_async
# ---------------------------------------------------------------------------

class TestRunHeuristicAsync:
    @pytest.mark.asyncio
    async def test_empty_inputs(self):
        result = await run_heuristic_async([], [], {})
        assert result["generated_goals"] == []
        assert result["motivation_type"] in _VALID_MOTIVATION_TYPES
        assert result["priority_score"] == 0.0
        assert result["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_gaps_only(self):
        result = await run_heuristic_async([_gap()], [], {})
        assert len(result["generated_goals"]) == 1
        assert result["motivation_type"] == "curiosity"

    @pytest.mark.asyncio
    async def test_activity_only(self):
        result = await run_heuristic_async([], [_activity()], {})
        assert len(result["generated_goals"]) == 1
        assert result["motivation_type"] == "prediction"

    @pytest.mark.asyncio
    async def test_tool_reliability_only(self):
        result = await run_heuristic_async([], [], {"bad_tool": _tool_stats(0.3, 10)})
        assert len(result["generated_goals"]) == 1
        assert result["motivation_type"] == "self_improvement"

    @pytest.mark.asyncio
    async def test_combined_goals_ranked(self):
        result = await run_heuristic_async(
            [_gap()],
            [_activity()],
            {"bad_tool": _tool_stats(0.2, 10)},
        )
        scores = [g["priority_score"] for g in result["generated_goals"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_goal_count_matches_list(self):
        result = await run_heuristic_async([_gap()], [_activity()], {})
        assert result["goal_count"] == len(result["generated_goals"])

    @pytest.mark.asyncio
    async def test_many_inputs_capped(self):
        gaps = [_gap(domain=f"d{i}") for i in range(8)]
        domains = [_activity(f"domain{i}", 10) for i in range(8)]
        tools = {f"tool{i}": _tool_stats(0.2, 10) for i in range(8)}
        result = await run_heuristic_async(gaps, domains, tools)
        assert len(result["generated_goals"]) <= _MAX_GOALS


# ---------------------------------------------------------------------------
# AutonomousGoalGenerationAgent — heuristic path
# ---------------------------------------------------------------------------

class TestAgentHeuristicPath:
    @pytest.mark.asyncio
    async def test_run_heuristic_empty(self):
        agent = AutonomousGoalGenerationAgent()
        with patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _mk_context())
        assert result.status == "success"
        assert result.outputs["generated_goals"] == []
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_run_heuristic_with_gap(self):
        agent = AutonomousGoalGenerationAgent()
        ctx = _mk_context(knowledge_gaps=[_gap()])
        with patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert len(result.outputs["generated_goals"]) == 1
        assert result.outputs["motivation_type"] == "curiosity"

    @pytest.mark.asyncio
    async def test_run_heuristic_with_activity(self):
        agent = AutonomousGoalGenerationAgent()
        ctx = _mk_context(activity_domains=[_activity()])
        with patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["motivation_type"] == "prediction"

    @pytest.mark.asyncio
    async def test_run_heuristic_with_tool_reliability(self):
        agent = AutonomousGoalGenerationAgent()
        ctx = _mk_context(tool_reliability={"slow_tool": _tool_stats(0.3, 10)})
        with patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["motivation_type"] == "self_improvement"

    @pytest.mark.asyncio
    async def test_outputs_validate(self):
        agent = AutonomousGoalGenerationAgent()
        ctx = _mk_context(knowledge_gaps=[_gap()], activity_domains=[_activity()])
        with patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        # validate_outputs should not raise
        agent.validate_outputs(result)

    @pytest.mark.asyncio
    async def test_confidence_set(self):
        agent = AutonomousGoalGenerationAgent()
        ctx = _mk_context(knowledge_gaps=[_gap()])
        with patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_trace_id_set(self):
        agent = AutonomousGoalGenerationAgent()
        ctx = _mk_context(job_id="job-abc")
        with patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.trace_id == "job-abc"

    @pytest.mark.asyncio
    async def test_no_trace_id(self):
        agent = AutonomousGoalGenerationAgent()
        ctx = _mk_context(job_id=None)
        with patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.trace_id is None

    @pytest.mark.asyncio
    async def test_missing_enrichment_key(self):
        agent = AutonomousGoalGenerationAgent()
        ctx = {"tenant": {"org_id": "org-1"}, "memory": {}, "runtime": {}}
        with patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"


# ---------------------------------------------------------------------------
# AutonomousGoalGenerationAgent — LLM path
# ---------------------------------------------------------------------------

class TestAgentLlmPath:
    def _mk_llm_resp(self, motivation: str = "curiosity") -> dict:
        return {
            "generated_goals": [
                {
                    "id": "goal-llm-1",
                    "title": "LLM-generated curiosity goal",
                    "description": "Investigate unknown domain",
                    "motivation_type": motivation,
                    "priority_score": 0.75,
                    "trigger_ids": [],
                    "confidence": 0.8,
                    "metadata": {},
                }
            ],
            "motivation_type": motivation,
            "priority_score": 0.75,
        }

    @pytest.mark.asyncio
    async def test_llm_success(self):
        agent = AutonomousGoalGenerationAgent()
        ctx = _mk_context()

        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=self._mk_llm_resp())

        with (
            patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings,
            patch("app.agents.autonomous_goal_generation_agent.create_ollama_client", return_value=mock_client),
        ):
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "qwen2.5:7b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)

        assert result.status == "success"
        assert result.outputs["rationale"] == "llm"
        assert result.outputs["motivation_type"] == "curiosity"

    @pytest.mark.asyncio
    async def test_llm_invalid_falls_back_to_heuristic(self):
        agent = AutonomousGoalGenerationAgent()
        ctx = _mk_context()

        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "response"})

        with (
            patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings,
            patch("app.agents.autonomous_goal_generation_agent.create_ollama_client", return_value=mock_client),
        ):
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "qwen2.5:7b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)

        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_heuristic(self):
        agent = AutonomousGoalGenerationAgent()
        ctx = _mk_context()

        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(side_effect=RuntimeError("timeout"))

        with (
            patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings,
            patch("app.agents.autonomous_goal_generation_agent.create_ollama_client", return_value=mock_client),
        ):
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "qwen2.5:7b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)

        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_all_motivation_types(self):
        agent = AutonomousGoalGenerationAgent()
        for mt in _VALID_MOTIVATION_TYPES:
            ctx = _mk_context()
            mock_client = MagicMock()
            mock_client.complete_json = AsyncMock(return_value=self._mk_llm_resp(mt))

            with (
                patch("app.agents.autonomous_goal_generation_agent.settings") as mock_settings,
                patch("app.agents.autonomous_goal_generation_agent.create_ollama_client", return_value=mock_client),
            ):
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_settings.OLLAMA_MODEL = "qwen2.5:7b"
                mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_settings.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-1", ctx)

            assert result.outputs["motivation_type"] == mt


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _make_result(self, outputs: dict):
        from app.agents.types import AgentResult
        return AgentResult(
            agent_name="AutonomousGoalGenerationAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="success",
            confidence=0.5,
            outputs=outputs,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

    def test_valid_outputs(self):
        agent = AutonomousGoalGenerationAgent()
        result = self._make_result({
            "generated_goals": [],
            "motivation_type": "curiosity",
            "priority_score": 0.5,
            "goal_count": 0,
            "rationale": "heuristic",
        })
        agent.validate_outputs(result)  # no raise

    def test_goals_not_list(self):
        agent = AutonomousGoalGenerationAgent()
        result = self._make_result({
            "generated_goals": "wrong",
            "motivation_type": "curiosity",
            "priority_score": 0.5,
            "goal_count": 0,
        })
        with pytest.raises(ValueError, match="generated_goals"):
            agent.validate_outputs(result)

    def test_invalid_motivation_type(self):
        agent = AutonomousGoalGenerationAgent()
        result = self._make_result({
            "generated_goals": [],
            "motivation_type": "bad_type",
            "priority_score": 0.5,
            "goal_count": 0,
        })
        with pytest.raises(ValueError, match="motivation_type"):
            agent.validate_outputs(result)

    def test_priority_score_out_of_range(self):
        agent = AutonomousGoalGenerationAgent()
        result = self._make_result({
            "generated_goals": [],
            "motivation_type": "curiosity",
            "priority_score": 1.5,
            "goal_count": 0,
        })
        with pytest.raises(ValueError, match="priority_score"):
            agent.validate_outputs(result)

    def test_negative_goal_count(self):
        agent = AutonomousGoalGenerationAgent()
        result = self._make_result({
            "generated_goals": [],
            "motivation_type": "curiosity",
            "priority_score": 0.5,
            "goal_count": -1,
        })
        with pytest.raises(ValueError, match="goal_count"):
            agent.validate_outputs(result)

    def test_skips_validation_on_non_success(self):
        from app.agents.types import AgentResult
        agent = AutonomousGoalGenerationAgent()
        result = AgentResult(
            agent_name="AutonomousGoalGenerationAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="failed",
            confidence=0.0,
            outputs={},
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )
        agent.validate_outputs(result)  # no raise


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered_canonical(self):
        agent = get_agent("autonomous_goal_generation")
        assert isinstance(agent, AutonomousGoalGenerationAgent)

    def test_registered_camel_case(self):
        agent = get_agent("autonomousgoalgeneration")
        assert isinstance(agent, AutonomousGoalGenerationAgent)

    def test_registered_full_name(self):
        agent = get_agent("autonomousgoalgenerationagent")
        assert isinstance(agent, AutonomousGoalGenerationAgent)

    def test_unknown_returns_none(self):
        assert get_agent("notarealphase37agent") is None


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

class TestAgentMetadata:
    def test_name(self):
        assert AutonomousGoalGenerationAgent.name == "AutonomousGoalGenerationAgent"

    def test_version(self):
        assert AutonomousGoalGenerationAgent.version == "v1"

    def test_dependencies(self):
        deps = AutonomousGoalGenerationAgent().dependencies()
        assert "GoalDecompositionAgent" in deps
        assert "ProactiveMemoryPushAgent" in deps
        assert "MetaCognitivePlanningAgent" in deps
