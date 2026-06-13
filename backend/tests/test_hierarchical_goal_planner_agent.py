from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.hierarchical_goal_planner_agent import (
    HierarchicalGoalPlannerAgent,
    _critical_path,
    _effort_for_depth,
    _leaf_tasks,
    _split_subgoals,
    run_heuristic,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult
from app.models.goal_hierarchy_node import GoalHierarchyNode


def _ctx(*, root_goal: str, depth_limit: int = 3, domain_context: str = "") -> dict:
    return {
        "memory": {
            "enrichment": {
                "root_goal": root_goal,
                "depth_limit": depth_limit,
                "domain_context": domain_context,
            }
        },
        "runtime": {"job_id": "trace-65"},
    }


class TestHelpers:
    def test_effort_by_depth(self):
        assert _effort_for_depth(0) == "large"
        assert _effort_for_depth(1) == "medium"
        assert _effort_for_depth(2) == "small"
        assert _effort_for_depth(3) == "trivial"

    def test_split_subgoals_by_and(self):
        out = _split_subgoals("collect logs and deploy fix")
        assert len(out) == 2

    def test_split_subgoals_handles_then(self):
        out = _split_subgoals("collect logs then deploy")
        assert len(out) == 2

    def test_leaf_tasks_identifies_non_parents(self):
        nodes = [
            {"title": "root", "parent_index": None},
            {"title": "child", "parent_index": 0},
        ]
        assert _leaf_tasks(nodes) == ["child"]

    def test_critical_path_walks_to_root(self):
        nodes = [
            {"title": "root", "depth": 0, "parent_index": None},
            {"title": "child", "depth": 1, "parent_index": 0},
            {"title": "leaf", "depth": 2, "parent_index": 1},
        ]
        assert _critical_path(nodes) == ["root", "child", "leaf"]


class TestHeuristic:
    def test_root_goal_with_and_creates_two_subgoals(self):
        out = run_heuristic(root_goal="collect logs and deploy fix", depth_limit=1)
        depth1 = [n for n in out["hierarchy"] if n["depth"] == 1]
        assert len(depth1) == 2

    def test_root_goal_without_conjunction_creates_default_subgoals(self):
        goal = "improve reliability"
        out = run_heuristic(root_goal=goal, depth_limit=1)
        depth1 = [n for n in out["hierarchy"] if n["depth"] == 1]
        assert len(depth1) == 2
        assert depth1[0]["title"].startswith("Gather information for:")

    def test_depth_limit_zero_only_root(self):
        out = run_heuristic(root_goal="goal", depth_limit=0)
        assert len(out["hierarchy"]) == 1
        assert out["max_depth_reached"] == 0

    def test_depth_limit_two_max_depth_two(self):
        out = run_heuristic(root_goal="collect and deploy", depth_limit=2)
        assert out["max_depth_reached"] == 2

    def test_leaf_tasks_depth_two_when_limit_two(self):
        out = run_heuristic(root_goal="collect and deploy", depth_limit=2)
        by_title = {n["title"]: n for n in out["hierarchy"]}
        assert all(by_title[t]["depth"] == 2 for t in out["leaf_tasks"])

    def test_critical_path_starts_with_root(self):
        goal = "collect and deploy"
        out = run_heuristic(root_goal=goal, depth_limit=3)
        assert out["critical_path"][0] == goal

    def test_total_nodes_matches_hierarchy_length(self):
        out = run_heuristic(root_goal="collect and deploy", depth_limit=3)
        assert out["total_nodes"] == len(out["hierarchy"])

    def test_estimated_effort_correct_by_depth(self):
        out = run_heuristic(root_goal="collect and deploy", depth_limit=3)
        for node in out["hierarchy"]:
            assert node["estimated_effort"] == _effort_for_depth(node["depth"])

    def test_parent_index_none_only_for_root(self):
        out = run_heuristic(root_goal="collect and deploy", depth_limit=3)
        root_nodes = [n for n in out["hierarchy"] if n["parent_index"] is None]
        assert len(root_nodes) == 1
        assert root_nodes[0]["depth"] == 0

    def test_depth_one_nodes_parent_zero(self):
        out = run_heuristic(root_goal="collect and deploy", depth_limit=1)
        for node in out["hierarchy"]:
            if node["depth"] == 1:
                assert node["parent_index"] == 0

    def test_depth_two_nodes_have_depth_one_parent(self):
        out = run_heuristic(root_goal="collect and deploy", depth_limit=2)
        for node in out["hierarchy"]:
            if node["depth"] == 2:
                parent = out["hierarchy"][node["parent_index"]]
                assert parent["depth"] == 1

    def test_depth_three_nodes_have_depth_two_parent(self):
        out = run_heuristic(root_goal="collect and deploy", depth_limit=3)
        for node in out["hierarchy"]:
            if node["depth"] == 3:
                parent = out["hierarchy"][node["parent_index"]]
                assert parent["depth"] == 2

    def test_status_pending_for_all_nodes(self):
        out = run_heuristic(root_goal="collect and deploy", depth_limit=3)
        assert all(n["status"] == "pending" for n in out["hierarchy"])

    def test_domain_context_appended_to_depth1(self):
        out = run_heuristic(root_goal="collect and deploy", depth_limit=1, domain_context="infra")
        assert all("(infra)" in n["title"] for n in out["hierarchy"] if n["depth"] == 1)

    def test_confidence_constant(self):
        out = run_heuristic(root_goal="goal", depth_limit=0)
        assert out["confidence"] == 0.7

    def test_blank_goal_uses_untitled(self):
        out = run_heuristic(root_goal="", depth_limit=0)
        assert out["hierarchy"][0]["title"] == "Untitled goal"

    def test_negative_depth_limit_treated_as_zero(self):
        out = run_heuristic(root_goal="goal", depth_limit=-3)
        assert len(out["hierarchy"]) == 1


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_heuristic(self):
        agent = HierarchicalGoalPlannerAgent()
        with patch("app.agents.hierarchical_goal_planner_agent.settings") as mock_settings:
            mock_settings.HIERARCHICAL_GOAL_PLANNER_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(root_goal="collect and deploy"))
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = HierarchicalGoalPlannerAgent()
        with patch("app.agents.hierarchical_goal_planner_agent.settings") as mock_settings:
            mock_settings.HIERARCHICAL_GOAL_PLANNER_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(root_goal="collect and deploy"))
        assert result.trace_id == "trace-65"

    @pytest.mark.asyncio
    async def test_strategy_fallback(self):
        agent = HierarchicalGoalPlannerAgent()
        with patch("app.agents.hierarchical_goal_planner_agent.settings") as mock_settings:
            mock_settings.HIERARCHICAL_GOAL_PLANNER_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(root_goal="collect and deploy"))
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = HierarchicalGoalPlannerAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "hierarchy": [{"title": "r", "depth": 0, "parent_index": None, "estimated_effort": "large", "status": "pending"}],
                "total_nodes": 1,
                "max_depth_reached": 0,
                "critical_path": ["r"],
                "leaf_tasks": ["r"],
                "confidence": 0.7,
                "rationale": "llm",
            }
        )
        with patch("app.agents.hierarchical_goal_planner_agent.settings") as mock_settings, patch(
            "app.agents.hierarchical_goal_planner_agent.create_llm_client", return_value=mock_client
        ):
            mock_settings.HIERARCHICAL_GOAL_PLANNER_STRATEGY = "llm"
            mock_settings.VLLM_BASE_URL = "http://localhost:11434"
            mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
            mock_settings.VLLM_MAX_CONCURRENCY = 2
            mock_settings.get_vllm_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(root_goal="collect and deploy"))
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_invalid_llm_falls_back(self):
        agent = HierarchicalGoalPlannerAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.hierarchical_goal_planner_agent.settings") as mock_settings, patch(
            "app.agents.hierarchical_goal_planner_agent.create_llm_client", return_value=mock_client
        ):
            mock_settings.HIERARCHICAL_GOAL_PLANNER_STRATEGY = "llm"
            mock_settings.VLLM_BASE_URL = "http://localhost:11434"
            mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
            mock_settings.VLLM_MAX_CONCURRENCY = 2
            mock_settings.get_vllm_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(root_goal="collect and deploy"))
        assert result.outputs["rationale"] == "heuristic"


class TestValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="HierarchicalGoalPlannerAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.7,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )

    def _valid_outputs(self) -> dict:
        return {
            "hierarchy": [],
            "total_nodes": 0,
            "max_depth_reached": 0,
            "critical_path": [],
            "leaf_tasks": [],
            "confidence": 0.7,
        }

    def test_validate_outputs_passes(self):
        HierarchicalGoalPlannerAgent().validate_outputs(self._result(self._valid_outputs()))

    def test_hierarchy_type_raises(self):
        with pytest.raises(ValueError, match="hierarchy"):
            HierarchicalGoalPlannerAgent().validate_outputs(self._result(dict(self._valid_outputs(), hierarchy="x")))

    def test_total_nodes_type_raises(self):
        with pytest.raises(ValueError, match="total_nodes"):
            HierarchicalGoalPlannerAgent().validate_outputs(self._result(dict(self._valid_outputs(), total_nodes="x")))

    def test_max_depth_type_raises(self):
        with pytest.raises(ValueError, match="max_depth_reached"):
            HierarchicalGoalPlannerAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), max_depth_reached="x"))
            )

    def test_critical_path_type_raises(self):
        with pytest.raises(ValueError, match="critical_path"):
            HierarchicalGoalPlannerAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), critical_path="x"))
            )

    def test_leaf_tasks_type_raises(self):
        with pytest.raises(ValueError, match="leaf_tasks"):
            HierarchicalGoalPlannerAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), leaf_tasks="x"))
            )

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        res = AgentResult(
            agent_name="HierarchicalGoalPlannerAgent",
            agent_version="v1",
            memory_id="m1",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )
        HierarchicalGoalPlannerAgent().validate_outputs(res)


class TestRegistry:
    def test_alias_primary(self):
        assert isinstance(get_agent("hierarchical_goal_planner"), HierarchicalGoalPlannerAgent)

    def test_alias_compact(self):
        assert isinstance(get_agent("hierarchicalgoalplanner"), HierarchicalGoalPlannerAgent)

    def test_alias_agent(self):
        assert isinstance(get_agent("HierarchicalGoalPlannerAgent"), HierarchicalGoalPlannerAgent)

    def test_alias_short(self):
        assert isinstance(get_agent("goal_hierarchy"), HierarchicalGoalPlannerAgent)


class TestModelShape:
    def test_model_has_required_fields(self):
        fields = GoalHierarchyNode.__table__.columns.keys()
        for name in (
            "org_id",
            "root_goal_id",
            "parent_node_id",
            "goal_id",
            "title",
            "depth",
            "status",
            "estimated_effort",
            "created_at",
        ):
            assert name in fields
