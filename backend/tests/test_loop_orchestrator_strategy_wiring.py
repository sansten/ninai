"""Tests verifying AdaptiveStrategyService and StrategyLearningService are wired into
LoopOrchestrator correctly — strategy hints fetched before the loop, tool recommendations
per iteration, both passed to the planner, and ingest called post-session.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.cognitive import CriticOutput, ExecutorOutput, PlannerOutput
from app.services.cognitive_loop.orchestrator import LoopOrchestrator, OrchestratorConfig
from app.services.cognitive_tooling.policy_guard import ToolContext
from app.services.simulation_service import SimulationService


# ---------------------------------------------------------------------------
# Test doubles (re-used from test_loop_orchestrator.py pattern)
# ---------------------------------------------------------------------------

@dataclass
class FakeSession:
    id: str
    status: str
    goal: str
    context_snapshot: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.context_snapshot is None:
            self.context_snapshot = {}


@dataclass
class FakeIteration:
    id: str
    session_id: str
    iteration_num: int
    critique_json: dict
    evaluation: str


class FakeRepo:
    def __init__(self, session: FakeSession):
        self.session = session
        self.iterations: dict[int, FakeIteration] = {}

    async def get_session(self, session_id: str):
        return self.session if self.session.id == session_id else None

    async def save_session_status(self, sess, status: str) -> None:
        sess.status = status

    async def get_iteration(self, *, session_id: str, iteration_num: int):
        return self.iterations.get(iteration_num)

    async def create_iteration(self, *, session_id: str, iteration_num: int):
        it = FakeIteration(
            id=f"it-{iteration_num}",
            session_id=session_id,
            iteration_num=iteration_num,
            critique_json={},
            evaluation="retry",
        )
        self.iterations[iteration_num] = it
        return it

    async def finalize_iteration(self, *, iteration, plan_json, execution_json,
                                  critique_json, evaluation, metrics, finished_at=None):
        iteration.critique_json = critique_json
        iteration.evaluation = evaluation


class FakeEvidence:
    async def retrieve_evidence(self, **kwargs):
        return [{"id": "m1", "summary": "evidence"}]


class TrackingPlanner:
    """Planner stub that records every call to plan()."""

    def __init__(self):
        self.calls: list[dict] = []

    async def plan(self, **kwargs) -> PlannerOutput:
        self.calls.append(dict(kwargs))
        return PlannerOutput(
            objective="goal",
            assumptions=[],
            constraints=[],
            required_tools=[],
            steps=[
                {
                    "step_id": "S1",
                    "action": "do",
                    "tool": None,
                    "tool_input_hint": {},
                    "expected_output": "x",
                    "success_criteria": ["ok"],
                    "risk_notes": [],
                }
            ],
            stop_conditions=[],
            confidence=0.5,
        )


class FakeExecutor:
    async def execute(self, **kwargs):
        return ExecutorOutput(step_results=[], overall_status="success", errors=[])


class FakeCritic:
    def __init__(self, evaluations: list[str]):
        self._evals = evaluations
        self._i = 0

    async def critique(self, **kwargs):
        ev = self._evals[min(self._i, len(self._evals) - 1)]
        self._i += 1
        return CriticOutput(evaluation=ev, strengths=[], issues=[], followup_questions=[], confidence=0.7)


@dataclass
class FakeSimRow:
    id: str


class FakeSimulationReports:
    async def create(self, **kwargs):
        return FakeSimRow(id="sim-1")


def _make_fake_adaptive_strategy(tool_recs=None):
    svc = MagicMock()
    svc.get_tool_recommendations = AsyncMock(return_value=tool_recs or [
        {"tool_name": "memory.search", "score": 0.9, "success_rate": 0.85, "total_uses": 10}
    ])
    return svc


def _make_fake_strategy_learning(hints=None):
    svc = MagicMock()
    svc.get_strategy_hints = AsyncMock(return_value=hints)
    svc.ingest_session_outcome = AsyncMock()
    return svc


def _make_orchestrator(
    session: FakeSession,
    planner: TrackingPlanner | None = None,
    adaptive_strategy=None,
    strategy_learning=None,
    critic_evals: list[str] | None = None,
    max_iterations: int = 1,
) -> LoopOrchestrator:
    return LoopOrchestrator(
        repo=FakeRepo(session),
        evidence=FakeEvidence(),
        planner=planner or TrackingPlanner(),
        simulator=SimulationService(),
        simulation_reports=FakeSimulationReports(),
        executor=FakeExecutor(),
        critic=FakeCritic(critic_evals or ["pass"]),
        available_tools=["memory.search"],
        adaptive_strategy=adaptive_strategy,
        strategy_learning=strategy_learning,
        config=OrchestratorConfig(max_iterations=max_iterations),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_strategy_hints_fetched_once_before_loop():
    """get_strategy_hints is called exactly once, before any planning iteration."""
    sess = FakeSession(id="s1", status="running", goal="analyze something",
                       context_snapshot={"domain": "ops"})
    strategy_hints = {"goal_type": "analysis", "domain": "ops", "top_strategies": []}
    strategy_learning = _make_fake_strategy_learning(hints=strategy_hints)
    planner = TrackingPlanner()

    orch = _make_orchestrator(sess, planner=planner, strategy_learning=strategy_learning)
    await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    strategy_learning.get_strategy_hints.assert_called_once_with(
        org_id="org1", goal_type="analysis", domain="ops"
    )


@pytest.mark.asyncio
async def test_strategy_hints_passed_to_planner():
    """The strategy_hints dict returned by get_strategy_hints is forwarded into planner.plan()."""
    sess = FakeSession(id="s1", status="running", goal="plan the sprint")
    hints = {"goal_type": "planning", "domain": "general", "top_strategies": [
        {"tool_sequence": ["memory.search"], "success_rate": 0.9, "avg_iterations": 2.0,
         "avg_plan_confidence": 0.8, "sample_count": 5}
    ]}
    strategy_learning = _make_fake_strategy_learning(hints=hints)
    planner = TrackingPlanner()

    orch = _make_orchestrator(sess, planner=planner, strategy_learning=strategy_learning)
    await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    assert planner.calls, "planner.plan() was never called"
    assert planner.calls[0]["strategy_hints"] == hints


@pytest.mark.asyncio
async def test_strategy_hints_none_when_service_absent():
    """When no strategy_learning service, strategy_hints=None is passed to planner."""
    sess = FakeSession(id="s1", status="running", goal="search for data")
    planner = TrackingPlanner()

    orch = _make_orchestrator(sess, planner=planner, strategy_learning=None)
    await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    assert planner.calls[0]["strategy_hints"] is None


@pytest.mark.asyncio
async def test_tool_recommendations_fetched_per_iteration():
    """get_tool_recommendations is called once per iteration (not once per session)."""
    sess = FakeSession(id="s1", status="running", goal="do something")
    adaptive_strategy = _make_fake_adaptive_strategy()

    orch = _make_orchestrator(
        sess,
        adaptive_strategy=adaptive_strategy,
        critic_evals=["retry", "pass"],  # 2 iterations
        max_iterations=3,
    )
    await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    assert adaptive_strategy.get_tool_recommendations.call_count == 2


@pytest.mark.asyncio
async def test_tool_recommendations_passed_to_planner():
    """Tool recommendations from adaptive_strategy are forwarded into planner.plan()."""
    sess = FakeSession(id="s1", status="running", goal="search for data")
    recs = [{"tool_name": "memory.search", "score": 0.95, "success_rate": 0.9, "total_uses": 10}]
    adaptive_strategy = _make_fake_adaptive_strategy(tool_recs=recs)
    planner = TrackingPlanner()

    orch = _make_orchestrator(sess, planner=planner, adaptive_strategy=adaptive_strategy)
    await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    assert planner.calls[0]["tool_recommendations"] == recs


@pytest.mark.asyncio
async def test_tool_recommendations_empty_when_service_absent():
    """When no adaptive_strategy service, tool_recommendations=[] is passed to planner."""
    sess = FakeSession(id="s1", status="running", goal="find something")
    planner = TrackingPlanner()

    orch = _make_orchestrator(sess, planner=planner, adaptive_strategy=None)
    await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    assert planner.calls[0]["tool_recommendations"] == []


@pytest.mark.asyncio
async def test_ingest_called_after_succeeded_session():
    """ingest_session_outcome is called once after the session succeeds."""
    sess = FakeSession(id="s1", status="running", goal="analyze metrics")
    strategy_learning = _make_fake_strategy_learning()

    orch = _make_orchestrator(
        sess,
        strategy_learning=strategy_learning,
        critic_evals=["pass"],
    )
    status = await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    assert status == "succeeded"
    strategy_learning.ingest_session_outcome.assert_called_once_with(
        org_id="org1", session_id="s1"
    )


@pytest.mark.asyncio
async def test_ingest_called_after_failed_session():
    """ingest_session_outcome is also called when the session exhausts all iterations (failed)."""
    sess = FakeSession(id="s1", status="running", goal="do something hard")
    strategy_learning = _make_fake_strategy_learning()

    orch = _make_orchestrator(
        sess,
        strategy_learning=strategy_learning,
        critic_evals=["retry", "retry", "retry"],
        max_iterations=2,
    )
    status = await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    assert status == "failed"
    strategy_learning.ingest_session_outcome.assert_called_once_with(
        org_id="org1", session_id="s1"
    )


@pytest.mark.asyncio
async def test_ingest_not_called_when_service_absent():
    """No error when strategy_learning is None — ingest is simply skipped."""
    sess = FakeSession(id="s1", status="running", goal="do something")

    orch = _make_orchestrator(sess, strategy_learning=None, critic_evals=["pass"])
    status = await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    assert status == "succeeded"  # still works without service


@pytest.mark.asyncio
async def test_strategy_hints_not_fetched_for_already_terminal_session():
    """If session is already in a terminal state, strategy_hints are never fetched."""
    sess = FakeSession(id="s1", status="succeeded", goal="some goal")
    strategy_learning = _make_fake_strategy_learning()

    orch = _make_orchestrator(sess, strategy_learning=strategy_learning)
    status = await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    assert status == "succeeded"
    strategy_learning.get_strategy_hints.assert_not_called()
    strategy_learning.ingest_session_outcome.assert_not_called()


@pytest.mark.asyncio
async def test_strategy_learning_exception_does_not_block_session():
    """If get_strategy_hints raises, the session still proceeds normally."""
    sess = FakeSession(id="s1", status="running", goal="analyze something")
    strategy_learning = _make_fake_strategy_learning()
    strategy_learning.get_strategy_hints = AsyncMock(side_effect=RuntimeError("service down"))
    planner = TrackingPlanner()

    orch = _make_orchestrator(sess, planner=planner, strategy_learning=strategy_learning)
    status = await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    assert status == "succeeded"
    # strategy_hints falls back to None
    assert planner.calls[0]["strategy_hints"] is None
