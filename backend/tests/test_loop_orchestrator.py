from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from app.services.cognitive_loop.orchestrator import LoopOrchestrator, OrchestratorConfig
from app.services.cognitive_tooling.policy_guard import ToolContext
from app.schemas.cognitive import PlannerOutput, ExecutorOutput, CriticOutput
from app.services.simulation_service import SimulationService


@dataclass
class FakeSession:
    id: str
    status: str
    goal: str


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

    async def finalize_iteration(
        self,
        *,
        iteration,
        plan_json: dict,
        execution_json: dict,
        critique_json: dict,
        evaluation: str,
        metrics: dict,
        finished_at: datetime | None = None,
    ) -> None:
        iteration.critique_json = critique_json
        iteration.evaluation = evaluation


class FakeEvidence:
    def __init__(self):
        self.last_kwargs = None

    async def retrieve_evidence(self, **kwargs):
        self.last_kwargs = dict(kwargs)
        return [{"id": "m1", "summary": "s"}]


class FakePlanner:
    async def plan(self, **kwargs):
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
        # Deterministic id for assertions if needed.
        return FakeSimRow(id="sim-1")


@pytest.mark.asyncio
async def test_orchestrator_stops_on_pass() -> None:
    repo = FakeRepo(FakeSession(id="s1", status="running", goal="g"))
    orch = LoopOrchestrator(
        repo=repo,
        evidence=FakeEvidence(),
        planner=FakePlanner(),
        simulator=SimulationService(),
        simulation_reports=FakeSimulationReports(),
        executor=FakeExecutor(),
        critic=FakeCritic(["retry", "pass"]),
        available_tools=[],
        config=OrchestratorConfig(max_iterations=3),
    )

    status = await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="o"))
    assert status == "succeeded"
    assert repo.session.status == "succeeded"


@pytest.mark.asyncio
async def test_orchestrator_respects_max_iterations() -> None:
    repo = FakeRepo(FakeSession(id="s1", status="running", goal="g"))
    orch = LoopOrchestrator(
        repo=repo,
        evidence=FakeEvidence(),
        planner=FakePlanner(),
        simulator=SimulationService(),
        simulation_reports=FakeSimulationReports(),
        executor=FakeExecutor(),
        critic=FakeCritic(["retry", "retry", "retry"]),
        available_tools=[],
        config=OrchestratorConfig(max_iterations=2),
    )

    status = await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="o"))
    assert status == "failed"
    assert repo.session.status == "failed"


@pytest.mark.asyncio
async def test_orchestrator_scales_evidence_by_self_model_multiplier() -> None:
    repo = FakeRepo(FakeSession(id="s1", status="running", goal="g"))
    evidence = FakeEvidence()
    orch = LoopOrchestrator(
        repo=repo,
        evidence=evidence,
        planner=FakePlanner(),
        simulator=SimulationService(),
        simulation_reports=FakeSimulationReports(),
        executor=FakeExecutor(),
        critic=FakeCritic(["pass"]),
        available_tools=[],
        self_model_summary={"recommended_evidence_multiplier": 2},
        config=OrchestratorConfig(max_iterations=1),
    )

    status = await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="o"))
    assert status == "succeeded"
    assert isinstance(evidence.last_kwargs, dict)
    assert evidence.last_kwargs.get("limit") == 20
