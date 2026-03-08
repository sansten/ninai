"""Tests for CognitiveContextAggregator and its orchestrator wiring."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.cognitive_context_aggregator import CognitiveContextAggregator


# ---------------------------------------------------------------------------
# Fake services
# ---------------------------------------------------------------------------

def _make_meta_cognitive(
    epistemic_state: dict | None = None,
    complexity: float = 0.3,
    raises: bool = False,
):
    svc = MagicMock()
    if raises:
        svc.get_epistemic_state = AsyncMock(side_effect=RuntimeError("meta down"))
        svc.estimate_query_complexity = AsyncMock(side_effect=RuntimeError("meta down"))
    else:
        svc.get_epistemic_state = AsyncMock(return_value=epistemic_state or {
            "known_domains": ["finance", "hr"],
            "uncertain_domains": ["legal"],
            "unknown_domains": [],
            "confidence_calibration": 0.82,
        })
        svc.estimate_query_complexity = AsyncMock(return_value=complexity)
    return svc


def _make_intrinsic_motivation(gaps: list | None = None, raises: bool = False):
    svc = MagicMock()
    if raises:
        svc.detect_knowledge_gaps = AsyncMock(side_effect=RuntimeError("motivation down"))
    else:
        svc.detect_knowledge_gaps = AsyncMock(return_value=gaps or [])
    return svc


# ---------------------------------------------------------------------------
# aggregate() — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_aggregate_returns_epistemic_state():
    """epistemic_state from MetaCognitiveService appears in the aggregated context."""
    meta = _make_meta_cognitive(
        epistemic_state={
            "known_domains": ["finance"],
            "uncertain_domains": ["legal"],
            "unknown_domains": ["quantum"],
            "confidence_calibration": 0.75,
        },
        complexity=0.4,
    )
    agg = CognitiveContextAggregator(meta_cognitive=meta)

    ctx = await agg.aggregate(org_id="org1", goal="analyze the legal exposure")

    assert "epistemic_state" in ctx
    es = ctx["epistemic_state"]
    assert es["known_domains"] == ["finance"]
    assert es["uncertain_domains"] == ["legal"]
    assert es["unknown_domains"] == ["quantum"]
    assert es["confidence_calibration"] == pytest.approx(0.75)
    assert es["query_complexity"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_aggregate_returns_top_3_knowledge_gaps():
    """Top-3 knowledge gaps by confidence_in_gap are returned."""
    gaps = [
        {"gap_type": "missing_fact", "domain": "legal", "description": "missing contract terms",
         "confidence_in_gap": 0.9},
        {"gap_type": "low_confidence", "domain": "hr", "description": "uncertain headcount",
         "confidence_in_gap": 0.7},
        {"gap_type": "outdated", "domain": "finance", "description": "stale Q3 revenue",
         "confidence_in_gap": 0.8},
        {"gap_type": "contradiction", "domain": "ops", "description": "conflicting metrics",
         "confidence_in_gap": 0.5},
    ]
    motivation = _make_intrinsic_motivation(gaps=gaps)
    agg = CognitiveContextAggregator(intrinsic_motivation=motivation)

    ctx = await agg.aggregate(org_id="org1", goal="something")

    assert "knowledge_gaps" in ctx
    returned_gaps = ctx["knowledge_gaps"]
    assert len(returned_gaps) == 3
    # Sorted by confidence desc: 0.9, 0.8, 0.7
    assert returned_gaps[0]["confidence"] == pytest.approx(0.9)
    assert returned_gaps[1]["confidence"] == pytest.approx(0.8)
    assert returned_gaps[2]["confidence"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_aggregate_knowledge_gaps_fields():
    """Returned gap dicts have the correct compact fields."""
    gaps = [
        {"gap_type": "missing_fact", "domain": "legal", "description": "key fact missing",
         "confidence_in_gap": 0.85, "related_memories": ["m1"]},
    ]
    agg = CognitiveContextAggregator(
        intrinsic_motivation=_make_intrinsic_motivation(gaps=gaps)
    )
    ctx = await agg.aggregate(org_id="org1", goal="x")

    gap = ctx["knowledge_gaps"][0]
    assert set(gap.keys()) == {"gap_type", "domain", "description", "confidence"}
    # related_memories stripped (not in compact format)


@pytest.mark.asyncio
async def test_aggregate_omits_knowledge_gaps_when_empty():
    """knowledge_gaps key absent when no gaps returned."""
    agg = CognitiveContextAggregator(
        intrinsic_motivation=_make_intrinsic_motivation(gaps=[])
    )
    ctx = await agg.aggregate(org_id="org1", goal="something")

    assert "knowledge_gaps" not in ctx


@pytest.mark.asyncio
async def test_aggregate_returns_empty_when_no_services():
    """Returns {} when neither service is wired."""
    agg = CognitiveContextAggregator()
    ctx = await agg.aggregate(org_id="org1", goal="anything")
    assert ctx == {}


# ---------------------------------------------------------------------------
# Fault isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_meta_cognitive_failure_does_not_block_gaps():
    """MetaCognitive failure → no epistemic_state, but gaps still returned."""
    gaps = [{"gap_type": "missing_fact", "domain": "hr", "description": "d",
              "confidence_in_gap": 0.6}]
    agg = CognitiveContextAggregator(
        meta_cognitive=_make_meta_cognitive(raises=True),
        intrinsic_motivation=_make_intrinsic_motivation(gaps=gaps),
    )
    ctx = await agg.aggregate(org_id="org1", goal="find something")

    assert "epistemic_state" not in ctx
    assert "knowledge_gaps" in ctx
    assert len(ctx["knowledge_gaps"]) == 1


@pytest.mark.asyncio
async def test_intrinsic_motivation_failure_does_not_block_epistemic_state():
    """IntrinsicMotivation failure → no knowledge_gaps, but epistemic_state still returned."""
    agg = CognitiveContextAggregator(
        meta_cognitive=_make_meta_cognitive(),
        intrinsic_motivation=_make_intrinsic_motivation(raises=True),
    )
    ctx = await agg.aggregate(org_id="org1", goal="analyze something")

    assert "epistemic_state" in ctx
    assert "knowledge_gaps" not in ctx


@pytest.mark.asyncio
async def test_all_services_fail_returns_empty_dict():
    """Both services failing returns {} without raising."""
    agg = CognitiveContextAggregator(
        meta_cognitive=_make_meta_cognitive(raises=True),
        intrinsic_motivation=_make_intrinsic_motivation(raises=True),
    )
    ctx = await agg.aggregate(org_id="org1", goal="x")
    assert ctx == {}


# ---------------------------------------------------------------------------
# Orchestrator wiring tests (via existing test doubles)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_passes_cognitive_context_to_planner():
    """context_aggregator result is forwarded into planner.plan() as cognitive_context."""
    from dataclasses import dataclass, field as dc_field

    from app.schemas.cognitive import CriticOutput, ExecutorOutput, PlannerOutput
    from app.services.cognitive_loop.orchestrator import LoopOrchestrator, OrchestratorConfig
    from app.services.cognitive_tooling.policy_guard import ToolContext
    from app.services.simulation_service import SimulationService

    @dataclass
    class FakeSession:
        id: str
        status: str
        goal: str
        context_snapshot: dict = dc_field(default_factory=dict)

    @dataclass
    class FakeIteration:
        id: str
        session_id: str
        iteration_num: int
        critique_json: dict = dc_field(default_factory=dict)
        evaluation: str = "retry"

    class FakeRepo:
        def __init__(self, sess):
            self.session = sess
            self.iterations = {}

        async def get_session(self, session_id):
            return self.session

        async def save_session_status(self, sess, status):
            sess.status = status

        async def get_iteration(self, *, session_id, iteration_num):
            return self.iterations.get(iteration_num)

        async def create_iteration(self, *, session_id, iteration_num):
            it = FakeIteration(id=f"it-{iteration_num}", session_id=session_id,
                               iteration_num=iteration_num)
            self.iterations[iteration_num] = it
            return it

        async def finalize_iteration(self, *, iteration, plan_json, execution_json,
                                      critique_json, evaluation, metrics, finished_at=None):
            iteration.evaluation = evaluation
            iteration.critique_json = critique_json

    class TrackingPlanner:
        def __init__(self):
            self.calls = []

        async def plan(self, **kwargs):
            self.calls.append(dict(kwargs))
            return PlannerOutput(
                objective="g", assumptions=[], constraints=[], required_tools=[],
                steps=[{"step_id": "S1", "action": "a", "tool": None, "tool_input_hint": {},
                        "expected_output": "x", "success_criteria": ["ok"], "risk_notes": []}],
                stop_conditions=[], confidence=0.5,
            )

    class FakeEvidence:
        async def retrieve_evidence(self, **kwargs):
            return []

    @dataclass
    class FakeSimRow:
        id: str

    class FakeSimReports:
        async def create(self, **kwargs):
            return FakeSimRow(id="sim-1")

    expected_ctx = {"epistemic_state": {"known_domains": ["finance"], "uncertain_domains": [],
                                         "unknown_domains": [], "confidence_calibration": 0.8,
                                         "query_complexity": 0.3}}
    fake_aggregator = MagicMock()
    fake_aggregator.aggregate = AsyncMock(return_value=expected_ctx)

    sess = FakeSession(id="s1", status="running", goal="analyze revenue trends")
    planner = TrackingPlanner()

    orch = LoopOrchestrator(
        repo=FakeRepo(sess),
        evidence=FakeEvidence(),
        planner=planner,
        simulator=SimulationService(),
        simulation_reports=FakeSimReports(),
        executor=MagicMock(execute=AsyncMock(return_value=ExecutorOutput(step_results=[], overall_status="success", errors=[]))),
        critic=MagicMock(critique=AsyncMock(return_value=CriticOutput(evaluation="pass", strengths=[], issues=[], followup_questions=[], confidence=0.7))),
        available_tools=["memory.search"],
        context_aggregator=fake_aggregator,
        config=OrchestratorConfig(max_iterations=1),
    )

    await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    fake_aggregator.aggregate.assert_called_once_with(org_id="org1", goal="analyze revenue trends")
    assert planner.calls[0]["cognitive_context"] == expected_ctx


@pytest.mark.asyncio
async def test_orchestrator_cognitive_context_none_when_no_aggregator():
    """When context_aggregator is None, cognitive_context=None is passed to planner."""
    from dataclasses import dataclass, field as dc_field

    from app.schemas.cognitive import CriticOutput, ExecutorOutput, PlannerOutput
    from app.services.cognitive_loop.orchestrator import LoopOrchestrator, OrchestratorConfig
    from app.services.cognitive_tooling.policy_guard import ToolContext
    from app.services.simulation_service import SimulationService

    @dataclass
    class FakeSession:
        id: str
        status: str
        goal: str
        context_snapshot: dict = dc_field(default_factory=dict)

    @dataclass
    class FakeIteration:
        id: str
        session_id: str
        iteration_num: int
        critique_json: dict = dc_field(default_factory=dict)
        evaluation: str = "retry"

    class FakeRepo:
        def __init__(self, sess):
            self.session = sess
            self.iterations = {}

        async def get_session(self, sid):
            return self.session

        async def save_session_status(self, sess, status):
            sess.status = status

        async def get_iteration(self, *, session_id, iteration_num):
            return self.iterations.get(iteration_num)

        async def create_iteration(self, *, session_id, iteration_num):
            it = FakeIteration(id=f"it-{iteration_num}", session_id=session_id,
                               iteration_num=iteration_num)
            self.iterations[iteration_num] = it
            return it

        async def finalize_iteration(self, **kwargs):
            kwargs["iteration"].evaluation = kwargs["evaluation"]

    class TrackingPlanner:
        def __init__(self):
            self.calls = []

        async def plan(self, **kwargs):
            self.calls.append(dict(kwargs))
            return PlannerOutput(
                objective="g", assumptions=[], constraints=[], required_tools=[],
                steps=[{"step_id": "S1", "action": "a", "tool": None, "tool_input_hint": {},
                        "expected_output": "x", "success_criteria": ["ok"], "risk_notes": []}],
                stop_conditions=[], confidence=0.5,
            )

    @dataclass
    class FakeSimRow:
        id: str

    sess = FakeSession(id="s1", status="running", goal="do something")
    planner = TrackingPlanner()

    orch = LoopOrchestrator(
        repo=FakeRepo(sess),
        evidence=MagicMock(retrieve_evidence=AsyncMock(return_value=[])),
        planner=planner,
        simulator=SimulationService(),
        simulation_reports=MagicMock(create=AsyncMock(return_value=FakeSimRow(id="r1"))),
        executor=MagicMock(execute=AsyncMock(return_value=ExecutorOutput(step_results=[], overall_status="success", errors=[]))),
        critic=MagicMock(critique=AsyncMock(return_value=CriticOutput(evaluation="pass", strengths=[], issues=[], followup_questions=[], confidence=0.7))),
        available_tools=[],
        context_aggregator=None,
        config=OrchestratorConfig(max_iterations=1),
    )

    await orch.run(session_id="s1", tool_ctx=ToolContext(user_id="u", org_id="org1"))

    assert planner.calls[0]["cognitive_context"] is None
