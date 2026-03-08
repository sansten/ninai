"""Tests for StrategyLearningService and _classify_goal_type."""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.strategy_learning_service import (
    StrategyLearningService,
    _classify_goal_type,
    _EMA_ALPHA,
    _MIN_SAMPLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeSession:
    id: str
    status: str
    goal: str
    organization_id: str
    context_snapshot: dict = field(default_factory=dict)


@dataclass
class FakeIteration:
    id: str
    session_id: str
    iteration_num: int
    plan_json: dict = field(default_factory=dict)
    critique_json: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


@dataclass
class FakeEntry:
    """Minimal in-memory StrategyLibraryEntry stand-in."""
    organization_id: str
    goal_type: str
    domain: str
    tool_sequence: list
    evidence_pattern: list
    avg_plan_confidence: float
    avg_iterations: float
    success_rate: float
    sample_count: int


def _make_result(*, scalar=None, all_results=None):
    """Build a MagicMock mimicking AsyncSession.execute() return value."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar
    result.scalars.return_value.all.return_value = all_results or []
    return result


def _make_db(execute_side_effects: list):
    """Return an AsyncMock DB session with sequential execute() results."""
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=execute_side_effects)
    db.flush = AsyncMock()
    db.add = MagicMock()
    return db


# ---------------------------------------------------------------------------
# _classify_goal_type
# ---------------------------------------------------------------------------

def test_classify_information_retrieval():
    assert _classify_goal_type("search for recent papers") == "information_retrieval"
    assert _classify_goal_type("find the user record") == "information_retrieval"
    assert _classify_goal_type("retrieve the document") == "information_retrieval"


def test_classify_analysis():
    assert _classify_goal_type("analyze the performance trend") == "analysis"
    assert _classify_goal_type("explain why the latency spikes") == "analysis"
    assert _classify_goal_type("summarize the meeting") == "analysis"


def test_classify_generation():
    assert _classify_goal_type("create a new onboarding flow") == "generation"
    assert _classify_goal_type("generate a weekly report") == "generation"
    assert _classify_goal_type("write a blog post") == "generation"


def test_classify_evaluation():
    assert _classify_goal_type("compare options A and B") == "evaluation"
    assert _classify_goal_type("evaluate the vendor proposals") == "evaluation"
    assert _classify_goal_type("rank the candidates") == "evaluation"


def test_classify_planning():
    assert _classify_goal_type("plan the sprint") == "planning"
    assert _classify_goal_type("how should we approach this?") == "planning"
    assert _classify_goal_type("develop a roadmap for Q3") == "planning"


def test_classify_self_improvement():
    assert _classify_goal_type("improve the algorithm accuracy") == "self_improvement"
    assert _classify_goal_type("optimize the query") == "self_improvement"
    assert _classify_goal_type("debug the connection issue") == "self_improvement"


def test_classify_general_fallback():
    assert _classify_goal_type("hello there") == "general"
    assert _classify_goal_type("") == "general"


def test_classify_case_insensitive():
    assert _classify_goal_type("FIND the answer") == "information_retrieval"
    assert _classify_goal_type("ANALYZE this data") == "analysis"
    assert _classify_goal_type("GENERATE output") == "generation"


# ---------------------------------------------------------------------------
# StrategyLearningService.get_strategy_hints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_strategy_hints_returns_top_strategies():
    """Returns formatted dict when qualifying entries exist."""
    entries = [
        FakeEntry(
            organization_id="org1",
            goal_type="analysis",
            domain="finance",
            tool_sequence=["memory.search", "memory.get"],
            evidence_pattern=["e1"],
            avg_plan_confidence=0.8,
            avg_iterations=2.0,
            success_rate=0.9,
            sample_count=5,
        ),
        FakeEntry(
            organization_id="org1",
            goal_type="analysis",
            domain="finance",
            tool_sequence=["memory.search"],
            evidence_pattern=[],
            avg_plan_confidence=0.7,
            avg_iterations=3.0,
            success_rate=0.75,
            sample_count=4,
        ),
    ]
    db = _make_db([_make_result(all_results=entries)])
    svc = StrategyLearningService(session=db)

    result = await svc.get_strategy_hints(org_id="org1", goal_type="analysis", domain="finance")

    assert result is not None
    assert result["goal_type"] == "analysis"
    assert result["domain"] == "finance"
    assert len(result["top_strategies"]) == 2
    assert result["top_strategies"][0]["success_rate"] == 0.9
    assert result["top_strategies"][0]["tool_sequence"] == ["memory.search", "memory.get"]
    assert "note" in result


@pytest.mark.asyncio
async def test_get_strategy_hints_returns_none_when_no_entries():
    """Returns None when no qualifying entries in DB."""
    db = _make_db([_make_result(all_results=[])])
    svc = StrategyLearningService(session=db)

    result = await svc.get_strategy_hints(org_id="org1", goal_type="planning", domain="general")

    assert result is None


# ---------------------------------------------------------------------------
# StrategyLearningService.ingest_session_outcome — new entry creation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_creates_new_entry_for_succeeded_session():
    """Ingesting a succeeded session with no prior entry creates a new StrategyLibraryEntry."""
    sess = FakeSession(
        id="s1",
        status="succeeded",
        goal="analyze the performance metrics",
        organization_id="org1",
        context_snapshot={"domain": "ops"},
    )
    iteration = FakeIteration(
        id="it1",
        session_id="s1",
        iteration_num=1,
        plan_json={"steps": [{"tool": "memory.search"}, {"tool": "memory.get"}]},
        critique_json={"confidence": 0.85},
        metrics={"evidence_memory_ids": ["m1", "m2"]},
    )

    db = _make_db([
        _make_result(scalar=sess),           # session lookup
        _make_result(all_results=[iteration]),  # iterations lookup
        _make_result(scalar=None),           # no existing entry
    ])
    svc = StrategyLearningService(session=db)

    await svc.ingest_session_outcome(org_id="org1", session_id="s1")

    assert db.add.called
    entry = db.add.call_args[0][0]
    assert entry.organization_id == "org1"
    assert entry.goal_type == "analysis"  # "analyze" maps to analysis
    assert entry.domain == "ops"
    assert entry.tool_sequence == ["memory.search", "memory.get"]
    assert entry.success_rate == 1.0  # succeeded
    assert entry.sample_count == 1
    assert db.flush.called


@pytest.mark.asyncio
async def test_ingest_creates_new_entry_for_failed_session():
    """A failed session produces success_rate=0.0 in the new entry."""
    sess = FakeSession(
        id="s2",
        status="failed",
        goal="generate a report",
        organization_id="org1",
        context_snapshot={},
    )
    iteration = FakeIteration(
        id="it1",
        session_id="s2",
        iteration_num=1,
        plan_json={"steps": [{"tool": "memory.search"}]},
        critique_json={"confidence": 0.4},
        metrics={},
    )

    db = _make_db([
        _make_result(scalar=sess),
        _make_result(all_results=[iteration]),
        _make_result(scalar=None),
    ])
    svc = StrategyLearningService(session=db)

    await svc.ingest_session_outcome(org_id="org1", session_id="s2")

    entry = db.add.call_args[0][0]
    assert entry.success_rate == 0.0
    assert entry.goal_type == "generation"
    assert entry.domain == "general"  # fallback


# ---------------------------------------------------------------------------
# StrategyLearningService.ingest_session_outcome — EMA update
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_ema_update_on_existing_entry():
    """EMA update: new_rate = (1 - alpha) * prev_rate + alpha * outcome."""
    sess = FakeSession(
        id="s3",
        status="succeeded",
        goal="search for records",
        organization_id="org1",
        context_snapshot={"domain": "hr"},
    )
    iteration = FakeIteration(
        id="it1",
        session_id="s3",
        iteration_num=1,
        plan_json={"steps": [{"tool": "memory.search"}]},
        critique_json={"confidence": 0.7},
        metrics={},
    )
    existing_entry = FakeEntry(
        organization_id="org1",
        goal_type="information_retrieval",
        domain="hr",
        tool_sequence=["memory.get"],
        evidence_pattern=[],
        avg_plan_confidence=0.6,
        avg_iterations=2.0,
        success_rate=1.0,  # prior rate was 1.0
        sample_count=3,
    )

    db = _make_db([
        _make_result(scalar=sess),
        _make_result(all_results=[iteration]),
        _make_result(scalar=existing_entry),
    ])
    svc = StrategyLearningService(session=db)

    await svc.ingest_session_outcome(org_id="org1", session_id="s3")

    # EMA: (1 - 0.25) * 1.0 + 0.25 * 1.0 = 1.0 (succeeded → outcome=1.0)
    assert existing_entry.success_rate == pytest.approx(1.0)
    assert existing_entry.sample_count == 4
    assert db.flush.called


@pytest.mark.asyncio
async def test_ingest_ema_decreases_on_failure():
    """EMA decreases when a previously successful entry gets a failed session."""
    sess = FakeSession(
        id="s4",
        status="failed",
        goal="plan the sprint",
        organization_id="org1",
        context_snapshot={"domain": "eng"},
    )
    iteration = FakeIteration(
        id="it1",
        session_id="s4",
        iteration_num=1,
        plan_json={"steps": [{"tool": "memory.search"}]},
        critique_json={"confidence": 0.5},
        metrics={},
    )
    existing_entry = FakeEntry(
        organization_id="org1",
        goal_type="planning",
        domain="eng",
        tool_sequence=["memory.search"],
        evidence_pattern=[],
        avg_plan_confidence=0.8,
        avg_iterations=2.0,
        success_rate=1.0,
        sample_count=5,
    )

    db = _make_db([
        _make_result(scalar=sess),
        _make_result(all_results=[iteration]),
        _make_result(scalar=existing_entry),
    ])
    svc = StrategyLearningService(session=db)

    await svc.ingest_session_outcome(org_id="org1", session_id="s4")

    # EMA: (1 - 0.25) * 1.0 + 0.25 * 0.0 = 0.75
    assert existing_entry.success_rate == pytest.approx(0.75)
    assert existing_entry.sample_count == 6


# ---------------------------------------------------------------------------
# StrategyLearningService.ingest_session_outcome — skip conditions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_skips_running_session():
    """Sessions with status='running' are not ingested."""
    sess = FakeSession(
        id="s5", status="running", goal="do something",
        organization_id="org1",
    )

    db = _make_db([
        _make_result(scalar=sess),
        # No further calls should happen
    ])
    svc = StrategyLearningService(session=db)

    await svc.ingest_session_outcome(org_id="org1", session_id="s5")

    assert not db.add.called
    assert not db.flush.called


@pytest.mark.asyncio
async def test_ingest_skips_missing_session():
    """If session_id not found, no entry is created."""
    db = _make_db([_make_result(scalar=None)])
    svc = StrategyLearningService(session=db)

    await svc.ingest_session_outcome(org_id="org1", session_id="missing")

    assert not db.add.called


@pytest.mark.asyncio
async def test_ingest_skips_session_with_no_iterations():
    """Sessions with no iterations are silently skipped."""
    sess = FakeSession(
        id="s6", status="succeeded", goal="do something",
        organization_id="org1",
    )

    db = _make_db([
        _make_result(scalar=sess),
        _make_result(all_results=[]),  # no iterations
    ])
    svc = StrategyLearningService(session=db)

    await svc.ingest_session_outcome(org_id="org1", session_id="s6")

    assert not db.add.called


@pytest.mark.asyncio
async def test_ingest_exception_is_caught():
    """Exceptions inside _do_ingest are caught by the public wrapper (fire-and-forget safe)."""
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=RuntimeError("DB down"))
    svc = StrategyLearningService(session=db)

    # Should not raise
    await svc.ingest_session_outcome(org_id="org1", session_id="s7")
