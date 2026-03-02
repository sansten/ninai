"""
PR-6: Meta-Cognitive Planning tests.

Validates model persistence and core service decision logic.
"""

from datetime import datetime
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import CognitiveStrategy, EpistemicState, StrategySelected
from app.services.meta_cognitive_service import MetaCognitiveService


@pytest.fixture
async def test_org_id() -> str:
    return str(uuid.uuid4())


@pytest.mark.asyncio
async def test_cognitive_strategy_model_creation(db_session: AsyncSession, test_org_id: str):
    strategy = CognitiveStrategy(
        organization_id=test_org_id,
        query_id="query-123",
        complexity_estimated=0.72,
        strategy_selected=StrategySelected.DELIBERATIVE.value,
        retrieval_budget=28,
        reasoning_depth=4,
        verification_required=True,
        confidence_threshold=0.75,
        time_budget_seconds=45,
        expected_answer_quality=0.9,
        started_at=datetime.utcnow(),
    )

    db_session.add(strategy)
    await db_session.commit()

    retrieved = await db_session.get(CognitiveStrategy, strategy.id)
    assert retrieved is not None
    assert retrieved.query_id == "query-123"
    assert retrieved.strategy_selected == StrategySelected.DELIBERATIVE.value
    assert retrieved.verification_required is True


@pytest.mark.asyncio
async def test_epistemic_state_model_creation(db_session: AsyncSession, test_org_id: str):
    state = EpistemicState(
        organization_id=test_org_id,
        timestamp=datetime.utcnow(),
        known_domains=["memory_retrieval", "tool_usage"],
        uncertain_domains=["causal_transfer"],
        unknown_domains=["org_private_constraints"],
        confidence_calibration=0.8,
        surprise_frequency=0.12,
    )

    db_session.add(state)
    await db_session.commit()

    retrieved = await db_session.get(EpistemicState, state.id)
    assert retrieved is not None
    assert "memory_retrieval" in retrieved.known_domains
    assert "causal_transfer" in retrieved.uncertain_domains


@pytest.mark.asyncio
async def test_strategy_selected_enum_values():
    assert StrategySelected.HEURISTIC.value == "heuristic"
    assert StrategySelected.DELIBERATIVE.value == "deliberative"
    assert StrategySelected.MIXED.value == "mixed"
    assert StrategySelected.ESCALATE.value == "escalate"


@pytest.mark.asyncio
async def test_query_complexity_estimation_range():
    svc = MetaCognitiveService(session=None)

    simple = await svc.estimate_query_complexity("What is the status?")
    complex_q = await svc.estimate_query_complexity(
        "Compare tradeoffs, forecast risk, and optimize strategy for an irreversible decision"
    )

    assert 0.0 <= simple <= 1.0
    assert 0.0 <= complex_q <= 1.0
    assert complex_q > simple


@pytest.mark.asyncio
async def test_allocate_retrieval_budget_bounds():
    svc = MetaCognitiveService(session=None)

    low = await svc.allocate_retrieval_budget(0.1, 20)
    high = await svc.allocate_retrieval_budget(0.9, 120)

    assert low >= 6
    assert high >= low


@pytest.mark.asyncio
async def test_allocate_reasoning_depth_increases_with_complexity():
    svc = MetaCognitiveService(session=None)

    d1 = await svc.allocate_reasoning_depth(0.1)
    d2 = await svc.allocate_reasoning_depth(0.5)
    d3 = await svc.allocate_reasoning_depth(0.95)

    assert d1 <= d2 <= d3
    assert 1 <= d1 <= 5
    assert 1 <= d3 <= 5


@pytest.mark.asyncio
async def test_select_strategy_heuristic_for_simple_query():
    svc = MetaCognitiveService(session=None)

    result = await svc.select_strategy(
        query="status?",
        complexity=0.12,
        time_budget_seconds=10,
    )

    assert result["strategy_selected"] == StrategySelected.HEURISTIC.value
    assert result["reasoning_depth"] == 1


@pytest.mark.asyncio
async def test_select_strategy_deliberative_for_complex_query():
    svc = MetaCognitiveService(session=None)

    result = await svc.select_strategy(
        query="optimize multi-step counterfactual plan",
        complexity=0.88,
        time_budget_seconds=90,
    )

    assert result["strategy_selected"] == StrategySelected.DELIBERATIVE.value
    assert result["reasoning_depth"] >= 4
    assert result["verification_required"] is True


@pytest.mark.asyncio
async def test_select_strategy_escalate_for_high_risk():
    svc = MetaCognitiveService(session=None)

    result = await svc.select_strategy(
        query="legal compliance for irreversible contract decision",
        complexity=0.8,
        time_budget_seconds=120,
    )

    assert result["strategy_selected"] == StrategySelected.ESCALATE.value


@pytest.mark.asyncio
async def test_should_verify_when_confidence_low_or_risky():
    svc = MetaCognitiveService(session=None)

    verify_low = await svc.should_verify("normal question", "clear answer", 0.55)
    verify_risky = await svc.should_verify("safety action", "answer", 0.9)
    no_verify = await svc.should_verify("normal question", "definitive", 0.95)

    assert verify_low is True
    assert verify_risky is True
    assert no_verify is False


@pytest.mark.asyncio
async def test_uncertainty_aware_response_adds_prefix_when_needed():
    svc = MetaCognitiveService(session=None)

    plain = await svc.uncertainty_aware_response("Strong answer", 0.9, 0.7)
    hedged = await svc.uncertainty_aware_response("Tentative answer", 0.6, 0.7)

    assert plain == "Strong answer"
    assert "confident" in hedged.lower()


@pytest.mark.asyncio
async def test_learn_strategy_effectiveness_returns_scored_result():
    svc = MetaCognitiveService(session=None)

    result = await svc.learn_strategy_effectiveness(
        strategy_id="strategy-1",
        actual_confidence=0.82,
        expected_answer_quality=0.8,
    )

    assert result["strategy_id"] == "strategy-1"
    assert 0.0 <= result["strategy_effectiveness"] <= 1.0


@pytest.mark.asyncio
async def test_get_confidence_calibration_shape():
    svc = MetaCognitiveService(session=None)
    result = await svc.get_confidence_calibration()

    assert "confidence_calibration" in result
    assert "surprise_frequency" in result
    assert "well_calibrated" in result


@pytest.mark.asyncio
async def test_get_epistemic_state_shape():
    svc = MetaCognitiveService(session=None)
    result = await svc.get_epistemic_state()

    assert "known_domains" in result
    assert "uncertain_domains" in result
    assert "unknown_domains" in result
    assert isinstance(result["known_domains"], list)


@pytest.mark.asyncio
async def test_persist_multiple_epistemic_states(db_session: AsyncSession, test_org_id: str):
    first = EpistemicState(
        organization_id=test_org_id,
        timestamp=datetime.utcnow(),
        known_domains=["domain_a"],
        uncertain_domains=[],
        unknown_domains=[],
    )
    second = EpistemicState(
        organization_id=test_org_id,
        timestamp=datetime.utcnow(),
        known_domains=["domain_b"],
        uncertain_domains=["domain_c"],
        unknown_domains=[],
    )

    db_session.add(first)
    db_session.add(second)
    await db_session.commit()

    result = await db_session.execute(
        select(EpistemicState).where(EpistemicState.organization_id == test_org_id)
    )
    states = result.scalars().all()

    assert len(states) >= 2
