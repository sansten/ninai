"""
PR-5: Temporal Reasoning Engine - Core Model Tests

Test suite for temporal reasoning models:
- Temporal fact validity tracking
- Event sequence detection and patterns
- Trajectory analysis with trend detection
"""

import pytest
from datetime import datetime, timedelta
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import (
    TemporalFact,
    TemporalSequence,
    TemporalTrajectory,
    TemporalChangetype,
    PatternType,
    TrendDirection,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
async def test_org_id():
    """Return test organization ID as valid UUID"""
    return str(uuid.uuid4())


# ============================================================================
# Test 1-3: TemporalFactModel
# ============================================================================

@pytest.mark.asyncio
async def test_temporal_fact_creation(db_session: AsyncSession, test_org_id: str):
    """Test basic TemporalFact model creation"""
    now = datetime.utcnow()
    
    fact = TemporalFact(
        organization_id=test_org_id,
        fact_id="user_sentiment_elevated",
        valid_from=now,
        valid_to=now + timedelta(hours=2),
        confidence_at_time=0.95,
        change_type=TemporalChangetype.ONSET,
    )
    
    db_session.add(fact)
    await db_session.commit()
    
    # Verify creation
    retrieved = await db_session.get(TemporalFact, fact.id)
    assert retrieved is not None
    assert retrieved.fact_id == "user_sentiment_elevated"
    assert retrieved.change_type == TemporalChangetype.ONSET


@pytest.mark.asyncio
async def test_temporal_fact_validity_intervals(db_session: AsyncSession, test_org_id: str):
    """Test TemporalFact validity interval tracking"""
    now = datetime.utcnow()
    start_time = now
    end_time = now + timedelta(hours=1)
    
    fact = TemporalFact(
        organization_id=test_org_id,
        fact_id="symptom_present",
        valid_from=start_time,
        valid_to=end_time,
        confidence_at_time=0.85,
        change_type=TemporalChangetype.STABLE,
    )
    
    db_session.add(fact)
    await db_session.commit()
    
    # Check interval properties
    assert fact.valid_from == start_time
    assert fact.valid_to == end_time
    assert (fact.valid_to - fact.valid_from).seconds == 3600


@pytest.mark.asyncio
async def test_temporal_fact_change_types(db_session: AsyncSession, test_org_id: str):
    """Test all TemporalFact change types"""
    now = datetime.utcnow()
    change_types = [
        TemporalChangetype.ONSET,
        TemporalChangetype.OFFSET,
        TemporalChangetype.STABLE,
        TemporalChangetype.TRANSIENT,
    ]
    
    for idx, change_type in enumerate(change_types):
        fact = TemporalFact(
            organization_id=test_org_id,
            fact_id=f"test_fact_{idx}",
            valid_from=now,
            valid_to=now + timedelta(hours=1),
            confidence_at_time=0.9,
            change_type=change_type,
        )
        db_session.add(fact)
    
    await db_session.commit()
    
    # Verify all change types persisted
    result = await db_session.execute(
        select(TemporalFact).filter(TemporalFact.organization_id == test_org_id)
    )
    facts = result.scalars().all()
    assert len(facts) == 4
    persisted_types = {f.change_type for f in facts}
    assert persisted_types == set(change_types)


# ============================================================================
# Test 4-6: TemporalSequenceModel
# ============================================================================

@pytest.mark.asyncio
async def test_temporal_sequence_creation(db_session: AsyncSession, test_org_id: str):
    """Test TemporalSequence model creation"""
    sequence = TemporalSequence(
        organization_id=test_org_id,
        entities=["event_a", "event_b", "event_c"],
        temporal_gaps=[300, 600],
        pattern_type=PatternType.ESCALATION,
        pattern_strength=0.92,
        predicted_next_event={"entity": "event_d", "confidence": 0.88},
        observation_count=15,
    )
    
    db_session.add(sequence)
    await db_session.commit()
    
    # Verify creation
    retrieved = await db_session.get(TemporalSequence, sequence.id)
    assert retrieved is not None
    assert retrieved.entities == ["event_a", "event_b", "event_c"]
    assert retrieved.temporal_gaps == [300, 600]


@pytest.mark.asyncio
async def test_temporal_sequence_pattern_types(db_session: AsyncSession, test_org_id: str):
    """Test all TemporalSequence pattern types"""
    pattern_types = [
        PatternType.ESCALATION,
        PatternType.RESOLUTION,
        PatternType.OSCILLATION,
        PatternType.TREND,
    ]
    
    for idx, pattern_type in enumerate(pattern_types):
        sequence = TemporalSequence(
            organization_id=test_org_id,
            entities=[f"entity_{idx}_a", f"entity_{idx}_b"],
            temporal_gaps=[100, 200],
            pattern_type=pattern_type,
            pattern_strength=0.85,
            predicted_next_event={"entity": f"entity_{idx}_c", "confidence": 0.80},
            observation_count=10,
        )
        db_session.add(sequence)
    
    await db_session.commit()
    
    # Verify all pattern types
    result = await db_session.execute(
        select(TemporalSequence).filter(TemporalSequence.organization_id == test_org_id)
    )
    sequences = result.scalars().all()
    assert len(sequences) == 4
    persisted_patterns = {s.pattern_type for s in sequences}
    assert persisted_patterns == set(pattern_types)


@pytest.mark.asyncio
async def test_temporal_sequence_gap_analysis(db_session: AsyncSession, test_org_id: str):
    """Test temporal gap analysis in sequences"""
    # Regular intervals
    seq1 = TemporalSequence(
        organization_id=test_org_id,
        entities=["a", "b", "c", "d"],
        temporal_gaps=[300, 300, 300],  # Regular: 5 minutes
        pattern_type=PatternType.TREND,
        pattern_strength=0.95,
        observation_count=20,
    )
    
    # Irregular intervals
    seq2 = TemporalSequence(
        organization_id=test_org_id,
        entities=["x", "y", "z"],
        temporal_gaps=[100, 500],  # Irregular
        pattern_type=PatternType.OSCILLATION,
        pattern_strength=0.65,
        observation_count=8,
    )
    
    db_session.add(seq1)
    db_session.add(seq2)
    await db_session.commit()
    
    # Verify gap consistency metrics
    assert seq1.temporal_gaps == [300, 300, 300]
    assert seq2.temporal_gaps == [100, 500]


# ============================================================================
# Test 7-10: TemporalTrajectoryModel
# ============================================================================

@pytest.mark.asyncio
async def test_temporal_trajectory_creation(db_session: AsyncSession, test_org_id: str):
    """Test TemporalTrajectory model creation"""
    measurements = [
        {"timestamp": datetime.utcnow().isoformat(), "value": 10.5},
        {"timestamp": (datetime.utcnow() + timedelta(hours=1)).isoformat(), "value": 12.3},
        {"timestamp": (datetime.utcnow() + timedelta(hours=2)).isoformat(), "value": 14.1},
    ]
    
    trajectory = TemporalTrajectory(
        organization_id=test_org_id,
        entity_id="memory_strength_metric",
        quantity="memory_activation_level",
        measurements=measurements,
        trend_direction=TrendDirection.INCREASING,
        trend_strength=0.88,
        predicted_future=[15.2, 16.5, 18.1],
        inflection_points=[],
        seasonality=None,
    )
    
    db_session.add(trajectory)
    await db_session.commit()
    
    # Verify creation
    retrieved = await db_session.get(TemporalTrajectory, trajectory.id)
    assert retrieved is not None
    assert retrieved.entity_id == "memory_strength_metric"
    assert retrieved.trend_direction == TrendDirection.INCREASING
    assert len(retrieved.measurements) == 3


@pytest.mark.asyncio
async def test_temporal_trajectory_trend_directions(db_session: AsyncSession, test_org_id: str):
    """Test all trend direction types"""
    trend_types = [
        TrendDirection.INCREASING,
        TrendDirection.DECREASING,
        TrendDirection.STABLE,
        TrendDirection.CYCLIC,
    ]
    
    measurements = [
        {"timestamp": datetime.utcnow().isoformat(), "value": 10.0},
        {"timestamp": (datetime.utcnow() + timedelta(hours=1)).isoformat(), "value": 11.0},
    ]
    
    for idx, trend_type in enumerate(trend_types):
        trajectory = TemporalTrajectory(
            organization_id=test_org_id,
            entity_id=f"metric_{idx}",
            quantity=f"quantity_{idx}",
            measurements=measurements,
            trend_direction=trend_type,
            trend_strength=0.80,
            predicted_future=[12.0, 13.0],
            inflection_points=[],
            seasonality=None,
        )
        db_session.add(trajectory)
    
    await db_session.commit()
    
    # Verify all trend types
    result = await db_session.execute(
        select(TemporalTrajectory).filter(TemporalTrajectory.organization_id == test_org_id)
    )
    trajectories = result.scalars().all()
    assert len(trajectories) == 4
    persisted_trends = {t.trend_direction for t in trajectories}
    assert persisted_trends == set(trend_types)


@pytest.mark.asyncio
async def test_temporal_trajectory_with_inflection_points(db_session: AsyncSession, test_org_id: str):
    """Test trajectory with inflection points (trend changes)"""
    measurements = [
        {"timestamp": (datetime.utcnow() - timedelta(hours=3)).isoformat(), "value": 5.0},
        {"timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(), "value": 8.0},
        {"timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(), "value": 10.0},
        {"timestamp": datetime.utcnow().isoformat(), "value": 9.5},
    ]
    
    inflection_points = [
        {
            "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            "severity": 0.75,
            "trend_change": "increasing->decreasing",
        }
    ]
    
    trajectory = TemporalTrajectory(
        organization_id=test_org_id,
        entity_id="sentiment_score",
        quantity="sentiment",
        measurements=measurements,
        trend_direction=TrendDirection.STABLE,
        trend_strength=0.45,
        predicted_future=[9.0, 8.5, 8.2],
        inflection_points=inflection_points,
        seasonality=None,
    )
    
    db_session.add(trajectory)
    await db_session.commit()
    
    # Verify inflection points
    retrieved = await db_session.get(TemporalTrajectory, trajectory.id)
    assert len(retrieved.inflection_points) == 1
    assert retrieved.inflection_points[0]["severity"] == 0.75


@pytest.mark.asyncio
async def test_temporal_trajectory_with_seasonality(db_session: AsyncSession, test_org_id: str):
    """Test trajectory seasonality metadata"""
    seasonality_info = {
        "period_hours": 24,
        "strength": 0.72,
        "peak_hour": 14,
        "trough_hour": 3,
    }
    
    measurements = [
        {"timestamp": (datetime.utcnow() - timedelta(days=1, hours=2)).isoformat(), "value": 15.0},
        {"timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(), "value": 25.0},
        {"timestamp": datetime.utcnow().isoformat(), "value": 18.0},
    ]
    
    trajectory = TemporalTrajectory(
        organization_id=test_org_id,
        entity_id="daily_activity_level",
        quantity="activity",
        measurements=measurements,
        trend_direction=TrendDirection.CYCLIC,
        trend_strength=0.72,
        predicted_future=[20.0, 24.0, 19.0],
        inflection_points=[],
        seasonality=seasonality_info,
    )
    
    db_session.add(trajectory)
    await db_session.commit()
    
    # Verify seasonality
    retrieved = await db_session.get(TemporalTrajectory, trajectory.id)
    assert retrieved.seasonality["period_hours"] == 24
    assert retrieved.seasonality["peak_hour"] == 14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
