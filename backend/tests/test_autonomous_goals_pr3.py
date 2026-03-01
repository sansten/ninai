"""
Tests for PR-3: Autonomous Goals & Intrinsic Motivation

Comprehensive test suite for:
- AutonomousGoal model and CRUD
- KnowledgeGap detection and management
- IntrinsicMotivationService
- API endpoints
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    AutonomousGoal,
    KnowledgeGap,
    AutonomousGoalOutcome,
)
from app.services.intrinsic_motivation_service import IntrinsicMotivationService
from app.schemas.intrinsic_motivation_pr3 import (
    AutonomousGoalCreate,
    KnowledgeGapCreate,
    GoalInitiator,
    KnowledgeGapType,
)


class TestAutonomousGoalModel:
    """Tests for AutonomousGoal model."""
    
    @pytest.mark.asyncio
    async def test_create_autonomous_goal(self, db_session: AsyncSession):
        """Test creating a new autonomous goal."""
        org_id = uuid4()
        goal = AutonomousGoal(
            organization_id=org_id,
            initiator="curiosity",
            title="Investigate customer preferences",
            description="Need to learn more about customer preferences",
            expected_value=0.75,
            urgency=0.6,
            confidence=0.8,
            status="proposed",
        )
        
        db_session.add(goal)
        await db_session.commit()
        
        # Verify goal was created
        stmt = select(AutonomousGoal).where(AutonomousGoal.id == goal.id)
        result = await db_session.execute(stmt)
        retrieved = result.scalar_one_or_none()
        
        assert retrieved is not None
        assert retrieved.initiator == "curiosity"
        assert retrieved.expected_value == 0.75
        assert retrieved.status == "proposed"
    
    @pytest.mark.asyncio
    async def test_goal_status_transitions(self, db_session: AsyncSession):
        """Test goal status transitions."""
        org_id = uuid4()
        goal = AutonomousGoal(
            organization_id=org_id,
            initiator="curiosity",
            title="Test goal",
            description="Test",
            status="proposed",
        )
        
        db_session.add(goal)
        await db_session.commit()
        
        # Activate goal
        goal.status = "active"
        goal.activated_at = datetime.utcnow()
        db_session.add(goal)
        await db_session.commit()
        
        # Verify activation
        stmt = select(AutonomousGoal).where(AutonomousGoal.id == goal.id)
        result = await db_session.execute(stmt)
        updated = result.scalar_one()
        assert updated.status == "active"
        assert updated.activated_at is not None
        
        # Complete goal
        updated.status = "completed"
        updated.completed_at = datetime.utcnow()
        updated.completion_evidence = ["memory_1", "memory_2"]
        db_session.add(updated)
        await db_session.commit()
        
        # Verify completion
        stmt = select(AutonomousGoal).where(AutonomousGoal.id == goal.id)
        result = await db_session.execute(stmt)
        completed = result.scalar_one()
        assert completed.status == "completed"
        assert completed.completed_at is not None
        assert len(completed.completion_evidence) == 2


class TestKnowledgeGapModel:
    """Tests for KnowledgeGap model."""
    
    @pytest.mark.asyncio
    async def test_create_knowledge_gap(self, db_session: AsyncSession):
        """Test creating a knowledge gap."""
        org_id = uuid4()
        gap = KnowledgeGap(
            organization_id=org_id,
            gap_type="low_confidence",
            domain="customer_preferences",
            description="Uncertain about customer spending patterns",
            confidence_in_gap=0.75,
            related_memories=["mem_1", "mem_2"],
        )
        
        db_session.add(gap)
        await db_session.commit()
        
        stmt = select(KnowledgeGap).where(KnowledgeGap.id == gap.id)
        result = await db_session.execute(stmt)
        retrieved = result.scalar_one()
        
        assert retrieved.gap_type == "low_confidence"
        assert retrieved.domain == "customer_preferences"
        assert retrieved.confidence_in_gap == 0.75
    
    @pytest.mark.asyncio
    async def test_resolve_knowledge_gap(self, db_session: AsyncSession):
        """Test marking a knowledge gap as resolved."""
        org_id = uuid4()
        gap = KnowledgeGap(
            organization_id=org_id,
            gap_type="missing_fact",
            domain="system_behavior",
            description="Missing information about system limits",
            resolved_at=None,
        )
        
        db_session.add(gap)
        await db_session.commit()
        
        # Mark as resolved
        gap.resolved_at = datetime.utcnow()
        db_session.add(gap)
        await db_session.commit()
        
        stmt = select(KnowledgeGap).where(KnowledgeGap.id == gap.id)
        result = await db_session.execute(stmt)
        resolved = result.scalar_one()
        
        assert resolved.resolved_at is not None


class TestIntrinsicMotivationService:
    """Tests for IntrinsicMotivationService."""
    
    @pytest.mark.asyncio
    async def test_detect_knowledge_gaps(self, db_session: AsyncSession):
        """Test detecting unresolved knowledge gaps."""
        org_id = uuid4()
        
        # Create a few gaps
        gap1 = KnowledgeGap(
            organization_id=org_id,
            gap_type="low_confidence",
            domain="customer_data",
            description="Uncertain about data accuracy",
            confidence_in_gap=0.6,
        )
        gap2 = KnowledgeGap(
            organization_id=org_id,
            gap_type="missing_fact",
            domain="billing",
            description="Missing billing edge cases",
            confidence_in_gap=0.7,
            resolved_at=datetime.utcnow(),  # Already resolved
        )
        
        db_session.add(gap1)
        db_session.add(gap2)
        await db_session.commit()
        
        # Detect unresolved gaps
        svc = IntrinsicMotivationService(db_session)
        gaps = await svc.detect_knowledge_gaps(org_id)
        
        # Should only return gap1 (unresolved)
        assert len(gaps) == 1
        assert gaps[0]["description"] == "Uncertain about data accuracy"
    
    @pytest.mark.asyncio
    async def test_generate_curiosity_goals(self, db_session: AsyncSession):
        """Test generating curiosity goals from gaps."""
        org_id = uuid4()
        
        gaps = [
            {
                "id": uuid4(),
                "gap_type": "low_confidence",
                "domain": "analytics",
                "description": "Uncertain about trend analysis accuracy",
                "confidence_in_gap": 0.75,
                "related_memories": ["mem_1", "mem_2"],
                "suggested_learning_approach": "experiment",
            }
        ]
        
        svc = IntrinsicMotivationService(db_session)
        goals = await svc.generate_curiosity_goals(org_id, gaps)
        
        assert len(goals) == 1
        assert goals[0]["initiator"] == "curiosity"
        assert goals[0]["title"].startswith("Investigate")
        assert goals[0]["expected_value"] > 0.3
    
    @pytest.mark.asyncio
    async def test_predict_user_needs(self, db_session: AsyncSession):
        """Test predicting user needs."""
        org_id = uuid4()
        
        svc = IntrinsicMotivationService(db_session)
        goals = await svc.predict_user_needs(org_id, lookback_days=90)
        
        # Should generate multiple predictive goals
        assert len(goals) > 0
        assert all(g["initiator"] == "prediction" for g in goals)
        assert all("based on" in g["description"].lower() for g in goals)
    
    @pytest.mark.asyncio
    async def test_detect_self_improvement_needs(self, db_session: AsyncSession):
        """Test detecting self-improvement opportunities."""
        org_id = uuid4()
        
        svc = IntrinsicMotivationService(db_session)
        goals = await svc.detect_self_improvement_needs(org_id, tool_success_threshold=0.8)
        
        # Should generate self-improvement goals for under-performing tools
        assert all(g["initiator"] == "self_improvement" for g in goals)
        assert all("success rate" in g["description"].lower() for g in goals)
    
    @pytest.mark.asyncio
    async def test_estimate_goal_value_curiosity(self, db_session: AsyncSession):
        """Test value estimation for curiosity goals."""
        svc = IntrinsicMotivationService(session=None)
        
        curiosity_goal = {
            "initiator": "curiosity",
            "confidence": 0.8,
            "expected_value": 0.6,
        }
        
        value = await svc.estimate_goal_value(curiosity_goal)
        
        assert 0.0 <= value <= 1.0
        assert value > 0.3  # Should be moderately valuable
    
    @pytest.mark.asyncio
    async def test_estimate_goal_value_self_improvement(self, db_session: AsyncSession):
        """Test value estimation for self-improvement goals."""
        svc = IntrinsicMotivationService(session=None)
        
        self_improvement_goal = {
            "initiator": "self_improvement",
            "expected_value": 0.7,
            "metadata": {
                "current_success_rate": 0.4,
                "target_success_rate": 0.9,
            },
        }
        
        value = await svc.estimate_goal_value(self_improvement_goal)
        
        assert 0.0 <= value <= 1.0
        assert value > 0.2  # Should have meaningful value
    
    @pytest.mark.asyncio
    async def test_record_goal_outcome(self, db_session: AsyncSession):
        """Test recording goal outcome."""
        org_id = uuid4()
        goal = AutonomousGoal(
            organization_id=org_id,
            initiator="curiosity",
            title="Test goal",
            description="For testing",
            status="completed",
        )
        
        db_session.add(goal)
        await db_session.commit()
        
        svc = IntrinsicMotivationService(db_session)
        outcome = await svc.record_goal_outcome(
            org_id=org_id,
            goal_id=goal.id,
            outcome_type="valuable",
            impact_description="Led to improved analysis",
            feedback_from="user",
            was_user_expecting=False,
        )
        
        assert outcome["outcome_type"] == "valuable"
        assert outcome["goal_id"] == goal.id
    
    @pytest.mark.asyncio
    async def test_get_goal_success_metrics(self, db_session: AsyncSession):
        """Test retrieving goal success metrics."""
        org_id = uuid4()
        
        # Create a goal and outcome
        goal = AutonomousGoal(
            organization_id=org_id,
            initiator="curiosity",
            title="Test",
            description="Test",
            status="completed",
        )
        db_session.add(goal)
        await db_session.commit()
        
        outcome = AutonomousGoalOutcome(
            organization_id=org_id,
            goal_id=goal.id,
            outcome_type="valuable",
            was_user_expecting=True,
        )
        db_session.add(outcome)
        await db_session.commit()
        
        svc = IntrinsicMotivationService(db_session)
        metrics = await svc.get_goal_success_metrics(org_id)
        
        assert metrics["total_outcomes"] == 1
        assert metrics["valuable_rate"] == 1.0
        assert metrics["user_expectation_alignment"] == 1.0


class TestAutonomousGoalOutcomes:
    """Tests for goal outcomes."""
    
    @pytest.mark.asyncio
    async def test_create_goal_outcome(self, db_session: AsyncSession):
        """Test creating goal outcome."""
        org_id = uuid4()
        goal_id = uuid4()
        
        outcome = AutonomousGoalOutcome(
            organization_id=org_id,
            goal_id=goal_id,
            outcome_type="valuable",
            impact_description="Helped improve customer interactions",
            feedback_from="metrics",
            was_user_expecting=False,
        )
        
        db_session.add(outcome)
        await db_session.commit()
        
        stmt = select(AutonomousGoalOutcome).where(AutonomousGoalOutcome.id == outcome.id)
        result = await db_session.execute(stmt)
        retrieved = result.scalar_one()
        
        assert retrieved.outcome_type == "valuable"
        assert retrieved.goal_id == goal_id
    
    @pytest.mark.asyncio
    async def test_outcome_types(self, db_session: AsyncSession):
        """Test all outcome types."""
        org_id = uuid4()
        outcome_types = ["valuable", "not_valuable", "premature"]
        
        for outcome_type in outcome_types:
            outcome = AutonomousGoalOutcome(
                organization_id=org_id,
                goal_id=uuid4(),
                outcome_type=outcome_type,
            )
            db_session.add(outcome)
        
        await db_session.commit()
        
        stmt = select(AutonomousGoalOutcome).where(
            AutonomousGoalOutcome.organization_id == org_id
        )
        result = await db_session.execute(stmt)
        outcomes = result.scalars().all()
        
        assert len(outcomes) == 3
        assert {o.outcome_type for o in outcomes} == set(outcome_types)


class TestGoalInitiators:
    """Tests for different goal initiator types."""
    
    @pytest.mark.asyncio
    async def test_all_initiator_types(self, db_session: AsyncSession):
        """Test creating goals with all initiator types."""
        org_id = uuid4()
        initiators = ["curiosity", "prediction", "self_improvement", "priority_rebalance"]
        
        for initiator in initiators:
            goal = AutonomousGoal(
                organization_id=org_id,
                initiator=initiator,
                title=f"Test {initiator} goal",
                description="Test",
                status="proposed",
            )
            db_session.add(goal)
        
        await db_session.commit()
        
        stmt = select(AutonomousGoal).where(AutonomousGoal.organization_id == org_id)
        result = await db_session.execute(stmt)
        goals = result.scalars().all()
        
        assert len(goals) == 4
        assert {g.initiator for g in goals} == set(initiators)


@pytest.fixture
def db_session():
    """Fixture for database session."""
    # This would be provided by your test setup
    # Return actual AsyncSession for testing
    pass
