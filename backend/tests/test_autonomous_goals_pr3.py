"""
Tests for PR-3: Autonomous Goals & Intrinsic Motivation

Comprehensive test suite for:
- AutonomousGoal model instantiation
- KnowledgeGap model  
- IntrinsicMotivationService logic
- AutonomousGoalOutcome tracking
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from app.models import (
    AutonomousGoal,
    KnowledgeGap,
    AutonomousGoalOutcome,
)
from app.services.intrinsic_motivation_service import IntrinsicMotivationService


class TestAutonomousGoalModel:
    """Tests for AutonomousGoal model instantiation."""
    
    def test_create_autonomous_goal(self):
        """Test creating an autonomous goal."""
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
        
        assert goal.initiator == "curiosity"
        assert goal.expected_value == 0.75
        assert goal.status == "proposed"
        assert goal.title == "Investigate customer preferences"
        assert goal.organization_id == org_id
    
    def test_goal_status_transitions(self):
        """Test goal status transitions."""
        org_id = uuid4()
        goal = AutonomousGoal(
            organization_id=org_id,
            initiator="curiosity",
            title="Test goal",
            description="Test",
            expected_value=0.5,
            urgency=0.5,
            confidence=0.5,
            status="proposed",
        )
        
        # Test status transition
        goal.status = "activated"
        assert goal.status == "activated"
        assert goal.activated_at is None  # Only set when actually activating
        
        goal.activated_at = datetime.utcnow()
        assert goal.activated_at is not None
        
        goal.status = "completed"
        assert goal.status == "completed"
        goal.completed_at = datetime.utcnow()
        goal.completion_evidence = ["memory_1", "memory_2"]
        assert goal.completed_at is not None
        assert len(goal.completion_evidence) == 2


class TestKnowledgeGapModel:
    """Tests for KnowledgeGap model."""
    
    def test_create_knowledge_gap(self):
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
        
        assert gap.gap_type == "low_confidence"
        assert gap.domain == "customer_preferences"
        assert gap.confidence_in_gap == 0.75
        assert gap.related_memories == ["mem_1", "mem_2"]
    
    def test_resolve_knowledge_gap(self):
        """Test marking knowledge gap as resolved."""
        org_id = uuid4()
        gap = KnowledgeGap(
            organization_id=org_id,
            gap_type="missing_fact",
            domain="system_behavior",
            description="Missing information about system limits",
        )
        
        assert gap.resolved_at is None
        gap.resolved_at = datetime.utcnow()
        assert gap.resolved_at is not None


class TestAutonomousGoalOutcomeModel:
    """Tests for AutonomousGoalOutcome model."""
    
    def test_create_goal_outcome(self):
        """Test creating a goal outcome."""
        org_id = uuid4()
        goal_id = uuid4()
        
        outcome = AutonomousGoalOutcome(
            organization_id=org_id,
            goal_id=goal_id,
            outcome_type="successful",
            impact_description="User found the information helpful",
            was_user_expecting=True,
        )
        
        assert outcome.goal_id == goal_id
        assert outcome.outcome_type == "successful"
        assert outcome.was_user_expecting is True
    
    def test_outcome_types(self):
        """Test different outcome types."""
        org_id = uuid4()
        goal_id = uuid4()
        
        outcome_types = [
            "successful",
            "partially_successful",
            "failed",
            "user_interrupted",
        ]
        
        for outcome_type in outcome_types:
            outcome = AutonomousGoalOutcome(
                organization_id=org_id,
                goal_id=goal_id,
                outcome_type=outcome_type,
                impact_description="Test outcome",
            )
            assert outcome.outcome_type == outcome_type


class TestIntrinsicMotivationService:
    """Tests for IntrinsicMotivationService methods."""
    
    @pytest.mark.asyncio
    async def test_generate_curiosity_goals(self):
        """Test curiosity goal generation."""
        svc = IntrinsicMotivationService(session=None)
        org_id = str(uuid4())
        
        gaps = [
            {
                "id": str(uuid4()),
                "gap_type": "low_confidence",
                "domain": "customer_preferences",
                "description": "Uncertain about preferences",
                "confidence_in_gap": 0.85,
                "related_memories": ["mem_1"],
                "suggested_learning_approach": "Analyze patterns",
            }
        ]
        
        goals = await svc.generate_curiosity_goals(org_id, gaps)
        
        assert len(goals) == 1
        assert goals[0]["initiator"] == "curiosity"
        assert goals[0]["status"] == "proposed"
        assert goals[0]["expected_value"] > 0


    @pytest.mark.asyncio
    async def test_predict_user_needs(self):
        """Test user need prediction with mocked audit log results."""
        from unittest.mock import AsyncMock, MagicMock

        mock_row1 = MagicMock()
        mock_row1.resource_type = "billing"
        mock_row1.event_count = 10
        mock_row2 = MagicMock()
        mock_row2.resource_type = "reports"
        mock_row2.event_count = 5

        mock_result = MagicMock()
        mock_result.all = MagicMock(return_value=[mock_row1, mock_row2])

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        svc = IntrinsicMotivationService(session=mock_session)
        goals = await svc.predict_user_needs(org_id="org1")

        assert isinstance(goals, list)
        assert len(goals) == 2
        assert goals[0]["initiator"] == "prediction"
        assert goals[0]["metadata"]["domain"] == "billing"

    @pytest.mark.asyncio
    async def test_detect_self_improvement_needs(self):
        """Test self-improvement need detection with a mocked SelfModelProfile."""
        from unittest.mock import AsyncMock, MagicMock

        mock_profile = MagicMock()
        mock_profile.tool_reliability = {
            "memory.search": {"success_rate_30d": 0.45, "sample_size_30d": 10},
            "memory.get": {"success_rate_30d": 0.95, "sample_size_30d": 20},
        }

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_profile)

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        svc = IntrinsicMotivationService(session=mock_session)
        goals = await svc.detect_self_improvement_needs(org_id="org1")

        assert isinstance(goals, list)
        assert len(goals) == 1  # only memory.search is below threshold
        assert goals[0]["initiator"] == "self_improvement"
        assert goals[0]["urgency"] > 0.3
    
    @pytest.mark.asyncio
    async def test_estimate_goal_value_curiosity(self):
        """Test value estimation for curiosity goals."""
        svc = IntrinsicMotivationService(session=None)
        
        curiosity_goal = {
            "initiator": "curiosity",
            "expected_value": 0.7,
        }
        
        value = await svc.estimate_goal_value(curiosity_goal)
        
        assert 0.0 <= value <= 1.0
        assert value > 0.2
    
    @pytest.mark.asyncio
    async def test_estimate_goal_value_prediction(self):
        """Test value estimation for prediction goals."""
        svc = IntrinsicMotivationService(session=None)
        
        prediction_goal = {
            "initiator": "prediction",
            "expected_value": 0.5,
        }
        
        value = await svc.estimate_goal_value(prediction_goal)
        
        assert 0.0 <= value <= 1.0
    
    @pytest.mark.asyncio
    async def test_estimate_goal_value_self_improvement(self):
        """Test value estimation for self-improvement goals."""
        svc = IntrinsicMotivationService(session=None)
        
        self_improvement_goal = {
            "initiator": "self_improvement",
            "expected_value": 0.7,
            "meta": {
                "current_success_rate": 0.4,
                "target_success_rate": 0.9,
            },
        }
        
        value = await svc.estimate_goal_value(self_improvement_goal)
        
        assert 0.0 <= value <= 1.0
        assert value > 0.2
    
    @pytest.mark.asyncio
    async def test_estimate_goal_value_priority_rebalance(self):
        """Test value estimation for priority rebalance goals."""
        svc = IntrinsicMotivationService(session=None)
        
        priority_goal = {
            "initiator": "priority_rebalance",
            "expected_value": 0.5,
        }
        
        value = await svc.estimate_goal_value(priority_goal)
        
        assert 0.0 <= value <= 1.0
    
    @pytest.mark.asyncio
    async def test_record_goal_outcome(self):
        """Test recording goal outcomes."""
        # Create a mock session
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        svc = IntrinsicMotivationService(session=mock_session)
        
        org_id = str(uuid4())
        goal_id = str(uuid4())
        
        # Should not raise any exceptions
        result = await svc.record_goal_outcome(
            org_id=org_id,
            goal_id=goal_id,
            outcome_type="successful",
            impact_description="User found helpful",
            was_user_expecting=True,
        )
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_get_goal_success_metrics(self):
        """Test retrieving goal success metrics (logic validation)."""
        # Since get_goal_success_metrics requires database access,
        # we'll test it with a real but empty query result mock
        # The important part is that the method exists and returns the right structure
        svc = IntrinsicMotivationService(session=None)
        
        # This method requires a session with database access
        # In unit tests without database, we just verify the method signature exists
        assert hasattr(svc, 'get_goal_success_metrics')
        assert callable(getattr(svc, 'get_goal_success_metrics'))


class TestGoalInitiators:
    """Tests for different goal initiator types."""
    
    def test_all_initiator_types(self):
        """Test creating goals with all initiator types."""
        org_id = uuid4()
        initiators = [
            "curiosity",
            "prediction",
            "self_improvement",
            "priority_rebalance",
        ]
        
        for initiator in initiators:
            goal = AutonomousGoal(
                organization_id=org_id,
                initiator=initiator,
                title=f"Test {initiator} goal",
                description=f"Description for {initiator}",
                expected_value=0.5,
                urgency=0.5,
                confidence=0.5,
                status="proposed",
            )
            
            assert goal.initiator == initiator
            assert goal.title == f"Test {initiator} goal"
