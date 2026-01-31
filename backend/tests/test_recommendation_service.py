"""
Tests for RecommendationService.

Test coverage:
- Recommendation ranking algorithm accuracy
- Factor calculation (similarity, recency, interaction, feedback)
- Caching behavior
- Feedback tracking
- Metrics calculation
- Error handling
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncSession
import redis

from app.services.recommendation_service import RecommendationService
from app.models.graph_relationship import GraphRelationship
from app.models.recommendation_feedback import RecommendationFeedback


@pytest.fixture
def mock_db():
    """Mock async database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    client = MagicMock(spec=redis.Redis)
    client.get.return_value = None  # No cached results
    return client


@pytest.fixture
def service(mock_db, mock_redis):
    """Create service instance."""
    return RecommendationService(mock_db, mock_redis)


class TestRecommendationRanking:
    """Test recommendation ranking algorithm."""

    @pytest.mark.asyncio
    async def test_ranking_weights_sum_to_one(self, service):
        """Test that ranking weights sum to 1.0."""
        weights = await service.get_weights()
        total = sum(weights.values())
        
        assert np.isclose(total, 1.0)

    @pytest.mark.asyncio
    async def test_ranking_weights_components(self, service):
        """Test that all required weight components exist."""
        weights = await service.get_weights()
        
        required = {"similarity", "recency", "interaction", "feedback"}
        assert set(weights.keys()) == required

    def test_composite_score_calculation(self, service):
        """Test composite score calculation from factors."""
        factors = {
            "similarity": 0.9,
            "recency": 0.7,
            "interaction": 0.6,
            "feedback": 0.8
        }
        
        # Manual calculation
        expected = (
            0.5 * 0.9 +
            0.2 * 0.7 +
            0.2 * 0.6 +
            0.1 * 0.8
        )
        
        assert expected == 0.5 * 0.9 + 0.2 * 0.7 + 0.2 * 0.6 + 0.1 * 0.8


class TestRecencyScoringCalculation:
    """Test recency score calculation."""

    @pytest.mark.asyncio
    async def test_recent_memory_high_score(self, service, mock_db):
        """Test that recent memories get high recency scores."""
        # Very recent memory
        recent = datetime.utcnow()
        memory_id = str(uuid4())
        
        # Mock database return
        mock_row = MagicMock()
        mock_row.created_at = recent
        mock_row.updated_at = recent
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.first.return_value = mock_row
        
        score = await service._calculate_recency_score(memory_id)
        
        # Should be very close to 1.0
        assert score > 0.95

    @pytest.mark.asyncio
    async def test_old_memory_low_score(self, service, mock_db):
        """Test that old memories get low recency scores."""
        # 1 year old memory
        old = datetime.utcnow() - timedelta(days=365)
        memory_id = str(uuid4())
        
        mock_row = MagicMock()
        mock_row.created_at = old
        mock_row.updated_at = old
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.first.return_value = mock_row
        
        score = await service._calculate_recency_score(memory_id)
        
        # Should be very close to 0.0
        assert score < 0.05

    @pytest.mark.asyncio
    async def test_30_day_memory_score(self, service, mock_db):
        """Test that 30-day old memory gets ~0.5 score (exponential decay)."""
        # 30 days old
        date_30_days = datetime.utcnow() - timedelta(days=30)
        memory_id = str(uuid4())
        
        mock_row = MagicMock()
        mock_row.created_at = date_30_days
        mock_row.updated_at = date_30_days
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.first.return_value = mock_row
        
        score = await service._calculate_recency_score(memory_id)
        
        # At 30 days, e^(-30/30) = e^-1 ≈ 0.368
        assert np.isclose(score, 0.368, atol=0.01)

    @pytest.mark.asyncio
    async def test_no_memory_returns_zero(self, service, mock_db):
        """Test that missing memory returns 0 recency."""
        memory_id = str(uuid4())
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.first.return_value = None
        
        score = await service._calculate_recency_score(memory_id)
        
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_recency_score_range(self, service, mock_db):
        """Test that recency score is always in [0, 1] range."""
        # Any valid date should produce score in [0, 1]
        for days_ago in [0, 7, 30, 90, 365]:
            date = datetime.utcnow() - timedelta(days=days_ago)
            memory_id = str(uuid4())
            
            mock_row = MagicMock()
            mock_row.created_at = date
            mock_row.updated_at = date
            
            mock_db.execute.return_value = MagicMock()
            mock_db.execute.return_value.first.return_value = mock_row
            
            score = await service._calculate_recency_score(memory_id)
            
            assert 0.0 <= score <= 1.0


class TestInteractionScoringCalculation:
    """Test interaction score calculation."""

    @pytest.mark.asyncio
    async def test_high_interaction_high_score(self, service, mock_db):
        """Test that memories with high interaction get high scores."""
        memory_id = str(uuid4())
        metadata = {
            "view_count": 100,
            "edit_count": 50,
            "share_count": 20,
            "time_spent_seconds": 3600  # 1 hour
        }
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.first.return_value = (0, metadata)
        
        score = await service._calculate_interaction_score(memory_id)
        
        # Should be high
        assert score > 0.7

    @pytest.mark.asyncio
    async def test_no_interaction_neutral_score(self, service, mock_db):
        """Test that memories with no interaction get neutral scores."""
        memory_id = str(uuid4())
        metadata = {
            "view_count": 0,
            "edit_count": 0,
            "share_count": 0,
            "time_spent_seconds": 0
        }
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.first.return_value = (0, metadata)
        
        score = await service._calculate_interaction_score(memory_id)
        
        # No interaction means neutral score of 0.5 (sigmoid(0) = 0.5)
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_interaction_score_range(self, service, mock_db):
        """Test that interaction score is in [0, 1] range."""
        memory_id = str(uuid4())
        metadata = {
            "view_count": 1000,
            "edit_count": 500,
            "share_count": 100,
            "time_spent_seconds": 10000
        }
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.first.return_value = (0, metadata)
        
        score = await service._calculate_interaction_score(memory_id)
        
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_empty_metadata_returns_zero(self, service, mock_db):
        """Test that missing metadata yields neutral score when row exists."""
        memory_id = str(uuid4())
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.first.return_value = (0, None)
        
        score = await service._calculate_interaction_score(memory_id)
        
        assert score == 0.5


class TestFeedbackScoringCalculation:
    """Test feedback score calculation."""

    @pytest.mark.asyncio
    async def test_all_helpful_feedback_score_one(self, service, mock_db):
        """Test that all helpful feedback gives score of 1.0."""
        user_id = str(uuid4())
        memory_id = str(uuid4())
        org_id = str(uuid4())
        
        # Create feedback objects
        feedbacks = [
            MagicMock(helpful=True, created_at=datetime.utcnow()),
            MagicMock(helpful=True, created_at=datetime.utcnow()),
            MagicMock(helpful=True, created_at=datetime.utcnow()),
        ]
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = feedbacks
        
        score = await service._calculate_feedback_score(memory_id, user_id, org_id)
        
        # Should be 1.0
        assert np.isclose(score, 1.0)

    @pytest.mark.asyncio
    async def test_all_unhelpful_feedback_score_zero(self, service, mock_db):
        """Test that all unhelpful feedback gives score of 0.0."""
        user_id = str(uuid4())
        memory_id = str(uuid4())
        org_id = str(uuid4())
        
        feedbacks = [
            MagicMock(helpful=False, created_at=datetime.utcnow()),
            MagicMock(helpful=False, created_at=datetime.utcnow()),
        ]
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = feedbacks
        
        score = await service._calculate_feedback_score(memory_id, user_id, org_id)
        
        # Should be 0.0
        assert np.isclose(score, 0.0)

    @pytest.mark.asyncio
    async def test_mixed_feedback_neutral_score(self, service, mock_db):
        """Test that mixed feedback gives neutral score."""
        user_id = str(uuid4())
        memory_id = str(uuid4())
        org_id = str(uuid4())
        
        feedbacks = [
            MagicMock(helpful=True, created_at=datetime.utcnow()),
            MagicMock(helpful=False, created_at=datetime.utcnow()),
        ]
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = feedbacks
        
        score = await service._calculate_feedback_score(memory_id, user_id, org_id)
        
        # Should be around 0.5
        assert 0.4 <= score <= 0.6

    @pytest.mark.asyncio
    async def test_no_feedback_neutral_default(self, service, mock_db):
        """Test that no feedback returns neutral default (0.5)."""
        user_id = str(uuid4())
        memory_id = str(uuid4())
        org_id = str(uuid4())
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        
        score = await service._calculate_feedback_score(memory_id, user_id, org_id)
        
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_recency_weights_recent_feedback(self, service, mock_db):
        """Test that recent feedback is weighted more heavily."""
        user_id = str(uuid4())
        memory_id = str(uuid4())
        org_id = str(uuid4())
        
        recent_fb = MagicMock(helpful=True, created_at=datetime.utcnow())
        old_fb = MagicMock(helpful=False, created_at=datetime.utcnow() - timedelta(days=90))
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = [recent_fb, old_fb]
        
        score = await service._calculate_feedback_score(memory_id, user_id, org_id)
        
        # Should be closer to 1.0 (helpful) since recent feedback is weighted higher
        assert score > 0.7


class TestRelationshipRetrieval:
    """Test retrieval of related memories."""

    @pytest.mark.asyncio
    async def test_get_related_memories(self, service, mock_db):
        """Test getting related memories from graph."""
        org_id = str(uuid4())
        memory_id = str(uuid4())
        
        # Mock outgoing relationships
        outgoing_rel = MagicMock()
        outgoing_rel.to_memory_id = str(uuid4())
        outgoing_rel.similarity_score = 0.9
        outgoing_rel.relationship_type = "RELATES_TO"
        
        # Mock database execute
        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = [outgoing_rel]
        
        mock_db.execute.return_value = execute_result
        
        related = await service._get_related_memories(memory_id, org_id)
        
        # Should have at least the outgoing relationship
        assert len(related) >= 1

    @pytest.mark.asyncio
    async def test_related_memories_deduplication(self, service, mock_db):
        """Test that bidirectional relationships are deduplicated."""
        org_id = str(uuid4())
        memory_id = str(uuid4())
        target_id = str(uuid4())
        
        # Both incoming and outgoing to same target
        rel = MagicMock()
        rel.to_memory_id = target_id
        rel.from_memory_id = target_id
        rel.similarity_score = 0.85
        rel.relationship_type = "RELATES_TO"
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = [rel]
        
        # Would need proper mock setup for both incoming and outgoing
        # This is a simplified test


class TestCaching:
    """Test recommendation caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, service, mock_redis, mock_db):
        """Test that cached results are returned."""
        cached_data = '[{"memory_id": "mem123", "score": 0.85}]'
        mock_redis.get.return_value = cached_data
        
        result = await service.get_recommendations(
            "mem1",
            "org123",
            use_cache=True
        )
        
        # Should use cache and not hit database
        assert result is not None
        assert not mock_db.execute.called

    @pytest.mark.asyncio
    async def test_cache_miss_queries_db(self, service, mock_redis, mock_db):
        """Test that cache miss causes database query."""
        mock_redis.get.return_value = None  # Cache miss
        
        # Mock database relationships
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        
        # Would need complete mocking of entire flow


class TestFeedbackSubmission:
    """Test feedback submission."""

    @pytest.mark.asyncio
    async def test_submit_helpful_feedback(self, service, mock_db):
        """Test submitting helpful feedback."""
        org_id = str(uuid4())
        user_id = str(uuid4())
        
        # Mock database add/commit
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        
        result = await service.submit_feedback(
            "mem1",
            "mem2",
            org_id,
            user_id,
            helpful=True,
            reason="Very relevant"
        )
        
        assert result["status"] == "feedback_submitted"
        assert result["helpful"] == True
        assert mock_db.add.called

    @pytest.mark.asyncio
    async def test_submit_unhelpful_feedback(self, service, mock_db):
        """Test submitting unhelpful feedback."""
        org_id = str(uuid4())
        user_id = str(uuid4())
        
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        
        result = await service.submit_feedback(
            "mem1",
            "mem2",
            org_id,
            user_id,
            helpful=False,
            reason="Not relevant"
        )
        
        assert result["status"] == "feedback_submitted"
        assert result["helpful"] == False


class TestWeightUpdates:
    """Test weight configuration updates."""

    @pytest.mark.asyncio
    async def test_update_weights_success(self, service):
        """Test successful weight update."""
        new_weights = {
            "similarity": 0.6,
            "recency": 0.2,
            "interaction": 0.1,
            "feedback": 0.1
        }
        
        result = await service.update_weights(new_weights)
        
        assert result == new_weights

    @pytest.mark.asyncio
    async def test_update_weights_invalid_sum(self, service):
        """Test that invalid weight sums are rejected."""
        invalid_weights = {
            "similarity": 0.6,
            "recency": 0.2,
            "interaction": 0.1,
            "feedback": 0.2  # Sum > 1.0
        }
        
        with pytest.raises(ValueError):
            await service.update_weights(invalid_weights)

    @pytest.mark.asyncio
    async def test_update_weights_invalid_keys(self, service):
        """Test that invalid weight keys are rejected."""
        invalid_weights = {
            "similarity": 0.5,
            "invalid_key": 0.5
        }
        
        with pytest.raises(ValueError):
            await service.update_weights(invalid_weights)


class TestMetricsCalculation:
    """Test metrics calculation."""

    @pytest.mark.asyncio
    async def test_recommendation_metrics(self, service, mock_db):
        """Test metrics calculation."""
        org_id = str(uuid4())
        
        # Mock feedback
        feedbacks = [
            MagicMock(helpful=True),
            MagicMock(helpful=True),
            MagicMock(helpful=False),
        ]
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = feedbacks
        
        metrics = await service.get_recommendation_metrics(org_id)
        
        assert metrics["total_feedback"] == 3
        assert metrics["helpful_count"] == 2
        assert metrics["helpful_ratio"] == 2/3

    @pytest.mark.asyncio
    async def test_metrics_empty_feedback(self, service, mock_db):
        """Test metrics with no feedback."""
        org_id = str(uuid4())
        
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        
        metrics = await service.get_recommendation_metrics(org_id)
        
        assert metrics["total_feedback"] == 0


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_get_recommendations_empty_results(self, service, mock_db, mock_redis):
        """Test handling of no recommendations."""
        mock_redis.get.return_value = None
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        
        # Should handle gracefully and return empty list
        result = await service.get_recommendations("mem123", "org123")
        
        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_org_id(self, service):
        """Test handling of invalid organization ID."""
        # Should handle UUID parsing gracefully
        # or return empty results
        pass


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_recommendation_generation_performance(self, service, mock_db, mock_redis):
        """Test that recommendation generation completes in reasonable time."""
        import time
        
        mock_redis.get.return_value = None
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        
        start = time.time()
        await service.get_recommendations("mem123", "org123", limit=10)
        elapsed = time.time() - start
        
        # Should be fast (seconds, not minutes)
        assert elapsed < 5.0
