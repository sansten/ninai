"""Tests for Memory Activation Scorer

Tests cover:
- All 8 component calculations
- Activation monotonicity
- Sigmoid bounds
- Component normalization
- Neighbor boost
"""

import pytest
import math
from datetime import datetime, timezone, timedelta

from app.services.memory_activation.scoring import (
    ActivationScorer,
    ActivationScorerConfig,
    ActivationComponents,
    get_activation_scorer,
)


class TestActivationComponents:
    """Test ActivationComponents dataclass."""

    def test_valid_components(self):
        """Test creating valid components."""
        comp = ActivationComponents(
            rel=0.9,
            rec=0.8,
            freq=0.7,
            imp=0.85,
            conf=0.9,
            ctx=0.95,
            prov=0.6,
            risk=0.1,
        )
        assert comp.rel == 0.9
        assert comp.to_dict()["rel"] == 0.9

    def test_components_validate(self):
        """Test component validation."""
        valid = ActivationComponents(rel=0.5, rec=0.5, freq=0.5, imp=0.5, conf=0.5, ctx=0.5, prov=0.5, risk=0.5)
        assert valid.validate() is True

        invalid = ActivationComponents(rel=1.5, rec=0.5, freq=0.5, imp=0.5, conf=0.5, ctx=0.5, prov=0.5, risk=0.5)
        assert invalid.validate() is False


class TestRelevanceComponent:
    """Test relevance component calculation."""

    def test_high_similarity(self):
        """Test high similarity maps to high relevance."""
        scorer = ActivationScorer()
        rel = scorer.compute_relevance(0.95)
        assert rel == 0.95

    def test_low_similarity(self):
        """Test low similarity maps to low relevance."""
        scorer = ActivationScorer()
        rel = scorer.compute_relevance(0.1)
        assert rel == 0.1

    def test_zero_similarity(self):
        """Test zero similarity."""
        scorer = ActivationScorer()
        rel = scorer.compute_relevance(0.0)
        assert rel == 0.0

    def test_relevance_bounds(self):
        """Test relevance stays in [0, 1]."""
        scorer = ActivationScorer()
        assert 0.0 <= scorer.compute_relevance(0.5) <= 1.0
        assert 0.0 <= scorer.compute_relevance(-0.1) <= 1.0
        assert 0.0 <= scorer.compute_relevance(1.5) <= 1.0


class TestRecencyComponent:
    """Test recency component calculation."""

    def test_recent_access(self):
        """Test recent access has high recency."""
        scorer = ActivationScorer()
        now = datetime.now(timezone.utc)
        recent = now - timedelta(hours=1)
        rec = scorer.compute_recency(recent, now)
        assert rec > 0.9

    def test_old_access(self):
        """Test old access has low recency."""
        scorer = ActivationScorer()
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=365)
        rec = scorer.compute_recency(old, now)
        assert rec < 0.1

    def test_never_accessed(self):
        """Test never accessed memory has minimum recency."""
        scorer = ActivationScorer()
        rec = scorer.compute_recency(None)
        assert rec == 0.01

    def test_recency_is_decreasing(self):
        """Test recency decreases monotonically with time."""
        scorer = ActivationScorer()
        now = datetime.now(timezone.utc)
        recent = now - timedelta(hours=1)
        older = now - timedelta(days=1)
        very_old = now - timedelta(days=30)

        rec_recent = scorer.compute_recency(recent, now)
        rec_older = scorer.compute_recency(older, now)
        rec_very_old = scorer.compute_recency(very_old, now)

        assert rec_recent > rec_older > rec_very_old


class TestFrequencyComponent:
    """Test frequency component calculation."""

    def test_high_frequency(self):
        """Test high access count."""
        scorer = ActivationScorer()
        freq = scorer.compute_frequency(100)
        assert freq > 0.9

    def test_low_frequency(self):
        """Test low access count."""
        scorer = ActivationScorer()
        freq = scorer.compute_frequency(1)
        assert 0.3 < freq < 0.7

    def test_zero_frequency(self):
        """Test zero access count."""
        scorer = ActivationScorer()
        freq = scorer.compute_frequency(0)
        assert freq < 0.01

    def test_frequency_saturation(self):
        """Test frequency saturates at high counts."""
        scorer = ActivationScorer()
        freq_50 = scorer.compute_frequency(50)
        freq_100 = scorer.compute_frequency(100)
        freq_1000 = scorer.compute_frequency(1000)

        # Should increase but eventually plateau
        assert freq_50 < freq_100
        # At 1000, should be very close to 1.0 (saturated)
        assert freq_100 <= freq_1000
        assert freq_1000 > 0.99


class TestImportanceComponent:
    """Test importance component calculation."""

    def test_high_importance(self):
        """Test high importance."""
        scorer = ActivationScorer()
        imp = scorer.compute_importance(0.9, age_days=0)
        assert imp > 0.8

    def test_low_importance(self):
        """Test low importance."""
        scorer = ActivationScorer()
        imp = scorer.compute_importance(0.1, age_days=0)
        assert imp < 0.3

    def test_importance_decay(self):
        """Test importance decays with age."""
        scorer = ActivationScorer()
        fresh = scorer.compute_importance(0.8, age_days=0)
        old = scorer.compute_importance(0.8, age_days=30)

        assert fresh > old


class TestConfidenceComponent:
    """Test confidence component calculation."""

    def test_high_confidence(self):
        """Test high confidence."""
        scorer = ActivationScorer()
        conf = scorer.compute_confidence(0.9)
        assert conf == 0.9

    def test_confidence_contradiction_penalty(self):
        """Test confidence is penalized when contradicted."""
        scorer = ActivationScorer()
        conf_normal = scorer.compute_confidence(0.9, contradicted=False)
        conf_contradicted = scorer.compute_confidence(0.9, contradicted=True)

        assert conf_normal > conf_contradicted


class TestContextGateComponent:
    """Test context gate component calculation."""

    def test_perfect_context_match(self):
        """Test perfect context match."""
        scorer = ActivationScorer()
        ctx = scorer.compute_context_gate(scope_match=1.0, episode_match=1.0, goal_match=1.0)
        assert ctx == 1.0

    def test_no_context_match(self):
        """Test no context match."""
        scorer = ActivationScorer()
        ctx = scorer.compute_context_gate(scope_match=0.0, episode_match=0.0, goal_match=0.0)
        assert ctx == 0.0

    def test_partial_context_match(self):
        """Test partial context match."""
        scorer = ActivationScorer()
        ctx = scorer.compute_context_gate(scope_match=0.5, episode_match=0.5, goal_match=0.5)
        assert ctx == 0.5


class TestProvenanceComponent:
    """Test provenance component calculation."""

    def test_no_evidence(self):
        """Test no evidence links."""
        scorer = ActivationScorer()
        prov = scorer.compute_provenance(0)
        assert prov < 0.01

    def test_high_evidence(self):
        """Test high evidence links."""
        scorer = ActivationScorer()
        prov = scorer.compute_provenance(10)
        assert prov > 0.8


class TestRiskComponent:
    """Test risk component calculation."""

    def test_no_risk(self):
        """Test no risk."""
        scorer = ActivationScorer()
        risk = scorer.compute_risk(0.0)
        assert risk == 1.0

    def test_high_risk(self):
        """Test high risk reduces score."""
        scorer = ActivationScorer()
        risk = scorer.compute_risk(1.0)
        assert risk == 0.0

    def test_medium_risk(self):
        """Test medium risk."""
        scorer = ActivationScorer()
        risk = scorer.compute_risk(0.5)
        assert risk == 0.5


class TestActivationComputation:
    """Test final activation score computation."""

    def test_high_similarity_high_activation(self):
        """Test high similarity results in high activation."""
        scorer = ActivationScorer()
        comp_high = ActivationComponents(rel=0.95, rec=0.9, freq=0.8, imp=0.9, conf=0.95, ctx=0.9, prov=0.8, risk=0.05)
        comp_low = ActivationComponents(rel=0.1, rec=0.1, freq=0.1, imp=0.1, conf=0.1, ctx=0.1, prov=0.1, risk=0.9)

        act_high = scorer.compute_activation(comp_high)
        act_low = scorer.compute_activation(comp_low)

        assert act_high > act_low

    def test_activation_bounds(self):
        """Test activation stays in [0, 1]."""
        scorer = ActivationScorer()
        comp = ActivationComponents(rel=0.5, rec=0.5, freq=0.5, imp=0.5, conf=0.5, ctx=0.5, prov=0.5, risk=0.5)
        act = scorer.compute_activation(comp)
        assert 0.0 <= act <= 1.0

    def test_neighbor_boost(self):
        """Test neighbor boost increases activation."""
        scorer = ActivationScorer()
        comp = ActivationComponents(rel=0.5, rec=0.5, freq=0.5, imp=0.5, conf=0.5, ctx=0.5, prov=0.5, risk=0.5)

        act_no_boost = scorer.compute_activation(comp, neighbor_activation=None)
        act_with_boost = scorer.compute_activation(comp, neighbor_activation=0.9)

        assert act_with_boost > act_no_boost


class TestFullScoring:
    """Test end-to-end scoring."""

    def test_score_memory_defaults(self):
        """Test scoring with default parameters."""
        scorer = ActivationScorer()
        act, comp = scorer.score_memory(
            similarity=0.8,
            base_importance=0.7,
            confidence=0.85,
        )
        assert 0.0 <= act <= 1.0
        assert comp.validate()

    def test_score_memory_comprehensive(self):
        """Test comprehensive scoring with all parameters."""
        scorer = ActivationScorer()
        now = datetime.now(timezone.utc)
        recent = now - timedelta(hours=1)

        act, comp = scorer.score_memory(
            similarity=0.9,
            base_importance=0.8,
            confidence=0.9,
            contradicted=False,
            risk_factor=0.05,
            access_count=10,
            last_accessed_at=recent,
            evidence_link_count=3,
            scope_match=0.8,
            episode_match=0.9,
            goal_match=0.7,
            neighbor_activation=0.8,
            current_time=now,
            age_days=5,
        )
        assert 0.0 <= act <= 1.0
        assert comp.validate()
        assert comp.rel == 0.9
        assert comp.conf > 0.8

    def test_denied_memory_scoring(self):
        """Test denied memory gets lower activation (RBAC would block before scorer)."""
        scorer = ActivationScorer()
        # All zeros with high risk
        comp_denied = ActivationComponents(rel=0.0, rec=0.0, freq=0.0, imp=0.0, conf=0.0, ctx=0.0, prov=0.0, risk=1.0)
        # Normal memory
        comp_normal = ActivationComponents(rel=0.8, rec=0.8, freq=0.7, imp=0.8, conf=0.8, ctx=0.8, prov=0.7, risk=0.1)
        
        act_denied = scorer.compute_activation(comp_denied)
        act_normal = scorer.compute_activation(comp_normal)
        
        # Denied should be lower than normal
        assert act_denied < act_normal


class TestSingletonScorer:
    """Test global scorer singleton."""

    def test_get_activation_scorer(self):
        """Test getting singleton scorer."""
        scorer1 = get_activation_scorer()
        scorer2 = get_activation_scorer()
        assert scorer1 is scorer2

    def test_custom_config(self):
        """Test custom configuration."""
        config = ActivationScorerConfig(w_rel=0.5, w_rec=0.5)
        scorer = ActivationScorer(config)
        assert scorer.config.w_rel == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
