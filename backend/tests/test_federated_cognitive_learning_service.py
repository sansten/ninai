from __future__ import annotations

from app.services.federated_cognitive_learning_service import (
    FederatedCognitiveLearningService,
    _epsilon_for_org,
    _fedavg,
    _percentile_rank,
)


class TestFederatedCognitiveLearningService:

    # ------------------------------------------------------------------
    # Helper unit tests
    # ------------------------------------------------------------------

    def test_epsilon_for_org_tiers(self):
        assert _epsilon_for_org("high") < _epsilon_for_org("medium")
        assert _epsilon_for_org("medium") < _epsilon_for_org("low")
        assert _epsilon_for_org("unknown") == 1.0  # fallback

    def test_fedavg_equal_weights(self):
        weights = [{"a": 0.4, "b": 0.6}, {"a": 0.8, "b": 0.2}]
        counts = [1, 1]
        result = _fedavg(weights, counts)
        assert abs(result["a"] - 0.6) < 1e-9
        assert abs(result["b"] - 0.4) < 1e-9

    def test_fedavg_sample_weighted(self):
        # Org A has 1 sample, org B has 3 — B should dominate
        weights = [{"x": 0.0}, {"x": 1.0}]
        counts = [1, 3]
        result = _fedavg(weights, counts)
        assert result["x"] == 0.75

    def test_percentile_rank_boundaries(self):
        peers = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert _percentile_rank(0.5, peers) == 1.0
        assert _percentile_rank(0.05, peers) == 0.0
        assert _percentile_rank([], []) == 0.0

    # ------------------------------------------------------------------
    # aggregate_agent_weights
    # ------------------------------------------------------------------

    def test_aggregate_empty_submissions(self):
        svc = FederatedCognitiveLearningService()
        result = svc.aggregate_agent_weights(
            agent_name="test_agent",
            org_weight_submissions=[],
        )
        assert result.contributing_orgs == 0
        assert result.global_weights == {}

    def test_aggregate_uses_strictest_epsilon(self):
        svc = FederatedCognitiveLearningService()
        submissions = [
            {"weights": {"w1": 0.5}, "sample_count": 10, "sensitivity_tier": "high"},
            {"weights": {"w1": 0.7}, "sample_count": 10, "sensitivity_tier": "low"},
        ]
        result = svc.aggregate_agent_weights(
            agent_name="anomaly_detection",
            org_weight_submissions=submissions,
        )
        # Should use high-tier epsilon (0.1) — strictest
        assert result.aggregation_epsilon == 0.1
        assert result.contributing_orgs == 2
        assert "w1" in result.global_weights

    def test_aggregate_weights_multiple_orgs(self):
        svc = FederatedCognitiveLearningService()
        submissions = [
            {"weights": {"threshold": 0.3, "boost": 0.5}, "sample_count": 100, "sensitivity_tier": "medium"},
            {"weights": {"threshold": 0.5, "boost": 0.3}, "sample_count": 100, "sensitivity_tier": "medium"},
            {"weights": {"threshold": 0.4, "boost": 0.4}, "sample_count": 100, "sensitivity_tier": "medium"},
        ]
        result = svc.aggregate_agent_weights(
            agent_name="pattern_detection",
            org_weight_submissions=submissions,
        )
        assert result.contributing_orgs == 3
        assert "threshold" in result.global_weights
        assert "boost" in result.global_weights

    # ------------------------------------------------------------------
    # benchmark_org
    # ------------------------------------------------------------------

    def test_benchmark_no_peer_data(self):
        svc = FederatedCognitiveLearningService()
        insights = svc.benchmark_org(
            org_metrics={"response_time": 0.8},
            peer_metric_submissions=[],
        )
        assert len(insights) == 1
        assert insights[0].status == "no_peer_data"
        assert insights[0].metric == "response_time"

    def test_benchmark_on_track(self):
        svc = FederatedCognitiveLearningService()
        # Org is at the top — at or above 75th pctile
        peers = [{"quality": float(i) / 10} for i in range(10)]
        insights = svc.benchmark_org(
            org_metrics={"quality": 0.95},
            peer_metric_submissions=peers,
        )
        assert len(insights) == 1
        assert insights[0].status == "on_track"

    def test_benchmark_below_target(self):
        svc = FederatedCognitiveLearningService()
        peers = [{"quality": float(i) / 10} for i in range(10)]
        insights = svc.benchmark_org(
            org_metrics={"quality": 0.1},
            peer_metric_submissions=peers,
        )
        assert insights[0].status == "below_target"

    # ------------------------------------------------------------------
    # synthesize (full integration)
    # ------------------------------------------------------------------

    def test_synthesize_returns_full_result(self):
        svc = FederatedCognitiveLearningService()
        submissions = [
            {"weights": {"w": 0.5}, "sample_count": 50, "sensitivity_tier": "medium"},
            {"weights": {"w": 0.6}, "sample_count": 50, "sensitivity_tier": "medium"},
            {"weights": {"w": 0.55}, "sample_count": 50, "sensitivity_tier": "medium"},
            {"weights": {"w": 0.52}, "sample_count": 50, "sensitivity_tier": "medium"},
            {"weights": {"w": 0.58}, "sample_count": 50, "sensitivity_tier": "medium"},
        ]
        peers = [{"recall": float(i) / 10} for i in range(10)]
        result = svc.synthesize(
            agent_name="anomaly_detection",
            org_weight_submissions=submissions,
            org_metrics={"recall": 0.8},
            peer_metric_submissions=peers,
            sensitivity_tier="medium",
        )
        assert len(result.aggregated_weights) == 1
        assert result.aggregated_weights[0].contributing_orgs == 5
        assert len(result.benchmark_insights) == 1
        assert result.sharing_recommendation != ""
        assert 0.0 <= result.federation_confidence <= 1.0
        assert result.total_privacy_budget_spent > 0

    def test_synthesize_insufficient_contributors_recommendation(self):
        svc = FederatedCognitiveLearningService()
        result = svc.synthesize(
            agent_name="test",
            org_weight_submissions=[
                {"weights": {"w": 0.5}, "sample_count": 10, "sensitivity_tier": "medium"},
            ],
            org_metrics={},
            peer_metric_submissions=[],
        )
        assert "hold" in result.sharing_recommendation
