"""Phase 94 — AgentContributionEstimatorService tests."""
from __future__ import annotations

import pytest

from app.services.agent_contribution_estimator import (
    AgentContributionEstimatorService,
    ContributionReport,
)


# ---------------------------------------------------------------------------
# Deterministic metric helpers
# ---------------------------------------------------------------------------

def _coverage_metric(coalition: frozenset[str], outcomes: list[dict]) -> float:
    """Fraction of outcomes contributed by any agent in the coalition."""
    if not outcomes:
        return 0.0
    hits = sum(1 for o in outcomes if o.get("agent_id") in coalition)
    return hits / len(outcomes)


def _additive_metric(coalition: frozenset[str], outcomes: list[dict]) -> float:
    """Sum of per-agent scores for agents in the coalition."""
    return sum(o.get("score", 0.0) for o in outcomes if o.get("agent_id") in coalition)


def _constant_metric(coalition: frozenset[str], outcomes: list[dict]) -> float:
    """All coalitions have equal value — Shapley values should be equal."""
    return 1.0 if coalition else 0.0


def _empty_metric(coalition: frozenset[str], outcomes: list[dict]) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# Outcomes fixture
# ---------------------------------------------------------------------------

def _make_outcomes(agents: list[str]) -> list[dict]:
    return [{"agent_id": a, "score": 1.0} for a in agents]


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_n_samples(self):
        svc = AgentContributionEstimatorService()
        assert svc._n_samples >= 10

    def test_custom_n_samples(self):
        svc = AgentContributionEstimatorService(n_samples=50)
        assert svc._n_samples == 50

    def test_min_n_samples_clamped(self):
        svc = AgentContributionEstimatorService(n_samples=1)
        assert svc._n_samples >= 10

    def test_seeded_rng(self):
        svc1 = AgentContributionEstimatorService(n_samples=10, seed=42)
        svc2 = AgentContributionEstimatorService(n_samples=10, seed=42)
        agents = ["a", "b", "c"]
        outcomes = _make_outcomes(agents)
        r1 = svc1.estimate(agents, outcomes, _coverage_metric)
        r2 = svc2.estimate(agents, outcomes, _coverage_metric)
        assert r1.shapley_values == r2.shapley_values


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_agents_returns_empty_report(self):
        svc = AgentContributionEstimatorService()
        report = svc.estimate([], [], _coverage_metric)
        assert isinstance(report, ContributionReport)
        assert report.n_agents == 0
        assert report.shapley_values == {}
        assert report.top_contributors == []

    def test_single_agent(self):
        svc = AgentContributionEstimatorService(n_samples=20, seed=0)
        outcomes = [{"agent_id": "solo", "score": 1.0}]
        report = svc.estimate(["solo"], outcomes, _coverage_metric)
        assert report.n_agents == 1
        assert "solo" in report.shapley_values

    def test_zero_metric_all_zero_shapley(self):
        svc = AgentContributionEstimatorService(n_samples=20, seed=0)
        agents = ["a", "b", "c"]
        report = svc.estimate(agents, [], _empty_metric)
        for v in report.shapley_values.values():
            assert abs(v) < 1e-9

    def test_no_outcomes_still_returns_report(self):
        svc = AgentContributionEstimatorService(n_samples=10, seed=0)
        report = svc.estimate(["a", "b"], [], _coverage_metric)
        assert report is not None
        assert report.n_agents == 2


# ---------------------------------------------------------------------------
# Shapley properties
# ---------------------------------------------------------------------------

class TestShapleyProperties:
    def test_values_sum_to_grand_coalition_value(self):
        """Efficiency property: sum of Shapley values ≈ grand coalition value."""
        svc = AgentContributionEstimatorService(n_samples=200, seed=7)
        agents = ["a1", "a2", "a3"]
        outcomes = _make_outcomes(agents)
        report = svc.estimate(agents, outcomes, _coverage_metric)
        grand = _coverage_metric(frozenset(agents), outcomes)
        total = sum(report.shapley_values.values())
        assert abs(total - grand) < 0.05  # MC approximation tolerance

    def test_symmetry_equal_agents(self):
        """Symmetry: agents with identical contributions should have equal Shapley values."""
        svc = AgentContributionEstimatorService(n_samples=300, seed=99)
        agents = ["a1", "a2", "a3"]
        # All agents contribute equally (one outcome each, same score)
        outcomes = _make_outcomes(agents)
        report = svc.estimate(agents, outcomes, _coverage_metric)
        values = list(report.shapley_values.values())
        # All values should be close to each other
        mean_v = sum(values) / len(values)
        for v in values:
            assert abs(v - mean_v) < 0.10

    def test_dummy_agent_zero_contribution(self):
        """Null player: agent that never contributes should have ~zero Shapley value."""
        svc = AgentContributionEstimatorService(n_samples=200, seed=13)
        # dummy-agent is never in outcomes
        outcomes = [{"agent_id": "a1"}, {"agent_id": "a2"}]
        report = svc.estimate(["a1", "a2", "dummy-agent"], outcomes, _coverage_metric)
        dummy_v = report.shapley_values["dummy-agent"]
        assert abs(dummy_v) < 0.05

    def test_high_contribution_agent_has_higher_shapley(self):
        """Agent responsible for most outcomes should have the highest Shapley value."""
        svc = AgentContributionEstimatorService(n_samples=300, seed=5)
        # a1 appears 8 times, a2 appears 2 times
        outcomes = [{"agent_id": "a1"}] * 8 + [{"agent_id": "a2"}] * 2
        report = svc.estimate(["a1", "a2"], outcomes, _coverage_metric)
        assert report.shapley_values["a1"] > report.shapley_values["a2"]

    def test_normalised_values_sum_to_one(self):
        svc = AgentContributionEstimatorService(n_samples=100, seed=1)
        agents = ["x", "y", "z"]
        outcomes = _make_outcomes(agents)
        report = svc.estimate(agents, outcomes, _coverage_metric)
        if report.total_value > 0:
            total_norm = sum(report.normalised_values.values())
            assert abs(total_norm - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

class TestReportStructure:
    def test_report_contains_all_agents(self):
        svc = AgentContributionEstimatorService(n_samples=20, seed=0)
        agents = ["a", "b", "c", "d"]
        outcomes = _make_outcomes(agents)
        report = svc.estimate(agents, outcomes, _coverage_metric)
        assert set(report.shapley_values.keys()) == set(agents)

    def test_top_contributors_is_subset_of_agents(self):
        svc = AgentContributionEstimatorService(n_samples=20, seed=0)
        agents = ["a", "b", "c", "d", "e"]
        outcomes = _make_outcomes(agents)
        report = svc.estimate(agents, outcomes, _coverage_metric)
        assert all(a in agents for a in report.top_contributors)

    def test_low_contributors_is_subset_of_agents(self):
        svc = AgentContributionEstimatorService(n_samples=20, seed=0)
        agents = ["a", "b", "c", "d", "e"]
        outcomes = _make_outcomes(agents)
        report = svc.estimate(agents, outcomes, _coverage_metric)
        assert all(a in agents for a in report.low_contributors)

    def test_n_samples_recorded(self):
        svc = AgentContributionEstimatorService(n_samples=77, seed=0)
        report = svc.estimate(["a", "b"], _make_outcomes(["a", "b"]), _coverage_metric)
        assert report.n_samples == 77

    def test_most_valuable_is_none_for_empty(self):
        svc = AgentContributionEstimatorService()
        report = svc.estimate([], [], _coverage_metric)
        assert report.most_valuable is None

    def test_most_valuable_is_string(self):
        svc = AgentContributionEstimatorService(n_samples=50, seed=0)
        agents = ["x", "y", "z"]
        outcomes = _make_outcomes(agents)
        report = svc.estimate(agents, outcomes, _coverage_metric)
        assert isinstance(report.most_valuable, str)

    def test_total_value_non_negative(self):
        svc = AgentContributionEstimatorService(n_samples=20, seed=0)
        agents = ["a", "b"]
        outcomes = _make_outcomes(agents)
        report = svc.estimate(agents, outcomes, _coverage_metric)
        # Total value is sum of Shapley values; may be near zero but not wildly negative
        assert report.total_value > -0.01


# ---------------------------------------------------------------------------
# Additive metric (closed-form Shapley known)
# ---------------------------------------------------------------------------

class TestAdditiveCases:
    def test_additive_metric_proportional_shapley(self):
        """For an additive metric, Shapley = individual contribution."""
        svc = AgentContributionEstimatorService(n_samples=500, seed=17)
        # a1 score=0.8, a2 score=0.2 → Shapley should be proportional
        outcomes = [
            {"agent_id": "a1", "score": 0.8},
            {"agent_id": "a2", "score": 0.2},
        ]
        report = svc.estimate(["a1", "a2"], outcomes, _additive_metric)
        assert report.shapley_values["a1"] > report.shapley_values["a2"]
        # Ratio should be roughly 4:1
        ratio = report.shapley_values["a1"] / max(report.shapley_values["a2"], 1e-9)
        assert 2.0 < ratio < 8.0  # generous MC tolerance


# ---------------------------------------------------------------------------
# Incremental estimation (ablation)
# ---------------------------------------------------------------------------

class TestIncrementalEstimation:
    def test_exclude_reduces_agent_set(self):
        svc = AgentContributionEstimatorService(n_samples=30, seed=0)
        agents = ["a1", "a2", "a3"]
        outcomes = _make_outcomes(agents)
        values = svc.estimate_incremental(agents, outcomes, _coverage_metric, exclude={"a3"})
        assert "a3" not in values
        assert "a1" in values and "a2" in values

    def test_exclude_all_returns_empty(self):
        svc = AgentContributionEstimatorService(n_samples=30, seed=0)
        agents = ["a1", "a2"]
        outcomes = _make_outcomes(agents)
        values = svc.estimate_incremental(agents, outcomes, _coverage_metric, exclude={"a1", "a2"})
        assert values == {}

    def test_exclude_none_same_as_full_estimate(self):
        svc = AgentContributionEstimatorService(n_samples=50, seed=42)
        agents = ["x", "y", "z"]
        outcomes = _make_outcomes(agents)
        full = svc.estimate(agents, outcomes, _coverage_metric).shapley_values
        incr = svc.estimate_incremental(agents, outcomes, _coverage_metric, exclude=None)
        assert set(incr.keys()) == set(full.keys())


# ---------------------------------------------------------------------------
# Scalability sanity
# ---------------------------------------------------------------------------

class TestScalability:
    def test_large_agent_set_completes(self):
        """85-agent pipeline should complete without hanging."""
        svc = AgentContributionEstimatorService(n_samples=20, seed=0)
        agents = [f"agent-{i}" for i in range(85)]
        outcomes = [{"agent_id": a} for a in agents]
        report = svc.estimate(agents, outcomes, _coverage_metric)
        assert report.n_agents == 85
        assert len(report.shapley_values) == 85

    def test_custom_top_fraction(self):
        svc = AgentContributionEstimatorService(n_samples=20, seed=0, top_fraction=0.5)
        agents = ["a", "b", "c", "d"]
        outcomes = _make_outcomes(agents)
        report = svc.estimate(agents, outcomes, _coverage_metric)
        # 50% of 4 = 2 top contributors
        assert len(report.top_contributors) == 2
