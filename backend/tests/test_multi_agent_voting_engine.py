from __future__ import annotations

import pytest

from app.services.multi_agent_voting_engine import MultiAgentVotingEngine


class TestVote:
    def test_all_true_consensus_and_full_agreement(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(
            ballots=[
                {"agent_name": "credibility_agent", "verdict": True, "confidence": 0.9},
                {"agent_name": "anomaly_detection_agent", "verdict": True, "confidence": 0.9},
            ]
        )
        assert result["consensus"] is True
        assert result["agreement_rate"] == 1.0

    def test_three_true_one_false_low_confidence_still_true(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(
            ballots=[
                {"agent_name": "credibility_agent", "verdict": True, "confidence": 0.9},
                {"agent_name": "anomaly_detection_agent", "verdict": True, "confidence": 0.8},
                {"agent_name": "conflict_detection_agent", "verdict": True, "confidence": 0.8},
                {"agent_name": "hypothesis_service", "verdict": False, "confidence": 0.1},
            ]
        )
        assert result["consensus"] is True

    def test_equal_weighted_votes_have_zero_margin(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(
            ballots=[
                {"agent_name": "credibility_agent", "verdict": True, "confidence": 1.0},
                {"agent_name": "credibility_agent", "verdict": False, "confidence": 1.0},
            ]
        )
        assert result["margin"] == 0.0
        assert svc.is_contested(margin=result["margin"]) is True

    def test_dissenting_agents_listed(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(
            ballots=[
                {"agent_name": "a", "verdict": True, "confidence": 1.0},
                {"agent_name": "b", "verdict": False, "confidence": 1.0},
                {"agent_name": "c", "verdict": True, "confidence": 1.0},
            ]
        )
        assert result["dissenting_agents"] == ["b"]

    def test_unknown_agent_uses_default_weight(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(
            ballots=[
                {"agent_name": "unknown_agent", "verdict": True, "confidence": 1.0},
            ]
        )
        assert result["weighted_true"] == pytest.approx(0.7)

    def test_empty_ballots_graceful_defaults(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(ballots=[])
        assert result["consensus"] is False
        assert result["margin"] == 0.0

    def test_confidence_clamped_low(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(ballots=[{"agent_name": "credibility_agent", "verdict": True, "confidence": -5.0}])
        assert result["weighted_true"] == 0.0

    def test_confidence_clamped_high(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(ballots=[{"agent_name": "credibility_agent", "verdict": True, "confidence": 9.0}])
        assert result["weighted_true"] == pytest.approx(0.9)

    def test_agreement_rate_fraction(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(
            ballots=[
                {"agent_name": "a", "verdict": True, "confidence": 1.0},
                {"agent_name": "b", "verdict": True, "confidence": 1.0},
                {"agent_name": "c", "verdict": False, "confidence": 1.0},
                {"agent_name": "d", "verdict": False, "confidence": 1.0},
                {"agent_name": "e", "verdict": True, "confidence": 1.0},
            ]
        )
        assert result["consensus"] is True
        assert result["agreement_rate"] == pytest.approx(3 / 5)


class TestResolveNumeric:
    def test_weighted_average_computed_correctly(self):
        svc = MultiAgentVotingEngine()
        result = svc.resolve_numeric(
            ballots=[
                {"agent_name": "credibility_agent", "value": 1.0, "confidence": 1.0},
                {"agent_name": "hypothesis_service", "value": 0.0, "confidence": 1.0},
            ]
        )
        expected = (0.9 * 1.0 + 0.75 * 0.0) / (0.9 + 0.75)
        assert result["consensus_value"] == pytest.approx(round(expected, 6))

    def test_single_ballot_consensus_value_equals_ballot_value(self):
        svc = MultiAgentVotingEngine()
        result = svc.resolve_numeric(ballots=[{"agent_name": "credibility_agent", "value": 0.42, "confidence": 1.0}])
        assert result["consensus_value"] == pytest.approx(0.42)

    def test_std_deviation_zero_for_single_ballot(self):
        svc = MultiAgentVotingEngine()
        result = svc.resolve_numeric(ballots=[{"agent_name": "credibility_agent", "value": 0.42, "confidence": 1.0}])
        assert result["std_deviation"] == 0.0

    def test_unknown_agent_default_weight_used_numeric(self):
        svc = MultiAgentVotingEngine()
        result = svc.resolve_numeric(
            ballots=[
                {"agent_name": "unknown", "value": 0.1, "confidence": 1.0},
                {"agent_name": "unknown2", "value": 0.9, "confidence": 1.0},
            ]
        )
        assert result["consensus_value"] == pytest.approx(0.5)

    def test_empty_numeric_ballots_graceful_defaults(self):
        svc = MultiAgentVotingEngine()
        result = svc.resolve_numeric(ballots=[])
        assert result == {
            "consensus_value": 0.0,
            "std_deviation": 0.0,
            "min_value": 0.0,
            "max_value": 0.0,
        }

    def test_min_and_max_values_reported(self):
        svc = MultiAgentVotingEngine()
        result = svc.resolve_numeric(
            ballots=[
                {"agent_name": "a", "value": 0.2, "confidence": 1.0},
                {"agent_name": "b", "value": 0.8, "confidence": 1.0},
            ]
        )
        assert result["min_value"] == 0.2
        assert result["max_value"] == 0.8

    def test_numeric_confidence_zero_excludes_ballot(self):
        svc = MultiAgentVotingEngine()
        result = svc.resolve_numeric(
            ballots=[
                {"agent_name": "credibility_agent", "value": 1.0, "confidence": 1.0},
                {"agent_name": "credibility_agent", "value": 0.0, "confidence": 0.0},
            ]
        )
        assert result["consensus_value"] == 1.0

    def test_numeric_confidence_clamped_over_one(self):
        svc = MultiAgentVotingEngine()
        result = svc.resolve_numeric(ballots=[{"agent_name": "credibility_agent", "value": 0.3, "confidence": 9.0}])
        assert result["consensus_value"] == 0.3

    def test_numeric_values_can_be_negative(self):
        svc = MultiAgentVotingEngine()
        result = svc.resolve_numeric(
            ballots=[
                {"agent_name": "a", "value": -1.0, "confidence": 1.0},
                {"agent_name": "b", "value": 1.0, "confidence": 1.0},
            ]
        )
        assert result["min_value"] == -1.0


class TestContested:
    def test_is_contested_true_for_margin_point_one(self):
        svc = MultiAgentVotingEngine()
        assert svc.is_contested(margin=0.1) is True

    def test_is_contested_false_for_margin_point_five(self):
        svc = MultiAgentVotingEngine()
        assert svc.is_contested(margin=0.5) is False

    def test_is_contested_boundary_point_two_false(self):
        svc = MultiAgentVotingEngine()
        assert svc.is_contested(margin=0.2) is False


class TestSanity:
    def test_agent_weights_include_default(self):
        assert "default" in MultiAgentVotingEngine.AGENT_WEIGHTS

    def test_margin_in_range(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(ballots=[{"agent_name": "a", "verdict": True, "confidence": 1.0}])
        assert 0.0 <= result["margin"] <= 1.0

    def test_weighted_values_non_negative(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(ballots=[{"agent_name": "a", "verdict": False, "confidence": 1.0}])
        assert result["weighted_true"] >= 0.0
        assert result["weighted_false"] >= 0.0

    def test_consensus_false_when_false_weight_greater(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(
            ballots=[
                {"agent_name": "credibility_agent", "verdict": False, "confidence": 1.0},
                {"agent_name": "hypothesis_service", "verdict": True, "confidence": 0.1},
            ]
        )
        assert result["consensus"] is False

    def test_vote_rounding_fields_present(self):
        svc = MultiAgentVotingEngine()
        result = svc.vote(ballots=[{"agent_name": "a", "verdict": True, "confidence": 0.333333333}])
        assert isinstance(result["weighted_true"], float)
        assert isinstance(result["margin"], float)

    def test_numeric_std_deviation_positive_for_spread_values(self):
        svc = MultiAgentVotingEngine()
        result = svc.resolve_numeric(
            ballots=[
                {"agent_name": "a", "value": 0.0, "confidence": 1.0},
                {"agent_name": "b", "value": 1.0, "confidence": 1.0},
            ]
        )
        assert result["std_deviation"] > 0.0
