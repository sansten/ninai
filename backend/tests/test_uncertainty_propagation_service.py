from __future__ import annotations

from app.services.uncertainty_propagation_service import UncertaintyPropagationService


class TestPropagate:
    def test_zero_hops_zero_corroboration_returns_source(self):
        svc = UncertaintyPropagationService()
        out = svc.propagate(source_uncertainty=0.6, inference_hops=0, corroborating_evidence_count=0)
        assert out == 0.6

    def test_three_hops_attenuates_by_decay_power(self):
        svc = UncertaintyPropagationService()
        out = svc.propagate(source_uncertainty=1.0, inference_hops=3, corroborating_evidence_count=0)
        assert out == round(0.7 ** 3, 6)

    def test_corroboration_reduces_uncertainty(self):
        svc = UncertaintyPropagationService()
        without = svc.propagate(source_uncertainty=0.8, inference_hops=1, corroborating_evidence_count=0)
        with_evidence = svc.propagate(source_uncertainty=0.8, inference_hops=1, corroborating_evidence_count=3)
        assert with_evidence < without

    def test_min_propagated_floor_applied(self):
        svc = UncertaintyPropagationService()
        out = svc.propagate(source_uncertainty=0.01, inference_hops=10, corroborating_evidence_count=10)
        assert out == svc.MIN_PROPAGATED

    def test_source_uncertainty_clamped_high(self):
        svc = UncertaintyPropagationService()
        out = svc.propagate(source_uncertainty=2.0, inference_hops=0, corroborating_evidence_count=0)
        assert out == 1.0

    def test_source_uncertainty_clamped_low(self):
        svc = UncertaintyPropagationService()
        out = svc.propagate(source_uncertainty=-1.0, inference_hops=0, corroborating_evidence_count=0)
        assert out == svc.MIN_PROPAGATED

    def test_negative_hops_treated_as_zero(self):
        svc = UncertaintyPropagationService()
        out = svc.propagate(source_uncertainty=0.4, inference_hops=-2, corroborating_evidence_count=0)
        assert out == 0.4

    def test_negative_evidence_treated_as_zero(self):
        svc = UncertaintyPropagationService()
        out_neg = svc.propagate(source_uncertainty=0.4, inference_hops=1, corroborating_evidence_count=-3)
        out_zero = svc.propagate(source_uncertainty=0.4, inference_hops=1, corroborating_evidence_count=0)
        assert out_neg == out_zero

    def test_precision_rounded(self):
        svc = UncertaintyPropagationService()
        out = svc.propagate(source_uncertainty=0.123456789, inference_hops=2, corroborating_evidence_count=1)
        assert round(out, 6) == out


class TestPropagateChain:
    def test_chain_geometric_mean_two_sources(self):
        svc = UncertaintyPropagationService()
        sources = [
            {"memory_id": "m1", "uncertainty": 0.49},
            {"memory_id": "m2", "uncertainty": 0.81},
        ]
        out = svc.propagate_chain(sources=sources, chain_length=0)
        expected = round((0.49 * 0.81) ** 0.5, 6)
        assert out == expected

    def test_chain_applies_chain_length_as_hops(self):
        svc = UncertaintyPropagationService()
        sources = [{"memory_id": "m1", "uncertainty": 0.9}]
        out = svc.propagate_chain(sources=sources, chain_length=2)
        expected = svc.propagate(source_uncertainty=0.9, inference_hops=2, corroborating_evidence_count=0)
        assert out == expected

    def test_chain_uses_per_source_corroboration(self):
        svc = UncertaintyPropagationService()
        sources = [
            {"memory_id": "m1", "uncertainty": 0.8, "corroborating_evidence_count": 0},
            {"memory_id": "m2", "uncertainty": 0.8, "corroborating_evidence_count": 5},
        ]
        out = svc.propagate_chain(sources=sources, chain_length=1)
        base = svc.propagate(source_uncertainty=0.8, inference_hops=1, corroborating_evidence_count=0)
        assert out < base

    def test_chain_empty_sources_returns_min_floor(self):
        svc = UncertaintyPropagationService()
        assert svc.propagate_chain(sources=[], chain_length=2) == svc.MIN_PROPAGATED

    def test_chain_all_certain_sources_near_certain(self):
        svc = UncertaintyPropagationService()
        sources = [{"memory_id": f"m{i}", "uncertainty": 0.0} for i in range(4)]
        out = svc.propagate_chain(sources=sources, chain_length=0)
        assert out == svc.MIN_PROPAGATED

    def test_chain_one_highly_uncertain_source_above_point_one(self):
        svc = UncertaintyPropagationService()
        sources = [
            {"memory_id": "m1", "uncertainty": 0.0},
            {"memory_id": "m2", "uncertainty": 0.95},
        ]
        out = svc.propagate_chain(sources=sources, chain_length=0)
        assert out > 0.1

    def test_chain_negative_length_treated_as_zero(self):
        svc = UncertaintyPropagationService()
        sources = [{"memory_id": "m1", "uncertainty": 0.6}]
        out_neg = svc.propagate_chain(sources=sources, chain_length=-1)
        out_zero = svc.propagate_chain(sources=sources, chain_length=0)
        assert out_neg == out_zero

    def test_chain_missing_uncertainty_defaults_zero(self):
        svc = UncertaintyPropagationService()
        out = svc.propagate_chain(sources=[{"memory_id": "m1"}], chain_length=1)
        assert out == svc.MIN_PROPAGATED

    def test_chain_geometric_mean_three_sources(self):
        svc = UncertaintyPropagationService()
        vals = [0.2, 0.4, 0.8]
        sources = [{"memory_id": f"m{i}", "uncertainty": vals[i]} for i in range(3)]
        out = svc.propagate_chain(sources=sources, chain_length=0)
        expected = round((vals[0] * vals[1] * vals[2]) ** (1 / 3), 6)
        assert out == expected


class TestUncertaintyLabel:
    def test_label_certain(self):
        svc = UncertaintyPropagationService()
        assert svc.uncertainty_label(0.05) == "certain"

    def test_label_likely(self):
        svc = UncertaintyPropagationService()
        assert svc.uncertainty_label(0.2) == "likely"

    def test_label_uncertain(self):
        svc = UncertaintyPropagationService()
        assert svc.uncertainty_label(0.5) == "uncertain"

    def test_label_speculative(self):
        svc = UncertaintyPropagationService()
        assert svc.uncertainty_label(0.8) == "speculative"

    def test_label_boundaries(self):
        svc = UncertaintyPropagationService()
        assert svc.uncertainty_label(0.1) == "likely"
        assert svc.uncertainty_label(0.3) == "uncertain"
        assert svc.uncertainty_label(0.6) == "speculative"

    def test_label_clamps_negative(self):
        svc = UncertaintyPropagationService()
        assert svc.uncertainty_label(-2) == "certain"

    def test_label_clamps_above_one(self):
        svc = UncertaintyPropagationService()
        assert svc.uncertainty_label(3) == "speculative"


class TestShouldFlagForReview:
    def test_critical_flags_above_point_one(self):
        svc = UncertaintyPropagationService()
        assert svc.should_flag_for_review(propagated_uncertainty=0.11, decision_stakes="critical") is True

    def test_critical_not_flag_at_point_one(self):
        svc = UncertaintyPropagationService()
        assert svc.should_flag_for_review(propagated_uncertainty=0.1, decision_stakes="critical") is False

    def test_high_flags_above_point_two_five(self):
        svc = UncertaintyPropagationService()
        assert svc.should_flag_for_review(propagated_uncertainty=0.26, decision_stakes="high") is True

    def test_medium_flags_above_point_five(self):
        svc = UncertaintyPropagationService()
        assert svc.should_flag_for_review(propagated_uncertainty=0.51, decision_stakes="medium") is True

    def test_low_ignores_point_six(self):
        svc = UncertaintyPropagationService()
        assert svc.should_flag_for_review(propagated_uncertainty=0.6, decision_stakes="low") is False

    def test_low_flags_above_point_seven_five(self):
        svc = UncertaintyPropagationService()
        assert svc.should_flag_for_review(propagated_uncertainty=0.76, decision_stakes="low") is True

    def test_unknown_stakes_defaults_low_threshold(self):
        svc = UncertaintyPropagationService()
        assert svc.should_flag_for_review(propagated_uncertainty=0.8, decision_stakes="unknown") is True

    def test_flag_clamps_negative_uncertainty(self):
        svc = UncertaintyPropagationService()
        assert svc.should_flag_for_review(propagated_uncertainty=-1.0, decision_stakes="critical") is False

    def test_flag_clamps_high_uncertainty(self):
        svc = UncertaintyPropagationService()
        assert svc.should_flag_for_review(propagated_uncertainty=9.0, decision_stakes="critical") is True
