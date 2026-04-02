from __future__ import annotations

from app.services.confidence_ensemble_service import ConfidenceEnsembleService


class TestEnsemble:
    def test_all_half_is_half(self):
        svc = ConfidenceEnsembleService()
        score = svc.ensemble(signals={
            "credibility": 0.5,
            "anomaly": 0.5,
            "uncertainty": 0.5,
            "hypothesis": 0.5,
            "calibration": 0.5,
        })
        assert score == 0.5

    def test_credibility_one_increases_score(self):
        svc = ConfidenceEnsembleService()
        baseline = svc.ensemble(signals={})
        score = svc.ensemble(signals={"credibility": 1.0})
        assert score > baseline

    def test_anomaly_one_contributes_zero(self):
        svc = ConfidenceEnsembleService()
        score = svc.ensemble(signals={"anomaly": 1.0, "credibility": 0.5, "uncertainty": 0.5, "hypothesis": 0.5, "calibration": 0.5})
        assert score == 0.4

    def test_uncertainty_one_contributes_zero(self):
        svc = ConfidenceEnsembleService()
        score = svc.ensemble(signals={"uncertainty": 1.0, "credibility": 0.5, "anomaly": 0.5, "hypothesis": 0.5, "calibration": 0.5})
        assert score == 0.375

    def test_missing_signal_defaults_half(self):
        svc = ConfidenceEnsembleService()
        score_partial = svc.ensemble(signals={"credibility": 0.9})
        score_explicit = svc.ensemble(signals={
            "credibility": 0.9,
            "anomaly": 0.5,
            "uncertainty": 0.5,
            "hypothesis": 0.5,
            "calibration": 0.5,
        })
        assert score_partial == score_explicit

    def test_score_clamped_low(self):
        svc = ConfidenceEnsembleService()
        score = svc.ensemble(signals={"credibility": -2, "anomaly": 2, "uncertainty": 2, "hypothesis": -1, "calibration": -3})
        assert 0.0 <= score <= 1.0

    def test_score_clamped_high(self):
        svc = ConfidenceEnsembleService()
        score = svc.ensemble(signals={"credibility": 3, "anomaly": -1, "uncertainty": -5, "hypothesis": 10, "calibration": 3})
        assert 0.0 <= score <= 1.0

    def test_rounds_to_4_decimals(self):
        svc = ConfidenceEnsembleService()
        score = svc.ensemble(signals={"credibility": 0.123456})
        assert round(score, 4) == score

    def test_anomaly_zero_is_good(self):
        svc = ConfidenceEnsembleService()
        high = svc.ensemble(signals={"anomaly": 0.0})
        low = svc.ensemble(signals={"anomaly": 1.0})
        assert high > low

    def test_uncertainty_zero_is_good(self):
        svc = ConfidenceEnsembleService()
        high = svc.ensemble(signals={"uncertainty": 0.0})
        low = svc.ensemble(signals={"uncertainty": 1.0})
        assert high > low


class TestLabel:
    def test_label_high_confidence(self):
        svc = ConfidenceEnsembleService()
        assert svc.ensemble_label(0.85) == "high_confidence"

    def test_label_moderate_confidence(self):
        svc = ConfidenceEnsembleService()
        assert svc.ensemble_label(0.65) == "moderate_confidence"

    def test_label_low_confidence(self):
        svc = ConfidenceEnsembleService()
        assert svc.ensemble_label(0.45) == "low_confidence"

    def test_label_unreliable(self):
        svc = ConfidenceEnsembleService()
        assert svc.ensemble_label(0.35) == "unreliable"

    def test_label_boundaries(self):
        svc = ConfidenceEnsembleService()
        assert svc.ensemble_label(0.8) == "high_confidence"
        assert svc.ensemble_label(0.6) == "moderate_confidence"
        assert svc.ensemble_label(0.4) == "low_confidence"


class TestMissingSignalImpact:
    def test_credibility_weight(self):
        svc = ConfidenceEnsembleService()
        assert svc.missing_signal_impact(missing_key="credibility") == 0.30

    def test_unknown_missing_key_zero(self):
        svc = ConfidenceEnsembleService()
        assert svc.missing_signal_impact(missing_key="does_not_exist") == 0.0

    def test_case_insensitive(self):
        svc = ConfidenceEnsembleService()
        assert svc.missing_signal_impact(missing_key="CrEdIbIlItY") == 0.30


class TestDominantSignal:
    def test_dominant_signal_returns_highest_weighted(self):
        svc = ConfidenceEnsembleService()
        dominant = svc.dominant_signal(signals={
            "credibility": 1.0,
            "anomaly": 0.0,
            "uncertainty": 0.0,
            "hypothesis": 0.0,
            "calibration": 0.0,
        })
        assert dominant == "credibility"

    def test_dominant_with_inversions(self):
        svc = ConfidenceEnsembleService()
        dominant = svc.dominant_signal(signals={
            "credibility": 0.4,
            "anomaly": 0.9,
            "uncertainty": 0.1,
            "hypothesis": 0.4,
            "calibration": 0.4,
        })
        assert dominant == "uncertainty"

    def test_dominant_defaults_to_credibility_on_all_missing(self):
        svc = ConfidenceEnsembleService()
        assert svc.dominant_signal(signals={}) == "credibility"


class TestWeightsAndBehavior:
    def test_weights_sum_to_one(self):
        assert round(sum(ConfidenceEnsembleService.WEIGHTS.values()), 6) == 1.0

    def test_ensemble_more_credible_than_anomalous_case(self):
        svc = ConfidenceEnsembleService()
        s1 = svc.ensemble(signals={"credibility": 0.9, "anomaly": 0.1, "uncertainty": 0.2})
        s2 = svc.ensemble(signals={"credibility": 0.1, "anomaly": 0.9, "uncertainty": 0.8})
        assert s1 > s2

    def test_hypothesis_direct_contribution(self):
        svc = ConfidenceEnsembleService()
        low = svc.ensemble(signals={"hypothesis": 0.0})
        high = svc.ensemble(signals={"hypothesis": 1.0})
        assert high > low

    def test_calibration_direct_contribution(self):
        svc = ConfidenceEnsembleService()
        low = svc.ensemble(signals={"calibration": 0.0})
        high = svc.ensemble(signals={"calibration": 1.0})
        assert high > low
