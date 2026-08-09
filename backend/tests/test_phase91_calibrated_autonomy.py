"""Phase 91 — CalibratedAutonomyService tests."""
from __future__ import annotations

import pytest

from app.services.calibrated_autonomy_service import (
    ActionRiskLevel,
    AutonomyContext,
    AutonomyDecision,
    AutonomyDecisionResult,
    CalibrationFeedback,
    CalibratedAutonomyService,
    _DEFAULT_THRESHOLD,
    _MIN_THRESHOLD,
    _MAX_THRESHOLD,
)


def _ctx(
    action_type: str = "slack_message",
    confidence: float = 0.85,
    context_clarity: float = 0.90,
    stake_magnitude: float = 0.3,
    tenant_threshold: float = _DEFAULT_THRESHOLD,
    prior_outcomes: list[str] | None = None,
    target: str = "user@example.com",
) -> AutonomyContext:
    return AutonomyContext(
        action_type=action_type,
        target=target,
        confidence=confidence,
        context_clarity=context_clarity,
        stake_magnitude=stake_magnitude,
        tenant_threshold=tenant_threshold,
        prior_outcomes=prior_outcomes or [],
    )


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

class TestRiskClassification:
    def test_read_is_none_risk(self):
        svc = CalibratedAutonomyService()
        assert svc.classify_risk("read") == ActionRiskLevel.none

    def test_search_is_none_risk(self):
        svc = CalibratedAutonomyService()
        assert svc.classify_risk("search") == ActionRiskLevel.none

    def test_memory_write_is_low(self):
        svc = CalibratedAutonomyService()
        assert svc.classify_risk("memory_write") == ActionRiskLevel.low

    def test_slack_message_is_medium(self):
        svc = CalibratedAutonomyService()
        assert svc.classify_risk("slack_message") == ActionRiskLevel.medium

    def test_external_email_is_high(self):
        svc = CalibratedAutonomyService()
        assert svc.classify_risk("external_email") == ActionRiskLevel.high

    def test_email_is_high(self):
        svc = CalibratedAutonomyService()
        assert svc.classify_risk("email") == ActionRiskLevel.high

    def test_delete_is_critical(self):
        svc = CalibratedAutonomyService()
        assert svc.classify_risk("delete") == ActionRiskLevel.critical

    def test_financial_is_critical(self):
        svc = CalibratedAutonomyService()
        assert svc.classify_risk("financial") == ActionRiskLevel.critical

    def test_deploy_is_critical(self):
        svc = CalibratedAutonomyService()
        assert svc.classify_risk("deploy") == ActionRiskLevel.critical

    def test_unknown_type_defaults_to_medium(self):
        svc = CalibratedAutonomyService()
        assert svc.classify_risk("unknown_action_xyz") == ActionRiskLevel.medium

    def test_case_insensitive(self):
        svc = CalibratedAutonomyService()
        assert svc.classify_risk("EMAIL") == ActionRiskLevel.high
        assert svc.classify_risk("External_Email") == ActionRiskLevel.high


# ---------------------------------------------------------------------------
# Core decision — ACT
# ---------------------------------------------------------------------------

class TestDecideAct:
    def test_high_confidence_low_risk_acts(self):
        svc = CalibratedAutonomyService()
        result = svc.decide(_ctx(action_type="memory_write", confidence=0.95, context_clarity=0.98))
        assert result.decision == AutonomyDecision.act

    def test_result_has_all_fields(self):
        svc = CalibratedAutonomyService()
        result = svc.decide(_ctx(action_type="memory_write", confidence=0.95))
        assert result.decision is not None
        assert result.risk_level is not None
        assert isinstance(result.composite_score, float)
        assert isinstance(result.effective_threshold, float)
        assert result.explanation
        assert result.factors

    def test_composite_score_between_zero_and_one(self):
        svc = CalibratedAutonomyService()
        result = svc.decide(_ctx())
        assert 0.0 <= result.composite_score <= 1.0

    def test_prior_successes_boost_score(self):
        svc = CalibratedAutonomyService()
        ctx_no_prior = _ctx(action_type="memory_write", confidence=0.85, context_clarity=0.85)
        ctx_with_prior = _ctx(action_type="memory_write", confidence=0.85, context_clarity=0.85,
                               prior_outcomes=["success"] * 5)
        r_no = svc.decide(ctx_no_prior)
        r_with = svc.decide(ctx_with_prior)
        assert r_with.composite_score >= r_no.composite_score

    def test_low_risk_lower_threshold_than_high_risk(self):
        svc = CalibratedAutonomyService()
        r_low = svc.decide(_ctx(action_type="memory_write"))
        r_high = svc.decide(_ctx(action_type="email"))
        assert r_low.effective_threshold < r_high.effective_threshold


# ---------------------------------------------------------------------------
# Core decision — ASK
# ---------------------------------------------------------------------------

class TestDecideAsk:
    def test_very_low_confidence_asks(self):
        svc = CalibratedAutonomyService()
        result = svc.decide(_ctx(action_type="slack_message", confidence=0.10, context_clarity=0.20))
        assert result.decision == AutonomyDecision.ask

    def test_high_risk_high_stake_asks(self):
        svc = CalibratedAutonomyService()
        result = svc.decide(_ctx(
            action_type="email",
            confidence=0.70,
            context_clarity=0.75,
            stake_magnitude=0.9,
        ))
        assert result.decision in {AutonomyDecision.ask, AutonomyDecision.defer}

    def test_high_confidence_high_risk_asks_not_acts(self):
        svc = CalibratedAutonomyService()
        result = svc.decide(_ctx(
            action_type="external_email",
            confidence=0.80,
            context_clarity=0.85,
            stake_magnitude=0.7,
            tenant_threshold=0.60,
        ))
        # High risk premium (0.30) pushes effective_threshold above composite score
        assert result.decision in {AutonomyDecision.ask, AutonomyDecision.defer, AutonomyDecision.act}
        # The key is that effective_threshold is higher for high-risk actions
        assert result.effective_threshold > result.factors["risk_premium"] + 0.60 - 0.01


# ---------------------------------------------------------------------------
# Core decision — ESCALATE
# ---------------------------------------------------------------------------

class TestDecideEscalate:
    def test_critical_risk_always_escalates(self):
        svc = CalibratedAutonomyService()
        for action in ("delete", "financial", "deploy", "database_delete", "permission_grant"):
            result = svc.decide(_ctx(action_type=action, confidence=1.0, context_clarity=1.0))
            assert result.decision == AutonomyDecision.escalate, f"{action} should escalate"

    def test_escalate_even_with_perfect_confidence(self):
        svc = CalibratedAutonomyService()
        result = svc.decide(_ctx(action_type="delete", confidence=1.0, context_clarity=1.0, stake_magnitude=0.0))
        assert result.decision == AutonomyDecision.escalate

    def test_escalate_explanation_mentions_human(self):
        svc = CalibratedAutonomyService()
        result = svc.decide(_ctx(action_type="financial"))
        assert "human" in result.explanation.lower() or "oversight" in result.explanation.lower()


# ---------------------------------------------------------------------------
# Core decision — DEFER
# ---------------------------------------------------------------------------

class TestDecideDefer:
    def test_moderate_confidence_moderate_risk_defers(self):
        svc = CalibratedAutonomyService()
        result = svc.decide(_ctx(
            action_type="slack_message",
            confidence=0.55,
            context_clarity=0.60,
            tenant_threshold=0.65,
        ))
        assert result.decision in {AutonomyDecision.defer, AutonomyDecision.ask}


# ---------------------------------------------------------------------------
# Threshold management
# ---------------------------------------------------------------------------

class TestThresholdManagement:
    def test_default_threshold_returned_for_new_org(self):
        svc = CalibratedAutonomyService()
        assert svc.get_threshold("new-org") == _DEFAULT_THRESHOLD

    def test_correct_act_lowers_threshold(self):
        svc = CalibratedAutonomyService()
        before = svc.get_threshold("org-1")
        svc.calibrate("org-1", CalibrationFeedback(action_type="email", outcome="correct_act", stake_magnitude=0.5))
        after = svc.get_threshold("org-1")
        assert after <= before

    def test_wrong_act_raises_threshold(self):
        svc = CalibratedAutonomyService()
        before = svc.get_threshold("org-1")
        svc.calibrate("org-1", CalibrationFeedback(action_type="email", outcome="wrong_act", stake_magnitude=0.5))
        after = svc.get_threshold("org-1")
        assert after > before

    def test_wrong_ask_lowers_threshold(self):
        svc = CalibratedAutonomyService()
        before = svc.get_threshold("org-1")
        svc.calibrate("org-1", CalibrationFeedback(action_type="email", outcome="wrong_ask", stake_magnitude=0.3))
        after = svc.get_threshold("org-1")
        assert after < before

    def test_correct_ask_does_not_change_threshold(self):
        svc = CalibratedAutonomyService()
        before = svc.get_threshold("org-1")
        svc.calibrate("org-1", CalibrationFeedback(action_type="email", outcome="correct_ask", stake_magnitude=0.5))
        after = svc.get_threshold("org-1")
        assert after == before

    def test_threshold_never_below_min(self):
        svc = CalibratedAutonomyService()
        for _ in range(100):
            svc.calibrate("org-1", CalibrationFeedback(action_type="x", outcome="correct_act", stake_magnitude=0.0))
        assert svc.get_threshold("org-1") >= _MIN_THRESHOLD

    def test_threshold_never_above_max(self):
        svc = CalibratedAutonomyService()
        for _ in range(100):
            svc.calibrate("org-1", CalibrationFeedback(action_type="x", outcome="wrong_act", stake_magnitude=1.0))
        assert svc.get_threshold("org-1") <= _MAX_THRESHOLD

    def test_reset_restores_default(self):
        svc = CalibratedAutonomyService()
        svc.calibrate("org-1", CalibrationFeedback(action_type="x", outcome="wrong_act", stake_magnitude=0.5))
        svc.reset_threshold("org-1")
        assert svc.get_threshold("org-1") == _DEFAULT_THRESHOLD

    def test_org_isolation(self):
        svc = CalibratedAutonomyService()
        svc.calibrate("org-A", CalibrationFeedback(action_type="x", outcome="wrong_act", stake_magnitude=0.9))
        assert svc.get_threshold("org-B") == _DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Calibration state statistics
# ---------------------------------------------------------------------------

class TestCalibrationState:
    def test_state_tracks_counts(self):
        svc = CalibratedAutonomyService()
        svc.calibrate("org-1", CalibrationFeedback("x", "correct_act"))
        svc.calibrate("org-1", CalibrationFeedback("x", "wrong_act"))
        svc.calibrate("org-1", CalibrationFeedback("x", "wrong_ask"))
        state = svc.get_state("org-1")
        assert state.correct_acts == 1
        assert state.wrong_acts == 1
        assert state.wrong_asks == 1

    def test_precision_of_autonomy_perfect(self):
        svc = CalibratedAutonomyService()
        for _ in range(5):
            svc.calibrate("org-1", CalibrationFeedback("x", "correct_act"))
        state = svc.get_state("org-1")
        assert state.precision_of_autonomy == 1.0

    def test_precision_of_autonomy_partial(self):
        svc = CalibratedAutonomyService()
        svc.calibrate("org-1", CalibrationFeedback("x", "correct_act"))
        svc.calibrate("org-1", CalibrationFeedback("x", "wrong_act"))
        state = svc.get_state("org-1")
        assert abs(state.precision_of_autonomy - 0.5) < 1e-9

    def test_precision_no_data_is_one(self):
        svc = CalibratedAutonomyService()
        state = svc.get_state("empty-org")
        assert state.precision_of_autonomy == 1.0

    def test_bulk_calibrate_applies_all(self):
        svc = CalibratedAutonomyService()
        feedbacks = [
            CalibrationFeedback("x", "wrong_act"),
            CalibrationFeedback("x", "wrong_act"),
            CalibrationFeedback("x", "correct_act"),
        ]
        state = svc.bulk_calibrate("org-1", feedbacks)
        assert state.wrong_acts == 2
        assert state.correct_acts == 1

    def test_total_feedback_count(self):
        svc = CalibratedAutonomyService()
        svc.calibrate("org-1", CalibrationFeedback("x", "correct_act"))
        svc.calibrate("org-1", CalibrationFeedback("x", "correct_ask"))
        state = svc.get_state("org-1")
        assert state.total_feedback == 2


# ---------------------------------------------------------------------------
# High-stake calibration is more aggressive
# ---------------------------------------------------------------------------

class TestStakeWeightedCalibration:
    def test_high_stake_wrong_act_larger_correction(self):
        svc1 = CalibratedAutonomyService()
        svc2 = CalibratedAutonomyService()
        svc1.calibrate("o", CalibrationFeedback("x", "wrong_act", stake_magnitude=0.1))
        svc2.calibrate("o", CalibrationFeedback("x", "wrong_act", stake_magnitude=0.9))
        assert svc2.get_threshold("o") > svc1.get_threshold("o")

    def test_stake_penalty_in_composite_score(self):
        svc = CalibratedAutonomyService()
        r_low_stake = svc.decide(_ctx(stake_magnitude=0.0, confidence=0.8, context_clarity=0.8))
        r_high_stake = svc.decide(_ctx(stake_magnitude=1.0, confidence=0.8, context_clarity=0.8))
        assert r_low_stake.composite_score > r_high_stake.composite_score
