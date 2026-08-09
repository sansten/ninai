"""Calibrated Autonomy Service — Phase 91.

Solves the "when to act vs. when to ask" problem by combining four signals
into a globally calibrated decision:

  1. confidence      — from ConfidenceEnsemble / local agent score
  2. action_risk     — irreversibility and blast radius of the action type
  3. context_clarity — is the user's intent unambiguous?
  4. stake_magnitude — how much does a wrong outcome hurt?

The composite score is compared against a per-tenant threshold that adapts
from outcome feedback: right-to-act outcomes lower the threshold; wrong-to-act
or unnecessary-ask outcomes move it in the corrective direction.

Decision output:
  ACT       — proceed autonomously
  ASK       — request explicit confirmation before proceeding
  DEFER     — gather more context first (confidence too low to decide either way)
  ESCALATE  — critical-risk action; always requires human oversight

Calibration converges to a threshold that minimises the weighted sum of:
  - false autonomy (acted without asking; outcome bad)
  - unnecessary caution (asked; user said "just do it")

Usage::

    svc = CalibratedAutonomyService()
    ctx = AutonomyContext(
        action_type="external_email",
        target="customer@acme.com",
        confidence=0.82,
        context_clarity=0.90,
        stake_magnitude=0.6,
        tenant_threshold=svc.get_threshold("org-1"),
        prior_outcomes=[],
    )
    result = svc.decide(ctx)
    # result.decision == AutonomyDecision.ask  (high risk → threshold not met)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionRiskLevel(str, Enum):
    none = "none"        # read-only; zero side effects
    low = "low"          # internal writes; fully reversible
    medium = "medium"    # internal notifications; hard to unsend
    high = "high"        # external communications; irreversible
    critical = "critical"  # financial, legal, or destructive; never autonomous


class AutonomyDecision(str, Enum):
    act = "act"
    ask = "ask"
    defer = "defer"
    escalate = "escalate"


# Risk premium added to tenant threshold per risk level.
# critical is set to ∞ (always escalate) via the special-case logic.
_RISK_PREMIUM: dict[ActionRiskLevel, float] = {
    ActionRiskLevel.none:     0.00,
    ActionRiskLevel.low:      0.05,
    ActionRiskLevel.medium:   0.15,
    ActionRiskLevel.high:     0.30,
    ActionRiskLevel.critical: 9.99,  # unreachable → always escalate
}

# Default risk level for action types not in the map.
_ACTION_RISK_MAP: dict[str, ActionRiskLevel] = {
    "read":            ActionRiskLevel.none,
    "search":          ActionRiskLevel.none,
    "memory_read":     ActionRiskLevel.none,
    "memory_write":    ActionRiskLevel.low,
    "internal_note":   ActionRiskLevel.low,
    "tag":             ActionRiskLevel.low,
    "playbook_run":    ActionRiskLevel.low,
    "jira_ticket":     ActionRiskLevel.medium,
    "jira_comment":    ActionRiskLevel.medium,
    "slack_message":   ActionRiskLevel.medium,
    "teams_message":   ActionRiskLevel.medium,
    "webhook":         ActionRiskLevel.medium,
    "github_issue":    ActionRiskLevel.high,
    "email":           ActionRiskLevel.high,
    "external_email":  ActionRiskLevel.high,
    "notion_page":     ActionRiskLevel.high,
    "database_write":  ActionRiskLevel.high,
    "api_post":        ActionRiskLevel.high,
    "financial":       ActionRiskLevel.critical,
    "delete":          ActionRiskLevel.critical,
    "database_delete": ActionRiskLevel.critical,
    "deploy":          ActionRiskLevel.critical,
    "permission_grant": ActionRiskLevel.critical,
}

# Calibration constants
_ALPHA = 0.10      # learning rate for threshold updates
_MIN_THRESHOLD = 0.30
_MAX_THRESHOLD = 0.95
_DEFAULT_THRESHOLD = 0.65

# Deferred band: composite score is in [threshold * DEFER_LO, threshold)
_DEFER_LO = 0.55


@dataclass
class AutonomyContext:
    action_type: str
    target: str
    confidence: float         # 0–1 from model/ensemble
    context_clarity: float    # 0–1: how unambiguous the user intent is
    stake_magnitude: float    # 0–1: severity of a wrong outcome
    tenant_threshold: float   # calibrated per-tenant decision threshold
    prior_outcomes: list[str] = field(default_factory=list)   # recent action outcomes
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AutonomyDecisionResult:
    decision: AutonomyDecision
    risk_level: ActionRiskLevel
    composite_score: float       # combined signal 0–1
    effective_threshold: float   # tenant_threshold + risk_premium
    explanation: str
    factors: dict[str, float]    # individual signal values for transparency


@dataclass
class CalibrationFeedback:
    action_type: str
    outcome: str   # "correct_act" | "wrong_act" | "correct_ask" | "wrong_ask"
    stake_magnitude: float = 0.5


@dataclass
class CalibrationState:
    org_id: str
    threshold: float = _DEFAULT_THRESHOLD
    correct_acts: int = 0
    wrong_acts: int = 0
    correct_asks: int = 0
    wrong_asks: int = 0  # user said "just do it" — we were over-cautious

    @property
    def total_feedback(self) -> int:
        return self.correct_acts + self.wrong_acts + self.correct_asks + self.wrong_asks

    @property
    def precision_of_autonomy(self) -> float:
        """Fraction of autonomous actions that were correct."""
        denom = self.correct_acts + self.wrong_acts
        return self.correct_acts / denom if denom > 0 else 1.0

    @property
    def recall_of_autonomy(self) -> float:
        """Fraction of cases where acting would have been correct, but we asked."""
        denom = self.correct_acts + self.wrong_asks
        return self.correct_acts / denom if denom > 0 else 0.0


class CalibratedAutonomyService:
    """Multi-factor act-or-ask decision with per-tenant feedback calibration.

    Thread-safe for reads; calibrate() mutates _states and should be called
    from a single writer (the nightly learning beat or the outcome handler).
    """

    def __init__(self) -> None:
        # org_id → CalibrationState
        self._states: dict[str, CalibrationState] = {}

    # ------------------------------------------------------------------
    # Risk classification
    # ------------------------------------------------------------------

    def classify_risk(self, action_type: str) -> ActionRiskLevel:
        """Map an action type string to a risk level."""
        key = str(action_type or "").lower().strip()
        return _ACTION_RISK_MAP.get(key, ActionRiskLevel.medium)

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def decide(self, ctx: AutonomyContext) -> AutonomyDecisionResult:
        """Produce an act-or-ask decision for the given action context.

        The composite score is:

            score = confidence × context_clarity × stake_penalty

        where stake_penalty = 1.0 − 0.25 × stake_magnitude  (high stake → be cautious)

        The effective threshold adds a risk premium on top of the tenant threshold.
        A critical-risk action always escalates regardless of score.
        """
        risk_level = self.classify_risk(ctx.action_type)

        # Critical actions always escalate
        if risk_level == ActionRiskLevel.critical:
            return AutonomyDecisionResult(
                decision=AutonomyDecision.escalate,
                risk_level=risk_level,
                composite_score=0.0,
                effective_threshold=ctx.tenant_threshold,
                explanation="Critical-risk action requires human oversight.",
                factors={
                    "confidence": ctx.confidence,
                    "context_clarity": ctx.context_clarity,
                    "stake_magnitude": ctx.stake_magnitude,
                    "risk_premium": _RISK_PREMIUM[risk_level],
                },
            )

        stake_penalty = 1.0 - 0.25 * max(0.0, min(1.0, ctx.stake_magnitude))
        composite = (
            max(0.0, min(1.0, ctx.confidence))
            * max(0.0, min(1.0, ctx.context_clarity))
            * stake_penalty
        )

        premium = _RISK_PREMIUM[risk_level]
        effective_threshold = min(_MAX_THRESHOLD, ctx.tenant_threshold + premium)
        defer_band_lo = effective_threshold * _DEFER_LO

        # Boost from consistent prior successes (small positive signal)
        if ctx.prior_outcomes:
            recent = ctx.prior_outcomes[-5:]
            success_rate = sum(1 for o in recent if o in ("success", "completed", "correct")) / len(recent)
            composite = min(1.0, composite + 0.05 * success_rate)

        factors = {
            "confidence": ctx.confidence,
            "context_clarity": ctx.context_clarity,
            "stake_magnitude": ctx.stake_magnitude,
            "stake_penalty": stake_penalty,
            "risk_premium": premium,
            "composite_score": composite,
            "effective_threshold": effective_threshold,
        }

        if composite >= effective_threshold:
            decision = AutonomyDecision.act
            explanation = (
                f"Composite score {composite:.3f} ≥ threshold {effective_threshold:.3f} "
                f"(risk={risk_level.value}). Proceeding autonomously."
            )
        elif composite >= defer_band_lo:
            decision = AutonomyDecision.defer
            explanation = (
                f"Composite score {composite:.3f} is in the defer band "
                f"[{defer_band_lo:.3f}, {effective_threshold:.3f}). "
                "Gathering more context before deciding."
            )
        else:
            decision = AutonomyDecision.ask
            explanation = (
                f"Composite score {composite:.3f} < defer threshold {defer_band_lo:.3f} "
                f"(risk={risk_level.value}). Requesting user confirmation."
            )

        return AutonomyDecisionResult(
            decision=decision,
            risk_level=risk_level,
            composite_score=composite,
            effective_threshold=effective_threshold,
            explanation=explanation,
            factors=factors,
        )

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def get_threshold(self, org_id: str) -> float:
        """Return the current calibrated threshold for the given org."""
        return self._states.get(str(org_id), CalibrationState(org_id=str(org_id))).threshold

    def get_state(self, org_id: str) -> CalibrationState:
        return self._states.setdefault(
            str(org_id), CalibrationState(org_id=str(org_id))
        )

    def calibrate(self, org_id: str, feedback: CalibrationFeedback) -> CalibrationState:
        """Update the threshold based on one feedback event.

        Feedback types and their effect on the threshold:
          correct_act  — acted autonomously, good outcome  → lower threshold slightly
          wrong_act    — acted autonomously, bad outcome   → raise threshold significantly
          correct_ask  — asked, user provided critical info → keep threshold (asking was right)
          wrong_ask    — asked, user said "just do it"    → lower threshold moderately
        """
        state = self.get_state(org_id)
        outcome = str(feedback.outcome or "").lower()
        stake = max(0.0, min(1.0, feedback.stake_magnitude))

        if outcome == "correct_act":
            state.correct_acts += 1
            # Lower threshold gently — we were right to act
            delta = -_ALPHA * 0.5 * (1.0 - stake)  # less adjustment for high-stake correct acts
        elif outcome == "wrong_act":
            state.wrong_acts += 1
            # Raise threshold significantly — we acted when we shouldn't have
            delta = +_ALPHA * (1.0 + stake)  # larger correction for high-stake failures
        elif outcome == "correct_ask":
            state.correct_asks += 1
            # No adjustment — asking was right
            delta = 0.0
        elif outcome == "wrong_ask":
            state.wrong_asks += 1
            # Lower threshold — we were over-cautious
            delta = -_ALPHA * 0.7
        else:
            delta = 0.0

        state.threshold = max(_MIN_THRESHOLD, min(_MAX_THRESHOLD, state.threshold + delta))
        return state

    def reset_threshold(self, org_id: str) -> None:
        """Reset to the default threshold (e.g. for new tenants)."""
        if org_id in self._states:
            self._states[org_id].threshold = _DEFAULT_THRESHOLD

    # ------------------------------------------------------------------
    # Batch calibration
    # ------------------------------------------------------------------

    def bulk_calibrate(
        self, org_id: str, feedbacks: list[CalibrationFeedback]
    ) -> CalibrationState:
        """Apply a batch of feedback events sequentially."""
        for fb in feedbacks:
            self.calibrate(org_id, fb)
        return self.get_state(org_id)
