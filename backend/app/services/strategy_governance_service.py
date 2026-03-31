"""Strategy Governance Service - Phase 53 Slice 1.

Provides deterministic governance decisions for strategy promotion/demotion,
playbook efficacy checks, A/B routing, and convergence gating.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any


@dataclass(frozen=True)
class StrategyWindowMetrics:
    """Observed strategy performance over a recent evaluation window."""

    win_rate: float
    false_positive_rate: float
    sample_count: int
    drift_severity: str = "none"  # none | low | medium | high | critical


@dataclass(frozen=True)
class GovernanceDecision:
    """Promotion/demotion/revert decision with auditable rationale."""

    action: str  # promote | demote | hold | revert
    reason: str
    next_stage: str
    auto_revert: bool = False


@dataclass(frozen=True)
class PlaybookEfficacyDecision:
    """Governance decision for playbook lifecycle based on efficacy."""

    status: str  # promote | keep | demote
    reason: str


@dataclass(frozen=True)
class ConvergenceResult:
    """Result of A/B convergence evaluation."""

    converged: bool
    winner: str | None
    reason: str


class StrategyGovernanceService:
    """Policy engine for adaptive strategy governance controls."""

    MIN_SAMPLE_SIZE = 100
    MIN_CANARY_SAMPLE_SIZE = 50
    REQUIRED_WIN_RATE_UPLIFT = 0.10
    REQUIRED_FALSE_POSITIVE_REDUCTION = 0.20
    MAX_FALSE_POSITIVE_REGRESSION = 0.05

    MIN_PLAYBOOK_SAMPLE_SIZE = 30
    PLAYBOOK_PROMOTE_SUCCESS_RATE = 0.70
    PLAYBOOK_DEMOTE_SUCCESS_RATE = 0.40

    _STAGE_ORDER = {
        "experimental": 0,
        "canary": 1,
        "stable": 2,
    }

    _DRIFT_AUTO_REVERT = {"critical", "high"}

    def evaluate_strategy_transition(
        self,
        *,
        current_stage: str,
        baseline: StrategyWindowMetrics,
        candidate: StrategyWindowMetrics,
    ) -> GovernanceDecision:
        """Evaluate whether a strategy should be promoted/demoted/reverted."""
        stage = self._normalize_stage(current_stage)
        uplift = self._safe_ratio_delta(candidate.win_rate, baseline.win_rate)
        fp_reduction = self._safe_ratio_delta(
            baseline.false_positive_rate,
            candidate.false_positive_rate,
        )

        if candidate.drift_severity in self._DRIFT_AUTO_REVERT:
            return GovernanceDecision(
                action="revert",
                reason=f"drift severity {candidate.drift_severity} requires safe auto-revert",
                next_stage="stable",
                auto_revert=True,
            )

        if candidate.sample_count < self.MIN_SAMPLE_SIZE:
            return GovernanceDecision(
                action="hold",
                reason=f"insufficient sample size: {candidate.sample_count} < {self.MIN_SAMPLE_SIZE}",
                next_stage=stage,
            )

        if stage == "experimental":
            if candidate.sample_count >= self.MIN_CANARY_SAMPLE_SIZE and uplift >= 0.0:
                return GovernanceDecision(
                    action="promote",
                    reason="experimental strategy met canary entry requirements",
                    next_stage="canary",
                )
            return GovernanceDecision(
                action="hold",
                reason="experimental strategy has not met canary entry requirements",
                next_stage=stage,
            )

        if stage == "canary":
            if (
                uplift >= self.REQUIRED_WIN_RATE_UPLIFT
                and fp_reduction >= self.REQUIRED_FALSE_POSITIVE_REDUCTION
            ):
                return GovernanceDecision(
                    action="promote",
                    reason=(
                        f"candidate met promotion thresholds: uplift={uplift:.3f}, "
                        f"false_positive_reduction={fp_reduction:.3f}"
                    ),
                    next_stage="stable",
                )

            fp_regression = self._safe_ratio_delta(
                candidate.false_positive_rate,
                baseline.false_positive_rate,
            )
            if fp_regression > self.MAX_FALSE_POSITIVE_REGRESSION:
                return GovernanceDecision(
                    action="demote",
                    reason=f"false-positive regression {fp_regression:.3f} exceeds max allowed",
                    next_stage="experimental",
                )

            return GovernanceDecision(
                action="hold",
                reason="canary strategy inconclusive against promotion criteria",
                next_stage=stage,
            )

        # stable stage safety checks
        if uplift < -0.05:
            return GovernanceDecision(
                action="demote",
                reason=f"stable strategy win-rate degraded by {abs(uplift):.3f}",
                next_stage="canary",
            )

        return GovernanceDecision(
            action="hold",
            reason="stable strategy within guardrails",
            next_stage=stage,
        )

    def evaluate_playbook_efficacy(
        self,
        *,
        success_rate: float,
        sample_count: int,
    ) -> PlaybookEfficacyDecision:
        """Evaluate playbook promotion/demotion based on observed efficacy."""
        if sample_count < self.MIN_PLAYBOOK_SAMPLE_SIZE:
            return PlaybookEfficacyDecision(
                status="keep",
                reason=f"insufficient playbook samples: {sample_count} < {self.MIN_PLAYBOOK_SAMPLE_SIZE}",
            )

        if success_rate >= self.PLAYBOOK_PROMOTE_SUCCESS_RATE:
            return PlaybookEfficacyDecision(
                status="promote",
                reason=f"success_rate {success_rate:.3f} exceeds promote threshold",
            )

        if success_rate <= self.PLAYBOOK_DEMOTE_SUCCESS_RATE:
            return PlaybookEfficacyDecision(
                status="demote",
                reason=f"success_rate {success_rate:.3f} below demote threshold",
            )

        return PlaybookEfficacyDecision(
            status="keep",
            reason="playbook efficacy within neutral band",
        )

    def route_ab_strategy(
        self,
        *,
        org_id: str,
        subject_id: str,
        stable_strategy_id: str,
        candidate_strategy_id: str,
        candidate_rollout_pct: int,
    ) -> str:
        """Deterministically route traffic between stable/candidate strategies."""
        rollout = max(0, min(100, int(candidate_rollout_pct)))
        if rollout == 0:
            return stable_strategy_id
        if rollout == 100:
            return candidate_strategy_id

        bucket = self._bucket_0_99(org_id=org_id, subject_id=subject_id)
        if bucket < rollout:
            return candidate_strategy_id
        return stable_strategy_id

    def evaluate_ab_convergence(
        self,
        *,
        stable: StrategyWindowMetrics,
        candidate: StrategyWindowMetrics,
    ) -> ConvergenceResult:
        """Evaluate whether A/B testing has converged and identify winner."""
        min_samples = min(stable.sample_count, candidate.sample_count)
        if min_samples < self.MIN_SAMPLE_SIZE:
            return ConvergenceResult(
                converged=False,
                winner=None,
                reason="insufficient sample size for convergence",
            )

        uplift = self._safe_ratio_delta(candidate.win_rate, stable.win_rate)
        if uplift >= self.REQUIRED_WIN_RATE_UPLIFT:
            return ConvergenceResult(
                converged=True,
                winner="candidate",
                reason=f"candidate win-rate uplift {uplift:.3f} reached threshold",
            )
        if uplift <= -self.REQUIRED_WIN_RATE_UPLIFT:
            return ConvergenceResult(
                converged=True,
                winner="stable",
                reason=f"candidate underperformed stable by {abs(uplift):.3f}",
            )

        return ConvergenceResult(
            converged=False,
            winner=None,
            reason="insufficient separation between stable and candidate",
        )

    @staticmethod
    def build_audit_payload(
        *,
        org_id: str,
        strategy_id: str,
        stage: str,
        decision: GovernanceDecision,
        baseline: StrategyWindowMetrics,
        candidate: StrategyWindowMetrics,
    ) -> dict[str, Any]:
        """Build structured payload for governance audit logs."""
        return {
            "org_id": org_id,
            "strategy_id": strategy_id,
            "current_stage": stage,
            "action": decision.action,
            "next_stage": decision.next_stage,
            "auto_revert": decision.auto_revert,
            "reason": decision.reason,
            "baseline": {
                "win_rate": baseline.win_rate,
                "false_positive_rate": baseline.false_positive_rate,
                "sample_count": baseline.sample_count,
                "drift_severity": baseline.drift_severity,
            },
            "candidate": {
                "win_rate": candidate.win_rate,
                "false_positive_rate": candidate.false_positive_rate,
                "sample_count": candidate.sample_count,
                "drift_severity": candidate.drift_severity,
            },
        }

    def _normalize_stage(self, stage: str) -> str:
        s = str(stage or "").strip().lower()
        if s in self._STAGE_ORDER:
            return s
        return "experimental"

    @staticmethod
    def _safe_ratio_delta(current: float, baseline: float) -> float:
        if baseline <= 0:
            if current > 0:
                return 1.0
            return 0.0
        return (current - baseline) / baseline

    @staticmethod
    def _bucket_0_99(*, org_id: str, subject_id: str) -> int:
        raw = f"{org_id}|{subject_id}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 100
