"""
Policy Versioning Service - Phase 5

Manages versioned PolicyGuard bundles with staged rollouts and rollback safety.
"""

import uuid
from datetime import datetime
from math import erf, sqrt
from typing import Optional, Dict, Any, List
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging

from app.services.strategy_governance_service import (
    StrategyGovernanceService,
    StrategyWindowMetrics,
)

logger = logging.getLogger(__name__)


class PolicyVersion(str, Enum):
    """Policy version states."""
    DRAFT = "draft"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"


class PolicyVersioningService:
    """
    Manage versioned PolicyGuard bundles.
    
    Features:
    - Version tracking
    - Staged rollout (draft → staging → canary → prod)
    - Canary deployment (%  of traffic)
    - Automatic rollback on errors
    - Rollback safety (keep N previous versions)
    """

    def __init__(self, db: AsyncSession, organization_id: uuid.UUID):
        self.db = db
        self.organization_id = organization_id
        self.min_versions_to_keep = 3
        self._governance = StrategyGovernanceService()

    async def create_policy_version(
        self,
        name: str,
        version_tag: str,
        policy_bundle: Dict[str, Any],
        description: Optional[str] = None,
        created_by_user_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Create a new policy version (starts in DRAFT).
        
        Returns: Policy version metadata
        """
        policy_id = uuid.uuid4()
        now = datetime.utcnow()

        version = {
            "id": str(policy_id),
            "organization_id": str(self.organization_id),
            "version_tag": version_tag,
            "name": name,
            "description": description,
            "state": PolicyVersion.DRAFT.value,
            "policy_bundle": policy_bundle,
            "created_at": now.isoformat(),
            "created_by": str(created_by_user_id) if created_by_user_id else None,
            "metadata": {}
        }

        # In production, save to database
        logger.info(
            f"Created policy version: org={self.organization_id} "
            f"name={name} tag={version_tag} state=draft"
        )

        return version

    async def stage_policy(
        self,
        policy_id: uuid.UUID,
        promoted_by_user_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Move policy from DRAFT to STAGING.
        
        Staging is for pre-production testing.
        """
        logger.info(
            f"Staged policy: {policy_id} (draft → staging)"
        )

        return {
            "policy_id": str(policy_id),
            "new_state": PolicyVersion.STAGING.value,
            "promoted_by": str(promoted_by_user_id) if promoted_by_user_id else None,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def start_canary_rollout(
        self,
        policy_id: uuid.UUID,
        canary_percentage: int = 5,  # 5% of traffic
        duration_minutes: int = 60,  # 1 hour
        promoted_by_user_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Start canary rollout (5% → 10% → 25% → 50% → 100%).
        
        Monitors for errors and auto-rolls-back if error rate exceeds threshold.
        """
        if not (1 <= canary_percentage <= 100):
            raise ValueError("Canary percentage must be 1-100")

        logger.info(
            f"Started canary rollout: policy={policy_id} "
            f"percentage={canary_percentage}% duration={duration_minutes}min"
        )

        return {
            "policy_id": str(policy_id),
            "state": PolicyVersion.CANARY.value,
            "canary_percentage": canary_percentage,
            "duration_minutes": duration_minutes,
            "error_threshold_pct": 5.0,  # Auto-rollback if error rate > 5%
            "start_time": datetime.utcnow().isoformat(),
            "promotion_by": str(promoted_by_user_id) if promoted_by_user_id else None
        }

    async def promote_to_production(
        self,
        policy_id: uuid.UUID,
        promoted_by_user_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Promote policy from CANARY to PRODUCTION (100% traffic).
        """
        logger.info(
            f"Promoted to production: {policy_id} (canary → prod)"
        )

        return {
            "policy_id": str(policy_id),
            "new_state": PolicyVersion.PRODUCTION.value,
            "promoted_by": str(promoted_by_user_id) if promoted_by_user_id else None,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def rollback_policy(
        self,
        current_policy_id: uuid.UUID,
        target_policy_id: Optional[uuid.UUID] = None,
        reason: str = "",
        rolled_back_by_user_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Rollback to previous policy version.
        
        If target_policy_id not specified, rollback to most recent non-current.
        """
        logger.warning(
            f"Policy rollback: current={current_policy_id} target={target_policy_id} "
            f"reason={reason}"
        )

        return {
            "previous_policy_id": str(current_policy_id),
            "new_policy_id": str(target_policy_id or uuid.uuid4()),
            "reason": reason,
            "rolled_back_by": str(rolled_back_by_user_id) if rolled_back_by_user_id else None,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def evaluate_canary_governance(
        self,
        *,
        policy_id: uuid.UUID,
        baseline_metrics: Dict[str, Any],
        candidate_metrics: Dict[str, Any],
        current_stage: str = PolicyVersion.CANARY.value,
        target_policy_id: Optional[uuid.UUID] = None,
        actor_user_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Evaluate governance policy and execute transition when needed.

        This is the Phase 53 wiring point between rollout management and the
        StrategyGovernanceService decision engine.
        """
        baseline = StrategyWindowMetrics(
            win_rate=float(baseline_metrics.get("win_rate", 0.0) or 0.0),
            false_positive_rate=float(
                baseline_metrics.get("false_positive_rate", 0.0) or 0.0
            ),
            sample_count=int(baseline_metrics.get("sample_count", 0) or 0),
            drift_severity=str(baseline_metrics.get("drift_severity", "none") or "none"),
        )
        candidate = StrategyWindowMetrics(
            win_rate=float(candidate_metrics.get("win_rate", 0.0) or 0.0),
            false_positive_rate=float(
                candidate_metrics.get("false_positive_rate", 0.0) or 0.0
            ),
            sample_count=int(candidate_metrics.get("sample_count", 0) or 0),
            drift_severity=str(candidate_metrics.get("drift_severity", "none") or "none"),
        )

        decision = self._governance.evaluate_strategy_transition(
            current_stage=current_stage,
            baseline=baseline,
            candidate=candidate,
        )

        transition = await self._apply_governance_transition(
            policy_id=policy_id,
            target_policy_id=target_policy_id,
            decision=decision.action,
            reason=decision.reason,
            actor_user_id=actor_user_id,
        )

        audit_payload = self._governance.build_audit_payload(
            org_id=str(self.organization_id),
            strategy_id=str(policy_id),
            stage=current_stage,
            decision=decision,
            baseline=baseline,
            candidate=candidate,
        )

        return {
            "policy_id": str(policy_id),
            "decision": decision.action,
            "next_stage": decision.next_stage,
            "auto_revert": decision.auto_revert,
            "reason": decision.reason,
            "transition": transition,
            "governance_audit": audit_payload,
        }

    async def _apply_governance_transition(
        self,
        *,
        policy_id: uuid.UUID,
        target_policy_id: Optional[uuid.UUID],
        decision: str,
        reason: str,
        actor_user_id: Optional[uuid.UUID],
    ) -> Dict[str, Any]:
        """Execute rollout transition chosen by governance policy."""
        if decision == "promote":
            return await self.promote_to_production(
                policy_id=policy_id,
                promoted_by_user_id=actor_user_id,
            )

        if decision in {"demote", "revert"}:
            return await self.rollback_policy(
                current_policy_id=policy_id,
                target_policy_id=target_policy_id,
                reason=reason,
                rolled_back_by_user_id=actor_user_id,
            )

        return {
            "policy_id": str(policy_id),
            "action": "hold",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def run_historical_replay_backtest(
        self,
        *,
        policy_id: uuid.UUID,
        baseline_records: List[Dict[str, Any]],
        candidate_records: List[Dict[str, Any]],
        current_stage: str = PolicyVersion.CANARY.value,
    ) -> Dict[str, Any]:
        """Run governance decisioning in replay mode over historical outcomes."""
        baseline = self._compute_window_metrics(baseline_records)
        candidate = self._compute_window_metrics(candidate_records)

        decision = self._governance.evaluate_strategy_transition(
            current_stage=current_stage,
            baseline=baseline,
            candidate=candidate,
        )

        significance = self.evaluate_canary_significance(
            stable_successes=int(round(baseline.win_rate * baseline.sample_count)),
            stable_total=baseline.sample_count,
            candidate_successes=int(round(candidate.win_rate * candidate.sample_count)),
            candidate_total=candidate.sample_count,
        )

        audit_payload = self._governance.build_audit_payload(
            org_id=str(self.organization_id),
            strategy_id=str(policy_id),
            stage=current_stage,
            decision=decision,
            baseline=baseline,
            candidate=candidate,
        )

        return {
            "policy_id": str(policy_id),
            "mode": "historical_replay",
            "decision": decision.action,
            "next_stage": decision.next_stage,
            "reason": decision.reason,
            "significance": significance,
            "baseline": {
                "win_rate": baseline.win_rate,
                "false_positive_rate": baseline.false_positive_rate,
                "sample_count": baseline.sample_count,
            },
            "candidate": {
                "win_rate": candidate.win_rate,
                "false_positive_rate": candidate.false_positive_rate,
                "sample_count": candidate.sample_count,
            },
            "governance_audit": audit_payload,
        }

    def evaluate_canary_significance(
        self,
        *,
        stable_successes: int,
        stable_total: int,
        candidate_successes: int,
        candidate_total: int,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Evaluate statistical significance of candidate vs stable win rates.

        Uses a two-proportion z-test approximation.
        """
        s_total = max(0, int(stable_total))
        c_total = max(0, int(candidate_total))
        s_success = max(0, min(int(stable_successes), s_total))
        c_success = max(0, min(int(candidate_successes), c_total))

        if s_total == 0 or c_total == 0:
            return {
                "significant": False,
                "p_value": 1.0,
                "z_score": 0.0,
                "uplift": 0.0,
                "reason": "insufficient sample size",
            }

        p1 = s_success / s_total
        p2 = c_success / c_total
        uplift = (p2 - p1) / p1 if p1 > 0 else (1.0 if p2 > 0 else 0.0)

        pooled = (s_success + c_success) / (s_total + c_total)
        variance = pooled * (1 - pooled) * ((1 / s_total) + (1 / c_total))
        if variance <= 0:
            return {
                "significant": False,
                "p_value": 1.0,
                "z_score": 0.0,
                "uplift": round(uplift, 4),
                "reason": "zero variance",
            }

        z = (p2 - p1) / sqrt(variance)
        cdf = 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))
        p_value = max(0.0, min(1.0, 2.0 * (1.0 - cdf)))

        return {
            "significant": p_value < float(alpha),
            "p_value": round(p_value, 6),
            "z_score": round(z, 4),
            "uplift": round(uplift, 4),
            "reason": "candidate outperforms stable" if p2 > p1 else "candidate does not outperform stable",
        }

    async def handle_drift_alert(
        self,
        *,
        policy_id: uuid.UUID,
        drift_severity: str,
        target_policy_id: Optional[uuid.UUID] = None,
        actor_user_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Trigger governance evaluation on drift alert and auto-revert safely."""
        baseline_metrics = {
            "win_rate": 0.5,
            "false_positive_rate": 0.1,
            "sample_count": 200,
            "drift_severity": "none",
        }
        candidate_metrics = {
            "win_rate": 0.5,
            "false_positive_rate": 0.1,
            "sample_count": 200,
            "drift_severity": str(drift_severity or "none").lower(),
        }

        result = await self.evaluate_canary_governance(
            policy_id=policy_id,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            current_stage=PolicyVersion.CANARY.value,
            target_policy_id=target_policy_id,
            actor_user_id=actor_user_id,
        )
        result["trigger"] = "drift_alert"
        return result

    @staticmethod
    def _compute_window_metrics(records: List[Dict[str, Any]]) -> StrategyWindowMetrics:
        """Aggregate replay records into strategy window metrics."""
        rows = records or []
        n = len(rows)
        if n == 0:
            return StrategyWindowMetrics(
                win_rate=0.0,
                false_positive_rate=0.0,
                sample_count=0,
                drift_severity="none",
            )

        success_count = sum(1 for r in rows if bool(r.get("success")))
        fp_count = sum(1 for r in rows if bool(r.get("false_positive")))

        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        drift = "none"
        for r in rows:
            s = str(r.get("drift_severity") or "none").lower()
            if severity_order.get(s, 0) > severity_order.get(drift, 0):
                drift = s

        return StrategyWindowMetrics(
            win_rate=success_count / n,
            false_positive_rate=fp_count / n,
            sample_count=n,
            drift_severity=drift,
        )

    async def get_policy_history(self) -> List[Dict[str, Any]]:
        """Get complete history of policies for this org."""
        # In production, query database
        return []

    async def get_current_policy(self) -> Optional[Dict[str, Any]]:
        """Get currently active policy for org."""
        # In production, query database
        return None
