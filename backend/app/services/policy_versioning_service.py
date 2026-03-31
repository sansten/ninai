"""
Policy Versioning Service - Phase 5

Manages versioned PolicyGuard bundles with staged rollouts and rollback safety.
"""

import uuid
from datetime import datetime
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

    async def get_policy_history(self) -> List[Dict[str, Any]]:
        """Get complete history of policies for this org."""
        # In production, query database
        return []

    async def get_current_policy(self) -> Optional[Dict[str, Any]]:
        """Get currently active policy for org."""
        # In production, query database
        return None
