"""
Resource Accounting Service - Phase 4

Tracks and enforces resource budgets per organization:
- Tokens (LLM usage)
- Storage (memory store)
- Latency (SLO tracking)
- Cost (USD estimation)

Admission control rejects or degrades when budgets exceeded.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import logging

from app.models.organization import Organization
from app.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class ResourceBudget:
    """Resource budget configuration for an organization."""

    def __init__(
        self,
        organization_id: uuid.UUID,
        tokens_per_month: int = 1_000_000,  # 1M tokens
        storage_gb: int = 100,  # 100 GB
        cost_usd_per_month: float = 1000.0,  # $1000/month
        latency_slo_ms: int = 5000,  # 5 second P95
    ):
        self.organization_id = organization_id
        self.tokens_per_month = tokens_per_month
        self.storage_gb = storage_gb
        self.cost_usd_per_month = cost_usd_per_month
        self.latency_slo_ms = latency_slo_ms

    def tokens_remaining(self, tokens_used: int) -> int:
        return max(0, self.tokens_per_month - tokens_used)

    def storage_remaining_gb(self, storage_used_gb: float) -> float:
        return max(0.0, self.storage_gb - storage_used_gb)

    def cost_remaining(self, cost_used: float) -> float:
        return max(0.0, self.cost_usd_per_month - cost_used)

    def tokens_exceeded(self, tokens_used: int) -> bool:
        return tokens_used > self.tokens_per_month

    def storage_exceeded(self, storage_used_gb: float) -> bool:
        return storage_used_gb > self.storage_gb

    def cost_exceeded(self, cost_used: float) -> bool:
        return cost_used > self.cost_usd_per_month


class ResourceAccountingService:
    """
    Track and enforce resource budgets.
    
    Admission control modes:
    - STRICT: Reject when budget exceeded
    - DEGRADE: Reduce throughput or quality
    - WARN: Log warning but allow
    """

    def __init__(self, db: AsyncSession, organization_id: uuid.UUID):
        self.db = db
        self.organization_id = organization_id

    async def record_token_usage(
        self,
        tokens_used: int,
        model_name: str,
        user_id: Optional[uuid.UUID] = None
    ) -> None:
        """Record LLM token usage."""
        # In production, store in resource_usage table
        logger.info(
            f"Token usage: org={self.organization_id} "
            f"tokens={tokens_used} model={model_name}"
        )

    async def record_storage_usage(
        self,
        storage_bytes: int,
        resource_type: str,
        user_id: Optional[uuid.UUID] = None
    ) -> None:
        """Record storage usage."""
        storage_gb = storage_bytes / (1024 ** 3)
        logger.info(
            f"Storage usage: org={self.organization_id} "
            f"bytes={storage_bytes}GB type={resource_type}"
        )

    async def record_latency(
        self,
        latency_ms: int,
        operation: str,
        user_id: Optional[uuid.UUID] = None
    ) -> None:
        """Record operation latency for SLO tracking."""
        logger.info(
            f"Latency: org={self.organization_id} "
            f"op={operation} latency_ms={latency_ms}"
        )

    async def check_admission(
        self,
        tokens_needed: int,
        storage_bytes_needed: int,
        operation: str,
        user_id: Optional[uuid.UUID] = None,
        mode: str = "STRICT"  # STRICT, DEGRADE, WARN
    ) -> Dict[str, Any]:
        """
        Check if operation should be admitted given budget.
        
        Returns dict with:
        - admitted: bool
        - reason: Optional reason for denial
        - mode: admission control mode applied
        """
        # Get org budgets (in production, query database)
        budget = ResourceBudget(self.organization_id)

        # Simulate budget usage (in production, query actual usage)
        tokens_used = 500_000  # Placeholder
        storage_used_gb = 50.0  # Placeholder
        cost_used = 500.0  # Placeholder

        # Check budgets
        if budget.tokens_exceeded(tokens_used + tokens_needed):
            if mode == "STRICT":
                logger.warning(
                    f"Token budget exceeded: org={self.organization_id} "
                    f"need={tokens_needed} remaining={budget.tokens_remaining(tokens_used)}"
                )
                return {
                    "admitted": False,
                    "reason": "Token budget exceeded",
                    "mode": mode
                }
            elif mode == "DEGRADE":
                logger.warning(
                    f"Token budget exceeded, degrading: org={self.organization_id}"
                )
                return {
                    "admitted": True,
                    "mode": "degrade",
                    "degradation": "reduce_quality",
                    "reason": "Budget near limit"
                }

        if budget.storage_exceeded(storage_used_gb + (storage_bytes_needed / (1024 ** 3))):
            if mode == "STRICT":
                logger.warning(
                    f"Storage budget exceeded: org={self.organization_id} "
                    f"need_gb={storage_bytes_needed / (1024 ** 3)} "
                    f"remaining_gb={budget.storage_remaining_gb(storage_used_gb)}"
                )
                return {
                    "admitted": False,
                    "reason": "Storage budget exceeded",
                    "mode": mode
                }

        # Admitted
        return {
            "admitted": True,
            "mode": mode,
            "tokens_remaining": budget.tokens_remaining(tokens_used + tokens_needed),
            "storage_remaining_gb": budget.storage_remaining_gb(
                storage_used_gb + (storage_bytes_needed / (1024 ** 3))
            ),
            "cost_remaining": budget.cost_remaining(cost_used)
        }

    async def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage for org."""
        # In production, query actual resource_usage table
        return {
            "tokens_used": 500_000,
            "tokens_limit": 1_000_000,
            "storage_used_gb": 50.0,
            "storage_limit_gb": 100,
            "cost_used_usd": 500.0,
            "cost_limit_usd": 1000.0,
            "month_start": (datetime.utcnow() - timedelta(days=15)).isoformat(),
            "month_end": (datetime.utcnow() + timedelta(days=15)).isoformat()
        }

    async def get_usage_forecast(self, days_ahead: int = 30) -> Dict[str, Any]:
        """Forecast resource usage for next N days."""
        current = await self.get_current_usage()

        # Simple linear forecast
        days_into_month = 15  # Placeholder
        days_remaining = 30 - days_into_month

        token_rate = current["tokens_used"] / max(1, days_into_month)
        storage_rate = current["storage_used_gb"] / max(1, days_into_month)
        cost_rate = current["cost_used_usd"] / max(1, days_into_month)

        return {
            "forecasted_tokens_eom": int(
                current["tokens_used"] + (token_rate * days_remaining)
            ),
            "forecasted_storage_eom_gb": current["storage_used_gb"] + (
                storage_rate * days_remaining
            ),
            "forecasted_cost_eom_usd": current["cost_used_usd"] + (cost_rate * days_remaining),
            "token_budget_exceeded_by_eom": current["tokens_used"] + (
                token_rate * days_remaining
            ) > current["tokens_limit"],
            "daily_token_rate": int(token_rate),
            "daily_storage_rate_gb": storage_rate,
            "daily_cost_rate": cost_rate
        }

    async def alert_if_approaching_limit(self, threshold_pct: float = 0.9) -> List[Dict[str, Any]]:
        """Alert if org is approaching budget limits (>90% by default)."""
        alerts = []
        current = await self.get_current_usage()

        token_pct = current["tokens_used"] / current["tokens_limit"]
        if token_pct >= threshold_pct:
            alerts.append({
                "alert_type": "APPROACHING_TOKEN_LIMIT",
                "current_pct": int(token_pct * 100),
                "threshold_pct": int(threshold_pct * 100),
                "tokens_used": current["tokens_used"],
                "tokens_limit": current["tokens_limit"]
            })

        storage_pct = current["storage_used_gb"] / current["storage_limit_gb"]
        if storage_pct >= threshold_pct:
            alerts.append({
                "alert_type": "APPROACHING_STORAGE_LIMIT",
                "current_pct": int(storage_pct * 100),
                "threshold_pct": int(threshold_pct * 100),
                "storage_used_gb": current["storage_used_gb"],
                "storage_limit_gb": current["storage_limit_gb"]
            })

        cost_pct = current["cost_used_usd"] / current["cost_limit_usd"]
        if cost_pct >= threshold_pct:
            alerts.append({
                "alert_type": "APPROACHING_COST_LIMIT",
                "current_pct": int(cost_pct * 100),
                "threshold_pct": int(threshold_pct * 100),
                "cost_used_usd": current["cost_used_usd"],
                "cost_limit_usd": current["cost_limit_usd"]
            })

        if alerts:
            logger.warning(f"Resource alerts for org {self.organization_id}: {alerts}")

        return alerts
