"""Resource budget model for tenant quota tracking and admission control.

Tracks token consumption, storage usage, and latency budgets per tenant.
Enables admission control and graceful degradation when budgets are exhausted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, Integer, String, Boolean, BigInteger, Float
from sqlalchemy.sql import func

from app.models.base import Base, UUIDMixin, TimestampMixin, TenantMixin


class BudgetPeriod(PyEnum):
    """Budget tracking period."""

    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


class ResourceBudget(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """Resource budget tracking for tenant quotas and admission control."""

    __tablename__ = "resource_budgets"

    # Budget period
    period = Column(
        String(20),
        default=BudgetPeriod.DAILY.value,
        nullable=False,
        index=True,
        comment="Budget period: hourly, daily, monthly",
    )
    period_start = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Start of current budget period",
    )
    period_end = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="End of current budget period",
    )

    # Token budget (LLM API calls)
    token_budget = Column(
        BigInteger,
        default=1000000,
        nullable=False,
        comment="Total token budget for period",
    )
    tokens_used = Column(
        BigInteger,
        default=0,
        nullable=False,
        comment="Tokens consumed in current period",
    )
    tokens_reserved = Column(
        BigInteger,
        default=0,
        nullable=False,
        comment="Tokens reserved but not yet consumed",
    )

    # Storage budget (bytes)
    storage_budget_mb = Column(
        Integer,
        default=10000,
        nullable=False,
        comment="Total storage budget in MB",
    )
    storage_used_mb = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Storage consumed in MB",
    )

    # Latency budget (SLO tracking)
    latency_budget_ms = Column(
        Integer,
        default=30000,
        nullable=False,
        comment="Total latency budget in ms per period",
    )
    latency_consumed_ms = Column(
        BigInteger,
        default=0,
        nullable=False,
        comment="Latency consumed in current period",
    )
    slo_breach_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of SLO breaches in period",
    )

    # Request limits
    request_budget = Column(
        Integer,
        default=10000,
        nullable=False,
        comment="Total request budget for period",
    )
    requests_used = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Requests made in current period",
    )

    # Admission control flags
    admission_blocked = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Is admission currently blocked due to quota exhaustion?",
    )
    degraded_mode = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Is tenant in degraded mode (reduced quota)?",
    )

    # Throttling
    throttle_rate = Column(
        Float,
        default=1.0,
        nullable=False,
        comment="Current throttle rate (1.0 = no throttle, 0.5 = 50% throttle)",
    )
    last_throttle_update = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When throttle rate was last updated",
    )

    @property
    def token_utilization(self) -> float:
        """Calculate token budget utilization as percentage."""
        token_budget = self.token_budget or 0
        if token_budget == 0:
            return 0.0
        tokens_used = self.tokens_used or 0
        tokens_reserved = self.tokens_reserved or 0
        return (tokens_used + tokens_reserved) / token_budget

    @property
    def storage_utilization(self) -> float:
        """Calculate storage budget utilization as percentage."""
        storage_budget_mb = self.storage_budget_mb or 0
        if storage_budget_mb == 0:
            return 0.0
        storage_used_mb = self.storage_used_mb or 0
        return storage_used_mb / storage_budget_mb

    @property
    def request_utilization(self) -> float:
        """Calculate request budget utilization as percentage."""
        request_budget = self.request_budget or 0
        if request_budget == 0:
            return 0.0
        requests_used = self.requests_used or 0
        return requests_used / request_budget

    @property
    def tokens_available(self) -> int:
        """Calculate available tokens (budget - used - reserved)."""
        token_budget = self.token_budget or 0
        tokens_used = self.tokens_used or 0
        tokens_reserved = self.tokens_reserved or 0
        return max(0, token_budget - tokens_used - tokens_reserved)

    @property
    def is_token_quota_exhausted(self) -> bool:
        """Check if token quota is exhausted (>95% utilized)."""
        return self.token_utilization >= 0.95

    @property
    def is_storage_quota_exhausted(self) -> bool:
        """Check if storage quota is exhausted (>95% utilized)."""
        return self.storage_utilization >= 0.95

    @property
    def should_throttle(self) -> bool:
        """Check if throttling should be applied (>80% token utilization)."""
        return self.token_utilization >= 0.80

    def __repr__(self) -> str:
        return (
            f"ResourceBudget(org={self.organization_id}, period={self.period}, "
            f"token_util={self.token_utilization:.2%}, storage_util={self.storage_utilization:.2%})"
        )
