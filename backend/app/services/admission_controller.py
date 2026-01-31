"""Admission controller service for resource quota enforcement.

The implementation lives in the private `ninai-enterprise` repo/package.
This stub remains only to make accidental imports fail loudly.

ENTERPRISE ONLY - AdmissionController is an enterprise feature.
"""


raise ImportError(
    "AdmissionController is an enterprise feature. Install the private 'ninai-enterprise' package to use it."
)

    """Result of an admission control decision."""

    def __init__(
        self,
        *,
        admitted: bool,
        reason: str,
        throttle_rate: float = 1.0,
        retry_after_seconds: int | None = None,
    ):
        self.admitted = admitted
        self.reason = reason
        self.throttle_rate = throttle_rate
        self.retry_after_seconds = retry_after_seconds


class AdmissionController:
    """Admission controller for resource quota enforcement."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.audit = AuditService(db)

    async def get_or_create_budget(
        self,
        *,
        organization_id: str,
        period: str = BudgetPeriod.DAILY.value,
    ) -> ResourceBudget:
        """Get current budget or create new one for the period.

        Args:
            organization_id: Organization ID
            period: Budget period (hourly, daily, monthly)

        Returns:
            ResourceBudget for current period
        """
        now = datetime.now(timezone.utc)

        # Try to find current budget
        stmt = (
            select(ResourceBudget)
            .where(
                and_(
                    ResourceBudget.organization_id == organization_id,
                    ResourceBudget.period == period,
                    ResourceBudget.period_start <= now,
                    ResourceBudget.period_end >= now,
                )
            )
            .limit(1)
        )
        result = await self.db.execute(stmt)
        budget = result.scalar_one_or_none()

        if budget:
            return budget

        # Create new budget for current period
        if period == BudgetPeriod.HOURLY.value:
            period_start = now.replace(minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(hours=1)
        elif period == BudgetPeriod.MONTHLY.value:
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Next month
            if now.month == 12:
                period_end = period_start.replace(year=now.year + 1, month=1)
            else:
                period_end = period_start.replace(month=now.month + 1)
        else:  # DAILY
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1)

        budget = ResourceBudget(
            organization_id=organization_id,
            period=period,
            period_start=period_start,
            period_end=period_end,
            # Explicitly set defaults since they won't be applied until INSERT
            token_budget=1000000,
            tokens_used=0,
            tokens_reserved=0,
            storage_budget_mb=10000,
            storage_used_mb=0,
            latency_budget_ms=30000,
            latency_consumed_ms=0,
            slo_breach_count=0,
            request_budget=10000,
            requests_used=0,
            admission_blocked=False,
            degraded_mode=False,
            throttle_rate=1.0,
        )
        self.db.add(budget)
        await self.db.flush()  # Force flush to get defaults applied
        await self.db.refresh(budget)  # Refresh to get server defaults

        return budget

    async def check_admission(
        self,
        *,
        organization_id: str,
        estimated_tokens: int = 0,
        estimated_storage_mb: int = 0,
        estimated_latency_ms: int = 0,
        actor_user_id: str | None = None,
    ) -> AdmissionDecision:
        """Check if request should be admitted based on resource budgets.

        Args:
            organization_id: Organization ID
            estimated_tokens: Estimated token cost
            estimated_storage_mb: Estimated storage cost
            estimated_latency_ms: Estimated latency
            actor_user_id: User making the request

        Returns:
            AdmissionDecision with admit/reject and throttling info
        """
        budget = await self.get_or_create_budget(organization_id=organization_id)

        # Check if admission is blocked
        if budget.admission_blocked:
            await self.audit.log_event(
                event_type="admission.rejected",
                organization_id=organization_id,
                actor_id=actor_user_id,
                resource_type="resource_budget",
                resource_id=budget.id,
                success=False,
                error_message="Admission blocked due to quota exhaustion",
                details={
                    "token_utilization": budget.token_utilization,
                    "storage_utilization": budget.storage_utilization,
                },
            )
            return AdmissionDecision(
                admitted=False,
                reason="Quota exhausted - admission blocked",
                retry_after_seconds=int((budget.period_end - datetime.now(timezone.utc)).total_seconds()),
            )

        # Check token quota
        if estimated_tokens > 0:
            if budget.tokens_available < estimated_tokens:
                await self.audit.log_event(
                    event_type="admission.rejected.tokens",
                    organization_id=organization_id,
                    actor_id=actor_user_id,
                    resource_type="resource_budget",
                    resource_id=budget.id,
                    success=False,
                    error_message="Insufficient token budget",
                    details={
                        "estimated_tokens": estimated_tokens,
                        "tokens_available": budget.tokens_available,
                        "token_utilization": budget.token_utilization,
                    },
                )
                return AdmissionDecision(
                    admitted=False,
                    reason=f"Insufficient token budget (need {estimated_tokens}, available {budget.tokens_available})",
                    retry_after_seconds=int((budget.period_end - datetime.now(timezone.utc)).total_seconds()),
                )

        # Check storage quota
        if estimated_storage_mb > 0:
            available_storage = budget.storage_budget_mb - budget.storage_used_mb
            if available_storage < estimated_storage_mb:
                await self.audit.log_event(
                    event_type="admission.rejected.storage",
                    organization_id=organization_id,
                    actor_id=actor_user_id,
                    resource_type="resource_budget",
                    resource_id=budget.id,
                    success=False,
                    error_message="Insufficient storage budget",
                    details={
                        "estimated_storage_mb": estimated_storage_mb,
                        "available_storage_mb": available_storage,
                        "storage_utilization": budget.storage_utilization,
                    },
                )
                return AdmissionDecision(
                    admitted=False,
                    reason=f"Insufficient storage budget (need {estimated_storage_mb}MB, available {available_storage}MB)",
                )

        # Check if throttling needed
        throttle_rate = 1.0
        if budget.should_throttle:
            # Linear throttle: at 80% = 1.0x, at 95% = 0.5x, at 100% = 0.1x
            utilization = budget.token_utilization
            if utilization >= 0.95:
                throttle_rate = 0.1
            elif utilization >= 0.90:
                throttle_rate = 0.3
            elif utilization >= 0.85:
                throttle_rate = 0.6
            else:
                throttle_rate = 0.8

            budget.throttle_rate = throttle_rate
            budget.last_throttle_update = datetime.now(timezone.utc)

            if not self.db.info.get("auto_commit", True):
                await self.db.flush()

        await self.audit.log_event(
            event_type="admission.admitted",
            organization_id=organization_id,
            actor_id=actor_user_id,
            resource_type="resource_budget",
            resource_id=budget.id,
            success=True,
            details={
                "estimated_tokens": estimated_tokens,
                "throttle_rate": throttle_rate,
                "token_utilization": budget.token_utilization,
            },
        )

        return AdmissionDecision(
            admitted=True,
            reason="Admitted",
            throttle_rate=throttle_rate,
        )

    async def reserve_tokens(
        self,
        *,
        organization_id: str,
        tokens: int,
    ) -> bool:
        """Reserve tokens before execution.

        Args:
            organization_id: Organization ID
            tokens: Number of tokens to reserve

        Returns:
            True if reservation successful, False if insufficient budget
        """
        budget = await self.get_or_create_budget(organization_id=organization_id)

        if budget.tokens_available < tokens:
            return False

        budget.tokens_reserved += tokens

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

        return True

    async def consume_resources(
        self,
        *,
        organization_id: str,
        tokens_used: int = 0,
        storage_mb_used: int = 0,
        latency_ms: int = 0,
        was_reserved: bool = False,
    ) -> None:
        """Record resource consumption.

        Args:
            organization_id: Organization ID
            tokens_used: Tokens consumed
            storage_mb_used: Storage consumed in MB
            latency_ms: Latency in milliseconds
            was_reserved: Whether tokens were previously reserved
        """
        budget = await self.get_or_create_budget(organization_id=organization_id)

        if tokens_used > 0:
            budget.tokens_used += tokens_used
            if was_reserved:
                budget.tokens_reserved = max(0, budget.tokens_reserved - tokens_used)

        if storage_mb_used > 0:
            budget.storage_used_mb += storage_mb_used

        if latency_ms > 0:
            budget.latency_consumed_ms += latency_ms

        budget.requests_used += 1

        # Check if we should block admission
        if budget.is_token_quota_exhausted or budget.is_storage_quota_exhausted:
            budget.admission_blocked = True
            budget.degraded_mode = True

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

    async def record_slo_breach(
        self,
        *,
        organization_id: str,
        breach_details: dict | None = None,
    ) -> None:
        """Record an SLO breach.

        Args:
            organization_id: Organization ID
            breach_details: Details about the breach
        """
        budget = await self.get_or_create_budget(organization_id=organization_id)
        budget.slo_breach_count += 1

        await self.audit.log_event(
            event_type="slo.breach",
            organization_id=organization_id,
            resource_type="resource_budget",
            resource_id=budget.id,
            success=False,
            error_message="SLO breach recorded",
            details=breach_details or {},
        )

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

    async def get_budget_summary(
        self,
        *,
        organization_id: str,
    ) -> dict:
        """Get budget summary for organization.

        Args:
            organization_id: Organization ID

        Returns:
            Budget summary with utilization metrics
        """
        budget = await self.get_or_create_budget(organization_id=organization_id)

        return {
            "period": budget.period,
            "period_start": budget.period_start.isoformat(),
            "period_end": budget.period_end.isoformat(),
            "token_budget": budget.token_budget,
            "tokens_used": budget.tokens_used,
            "tokens_reserved": budget.tokens_reserved,
            "tokens_available": budget.tokens_available,
            "token_utilization": budget.token_utilization,
            "storage_budget_mb": budget.storage_budget_mb,
            "storage_used_mb": budget.storage_used_mb,
            "storage_utilization": budget.storage_utilization,
            "request_budget": budget.request_budget,
            "requests_used": budget.requests_used,
            "request_utilization": budget.request_utilization,
            "slo_breach_count": budget.slo_breach_count,
            "admission_blocked": budget.admission_blocked,
            "degraded_mode": budget.degraded_mode,
            "throttle_rate": budget.throttle_rate,
        }
