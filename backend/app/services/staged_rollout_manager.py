"""Staged rollout manager for safe policy deployment.

The implementation lives in the private `ninai-enterprise` repo/package.
This stub remains only to make accidental imports fail loudly.

ENTERPRISE ONLY - StagedRolloutManager is an enterprise feature.
"""


raise ImportError(
    "StagedRolloutManager is an enterprise feature. Install the private 'ninai-enterprise' package to use it."
)

    """Manager for staged policy rollouts with canary deployments."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.audit = AuditService(db)

    async def create_policy_version(
        self,
        *,
        organization_id: str,
        policy_name: str,
        policy_type: str,
        policy_config: dict,
        description: str | None = None,
        validation_schema: dict | None = None,
        created_by_user_id: str | None = None,
        change_notes: str | None = None,
    ) -> PolicyVersion:
        """Create a new policy version in DRAFT state.

        Args:
            organization_id: Organization ID
            policy_name: Policy name
            policy_type: Policy type (access_control, validation, etc.)
            policy_config: Policy configuration as dict
            description: Human-readable description
            validation_schema: JSON schema for config validation
            created_by_user_id: User creating this version
            change_notes: Notes about changes

        Returns:
            Created PolicyVersion
        """
        # Get next version number
        stmt = (
            select(func.max(PolicyVersion.version))
            .where(
                and_(
                    PolicyVersion.organization_id == organization_id,
                    PolicyVersion.policy_name == policy_name,
                )
            )
        )
        max_version = await self.db.scalar(stmt)
        next_version = (max_version or 0) + 1

        policy = PolicyVersion(
            id=str(uuid4()),
            organization_id=organization_id,
            policy_name=policy_name,
            policy_type=policy_type,
            version=next_version,
            policy_config=policy_config,
            validation_schema=validation_schema,
            description=description,
            rollout_status=RolloutStatus.DRAFT.value,
            rollout_percentage=0.0,
            created_by_user_id=created_by_user_id,
            change_notes=change_notes,
        )

        self.db.add(policy)

        await self.audit.log_event(
            event_type="policy.version.created",
            organization_id=organization_id,
            actor_id=created_by_user_id,
            resource_type="policy_version",
            resource_id=policy.id,
            success=True,
            details={
                "policy_name": policy_name,
                "version": next_version,
                "policy_type": policy_type,
            },
        )

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

        return policy

    async def deploy_to_canary(
        self,
        *,
        policy_id: str,
        canary_group_ids: list[str],
        actor_user_id: str | None = None,
    ) -> PolicyVersion:
        """Deploy policy version to canary group.

        Args:
            policy_id: Policy version ID
            canary_group_ids: List of user/org IDs for canary testing
            actor_user_id: User performing deployment

        Returns:
            Updated PolicyVersion
        """
        policy = await self.db.get(PolicyVersion, policy_id)
        if not policy:
            raise ValueError(f"Policy version {policy_id} not found")

        if policy.rollout_status != RolloutStatus.DRAFT.value:
            raise ValueError(f"Only DRAFT policies can be deployed to canary (current: {policy.rollout_status})")

        policy.rollout_status = RolloutStatus.CANARY.value
        policy.canary_group_ids = canary_group_ids
        policy.rollout_percentage = 0.0  # Canary is explicit group, not percentage

        await self.audit.log_event(
            event_type="policy.canary.deployed",
            organization_id=policy.organization_id,
            actor_id=actor_user_id,
            resource_type="policy_version",
            resource_id=policy.id,
            success=True,
            details={
                "policy_name": policy.policy_name,
                "version": policy.version,
                "canary_group_size": len(canary_group_ids),
            },
        )

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

        return policy

    async def promote_to_staged(
        self,
        *,
        policy_id: str,
        rollout_percentage: float,
        actor_user_id: str | None = None,
    ) -> PolicyVersion:
        """Promote policy from canary to staged rollout.

        Args:
            policy_id: Policy version ID
            rollout_percentage: Percentage of users to deploy to (0.0 - 1.0)
            actor_user_id: User performing promotion

        Returns:
            Updated PolicyVersion
        """
        policy = await self.db.get(PolicyVersion, policy_id)
        if not policy:
            raise ValueError(f"Policy version {policy_id} not found")

        if policy.rollout_status not in (RolloutStatus.CANARY.value, RolloutStatus.STAGED.value):
            raise ValueError(f"Only CANARY or STAGED policies can be promoted (current: {policy.rollout_status})")

        if not 0.0 <= rollout_percentage <= 1.0:
            raise ValueError(f"Rollout percentage must be between 0.0 and 1.0, got {rollout_percentage}")

        policy.rollout_status = RolloutStatus.STAGED.value
        policy.rollout_percentage = rollout_percentage

        await self.audit.log_event(
            event_type="policy.staged.promoted",
            organization_id=policy.organization_id,
            actor_id=actor_user_id,
            resource_type="policy_version",
            resource_id=policy.id,
            success=True,
            details={
                "policy_name": policy.policy_name,
                "version": policy.version,
                "rollout_percentage": rollout_percentage,
            },
        )

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

        return policy

    async def activate_fully(
        self,
        *,
        policy_id: str,
        actor_user_id: str | None = None,
    ) -> PolicyVersion:
        """Activate policy version fully (100% rollout).

        This also supersedes any previously active version.

        Args:
            policy_id: Policy version ID
            actor_user_id: User performing activation

        Returns:
            Updated PolicyVersion
        """
        policy = await self.db.get(PolicyVersion, policy_id)
        if not policy:
            raise ValueError(f"Policy version {policy_id} not found")

        # Supersede any currently active version
        stmt = (
            select(PolicyVersion)
            .where(
                and_(
                    PolicyVersion.organization_id == policy.organization_id,
                    PolicyVersion.policy_name == policy.policy_name,
                    PolicyVersion.rollout_status == RolloutStatus.ACTIVE.value,
                )
            )
        )
        result = await self.db.execute(stmt)
        current_active = result.scalar_one_or_none()

        if current_active:
            current_active.rollout_status = RolloutStatus.SUPERSEDED.value
            current_active.superseded_at = datetime.now(timezone.utc)
            current_active.superseded_by_version = policy.version

        policy.rollout_status = RolloutStatus.ACTIVE.value
        policy.rollout_percentage = 1.0
        policy.activated_at = datetime.now(timezone.utc)

        await self.audit.log_event(
            event_type="policy.activated",
            organization_id=policy.organization_id,
            actor_id=actor_user_id,
            resource_type="policy_version",
            resource_id=policy.id,
            success=True,
            details={
                "policy_name": policy.policy_name,
                "version": policy.version,
                "superseded_version": current_active.version if current_active else None,
            },
        )

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

        return policy

    async def rollback(
        self,
        *,
        policy_id: str,
        reason: str,
        rollback_to_version: int | None = None,
        actor_user_id: str | None = None,
    ) -> PolicyVersion:
        """Rollback a policy version.

        Args:
            policy_id: Policy version ID to rollback
            reason: Reason for rollback
            rollback_to_version: Version to rollback to (or None for previous active)
            actor_user_id: User performing rollback

        Returns:
            Updated PolicyVersion (the one being rolled back)
        """
        policy = await self.db.get(PolicyVersion, policy_id)
        if not policy:
            raise ValueError(f"Policy version {policy_id} not found")

        # Find version to rollback to
        if rollback_to_version is None:
            # Find most recent ACTIVE or SUPERSEDED version before this one
            stmt = (
                select(PolicyVersion)
                .where(
                    and_(
                        PolicyVersion.organization_id == policy.organization_id,
                        PolicyVersion.policy_name == policy.policy_name,
                        PolicyVersion.version < policy.version,
                        or_(
                            PolicyVersion.rollout_status == RolloutStatus.ACTIVE.value,
                            PolicyVersion.rollout_status == RolloutStatus.SUPERSEDED.value,
                        ),
                    )
                )
                .order_by(desc(PolicyVersion.version))
                .limit(1)
            )
            result = await self.db.execute(stmt)
            previous = result.scalar_one_or_none()
            if previous:
                rollback_to_version = previous.version
        else:
            # Validate specified version exists
            stmt = (
                select(PolicyVersion)
                .where(
                    and_(
                        PolicyVersion.organization_id == policy.organization_id,
                        PolicyVersion.policy_name == policy.policy_name,
                        PolicyVersion.version == rollback_to_version,
                    )
                )
            )
            result = await self.db.execute(stmt)
            previous = result.scalar_one_or_none()
            if not previous:
                raise ValueError(f"Rollback target version {rollback_to_version} not found")

        # Mark current as rolled back
        policy.rollout_status = RolloutStatus.ROLLED_BACK.value
        policy.rolled_back_at = datetime.now(timezone.utc)
        policy.rollback_reason = reason
        policy.rolled_back_to_version = rollback_to_version

        # Reactivate previous version if found
        if previous:
            previous.rollout_status = RolloutStatus.ACTIVE.value
            previous.activated_at = datetime.now(timezone.utc)
            previous.rollout_percentage = 1.0

        await self.audit.log_event(
            event_type="policy.rolled_back",
            organization_id=policy.organization_id,
            actor_id=actor_user_id,
            resource_type="policy_version",
            resource_id=policy.id,
            success=True,
            details={
                "policy_name": policy.policy_name,
                "version": policy.version,
                "rollback_to_version": rollback_to_version,
                "reason": reason,
            },
        )

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

        return policy

    async def record_evaluation(
        self,
        *,
        policy_id: str,
        success: bool,
    ) -> None:
        """Record policy evaluation result for metrics.

        Args:
            policy_id: Policy version ID
            success: Whether evaluation succeeded
        """
        policy = await self.db.get(PolicyVersion, policy_id)
        if not policy:
            return

        if success:
            policy.success_count += 1
        else:
            policy.failure_count += 1

        total = policy.success_count + policy.failure_count
        policy.error_rate = policy.failure_count / total if total > 0 else 0.0

        if not self.db.info.get("auto_commit", True):
            await self.db.flush()

    async def check_auto_rollback(
        self,
        *,
        policy_id: str,
        error_rate_threshold: float = 0.1,
        min_evaluations: int = 100,
    ) -> bool:
        """Check if policy should be auto-rolled back based on error rate.

        Args:
            policy_id: Policy version ID
            error_rate_threshold: Error rate threshold for auto-rollback
            min_evaluations: Minimum evaluations before auto-rollback

        Returns:
            True if policy was rolled back, False otherwise
        """
        policy = await self.db.get(PolicyVersion, policy_id)
        if not policy:
            return False

        total = policy.success_count + policy.failure_count
        if total < min_evaluations:
            return False

        if policy.error_rate > error_rate_threshold:
            await self.rollback(
                policy_id=policy_id,
                reason=f"Auto-rollback: error rate {policy.error_rate:.2%} exceeded threshold {error_rate_threshold:.2%}",
            )
            return True

        return False

    async def get_active_policy(
        self,
        *,
        organization_id: str,
        policy_name: str,
    ) -> PolicyVersion | None:
        """Get currently active policy version.

        Args:
            organization_id: Organization ID
            policy_name: Policy name

        Returns:
            Active PolicyVersion or None
        """
        stmt = (
            select(PolicyVersion)
            .where(
                and_(
                    PolicyVersion.organization_id == organization_id,
                    PolicyVersion.policy_name == policy_name,
                    PolicyVersion.rollout_status == RolloutStatus.ACTIVE.value,
                )
            )
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_policy_versions(
        self,
        *,
        organization_id: str,
        policy_name: str,
    ) -> list[PolicyVersion]:
        """List all versions of a policy.

        Args:
            organization_id: Organization ID
            policy_name: Policy name

        Returns:
            List of PolicyVersion ordered by version descending
        """
        stmt = (
            select(PolicyVersion)
            .where(
                and_(
                    PolicyVersion.organization_id == organization_id,
                    PolicyVersion.policy_name == policy_name,
                )
            )
            .order_by(desc(PolicyVersion.version))
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
