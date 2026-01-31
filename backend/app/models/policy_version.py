"""Policy version model for validation rules and access control.

The implementation lives in the private `ninai-enterprise` repo/package.
This stub remains only to make accidental imports fail loudly.

ENTERPRISE ONLY - PolicyVersion is an enterprise feature.
"""


raise ImportError(
    "PolicyVersion is an enterprise feature. Install the private 'ninai-enterprise' package to use it."
)

    """Versioned policy with staged rollout support."""

    __tablename__ = "policy_versions"

    # Policy identification
    policy_name = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Policy name (e.g., 'content_safety', 'rate_limiter')",
    )
    policy_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Policy type: access_control, validation, rate_limit, safety, routing",
    )
    version = Column(
        Integer,
        nullable=False,
        comment="Version number (monotonically increasing)",
    )

    # Policy content
    policy_config = Column(
        JSON,
        nullable=False,
        comment="Policy configuration as JSON",
    )
    validation_schema = Column(
        JSON,
        nullable=True,
        comment="JSON schema for validating policy config",
    )
    description = Column(
        Text,
        nullable=True,
        comment="Human-readable policy description",
    )

    # Rollout control
    rollout_status = Column(
        String(20),
        default=RolloutStatus.DRAFT.value,
        nullable=False,
        index=True,
        comment="Rollout status: draft, canary, staged, active, superseded, rolled_back",
    )
    rollout_percentage = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Percentage of users to apply this policy to (0.0 - 1.0)",
    )
    canary_group_ids = Column(
        JSON,
        default=[],
        nullable=False,
        comment="List of user/org IDs in canary group",
    )

    # Activation tracking
    activated_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When this version became active",
    )
    superseded_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When this version was superseded",
    )
    superseded_by_version = Column(
        Integer,
        nullable=True,
        comment="Version number that superseded this one",
    )

    # Rollback tracking
    rolled_back_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When this version was rolled back",
    )
    rollback_reason = Column(
        Text,
        nullable=True,
        comment="Reason for rollback",
    )
    rolled_back_to_version = Column(
        Integer,
        nullable=True,
        comment="Version rolled back to",
    )

    # Metrics for rollout decision-making
    success_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of successful policy evaluations",
    )
    failure_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of failed policy evaluations",
    )
    error_rate = Column(
        Float,
        default=0.0,
        nullable=False,
        comment="Current error rate (failures / total)",
    )

    # Metadata
    created_by_user_id = Column(
        String(100),
        nullable=True,
        comment="User who created this version",
    )
    change_notes = Column(
        Text,
        nullable=True,
        comment="Notes about what changed in this version",
    )

    @property
    def is_active(self) -> bool:
        """Check if this version is currently active."""
        return self.rollout_status == RolloutStatus.ACTIVE.value

    @property
    def is_deployed(self) -> bool:
        """Check if this version is deployed (canary, staged, or active)."""
        return self.rollout_status in (
            RolloutStatus.CANARY.value,
            RolloutStatus.STAGED.value,
            RolloutStatus.ACTIVE.value,
        )

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

    def __repr__(self) -> str:
        return (
            f"PolicyVersion(name={self.policy_name}, version={self.version}, "
            f"status={self.rollout_status}, rollout={self.rollout_percentage:.1%})"
        )
