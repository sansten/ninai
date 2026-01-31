"""Memory snapshot model for point-in-time state capture."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import String, Text, BigInteger, Boolean, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin, TimestampMixin, TenantMixin


class SnapshotStatus(str, Enum):
    """Snapshot creation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class SnapshotType(str, Enum):
    """Type of snapshot."""
    FULL = "full"  # Complete memory state
    INCREMENTAL = "incremental"  # Changes since last snapshot
    DIFFERENTIAL = "differential"  # Changes since last full


class MemorySnapshot(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """Point-in-time snapshot of memory state for backup/restore.

    Captures complete or incremental memory state including:
    - Long-term memory entries
    - Vector embeddings
    - Metadata and relationships
    - Agent process state
    - Configuration
    """

    __tablename__ = "memory_snapshots"

    # Snapshot identification
    snapshot_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    snapshot_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=SnapshotStatus.PENDING.value, index=True)

    # Snapshot content
    snapshot_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    memory_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    embedding_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    # Snapshot metadata
    snapshot_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    """
    Metadata includes:
    - total_tokens: Total token count in snapshot
    - embedding_model: Model used for embeddings
    - schema_version: Schema version at time of snapshot
    - includes_qdrant: Whether Qdrant vectors are included
    - checksum: SHA256 checksum for integrity verification
    """

    # Storage
    storage_location: Mapped[str | None] = mapped_column(Text, nullable=True)
    """
    Storage location URI:
    - s3://bucket/path/snapshot.tar.gz
    - file:///local/path/snapshot.tar.gz
    - azure://container/path/snapshot.tar.gz
    """

    compression_format: Mapped[str | None] = mapped_column(String(20), nullable=True)
    """Compression: gzip, zstd, lz4, none"""

    # Incremental snapshots
    parent_snapshot_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True, index=True)
    """For incremental/differential: ID of base snapshot"""

    # Lifecycle
    retention_days: Mapped[int] = mapped_column(BigInteger, nullable=False, default=30)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Execution tracking
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Verification
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    """SHA256 checksum of snapshot data"""

    # Replication
    replicated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    replication_targets: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    """
    List of replication targets:
    - region: Target region
    - location: Storage location in region
    - status: pending, in_progress, completed, failed
    - synced_at: Timestamp of last sync
    """

    # Created by
    created_by_user_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)

    __table_args__ = (
        Index("ix_memory_snapshots_org_status", "organization_id", "status"),
        Index("ix_memory_snapshots_org_type", "organization_id", "snapshot_type"),
        Index("ix_memory_snapshots_parent", "parent_snapshot_id"),
        Index("ix_memory_snapshots_expires", "expires_at"),
    )

    @property
    def is_complete(self) -> bool:
        """Check if snapshot is completed successfully."""
        return self.status == SnapshotStatus.COMPLETED.value

    @property
    def is_expired(self) -> bool:
        """Check if snapshot has expired."""
        if self.expires_at is None:
            return False
        from datetime import datetime, timezone
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def compression_ratio(self) -> float:
        """Estimate compression ratio (if metadata available)."""
        if "uncompressed_size_bytes" in self.snapshot_metadata:
            uncompressed = self.snapshot_metadata["uncompressed_size_bytes"]
            if uncompressed > 0:
                return self.snapshot_size_bytes / uncompressed
        return 1.0

    @property
    def duration_seconds(self) -> float | None:
        """Calculate snapshot duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
