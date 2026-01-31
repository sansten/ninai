"""
Snapshot Model - For exports and compliance

Snapshots represent point-in-time exports of memory/knowledge items.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Boolean, Integer, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as SQLA_UUID
from app.core.database import Base


class Snapshot(Base):
    """
    Snapshot - point-in-time export of memory items.
    
    Snapshots are:
    - Org-scoped (tied to organization)
    - Format-flexible (JSON, CSV, PDF, etc.)
    - Stored in object storage or database
    - Expiring for compliance
    - Queryable for compliance audits
    """
    __tablename__ = "snapshots"

    # Identification
    id: Mapped[str] = mapped_column(SQLA_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Scoping
    organization_id: Mapped[str] = mapped_column(SQLA_UUID(as_uuid=False), nullable=False, index=True)
    
    # Snapshot metadata
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    
    # Format and content
    format: Mapped[str] = mapped_column(String(32), nullable=False)  # json, csv, pdf, jsonl, parquet
    compression: Mapped[str | None] = mapped_column(String(32), nullable=True)  # gzip, brotli, etc.
    
    # Filtering (what was included)
    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True)  # memory, knowledge, agent
    filters: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # Date range, tags, etc.
    
    # Storage
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)  # S3 path or local path
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    item_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Status
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)  # pending, processing, completed, failed
    progress_percent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    
    # Access control
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    download_token: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)  # Unsigned access
    
    # Retention
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # Auto-delete
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Metadata
    checksum: Mapped[str | None] = mapped_column(String(256), nullable=True)  # SHA256 for integrity
    created_by_user_id: Mapped[str | None] = mapped_column(SQLA_UUID(as_uuid=False), nullable=True)
    
    # Lifecycle
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Indices
    __table_args__ = (
        Index("idx_snapshot_org", "organization_id"),
        Index("idx_snapshot_status", "status"),
        Index("idx_snapshot_created", "created_at"),
    )

    def __repr__(self):
        return f"<Snapshot id={self.id} name={self.name} format={self.format}>"
