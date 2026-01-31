"""
Backup Models - BackupTask, BackupSchedule
"""

from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSON
from datetime import datetime
import uuid

from app.models.base import Base


class BackupTask(Base):
    """Database backup task"""
    __tablename__ = "backup_task"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    backup_type = Column(String(20), nullable=False, default="full")  # full, incremental
    size_bytes = Column(BigInteger, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    s3_path = Column(String(500), nullable=True)  # S3 URI
    s3_object_key = Column(String(500), nullable=True)  # S3 object key
    checksum_sha256 = Column(String(64), nullable=True)  # SHA256 hash for integrity
    retention_until = Column(DateTime, nullable=True)  # When to delete this backup
    error_message = Column(Text, nullable=True)
    backup_metadata = Column(JSON, nullable=True)  # Additional backup metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class BackupSchedule(Base):
    """Backup scheduling configuration"""
    __tablename__ = "backup_schedule"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    enabled = Column(Boolean, default=True)
    frequency = Column(String(20), nullable=False)  # daily, weekly, monthly
    retention_days = Column(Integer, default=30)
    s3_bucket = Column(String(255), nullable=False)
    s3_prefix = Column(String(100), default="backups/")
    backup_time = Column(String(5), default="02:00")  # HH:MM format, UTC
    last_run_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    last_success_at = Column(DateTime, nullable=True)
    consecutive_failures = Column(Integer, default=0)
    max_backup_size_gb = Column(Integer, default=100)  # Max allowed backup size
    enable_incremental = Column(Boolean, default=False)  # Use incremental backups
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BackupRestore(Base):
    """Track database restore operations"""
    __tablename__ = "backup_restore"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backup_id = Column(UUID(as_uuid=True), ForeignKey("backup_task.id"), nullable=False)
    initiated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    duration_seconds = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
