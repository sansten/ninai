"""
Pydantic schemas for Backup endpoints
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID


class BackupTaskResponse(BaseModel):
    """Response for backup task info"""
    id: UUID
    backup_type: str  # full, incremental
    status: str  # pending, running, completed, failed
    size_bytes: int
    duration_seconds: Optional[int] = None
    checksum_sha256: str
    s3_path: Optional[str] = None
    s3_object_key: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class BackupScheduleResponse(BaseModel):
    """Response for backup schedule info"""
    id: UUID
    frequency: str  # daily, weekly, monthly
    retention_days: int
    backup_time: str  # HH:MM format
    enabled: bool
    s3_bucket: str
    max_backup_size_gb: int
    enable_incremental: bool
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    consecutive_failures: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class BackupRestoreResponse(BaseModel):
    """Response for backup restore info"""
    id: UUID
    backup_id: UUID
    initiated_by: UUID
    status: str  # pending, in_progress, completed, failed
    duration_seconds: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CreateBackupRequest(BaseModel):
    """Request to create a backup"""
    backup_type: str = Field(default="full", description="Type of backup: full or incremental")


class CreateBackupResponse(BaseModel):
    """Response after creating backup"""
    backup_id: UUID
    status: str
    message: str


class RestoreBackupRequest(BaseModel):
    """Request to restore from backup"""
    backup_id: UUID = Field(..., description="ID of backup to restore from")
    confirm: bool = Field(..., description="Confirmation that database will be overwritten")


class RestoreBackupResponse(BaseModel):
    """Response after restore request"""
    restore_id: UUID
    backup_id: UUID
    status: str
    message: str


class ScheduleBackupRequest(BaseModel):
    """Request to schedule backups"""
    frequency: str = Field(default="daily", description="Frequency: daily, weekly, or monthly")
    retention_days: int = Field(default=30, ge=1, le=365, description="Days to retain backups")
    backup_time: str = Field(default="02:00", description="Time to run backup (HH:MM UTC)")
    s3_bucket: str = Field(..., description="S3 bucket for backup storage")
    max_backup_size_gb: int = Field(default=10, ge=1, le=1000, description="Maximum backup size")
    enable_incremental: bool = Field(default=True, description="Enable incremental backups")


class ScheduleBackupResponse(BaseModel):
    """Response after scheduling backup"""
    schedule_id: UUID
    status: str
    message: str


class BackupStatisticsResponse(BaseModel):
    """Response for backup statistics"""
    total_backups: int
    total_size_gb: float
    failed_backups: int
    last_backup_time: Optional[datetime] = None
    success_rate: float


class BackupListResponse(BaseModel):
    """Response for backup list"""
    backups: List[BackupTaskResponse]
    total: int
    page: int
    page_size: int


class UpdateScheduleRequest(BaseModel):
    """Request to update backup schedule"""
    frequency: Optional[str] = None
    retention_days: Optional[int] = None
    backup_time: Optional[str] = None
    enabled: Optional[bool] = None
    max_backup_size_gb: Optional[int] = None
    enable_incremental: Optional[bool] = None


class UpdateScheduleResponse(BaseModel):
    """Response after updating schedule"""
    schedule_id: UUID
    message: str
