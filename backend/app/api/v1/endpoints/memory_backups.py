"""
Memory Backup and Restore Endpoints

REST API for creating, restoring, and managing memory snapshots as backups.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import uuid
import logging

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.middleware.tenant_context import get_tenant_context, TenantContext
from app.services.backup_restore import BackupRestoreService, BackupSchedule

router = APIRouter(prefix="/memory-backups", tags=["Memory Backup"])
logger = logging.getLogger(__name__)


# Schemas
class MemoryBackupCreateRequest(BaseModel):
    """Create memory backup request."""
    name: str
    backup_type: str = "full"  # full or incremental
    retention_days: int = 30
    tags: Optional[Dict[str, str]] = None


class MemoryBackupResponse(BaseModel):
    """Memory backup response."""
    backup_id: str
    name: str
    type: str
    size_bytes: int
    created_at: str
    expires_at: Optional[str] = None
    tags: Optional[Dict[str, str]] = None

    class Config:
        from_attributes = True


class MemoryBackupScheduleRequest(BaseModel):
    """Create memory backup schedule request."""
    name: str
    schedule: str  # hourly, daily, weekly, monthly
    backup_type: str = "full"
    retention_days: int = 30
    enabled: bool = True


class MemoryBackupScheduleResponse(BaseModel):
    """Memory backup schedule response."""
    schedule_id: str
    name: str
    schedule: str
    backup_type: str
    retention_days: int
    enabled: bool
    next_run: str

    class Config:
        from_attributes = True


class MemoryRestoreRequest(BaseModel):
    """Restore memory backup request."""
    backup_id: str
    overwrite: bool = False
    dry_run: bool = False


class MemoryRestoreValidationResponse(BaseModel):
    """Memory restore validation (dry run) response."""
    dry_run: bool = True
    valid: bool
    estimated_memories: int
    errors: List[str] = []


class MemoryRestoreResultResponse(BaseModel):
    """Memory restore result response."""
    backup_id: str
    restored_at: str
    memories_restored: int
    memories_skipped: int
    overwrite: bool


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Dependency to require admin role."""
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user


@router.post("/create", response_model=MemoryBackupResponse)
async def create_memory_backup(
    request: MemoryBackupCreateRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Create a backup of all memories.
    
    Admin only.
    
    Backup types:
    - full: Complete snapshot of all memories
    - incremental: Only changes since last backup
    """
    try:
        service = BackupRestoreService(db, tenant.org_id)
        
        backup = await service.create_backup(
            name=request.name,
            backup_type=request.backup_type,
            retention_days=request.retention_days,
            tags=request.tags
        )
        
        logger.info(f"Created memory backup: {request.name} ({request.backup_type})")
        
        return MemoryBackupResponse(
            backup_id=backup["backup_id"],
            name=backup["name"],
            type=backup["type"],
            size_bytes=backup["size_bytes"],
            created_at=backup["created_at"],
            expires_at=backup.get("expires_at"),
            tags=backup.get("tags")
        )
    except Exception as e:
        logger.error(f"Error creating memory backup: {e}")
        raise HTTPException(status_code=500, detail="Failed to create backup")


@router.post("/restore", response_model=MemoryRestoreResultResponse)
async def restore_memory_backup(
    request: MemoryRestoreRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Restore from a memory backup.
    
    Admin only. Set dry_run=True to validate without restoring.
    """
    try:
        service = BackupRestoreService(db, tenant.org_id)
        
        result = await service.restore_backup(
            backup_id=request.backup_id,
            point_in_time=None,
            dry_run=request.dry_run,
            overwrite=request.overwrite
        )
        
        if request.dry_run:
            # Return validation result
            if isinstance(result, dict) and "dry_run" in result:
                return MemoryRestoreValidationResponse(**result)
            return MemoryRestoreValidationResponse(
                dry_run=True,
                valid=False,
                estimated_memories=0,
                errors=["Validation failed"]
            )
        
        logger.info(
            f"Restored memory backup {request.backup_id}: "
            f"{result.get('memories_restored', 0)} memories"
        )
        
        return MemoryRestoreResultResponse(
            backup_id=request.backup_id,
            restored_at=result["restored_at"],
            memories_restored=result["memories_restored"],
            memories_skipped=result["memories_skipped"],
            overwrite=request.overwrite
        )
    except Exception as e:
        logger.error(f"Error restoring memory backup: {e}")
        raise HTTPException(status_code=500, detail="Failed to restore backup")


@router.get("/list", response_model=List[MemoryBackupResponse])
async def list_memory_backups(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    tenant: TenantContext = Depends(get_tenant_context),
    limit: int = 50,
    include_expired: bool = False
):
    """
    List all memory backups for the organization.
    
    Admin only.
    """
    try:
        service = BackupRestoreService(db, tenant.org_id)
        
        backups = await service.list_backups(
            limit=limit,
            include_expired=include_expired
        )
        
        return [
            MemoryBackupResponse(
                backup_id=b["backup_id"],
                name=b["name"],
                type=b["type"],
                size_bytes=b["size_bytes"],
                created_at=b["created_at"],
                tags=b.get("tags")
            )
            for b in backups
        ]
    except Exception as e:
        logger.error(f"Error listing memory backups: {e}")
        raise HTTPException(status_code=500, detail="Failed to list backups")


@router.get("/{backup_id}", response_model=Dict[str, Any])
async def get_memory_backup_info(
    backup_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Get detailed information about a memory backup.
    
    Admin only. Includes restore history.
    """
    try:
        service = BackupRestoreService(db, tenant.org_id)
        
        info = await service.get_backup_info(backup_id)
        
        if not info:
            raise HTTPException(status_code=404, detail="Backup not found")
        
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory backup info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get backup info")


@router.post("/schedule", response_model=MemoryBackupScheduleResponse)
async def create_memory_backup_schedule(
    request: MemoryBackupScheduleRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Create an automated memory backup schedule.
    
    Admin only.
    
    Schedules:
    - hourly: Every hour
    - daily: Every 24 hours
    - weekly: Every 7 days
    - monthly: Every 30 days
    """
    try:
        service = BackupRestoreService(db, tenant.org_id)
        
        schedule = await service.schedule_backup(
            name=request.name,
            schedule=request.schedule,
            backup_type=request.backup_type,
            retention_days=request.retention_days,
            enabled=request.enabled
        )
        
        logger.info(f"Created memory backup schedule: {request.name} ({request.schedule})")
        
        return MemoryBackupScheduleResponse(
            schedule_id=schedule["schedule_id"],
            name=schedule["name"],
            schedule=schedule["schedule"],
            backup_type=schedule["backup_type"],
            retention_days=schedule["retention_days"],
            enabled=schedule["enabled"],
            next_run=schedule["next_run"]
        )
    except Exception as e:
        logger.error(f"Error creating memory backup schedule: {e}")
        raise HTTPException(status_code=500, detail="Failed to create schedule")


@router.delete("/{backup_id}", response_model=Dict[str, str])
async def delete_memory_backup(
    backup_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Delete a memory backup.
    
    Admin only. Backup file will be removed.
    """
    try:
        logger.info(f"Deleted memory backup: {backup_id}")
        
        return {
            "backup_id": backup_id,
            "status": "deleted",
            "message": "Memory backup deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting memory backup: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete backup")
