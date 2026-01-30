"""
Memory Snapshot Endpoints

REST API for creating, exporting, importing, and managing memory snapshots.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.models.memory_snapshot import SnapshotType, SnapshotStatus
from app.middleware.tenant_context import get_tenant_context, TenantContext
from app.services.snapshot_service import SnapshotService
from pydantic import BaseModel, Field
import logging

router = APIRouter(prefix="/snapshots", tags=["Memory Snapshots"])
logger = logging.getLogger(__name__)


# Schemas
class SnapshotCreateRequest(BaseModel):
    name: str = Field(..., description="Snapshot name")
    snapshot_type: SnapshotType = Field(default=SnapshotType.FULL, description="Snapshot type")
    memory_ids: Optional[List[str]] = Field(None, description="Specific memory IDs (optional)")
    filters: Optional[dict] = Field(None, description="Filters for selecting memories")
    format: str = Field(default="json", description="Export format (json/markdown/zip)")
    retention_days: int = Field(default=30, description="Days to retain snapshot")


class SnapshotResponse(BaseModel):
    id: str
    name: str
    snapshot_type: str
    status: str
    format: str
    memory_count: Optional[int] = None
    content_size_bytes: Optional[int] = None
    file_path: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    expires_at: Optional[str] = None
    error_message: Optional[str] = None


class ImportResult(BaseModel):
    imported: int
    skipped: int
    total: int
    errors: List[str]


@router.post("/", response_model=SnapshotResponse, status_code=status.HTTP_201_CREATED)
async def create_snapshot(
    request: SnapshotCreateRequest,
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new memory snapshot.
    
    **Formats**:
    - `json` - Structured JSON export
    - `markdown` - Human-readable Markdown
    - `zip` - ZIP archive with both formats
    
    **Snapshot Types**:
    - `FULL` - All memories matching filters
    - `INCREMENTAL` - Only new/changed since last snapshot
    - `DIFFERENTIAL` - Changes since last full snapshot
    """
    try:
        service = SnapshotService(db, tenant.user_id, tenant.org_id)
        
        snapshot = await service.create_snapshot(
            name=request.name,
            snapshot_type=request.snapshot_type,
            memory_ids=request.memory_ids,
            filters=request.filters,
            format=request.format,
            retention_days=request.retention_days
        )
        
        return SnapshotResponse(
            id=str(snapshot.id),
            name=snapshot.name,
            snapshot_type=snapshot.snapshot_type.value,
            status=snapshot.status.value,
            format=snapshot.format,
            memory_count=snapshot.memory_count,
            content_size_bytes=snapshot.content_size_bytes,
            file_path=snapshot.file_path,
            created_at=snapshot.created_at.isoformat(),
            started_at=snapshot.started_at.isoformat() if snapshot.started_at else None,
            completed_at=snapshot.completed_at.isoformat() if snapshot.completed_at else None,
            expires_at=snapshot.expires_at.isoformat() if snapshot.expires_at else None,
            error_message=snapshot.error_message
        )
    
    except Exception as e:
        logger.error(f"Error creating snapshot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create snapshot: {str(e)}"
        )


@router.get("/", response_model=List[SnapshotResponse])
async def list_snapshots(
    status_filter: Optional[SnapshotStatus] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
    current_user: User = Depends(get_current_user),
):
    """List all snapshots for the organization."""
    try:
        service = SnapshotService(db, tenant.user_id, tenant.org_id)
        
        snapshots = await service.list_snapshots(
            status=status_filter,
            limit=limit,
            offset=offset
        )
        
        return [
            SnapshotResponse(
                id=str(s.id),
                name=s.name,
                snapshot_type=s.snapshot_type.value,
                status=s.status.value,
                format=s.format,
                memory_count=s.memory_count,
                content_size_bytes=s.content_size_bytes,
                file_path=None,  # Don't expose file path
                created_at=s.created_at.isoformat(),
                started_at=s.started_at.isoformat() if s.started_at else None,
                completed_at=s.completed_at.isoformat() if s.completed_at else None,
                expires_at=s.expires_at.isoformat() if s.expires_at else None,
                error_message=s.error_message
            )
            for s in snapshots
        ]
    
    except Exception as e:
        logger.error(f"Error listing snapshots: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list snapshots: {str(e)}"
        )


@router.get("/{snapshot_id}", response_model=SnapshotResponse)
async def get_snapshot(
    snapshot_id: str,
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
    current_user: User = Depends(get_current_user),
):
    """Get snapshot details."""
    try:
        import uuid
        service = SnapshotService(db, tenant.user_id, tenant.org_id)
        
        snapshot = await service.get_snapshot(uuid.UUID(snapshot_id))
        
        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Snapshot {snapshot_id} not found"
            )
        
        return SnapshotResponse(
            id=str(snapshot.id),
            name=snapshot.name,
            snapshot_type=snapshot.snapshot_type.value,
            status=snapshot.status.value,
            format=snapshot.format,
            memory_count=snapshot.memory_count,
            content_size_bytes=snapshot.content_size_bytes,
            file_path=None,
            created_at=snapshot.created_at.isoformat(),
            started_at=snapshot.started_at.isoformat() if snapshot.started_at else None,
            completed_at=snapshot.completed_at.isoformat() if snapshot.completed_at else None,
            expires_at=snapshot.expires_at.isoformat() if snapshot.expires_at else None,
            error_message=snapshot.error_message
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting snapshot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get snapshot: {str(e)}"
        )


@router.get("/{snapshot_id}/download")
async def download_snapshot(
    snapshot_id: str,
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
    current_user: User = Depends(get_current_user),
):
    """
    Download snapshot file.
    
    Returns the snapshot content with appropriate Content-Type header.
    """
    try:
        import uuid
        service = SnapshotService(db, tenant.user_id, tenant.org_id)
        
        snapshot = await service.get_snapshot(uuid.UUID(snapshot_id))
        
        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Snapshot {snapshot_id} not found"
            )
        
        if snapshot.status != SnapshotStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Snapshot is not completed (status: {snapshot.status.value})"
            )
        
        content = await service.download_snapshot(uuid.UUID(snapshot_id))
        
        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Snapshot file not found"
            )
        
        # Determine content type
        content_types = {
            "json": "application/json",
            "markdown": "text/markdown",
            "zip": "application/zip"
        }
        content_type = content_types.get(snapshot.format, "application/octet-stream")
        
        # Determine filename
        filename = f"{snapshot.name}.{snapshot.format}"
        
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading snapshot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download snapshot: {str(e)}"
        )


@router.post("/import", response_model=ImportResult)
async def import_snapshot(
    file: UploadFile = File(...),
    format: str = "json",
    overwrite: bool = False,
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
    current_user: User = Depends(get_current_user),
):
    """
    Import memories from snapshot file.
    
    **Supported formats**: json, zip
    
    **Overwrite**: If true, existing memories will be updated.
    If false, existing memories are skipped.
    """
    try:
        service = SnapshotService(db, tenant.user_id, tenant.org_id)
        
        # Read file content
        content = await file.read()
        
        # Import
        result = await service.import_snapshot(
            content=content,
            format=format,
            overwrite=overwrite
        )
        
        return ImportResult(**result)
    
    except Exception as e:
        logger.error(f"Error importing snapshot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import snapshot: {str(e)}"
        )


@router.delete("/{snapshot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_snapshot(
    snapshot_id: str,
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
    current_user: User = Depends(get_current_user),
):
    """Delete a snapshot and its file."""
    try:
        import uuid
        service = SnapshotService(db, tenant.user_id, tenant.org_id)
        
        success = await service.delete_snapshot(uuid.UUID(snapshot_id))
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Snapshot {snapshot_id} not found"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting snapshot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete snapshot: {str(e)}"
        )


@router.post("/cleanup", status_code=status.HTTP_200_OK)
async def cleanup_expired_snapshots(
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
    current_user: User = Depends(get_current_user),
):
    """
    Cleanup expired snapshots.
    
    Admin operation to delete snapshots past their retention period.
    """
    try:
        if not current_user.is_admin and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        service = SnapshotService(db, tenant.user_id, tenant.org_id)
        
        count = await service.cleanup_expired_snapshots()
        
        return {
            "deleted": count,
            "message": f"Deleted {count} expired snapshots"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up snapshots: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup snapshots: {str(e)}"
        )
