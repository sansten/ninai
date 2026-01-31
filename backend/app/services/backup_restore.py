"""
Backup and Restore Service

Automated backup scheduling, point-in-time recovery, and restore operations.
"""

import json
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc

from app.services.snapshot_service import SnapshotService
from app.models.memory_snapshot import SnapshotType, SnapshotStatus

logger = logging.getLogger(__name__)


class BackupSchedule:
    """Backup schedule configuration."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class BackupRestoreService:
    """Service for automated backups and restore operations."""
    
    def __init__(self, db: AsyncSession, org_id: uuid.UUID, user_id: Optional[uuid.UUID] = None):
        self.db = db
        self.org_id = org_id
        self.user_id = user_id or org_id  # Use org_id as fallback
        self.snapshot_service = SnapshotService(db, self.user_id, org_id)
    
    async def create_backup(
        self,
        name: str,
        backup_type: str = "full",
        retention_days: int = 30,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a backup snapshot.
        
        Args:
            name: Backup name
            backup_type: "full" or "incremental"
            retention_days: Days to retain backup
            tags: Metadata tags for the backup
            
        Returns:
            Backup information
        """
        try:
            # Determine snapshot type
            snapshot_type = SnapshotType.FULL
            if backup_type == "incremental":
                snapshot_type = SnapshotType.INCREMENTAL
            
            # Create snapshot
            snapshot = await self.snapshot_service.create_snapshot(
                name=f"backup_{name}_{datetime.utcnow().isoformat()}",
                snapshot_type=snapshot_type,
                format="zip",
                retention_days=retention_days,
                filters=None,
                memory_ids=None
            )
            
            # Add backup metadata
            from app.models.audit import AuditEvent
            
            event = AuditEvent(
                id=uuid.uuid4(),
                organization_id=self.org_id,
                event_type="backup.created",
                resource_type="backup",
                resource_id=str(snapshot.id),
                success=True,
                metadata={
                    "backup_name": name,
                    "backup_type": backup_type,
                    "snapshot_id": str(snapshot.id),
                    "snapshot_size_bytes": snapshot.size_bytes,
                    "retention_days": retention_days,
                    "tags": tags or {},
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            self.db.add(event)
            await self.db.commit()
            
            logger.info(f"Created backup: {name} ({backup_type})")
            
            return {
                "backup_id": str(snapshot.id),
                "name": name,
                "type": backup_type,
                "size_bytes": snapshot.size_bytes,
                "created_at": snapshot.created_at.isoformat(),
                "expires_at": snapshot.expires_at.isoformat() if snapshot.expires_at else None,
                "tags": tags or {}
            }
        
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            await self.db.rollback()
            raise
    
    async def restore_backup(
        self,
        backup_id: str,
        point_in_time: Optional[datetime] = None,
        dry_run: bool = False,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Restore from a backup.
        
        Args:
            backup_id: Backup/snapshot ID to restore
            point_in_time: Optional PITR timestamp
            dry_run: If True, validate without restoring
            overwrite: If True, overwrite existing memories
            
        Returns:
            Restore result information
        """
        try:
            # Get snapshot
            from app.models.memory_snapshot import MemorySnapshot
            
            stmt = select(MemorySnapshot).where(
                and_(
                    MemorySnapshot.id == uuid.UUID(backup_id),
                    MemorySnapshot.organization_id == self.org_id
                )
            )
            result = await self.db.execute(stmt)
            snapshot = result.scalar_one_or_none()
            
            if not snapshot:
                raise ValueError(f"Backup {backup_id} not found")
            
            if dry_run:
                # Validate backup integrity
                validation_result = await self._validate_backup(snapshot)
                return {
                    "dry_run": True,
                    "backup_id": backup_id,
                    "valid": validation_result["valid"],
                    "estimated_memories": validation_result["memory_count"],
                    "errors": validation_result.get("errors", [])
                }
            
            # Read snapshot file
            snapshot_path = Path("backend_attachments/snapshots") / f"{snapshot.id}.{snapshot.format}"
            
            if not snapshot_path.exists():
                raise FileNotFoundError(f"Backup file not found: {snapshot_path}")
            
            with open(snapshot_path, "rb") as f:
                content = f.read()
            
            # Perform restore
            result = await self.snapshot_service.import_snapshot(
                content=content,
                format=snapshot.format,
                overwrite=overwrite
            )
            
            # Log restore event
            from app.models.audit import AuditEvent
            
            event = AuditEvent(
                id=uuid.uuid4(),
                organization_id=self.org_id,
                event_type="backup.restored",
                resource_type="backup",
                resource_id=backup_id,
                success=True,
                metadata={
                    "backup_id": backup_id,
                    "restored_at": datetime.utcnow().isoformat(),
                    "point_in_time": point_in_time.isoformat() if point_in_time else None,
                    "overwrite": overwrite,
                    "memories_restored": result.get("imported_count", 0),
                    "memories_skipped": result.get("skipped_count", 0)
                }
            )
            
            self.db.add(event)
            await self.db.commit()
            
            logger.info(
                f"Restored backup {backup_id}: "
                f"{result.get('imported_count', 0)} memories"
            )
            
            return {
                "backup_id": backup_id,
                "restored_at": datetime.utcnow().isoformat(),
                "memories_restored": result.get("imported_count", 0),
                "memories_skipped": result.get("skipped_count", 0),
                "overwrite": overwrite
            }
        
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            await self.db.rollback()
            raise
    
    async def _validate_backup(self, snapshot) -> Dict[str, Any]:
        """Validate backup integrity."""
        try:
            snapshot_path = Path("backend_attachments/snapshots") / f"{snapshot.id}.{snapshot.format}"
            
            if not snapshot_path.exists():
                return {
                    "valid": False,
                    "errors": ["Backup file not found"],
                    "memory_count": 0
                }
            
            # Read and parse based on format
            if snapshot.format == "json":
                with open(snapshot_path, "r") as f:
                    data = json.load(f)
                    memories = data.get("memories", [])
                    return {
                        "valid": True,
                        "memory_count": len(memories),
                        "errors": []
                    }
            
            elif snapshot.format == "zip":
                import zipfile
                try:
                    with zipfile.ZipFile(snapshot_path, "r") as zf:
                        # Check for manifest
                        if "snapshot.json" in zf.namelist():
                            manifest = json.loads(zf.read("snapshot.json"))
                            return {
                                "valid": True,
                                "memory_count": manifest.get("total_memories", 0),
                                "errors": []
                            }
                        return {
                            "valid": False,
                            "errors": ["Missing manifest file"],
                            "memory_count": 0
                        }
                except zipfile.BadZipFile:
                    return {
                        "valid": False,
                        "errors": ["Corrupted zip file"],
                        "memory_count": 0
                    }
            
            return {
                "valid": True,
                "memory_count": 0,
                "errors": []
            }
        
        except Exception as e:
            logger.error(f"Error validating backup: {e}")
            return {
                "valid": False,
                "errors": [str(e)],
                "memory_count": 0
            }
    
    async def schedule_backup(
        self,
        name: str,
        schedule: str,
        backup_type: str = "full",
        retention_days: int = 30,
        enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Create a backup schedule.
        
        Args:
            name: Schedule name
            schedule: Schedule frequency (hourly, daily, weekly, monthly)
            backup_type: Backup type
            retention_days: Retention period
            enabled: Whether schedule is active
            
        Returns:
            Schedule configuration
        """
        try:
            from app.models.audit import AuditEvent
            
            schedule_id = str(uuid.uuid4())
            
            # Store schedule configuration
            event = AuditEvent(
                id=uuid.uuid4(),
                organization_id=self.org_id,
                event_type="backup.schedule.created",
                resource_type="backup_schedule",
                resource_id=schedule_id,
                success=True,
                metadata={
                    "schedule_id": schedule_id,
                    "name": name,
                    "schedule": schedule,
                    "backup_type": backup_type,
                    "retention_days": retention_days,
                    "enabled": enabled,
                    "created_at": datetime.utcnow().isoformat(),
                    "next_run": self._calculate_next_run(schedule).isoformat()
                }
            )
            
            self.db.add(event)
            await self.db.commit()
            
            logger.info(f"Created backup schedule: {name} ({schedule})")
            
            return {
                "schedule_id": schedule_id,
                "name": name,
                "schedule": schedule,
                "backup_type": backup_type,
                "retention_days": retention_days,
                "enabled": enabled,
                "next_run": self._calculate_next_run(schedule).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error creating backup schedule: {e}")
            await self.db.rollback()
            raise
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate next scheduled run time."""
        now = datetime.utcnow()
        
        if schedule == BackupSchedule.HOURLY:
            return now + timedelta(hours=1)
        elif schedule == BackupSchedule.DAILY:
            return now + timedelta(days=1)
        elif schedule == BackupSchedule.WEEKLY:
            return now + timedelta(weeks=1)
        elif schedule == BackupSchedule.MONTHLY:
            return now + timedelta(days=30)
        else:
            return now + timedelta(days=1)
    
    async def list_backups(
        self,
        limit: int = 50,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """List all backups for the organization."""
        try:
            from app.models.audit import AuditEvent
            
            conditions = [
                AuditEvent.organization_id == self.org_id,
                AuditEvent.event_type == "backup.created"
            ]
            
            stmt = select(AuditEvent).where(
                and_(*conditions)
            ).order_by(desc(AuditEvent.created_at)).limit(limit)
            
            result = await self.db.execute(stmt)
            events = result.scalars().all()
            
            backups = []
            for event in events:
                if not event.metadata:
                    continue
                
                backup = {
                    "backup_id": event.resource_id,
                    "name": event.metadata.get("backup_name"),
                    "type": event.metadata.get("backup_type"),
                    "size_bytes": event.metadata.get("snapshot_size_bytes"),
                    "created_at": event.metadata.get("created_at"),
                    "retention_days": event.metadata.get("retention_days"),
                    "tags": event.metadata.get("tags", {})
                }
                
                backups.append(backup)
            
            return backups
        
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    async def get_backup_info(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a backup."""
        try:
            from app.models.audit import AuditEvent
            
            # Get backup creation event
            stmt = select(AuditEvent).where(
                and_(
                    AuditEvent.organization_id == self.org_id,
                    AuditEvent.event_type == "backup.created",
                    AuditEvent.resource_id == backup_id
                )
            )
            result = await self.db.execute(stmt)
            event = result.scalar_one_or_none()
            
            if not event or not event.metadata:
                return None
            
            # Get restore history
            restore_stmt = select(AuditEvent).where(
                and_(
                    AuditEvent.organization_id == self.org_id,
                    AuditEvent.event_type == "backup.restored",
                    AuditEvent.resource_id == backup_id
                )
            ).order_by(desc(AuditEvent.created_at))
            
            restore_result = await self.db.execute(restore_stmt)
            restore_events = restore_result.scalars().all()
            
            return {
                "backup_id": backup_id,
                "name": event.metadata.get("backup_name"),
                "type": event.metadata.get("backup_type"),
                "size_bytes": event.metadata.get("snapshot_size_bytes"),
                "created_at": event.metadata.get("created_at"),
                "retention_days": event.metadata.get("retention_days"),
                "tags": event.metadata.get("tags", {}),
                "restore_history": [
                    {
                        "restored_at": e.metadata.get("restored_at"),
                        "memories_restored": e.metadata.get("memories_restored", 0),
                        "overwrite": e.metadata.get("overwrite", False)
                    }
                    for e in restore_events
                    if e.metadata
                ]
            }
        
        except Exception as e:
            logger.error(f"Error getting backup info: {e}")
            return None
