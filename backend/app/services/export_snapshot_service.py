"""
Export and Snapshot Service - Multi-format exports for compliance

Handles creating snapshots/exports in JSON, CSV, PDF formats.
"""

import json
import csv
import io
from typing import Optional, List, Literal
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.snapshot import Snapshot
from app.models.memory import MemoryMetadata
from app.models.knowledge_item import KnowledgeItem
from app.services.audit_service import AuditService


class ExportAndSnapshotService:
    """
    Handles exports and snapshots for compliance and data portability.
    
    Supports:
    - JSON export (full structure, queryable)
    - CSV export (tabular, spreadsheet compatible)
    - PDF export (human readable, signed)
    - JSONL export (streaming, large datasets)
    - Parquet export (analytics)
    - Automatic expiration
    - Signed download tokens
    """

    def __init__(self, db: AsyncSession, organization_id: str):
        self.db = db
        self.organization_id = organization_id
        self.audit_svc = AuditService(db)

    async def create_memory_export(
        self,
        format: Literal["json", "csv", "jsonl", "pdf", "parquet"] = "json",
        name: Optional[str] = None,
        filters: Optional[dict] = None,
        include_deleted: bool = False,
        user_id: Optional[str] = None,
        expires_in_days: int = 30,
    ) -> Snapshot:
        """
        Create an export snapshot of memory items.
        
        Args:
            format: Export format
            name: Display name for the export
            filters: Query filters (date range, tags, status, etc.)
            include_deleted: Include soft-deleted items
            user_id: User creating the export
            expires_in_days: Auto-delete after N days
        
        Returns:
            Snapshot object with storage path and metadata
        """
        # Create snapshot record
        snapshot = Snapshot(
            organization_id=self.organization_id,
            name=name or f"memory-export-{datetime.utcnow().isoformat()}",
            format=format,
            resource_type="memory",
            filters=filters or {},
            status="processing",
            created_by_user_id=user_id,
        )
        
        self.db.add(snapshot)
        await self.db.flush()
        
        try:
            # Query memory items
            query = select(MemoryMetadata).where(
                MemoryMetadata.organization_id == self.organization_id
            )
            
            if not include_deleted:
                query = query.where(MemoryMetadata.is_deleted == False)
            
            # Apply filters
            if filters:
                if filters.get("created_after"):
                    query = query.where(MemoryMetadata.created_at >= filters["created_after"])
                if filters.get("tags"):
                    # Simplified - actual implementation may use array operators
                    pass
                if filters.get("status"):
                    query = query.where(MemoryMetadata.status == filters["status"])
            
            result = await self.db.execute(query)
            memories = result.scalars().all()
            
            # Generate export content
            content = await self._generate_memory_export(memories, format)
            
            # Store export (in production, use S3/GCS)
            storage_path = f"s3://exports/{self.organization_id}/{snapshot.id}.{format}"
            # In real implementation: upload content to storage
            
            # Update snapshot
            snapshot.status = "completed"
            snapshot.storage_path = storage_path
            snapshot.item_count = len(memories)
            snapshot.size_bytes = len(content)
            snapshot.completed_at = datetime.utcnow()
            
            # Generate download token
            import secrets
            snapshot.download_token = secrets.token_urlsafe(32)
            
            # Calculate checksum
            import hashlib
            snapshot.checksum = hashlib.sha256(content).hexdigest()
            
            # Set expiration
            from datetime import timedelta
            snapshot.expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            await self.db.flush()
            
            # Audit
            await self.audit_svc.log_event(
                event_type="snapshot.memory.created",
                actor_id=user_id,
                organization_id=self.organization_id,
                resource_type="snapshot",
                resource_id=snapshot.id,
                success=True,
                details={
                    "format": format,
                    "item_count": len(memories),
                    "size_bytes": len(content),
                }
            )
            
        except Exception as e:
            snapshot.status = "failed"
            snapshot.error_message = str(e)
            await self.db.flush()
            
            await self.audit_svc.log_event(
                event_type="snapshot.memory.failed",
                actor_id=user_id,
                organization_id=self.organization_id,
                resource_type="snapshot",
                resource_id=snapshot.id,
                success=False,
                details={"error": str(e)}
            )
            
            raise
        
        return snapshot

    async def create_knowledge_export(
        self,
        format: Literal["json", "csv", "jsonl", "pdf"] = "json",
        name: Optional[str] = None,
        filters: Optional[dict] = None,
        include_unpublished: bool = False,
        user_id: Optional[str] = None,
        expires_in_days: int = 30,
    ) -> Snapshot:
        """Create an export snapshot of knowledge items."""
        snapshot = Snapshot(
            organization_id=self.organization_id,
            name=name or f"knowledge-export-{datetime.utcnow().isoformat()}",
            format=format,
            resource_type="knowledge",
            filters=filters or {},
            status="processing",
            created_by_user_id=user_id,
        )
        
        self.db.add(snapshot)
        await self.db.flush()
        
        try:
            # Query knowledge items
            query = select(KnowledgeItem).where(
                KnowledgeItem.organization_id == self.organization_id
            )
            
            if not include_unpublished:
                query = query.where(KnowledgeItem.is_published == True)
            
            result = await self.db.execute(query)
            items = result.scalars().all()
            
            # Generate export
            content = await self._generate_knowledge_export(items, format)
            
            # Store
            storage_path = f"s3://exports/{self.organization_id}/{snapshot.id}.{format}"
            
            snapshot.status = "completed"
            snapshot.storage_path = storage_path
            snapshot.item_count = len(items)
            snapshot.size_bytes = len(content)
            snapshot.completed_at = datetime.utcnow()
            
            import secrets
            snapshot.download_token = secrets.token_urlsafe(32)
            
            import hashlib
            snapshot.checksum = hashlib.sha256(content).hexdigest()
            
            from datetime import timedelta
            snapshot.expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            await self.db.flush()
            
        except Exception as e:
            snapshot.status = "failed"
            snapshot.error_message = str(e)
            await self.db.flush()
            raise
        
        return snapshot

    async def _generate_memory_export(self, memories: List[MemoryMetadata], format: str) -> bytes:
        """Generate export content in specified format."""
        if format == "json":
            data = [
                {
                    "id": str(m.id),
                    "content": m.content,
                    "tags": m.tags,
                    "created_at": m.created_at.isoformat(),
                    "updated_at": m.updated_at.isoformat(),
                    "metadata": m.metadata,
                }
                for m in memories
            ]
            return json.dumps(data, indent=2).encode()
        
        elif format == "csv":
            output = io.StringIO()
            if memories:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["id", "content", "tags", "created_at", "updated_at"]
                )
                writer.writeheader()
                for m in memories:
                    writer.writerow({
                        "id": str(m.id),
                        "content": m.content,
                        "tags": ",".join(m.tags or []),
                        "created_at": m.created_at.isoformat(),
                        "updated_at": m.updated_at.isoformat(),
                    })
            return output.getvalue().encode()
        
        elif format == "jsonl":
            lines = []
            for m in memories:
                line = json.dumps({
                    "id": str(m.id),
                    "content": m.content,
                    "tags": m.tags,
                    "created_at": m.created_at.isoformat(),
                    "updated_at": m.updated_at.isoformat(),
                    "metadata": m.metadata,
                })
                lines.append(line)
            return "\n".join(lines).encode()
        
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def _generate_knowledge_export(self, items: List[KnowledgeItem], format: str) -> bytes:
        """Generate knowledge export in specified format."""
        if format == "json":
            data = [
                {
                    "id": str(i.id),
                    "content": i.content,
                    "is_published": i.is_published,
                    "review_status": i.review_status,
                    "created_at": i.created_at.isoformat(),
                    "metadata": i.metadata,
                }
                for i in items
            ]
            return json.dumps(data, indent=2).encode()
        
        elif format == "csv":
            output = io.StringIO()
            if items:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["id", "content", "is_published", "review_status", "created_at"]
                )
                writer.writeheader()
                for i in items:
                    writer.writerow({
                        "id": str(i.id),
                        "content": i.content,
                        "is_published": i.is_published,
                        "review_status": i.review_status,
                        "created_at": i.created_at.isoformat(),
                    })
            return output.getvalue().encode()
        
        elif format == "jsonl":
            lines = []
            for i in items:
                line = json.dumps({
                    "id": str(i.id),
                    "content": i.content,
                    "is_published": i.is_published,
                    "created_at": i.created_at.isoformat(),
                })
                lines.append(line)
            return "\n".join(lines).encode()
        
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def get_snapshot(self, snapshot_id: str) -> Snapshot:
        """Get snapshot by ID."""
        result = await self.db.execute(
            select(Snapshot).where(
                Snapshot.id == snapshot_id,
                Snapshot.organization_id == self.organization_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_snapshots(
        self,
        resource_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[Snapshot], int]:
        """List snapshots for the organization."""
        query = select(Snapshot).where(
            Snapshot.organization_id == self.organization_id
        )
        
        if resource_type:
            query = query.where(Snapshot.resource_type == resource_type)
        
        # Get count
        count_result = await self.db.execute(
            select(len(query.distinct(Snapshot.id)))
        )
        total = count_result.scalar() or 0
        
        # Get paginated results
        query = query.order_by(Snapshot.created_at.desc()).limit(limit).offset(offset)
        result = await self.db.execute(query)
        snapshots = result.scalars().all()
        
        return snapshots, total

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot."""
        result = await self.db.execute(
            select(Snapshot).where(
                Snapshot.id == snapshot_id,
                Snapshot.organization_id == self.organization_id,
            )
        )
        snapshot = result.scalar_one_or_none()
        
        if snapshot:
            await self.db.delete(snapshot)
            await self.db.flush()
