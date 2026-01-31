"""
Memory Snapshot and Export Service

Handles creation, export, import, and management of memory snapshots.
Supports JSON, Markdown, and S3 storage.
"""

import json
import zipfile
from typing import List, Optional, Dict, Any, BinaryIO
from datetime import datetime, timedelta
from pathlib import Path
import logging
import uuid
import io

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.memory import Memory
from app.models.memory_snapshot import MemorySnapshot, SnapshotType, SnapshotStatus
from app.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class SnapshotService:
    """Service for creating and managing memory snapshots."""
    
    def __init__(self, db: AsyncSession, user_id: uuid.UUID, org_id: uuid.UUID):
        self.db = db
        self.user_id = user_id
        self.org_id = org_id
        self.memory_service = MemoryService(db, user_id, org_id)
    
    async def create_snapshot(
        self,
        name: str,
        snapshot_type: SnapshotType = SnapshotType.FULL,
        memory_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        format: str = "json",
        retention_days: int = 30
    ) -> MemorySnapshot:
        """
        Create a new memory snapshot.
        
        Args:
            name: Snapshot name
            snapshot_type: FULL, INCREMENTAL, or DIFFERENTIAL
            memory_ids: Specific memory IDs to snapshot (optional)
            filters: Filters for selecting memories (optional)
            format: Export format (json, markdown, zip)
            retention_days: Days to retain snapshot
            
        Returns:
            Created snapshot record
        """
        try:
            # Create snapshot record
            snapshot = MemorySnapshot(
                id=uuid.uuid4(),
                organization_id=self.org_id,
                user_id=self.user_id,
                name=name,
                snapshot_type=snapshot_type,
                status=SnapshotStatus.PENDING,
                format=format,
                retention_days=retention_days,
                expires_at=datetime.utcnow() + timedelta(days=retention_days),
                metadata={
                    "filters": filters or {},
                    "memory_ids": memory_ids or [],
                    "created_by": str(self.user_id)
                }
            )
            
            self.db.add(snapshot)
            await self.db.flush()
            
            # Start export process
            await self._execute_snapshot(snapshot, memory_ids, filters)
            
            await self.db.commit()
            
            logger.info(f"Created snapshot {snapshot.id} for org {self.org_id}")
            return snapshot
        
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            await self.db.rollback()
            raise
    
    async def _execute_snapshot(
        self,
        snapshot: MemorySnapshot,
        memory_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> None:
        """Execute snapshot creation."""
        try:
            snapshot.status = SnapshotStatus.IN_PROGRESS
            snapshot.started_at = datetime.utcnow()
            await self.db.flush()
            
            # Fetch memories
            if memory_ids:
                memories = await self._get_memories_by_ids(memory_ids)
            else:
                memories = await self._get_memories_by_filters(filters or {})
            
            # Export based on format
            if snapshot.format == "json":
                content = await self._export_json(memories)
            elif snapshot.format == "markdown":
                content = await self._export_markdown(memories)
            elif snapshot.format == "zip":
                content = await self._export_zip(memories)
            else:
                raise ValueError(f"Unsupported format: {snapshot.format}")
            
            # Store content
            snapshot.content_size_bytes = len(content)
            snapshot.file_path = await self._store_snapshot(snapshot.id, content, snapshot.format)
            snapshot.memory_count = len(memories)
            snapshot.status = SnapshotStatus.COMPLETED
            snapshot.completed_at = datetime.utcnow()
            
            logger.info(f"Snapshot {snapshot.id} completed: {len(memories)} memories, {len(content)} bytes")
        
        except Exception as e:
            logger.error(f"Error executing snapshot: {e}")
            snapshot.status = SnapshotStatus.FAILED
            snapshot.error_message = str(e)
            raise
    
    async def _get_memories_by_ids(self, memory_ids: List[str]) -> List[Memory]:
        """Fetch memories by IDs."""
        stmt = select(Memory).where(
            and_(
                Memory.id.in_([uuid.UUID(mid) for mid in memory_ids]),
                Memory.organization_id == self.org_id
            )
        )
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def _get_memories_by_filters(self, filters: Dict[str, Any]) -> List[Memory]:
        """Fetch memories by filters."""
        stmt = select(Memory).where(
            Memory.organization_id == self.org_id
        )
        
        # Apply filters
        if filters.get("scope"):
            stmt = stmt.where(Memory.scope == filters["scope"])
        if filters.get("tags"):
            # PostgreSQL array contains
            for tag in filters["tags"]:
                stmt = stmt.where(Memory.tags.contains([tag]))
        if filters.get("created_after"):
            stmt = stmt.where(Memory.created_at >= filters["created_after"])
        if filters.get("created_before"):
            stmt = stmt.where(Memory.created_at <= filters["created_before"])
        
        stmt = stmt.limit(filters.get("limit", 10000))
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def _export_json(self, memories: List[Memory]) -> bytes:
        """Export memories to JSON format."""
        data = {
            "version": "1.0",
            "export_date": datetime.utcnow().isoformat(),
            "organization_id": str(self.org_id),
            "memory_count": len(memories),
            "memories": [
                {
                    "id": str(m.id),
                    "title": m.title,
                    "content": m.content,
                    "scope": m.scope,
                    "tags": m.tags,
                    "session_id": m.session_id,
                    "agent_name": m.agent_name,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                    "metadata": m.metadata
                }
                for m in memories
            ]
        }
        
        return json.dumps(data, indent=2).encode('utf-8')
    
    async def _export_markdown(self, memories: List[Memory]) -> bytes:
        """Export memories to Markdown format."""
        lines = [
            f"# Memory Export",
            f"",
            f"**Organization**: {self.org_id}",
            f"**Export Date**: {datetime.utcnow().isoformat()}",
            f"**Memory Count**: {len(memories)}",
            f"",
            f"---",
            f""
        ]
        
        for m in memories:
            lines.extend([
                f"",
                f"## {m.title}",
                f"",
                f"**ID**: `{m.id}`  ",
                f"**Scope**: {m.scope}  ",
                f"**Tags**: {', '.join(m.tags) if m.tags else 'None'}  ",
                f"**Created**: {m.created_at.isoformat() if m.created_at else 'N/A'}  ",
                f"**Session**: {m.session_id or 'N/A'}  ",
                f"**Agent**: {m.agent_name or 'N/A'}  ",
                f"",
                f"### Content",
                f"",
                m.content,
                f"",
                f"---",
                f""
            ])
        
        return "\n".join(lines).encode('utf-8')
    
    async def _export_zip(self, memories: List[Memory]) -> bytes:
        """Export memories to ZIP with both JSON and Markdown."""
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add JSON export
            json_content = await self._export_json(memories)
            zf.writestr("memories.json", json_content)
            
            # Add Markdown export
            md_content = await self._export_markdown(memories)
            zf.writestr("memories.md", md_content)
            
            # Add individual memory files
            for m in memories:
                filename = f"memories/{m.id}.md"
                content = f"# {m.title}\n\n{m.content}"
                zf.writestr(filename, content.encode('utf-8'))
        
        buffer.seek(0)
        return buffer.read()
    
    async def _store_snapshot(self, snapshot_id: uuid.UUID, content: bytes, format: str) -> str:
        """
        Store snapshot content to filesystem or S3.
        
        Returns file path or S3 URL.
        """
        # Local filesystem storage (production should use S3)
        storage_dir = Path("backend_attachments") / "snapshots"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = storage_dir / f"{snapshot_id}.{format}"
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return str(file_path)
    
    async def get_snapshot(self, snapshot_id: uuid.UUID) -> Optional[MemorySnapshot]:
        """Get snapshot by ID."""
        stmt = select(MemorySnapshot).where(
            and_(
                MemorySnapshot.id == snapshot_id,
                MemorySnapshot.organization_id == self.org_id
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def list_snapshots(
        self,
        status: Optional[SnapshotStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[MemorySnapshot]:
        """List snapshots for organization."""
        stmt = select(MemorySnapshot).where(
            MemorySnapshot.organization_id == self.org_id
        )
        
        if status:
            stmt = stmt.where(MemorySnapshot.status == status)
        
        stmt = stmt.order_by(MemorySnapshot.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def download_snapshot(self, snapshot_id: uuid.UUID) -> Optional[bytes]:
        """Download snapshot content."""
        snapshot = await self.get_snapshot(snapshot_id)
        if not snapshot or not snapshot.file_path:
            return None
        
        try:
            with open(snapshot.file_path, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Snapshot file not found: {snapshot.file_path}")
            return None
    
    async def import_snapshot(
        self,
        content: bytes,
        format: str = "json",
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Import memories from snapshot.
        
        Args:
            content: Snapshot content bytes
            format: Format (json, zip)
            overwrite: Whether to overwrite existing memories
            
        Returns:
            Import result with counts
        """
        try:
            if format == "json":
                data = json.loads(content.decode('utf-8'))
                memories_data = data.get("memories", [])
            elif format == "zip":
                # Extract JSON from ZIP
                buffer = io.BytesIO(content)
                with zipfile.ZipFile(buffer, 'r') as zf:
                    json_content = zf.read("memories.json")
                    data = json.loads(json_content.decode('utf-8'))
                    memories_data = data.get("memories", [])
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            imported = 0
            skipped = 0
            errors = []
            
            for mem_data in memories_data:
                try:
                    # Check if memory exists
                    memory_id = uuid.UUID(mem_data["id"])
                    existing = await self.memory_service.get_memory(str(memory_id))
                    
                    if existing and not overwrite:
                        skipped += 1
                        continue
                    
                    # Create or update memory
                    from app.schemas.memory import MemoryCreate
                    memory_create = MemoryCreate(
                        title=mem_data["title"],
                        content=mem_data["content"],
                        scope=mem_data.get("scope", "private"),
                        tags=mem_data.get("tags", []),
                        session_id=mem_data.get("session_id"),
                        agent_name=mem_data.get("agent_name"),
                        metadata=mem_data.get("metadata", {})
                    )
                    
                    await self.memory_service.create_memory(
                        memory_create,
                        user_id=str(self.user_id),
                        org_id=str(self.org_id)
                    )
                    
                    imported += 1
                
                except Exception as e:
                    errors.append(f"Memory {mem_data.get('id')}: {str(e)}")
            
            await self.db.commit()
            
            return {
                "imported": imported,
                "skipped": skipped,
                "errors": errors,
                "total": len(memories_data)
            }
        
        except Exception as e:
            logger.error(f"Error importing snapshot: {e}")
            await self.db.rollback()
            raise
    
    async def delete_snapshot(self, snapshot_id: uuid.UUID) -> bool:
        """Delete snapshot and its file."""
        snapshot = await self.get_snapshot(snapshot_id)
        if not snapshot:
            return False
        
        # Delete file
        if snapshot.file_path:
            try:
                Path(snapshot.file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Error deleting snapshot file: {e}")
        
        # Delete record
        await self.db.delete(snapshot)
        await self.db.commit()
        
        return True
    
    async def cleanup_expired_snapshots(self) -> int:
        """Delete expired snapshots. Returns count of deleted snapshots."""
        stmt = select(MemorySnapshot).where(
            and_(
                MemorySnapshot.expires_at < datetime.utcnow(),
                MemorySnapshot.status == SnapshotStatus.COMPLETED
            )
        )
        result = await self.db.execute(stmt)
        snapshots = result.scalars().all()
        
        count = 0
        for snapshot in snapshots:
            if await self.delete_snapshot(snapshot.id):
                count += 1
        
        return count
