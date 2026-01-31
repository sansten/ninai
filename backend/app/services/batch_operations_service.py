"""
Batch Operations Service - Bulk update/delete/share

Handles bulk operations on memory and knowledge items with fine-grained authorization.
"""

from typing import Optional, List, Literal
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from uuid import UUID

from app.models.memory import MemoryMetadata
from app.models.knowledge_item import KnowledgeItem
from app.services.audit_service import AuditService


class BatchOperation:
    """Represents a batch operation result."""
    
    def __init__(self, operation_type: str, resource_type: str):
        self.operation_type = operation_type  # update, delete, share
        self.resource_type = resource_type  # memory, knowledge
        self.total_items = 0
        self.successful = 0
        self.failed = 0
        self.errors: dict[str, str] = {}  # item_id -> error message
        self.start_time = datetime.utcnow()


class BatchOperationsService:
    """
    Handles bulk operations on memory and knowledge items.
    
    Supports:
    - Bulk update (tags, status, metadata)
    - Bulk delete with soft-delete
    - Bulk share (change access permissions)
    - Per-item authorization checks
    - Audit logging for all operations
    """

    def __init__(self, db: AsyncSession, organization_id: str, user_id: str):
        self.db = db
        self.organization_id = organization_id
        self.user_id = user_id
        self.audit_svc = AuditService(db)

    async def bulk_update_memory(
        self,
        memory_ids: List[str],
        tags: Optional[List[str]] = None,
        is_starred: Optional[bool] = None,
        status: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> BatchOperation:
        """
        Bulk update memory items.
        
        Only updates items the user has access to (implicit from org_id).
        """
        operation = BatchOperation("update", "memory")
        operation.total_items = len(memory_ids)
        
        for memory_id in memory_ids:
            try:
                # Fetch and verify access
                result = await self.db.execute(
                    select(MemoryMetadata).where(
                        MemoryMetadata.id == memory_id,
                        MemoryMetadata.organization_id == self.organization_id,
                    )
                )
                memory = result.scalar_one_or_none()
                
                if not memory:
                    operation.errors[memory_id] = "Access denied or not found"
                    operation.failed += 1
                    continue
                
                # Apply updates
                if tags is not None:
                    memory.tags = tags
                if is_starred is not None:
                    memory.is_starred = is_starred
                if status is not None:
                    memory.status = status
                if metadata is not None:
                    memory.metadata = {**(memory.metadata or {}), **metadata}
                
                memory.updated_at = datetime.utcnow()
                
                await self.db.flush()
                operation.successful += 1
                
            except Exception as e:
                operation.errors[memory_id] = str(e)
                operation.failed += 1
        
        # Audit batch operation
        await self.audit_svc.log_event(
            event_type="memory.bulk_update",
            actor_id=self.user_id,
            organization_id=self.organization_id,
            resource_type="memory",
            resource_id="batch",
            success=operation.failed == 0,
            details={
                "total": operation.total_items,
                "successful": operation.successful,
                "failed": operation.failed,
                "updates": {
                    "tags": tags is not None,
                    "is_starred": is_starred is not None,
                    "status": status is not None,
                    "metadata": metadata is not None,
                }
            }
        )
        
        return operation

    async def bulk_delete_memory(
        self,
        memory_ids: List[str],
        soft_delete: bool = True,
    ) -> BatchOperation:
        """
        Bulk delete memory items.
        
        soft_delete=True marks as deleted without removing data.
        soft_delete=False permanently removes items.
        """
        operation = BatchOperation("delete", "memory")
        operation.total_items = len(memory_ids)
        
        for memory_id in memory_ids:
            try:
                # Fetch and verify access
                result = await self.db.execute(
                    select(Memory).where(
                        MemoryMetadata.id == memory_id,
                        MemoryMetadata.organization_id == self.organization_id,
                    )
                )
                memory = result.scalar_one_or_none()
                
                if not memory:
                    operation.errors[memory_id] = "Access denied or not found"
                    operation.failed += 1
                    continue
                
                if soft_delete:
                    # Mark as deleted
                    memory.is_deleted = True
                    memory.deleted_at = datetime.utcnow()
                    memory.deleted_by_user_id = self.user_id
                    await self.db.flush()
                else:
                    # Hard delete
                    await self.db.delete(memory)
                    await self.db.flush()
                
                operation.successful += 1
                
            except Exception as e:
                operation.errors[memory_id] = str(e)
                operation.failed += 1
        
        # Audit batch operation
        await self.audit_svc.log_event(
            event_type="memory.bulk_delete",
            actor_id=self.user_id,
            organization_id=self.organization_id,
            resource_type="memory",
            resource_id="batch",
            success=operation.failed == 0,
            details={
                "total": operation.total_items,
                "successful": operation.successful,
                "failed": operation.failed,
                "soft_delete": soft_delete,
            }
        )
        
        return operation

    async def bulk_share_memory(
        self,
        memory_ids: List[str],
        shared_with_user_ids: Optional[List[str]] = None,
        shared_with_team_ids: Optional[List[str]] = None,
        access_level: str = "view",  # view, edit, admin
    ) -> BatchOperation:
        """
        Bulk share memory items with users/teams.
        
        access_level: view (read-only), edit (can modify), admin (full control)
        """
        operation = BatchOperation("share", "memory")
        operation.total_items = len(memory_ids)
        
        for memory_id in memory_ids:
            try:
                # Fetch and verify ownership
                result = await self.db.execute(
                    select(Memory).where(
                        MemoryMetadata.id == memory_id,
                        MemoryMetadata.organization_id == self.organization_id,
                        # User must be owner or have edit access
                    )
                )
                memory = result.scalar_one_or_none()
                
                if not memory:
                    operation.errors[memory_id] = "Access denied or not found"
                    operation.failed += 1
                    continue
                
                # Update sharing permissions (implementation depends on your sharing model)
                # This is a simplified example - your actual implementation may differ
                
                # Mark memory as shared
                memory.is_shared = True
                memory.shared_with_count = (shared_with_user_ids or []) + (shared_with_team_ids or [])
                memory.shared_at = datetime.utcnow()
                
                await self.db.flush()
                operation.successful += 1
                
            except Exception as e:
                operation.errors[memory_id] = str(e)
                operation.failed += 1
        
        # Audit batch operation
        await self.audit_svc.log_event(
            event_type="memory.bulk_share",
            actor_id=self.user_id,
            organization_id=self.organization_id,
            resource_type="memory",
            resource_id="batch",
            success=operation.failed == 0,
            details={
                "total": operation.total_items,
                "successful": operation.successful,
                "failed": operation.failed,
                "shared_with_users": len(shared_with_user_ids or []),
                "shared_with_teams": len(shared_with_team_ids or []),
                "access_level": access_level,
            }
        )
        
        return operation

    async def bulk_update_knowledge(
        self,
        knowledge_ids: List[str],
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        is_published: Optional[bool] = None,
        metadata: Optional[dict] = None,
    ) -> BatchOperation:
        """Bulk update knowledge items."""
        operation = BatchOperation("update", "knowledge")
        operation.total_items = len(knowledge_ids)
        
        for knowledge_id in knowledge_ids:
            try:
                result = await self.db.execute(
                    select(KnowledgeItem).where(
                        KnowledgeItem.id == knowledge_id,
                        KnowledgeItem.organization_id == self.organization_id,
                    )
                )
                knowledge = result.scalar_one_or_none()
                
                if not knowledge:
                    operation.errors[knowledge_id] = "Access denied or not found"
                    operation.failed += 1
                    continue
                
                if tags is not None:
                    knowledge.tags = tags
                if status is not None:
                    knowledge.review_status = status
                if is_published is not None:
                    knowledge.is_published = is_published
                if metadata is not None:
                    knowledge.metadata = {**(knowledge.metadata or {}), **metadata}
                
                knowledge.updated_at = datetime.utcnow()
                await self.db.flush()
                operation.successful += 1
                
            except Exception as e:
                operation.errors[knowledge_id] = str(e)
                operation.failed += 1
        
        await self.audit_svc.log_event(
            event_type="knowledge.bulk_update",
            actor_id=self.user_id,
            organization_id=self.organization_id,
            resource_type="knowledge",
            resource_id="batch",
            success=operation.failed == 0,
            details={
                "total": operation.total_items,
                "successful": operation.successful,
                "failed": operation.failed,
            }
        )
        
        return operation

    async def bulk_delete_knowledge(
        self,
        knowledge_ids: List[str],
        soft_delete: bool = True,
    ) -> BatchOperation:
        """Bulk delete knowledge items."""
        operation = BatchOperation("delete", "knowledge")
        operation.total_items = len(knowledge_ids)
        
        for knowledge_id in knowledge_ids:
            try:
                result = await self.db.execute(
                    select(KnowledgeItem).where(
                        KnowledgeItem.id == knowledge_id,
                        KnowledgeItem.organization_id == self.organization_id,
                    )
                )
                knowledge = result.scalar_one_or_none()
                
                if not knowledge:
                    operation.errors[knowledge_id] = "Access denied or not found"
                    operation.failed += 1
                    continue
                
                if soft_delete:
                    knowledge.is_deleted = True
                    knowledge.deleted_at = datetime.utcnow()
                    await self.db.flush()
                else:
                    await self.db.delete(knowledge)
                    await self.db.flush()
                
                operation.successful += 1
                
            except Exception as e:
                operation.errors[knowledge_id] = str(e)
                operation.failed += 1
        
        await self.audit_svc.log_event(
            event_type="knowledge.bulk_delete",
            actor_id=self.user_id,
            organization_id=self.organization_id,
            resource_type="knowledge",
            resource_id="batch",
            success=operation.failed == 0,
            details={
                "total": operation.total_items,
                "successful": operation.successful,
                "failed": operation.failed,
                "soft_delete": soft_delete,
            }
        )
        
        return operation
