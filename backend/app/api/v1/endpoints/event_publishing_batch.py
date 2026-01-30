"""
Event Publishing and Batch Operations API Endpoints - Phase 7

Exposes:
- Event publishing and listing
- Webhook subscription management
- Batch operations (update/delete/share)
- Snapshot/export creation and management
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import get_tenant_context, TenantContext, get_db_with_tenant
from app.services.event_publishing_service import EventPublishingService
from app.services.batch_operations_service import BatchOperationsService
from app.services.export_snapshot_service import ExportAndSnapshotService
from app.schemas.event_publishing import (
    EventResponse, EventListResponse,
    WebhookSubscriptionCreate, WebhookSubscriptionUpdate, WebhookSubscriptionResponse,
    WebhookSubscriptionListResponse,
    BatchUpdateMemoryRequest, BatchDeleteMemoryRequest, BatchShareMemoryRequest,
    BatchUpdateKnowledgeRequest, BatchDeleteKnowledgeRequest,
    BatchOperationResult,
    SnapshotCreateRequest, SnapshotResponse, SnapshotListResponse,
)


router = APIRouter(prefix="/events", tags=["events_and_batch"])


# ============================================================================
# EVENT ENDPOINTS
# ============================================================================

@router.get("/", response_model=EventListResponse)
async def list_events(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db_with_tenant),
    event_type: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    List events for the organization.
    
    Supports filtering by event_type, resource_type, resource_id.
    """
    svc = EventPublishingService(db, tenant.org_id)
    events, total = await svc.get_events(
        event_type=event_type,
        resource_type=resource_type,
        resource_id=resource_id,
        limit=limit,
        offset=offset,
    )
    
    return EventListResponse(
        events=[EventResponse.from_orm(e) for e in events],
        total=total,
        limit=limit,
        offset=offset,
    )


# ============================================================================
# WEBHOOK ENDPOINTS
# ============================================================================

@router.post("/webhooks", response_model=WebhookSubscriptionResponse)
async def create_webhook(
    request: WebhookSubscriptionCreate,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db_with_tenant),
):
    """Create a new webhook subscription."""
    svc = EventPublishingService(db, tenant.org_id)
    subscription = await svc.create_subscription(
        url=str(request.url),
        event_types=request.event_types,
        resource_types=request.resource_types,
        secret=request.secret or "",
        max_retries=request.max_retries,
        rate_limit_per_minute=request.rate_limit_per_minute,
        description=request.description,
        custom_headers=request.custom_headers,
        created_by_user_id=tenant.user_id,
    )
    await db.commit()
    return WebhookSubscriptionResponse.from_orm(subscription)


@router.get("/webhooks", response_model=WebhookSubscriptionListResponse)
async def list_webhooks(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db_with_tenant),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List webhook subscriptions for the organization."""
    svc = EventPublishingService(db, tenant.org_id)
    subscriptions, total = await svc.list_subscriptions(limit=limit, offset=offset)
    await db.commit()
    
    return WebhookSubscriptionListResponse(
        subscriptions=[WebhookSubscriptionResponse.from_orm(s) for s in subscriptions],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.patch("/webhooks/{subscription_id}", response_model=WebhookSubscriptionResponse)
async def update_webhook(
    subscription_id: str,
    request: WebhookSubscriptionUpdate,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db_with_tenant),
):
    """Update a webhook subscription."""
    svc = EventPublishingService(db, tenant.org_id)
    
    update_data = request.dict(exclude_unset=True)
    update_data["updated_by_user_id"] = tenant.user_id
    
    subscription = await svc.update_subscription(subscription_id, **update_data)
    await db.commit()
    return WebhookSubscriptionResponse.from_orm(subscription)


@router.delete("/webhooks/{subscription_id}")
async def delete_webhook(
    subscription_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db_with_tenant),
):
    """Delete a webhook subscription."""
    svc = EventPublishingService(db, tenant.org_id)
    await svc.delete_subscription(subscription_id)
    await db.commit()
    return {"status": "deleted"}


# ============================================================================
# BATCH OPERATIONS ENDPOINTS
# ============================================================================

@router.post("/batch/memory/update", response_model=BatchOperationResult)
async def batch_update_memory(
    request: BatchUpdateMemoryRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db_with_tenant),
):
    """Bulk update memory items."""
    svc = BatchOperationsService(db, tenant.org_id, tenant.user_id)
    
    operation = await svc.bulk_update_memory(
        memory_ids=request.memory_ids,
        tags=request.tags,
        is_starred=request.is_starred,
        status=request.status,
        metadata=request.metadata,
    )
    
    await db.commit()
    
    import time
    duration = (time.time() - operation.start_time.timestamp())
    
    return BatchOperationResult(
        operation_type=operation.operation_type,
        resource_type=operation.resource_type,
        total_items=operation.total_items,
        successful=operation.successful,
        failed=operation.failed,
        errors=operation.errors,
        duration_seconds=duration,
    )


@router.post("/batch/memory/delete", response_model=BatchOperationResult)
async def batch_delete_memory(
    request: BatchDeleteMemoryRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db_with_tenant),
):
    """Bulk delete memory items."""
    svc = BatchOperationsService(db, tenant.org_id, tenant.user_id)
    
    operation = await svc.bulk_delete_memory(
        memory_ids=request.memory_ids,
        soft_delete=request.soft_delete,
    )
    
    await db.commit()
    
    import time
    duration = (time.time() - operation.start_time.timestamp())
    
    return BatchOperationResult(
        operation_type=operation.operation_type,
        resource_type=operation.resource_type,
        total_items=operation.total_items,
        successful=operation.successful,
        failed=operation.failed,
        errors=operation.errors,
        duration_seconds=duration,
    )


@router.post("/batch/memory/share", response_model=BatchOperationResult)
async def batch_share_memory(
    request: BatchShareMemoryRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db_with_tenant),
):
    """Bulk share memory items."""
    svc = BatchOperationsService(db, tenant.org_id, tenant.user_id)
    
    operation = await svc.bulk_share_memory(
        memory_ids=request.memory_ids,
        shared_with_user_ids=request.shared_with_user_ids,
        shared_with_team_ids=request.shared_with_team_ids,
        access_level=request.access_level,
    )
    
    await db.commit()
    
    import time
    duration = (time.time() - operation.start_time.timestamp())
    
    return BatchOperationResult(
        operation_type=operation.operation_type,
        resource_type=operation.resource_type,
        total_items=operation.total_items,
        successful=operation.successful,
        failed=operation.failed,
        errors=operation.errors,
        duration_seconds=duration,
    )


@router.post("/batch/knowledge/update", response_model=BatchOperationResult)
async def batch_update_knowledge(
    request: BatchUpdateKnowledgeRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Bulk update knowledge items."""
    svc = BatchOperationsService(db, tenant.org_id, tenant.user_id)
    
    operation = await svc.bulk_update_knowledge(
        knowledge_ids=request.knowledge_ids,
        tags=request.tags,
        status=request.status,
        is_published=request.is_published,
        metadata=request.metadata,
    )
    
    await db.commit()
    
    import time
    duration = (time.time() - operation.start_time.timestamp())
    
    return BatchOperationResult(
        operation_type=operation.operation_type,
        resource_type=operation.resource_type,
        total_items=operation.total_items,
        successful=operation.successful,
        failed=operation.failed,
        errors=operation.errors,
        duration_seconds=duration,
    )


@router.post("/batch/knowledge/delete", response_model=BatchOperationResult)
async def batch_delete_knowledge(
    request: BatchDeleteKnowledgeRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Bulk delete knowledge items."""
    svc = BatchOperationsService(db, tenant.org_id, tenant.user_id)
    
    operation = await svc.bulk_delete_knowledge(
        knowledge_ids=request.knowledge_ids,
        soft_delete=request.soft_delete,
    )
    
    await db.commit()
    
    import time
    duration = (time.time() - operation.start_time.timestamp())
    
    return BatchOperationResult(
        operation_type=operation.operation_type,
        resource_type=operation.resource_type,
        total_items=operation.total_items,
        successful=operation.successful,
        failed=operation.failed,
        errors=operation.errors,
        duration_seconds=duration,
    )


# ============================================================================
# SNAPSHOT/EXPORT ENDPOINTS
# ============================================================================

@router.post("/snapshots", response_model=SnapshotResponse)
async def create_snapshot(
    request: SnapshotCreateRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Create a snapshot/export of memory or knowledge items."""
    svc = ExportAndSnapshotService(db, tenant.org_id)
    
    if request.resource_type == "memory":
        snapshot = await svc.create_memory_export(
            format=request.format,
            name=request.name,
            filters=request.filters,
            include_deleted=request.include_deleted,
            user_id=tenant.user_id,
            expires_in_days=request.expires_in_days,
        )
    elif request.resource_type == "knowledge":
        snapshot = await svc.create_knowledge_export(
            format=request.format,
            name=request.name,
            filters=request.filters,
            include_unpublished=request.include_unpublished,
            user_id=tenant.user_id,
            expires_in_days=request.expires_in_days,
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid resource_type")
    
    await db.commit()
    return SnapshotResponse.from_orm(snapshot)


@router.get("/snapshots", response_model=SnapshotListResponse)
async def list_snapshots(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    resource_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List snapshots for the organization."""
    svc = ExportAndSnapshotService(db, tenant.org_id)
    snapshots, total = await svc.list_snapshots(
        resource_type=resource_type,
        limit=limit,
        offset=offset,
    )
    
    return SnapshotListResponse(
        snapshots=[SnapshotResponse.from_orm(s) for s in snapshots],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/snapshots/{snapshot_id}", response_model=SnapshotResponse)
async def get_snapshot(
    snapshot_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific snapshot."""
    svc = ExportAndSnapshotService(db, tenant.org_id)
    snapshot = await svc.get_snapshot(snapshot_id)
    
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    
    return SnapshotResponse.from_orm(snapshot)


@router.delete("/snapshots/{snapshot_id}")
async def delete_snapshot(
    snapshot_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Delete a snapshot."""
    svc = ExportAndSnapshotService(db, tenant.org_id)
    await svc.delete_snapshot(snapshot_id)
    await db.commit()
    return {"status": "deleted"}
