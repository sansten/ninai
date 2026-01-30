"""
Audit Endpoints
===============

Audit event and access log queries.
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import (
    TenantContext,
    get_tenant_context,
    require_roles,
)
from app.models.audit import AuditEvent, MemoryAccessLog
from app.schemas.base import PaginatedResponse


router = APIRouter()


# =============================================================================
# Audit Schemas (local to this module)
# =============================================================================

from pydantic import BaseModel, Field, field_validator


class AuditEventResponse(BaseModel):
    """Audit event response."""
    
    id: str
    timestamp: datetime
    organization_id: Optional[str]
    event_type: str
    event_category: str
    severity: str
    actor_id: Optional[str]
    actor_type: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    request_id: Optional[str]
    ip_address: Optional[str]  # Will be converted from INET
    user_agent: Optional[str]
    success: bool
    error_message: Optional[str]
    details: dict
    changes: Optional[dict]
    
    @field_validator('ip_address', mode='before')
    @classmethod
    def validate_ip_address(cls, v):
        """Convert INET to string."""
        if v is None:
            return None
        return str(v)
    
    model_config = {"from_attributes": True}


class AccessLogResponse(BaseModel):
    """Memory access log response."""
    
    id: str
    timestamp: datetime
    memory_id: str
    user_id: str
    organization_id: str
    action: str
    authorized: bool
    authorization_method: str
    denial_reason: Optional[str]
    access_context: dict
    request_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    justification: Optional[str]
    case_id: Optional[str]
    
    @field_validator('ip_address', mode='before')
    @classmethod
    def validate_ip_address(cls, v):
        """Convert INET to string."""
        if v is None:
            return None
        return str(v)
    
    model_config = {"from_attributes": True}


class AuditStatsResponse(BaseModel):
    """Audit statistics."""
    
    total_events: int
    events_by_type: dict
    events_by_action: dict
    top_actors: List[dict]
    recent_denials: int
    period_start: datetime
    period_end: datetime


# =============================================================================
# Audit Event Endpoints
# =============================================================================

@router.get("/events", response_model=PaginatedResponse)
async def list_audit_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    event_type: Optional[str] = None,
    actor_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    action: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    tenant: TenantContext = Depends(require_roles("org_admin", "security_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    List audit events with filtering.
    
    **Required role:** org_admin, security_admin, or system_admin
    
    Common event types:
    - auth.login, auth.logout, auth.failed_login
    - memory.create, memory.read, memory.update, memory.delete
    - permission.granted, permission.denied, permission.change
    - admin.user_created, admin.role_assigned
    """
    # Build query
    query = select(AuditEvent).where(
        AuditEvent.organization_id == tenant.org_id
    )
    count_query = select(func.count(AuditEvent.id)).where(
        AuditEvent.organization_id == tenant.org_id
    )
    
    # Apply filters
    if event_type:
        query = query.where(AuditEvent.event_type == event_type)
        count_query = count_query.where(AuditEvent.event_type == event_type)
    
    if actor_id:
        query = query.where(AuditEvent.actor_id == actor_id)
        count_query = count_query.where(AuditEvent.actor_id == actor_id)
    
    if resource_type:
        query = query.where(AuditEvent.resource_type == resource_type)
        count_query = count_query.where(AuditEvent.resource_type == resource_type)
    
    if resource_id:
        query = query.where(AuditEvent.resource_id == resource_id)
        count_query = count_query.where(AuditEvent.resource_id == resource_id)
    
    if action:
        query = query.where(AuditEvent.action == action)
        count_query = count_query.where(AuditEvent.action == action)
    
    if start_date:
        query = query.where(AuditEvent.timestamp >= start_date)
        count_query = count_query.where(AuditEvent.timestamp >= start_date)
    
    if end_date:
        query = query.where(AuditEvent.timestamp <= end_date)
        count_query = count_query.where(AuditEvent.timestamp <= end_date)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(desc(AuditEvent.timestamp))
    
    result = await db.execute(query)
    events = result.scalars().all()
    
    return PaginatedResponse(
        items=[AuditEventResponse.model_validate(e) for e in events],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size if total > 0 else 0,
    )


@router.get("/events/{event_id}", response_model=AuditEventResponse)
async def get_audit_event(
    event_id: str,
    tenant: TenantContext = Depends(require_roles("org_admin", "security_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific audit event.
    
    **Required role:** org_admin, security_admin, or system_admin
    """
    result = await db.execute(
        select(AuditEvent).where(
            AuditEvent.id == event_id,
            AuditEvent.organization_id == tenant.org_id,
        )
    )
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audit event not found",
        )
    
    return AuditEventResponse.model_validate(event)


@router.get("/events/{event_id}/related", response_model=List[AuditEventResponse])
async def get_related_events(
    event_id: str,
    tenant: TenantContext = Depends(require_roles("org_admin", "security_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Get events related to a specific event (same request_id or parent chain).
    
    **Required role:** org_admin, security_admin, or system_admin
    """
    # Get the original event
    result = await db.execute(
        select(AuditEvent).where(
            AuditEvent.id == event_id,
            AuditEvent.organization_id == tenant.org_id,
        )
    )
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audit event not found",
        )
    
    # Find related events
    related_result = await db.execute(
        select(AuditEvent).where(
            AuditEvent.organization_id == tenant.org_id,
            AuditEvent.id != event_id,
            (
                (AuditEvent.request_id == event.request_id) |
                (AuditEvent.parent_event_id == event_id) |
                (AuditEvent.id == event.parent_event_id)
            ),
        ).order_by(AuditEvent.timestamp)
    )
    related = related_result.scalars().all()
    
    return [AuditEventResponse.model_validate(e) for e in related]


# =============================================================================
# Access Log Endpoints
# =============================================================================

@router.get("/access-logs", response_model=PaginatedResponse)
async def list_access_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    memory_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    access_type: Optional[str] = None,
    access_granted: Optional[bool] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    tenant: TenantContext = Depends(require_roles("org_admin", "security_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    List memory access logs with filtering.
    
    **Required role:** org_admin, security_admin, or system_admin
    
    Access types: read, update, delete, share, search
    """
    # Build query
    query = select(MemoryAccessLog).where(
        MemoryAccessLog.organization_id == tenant.org_id
    )
    count_query = select(func.count(MemoryAccessLog.id)).where(
        MemoryAccessLog.organization_id == tenant.org_id
    )
    
    # Apply filters
    if memory_id:
        query = query.where(MemoryAccessLog.memory_id == memory_id)
        count_query = count_query.where(MemoryAccessLog.memory_id == memory_id)
    
    if user_id:
        query = query.where(MemoryAccessLog.user_id == user_id)
        count_query = count_query.where(MemoryAccessLog.user_id == user_id)
    
    if agent_id:
        # Kept for backward-compat with older clients; model has no agent_id.
        pass
    
    if access_type:
        # access_type maps to action in the current model
        query = query.where(MemoryAccessLog.action == access_type)
        count_query = count_query.where(MemoryAccessLog.action == access_type)
    
    if access_granted is not None:
        # access_granted maps to authorized in the current model
        query = query.where(MemoryAccessLog.authorized == access_granted)
        count_query = count_query.where(MemoryAccessLog.authorized == access_granted)
    
    if start_date:
        query = query.where(MemoryAccessLog.timestamp >= start_date)
        count_query = count_query.where(MemoryAccessLog.timestamp >= start_date)
    
    if end_date:
        query = query.where(MemoryAccessLog.timestamp <= end_date)
        count_query = count_query.where(MemoryAccessLog.timestamp <= end_date)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(desc(MemoryAccessLog.timestamp))
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return PaginatedResponse(
        items=[AccessLogResponse.model_validate(l) for l in logs],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size if total > 0 else 0,
    )


@router.get("/access-logs/memory/{memory_id}", response_model=List[AccessLogResponse])
async def get_memory_access_history(
    memory_id: str,
    limit: int = Query(100, ge=1, le=1000),
    tenant: TenantContext = Depends(require_roles("org_admin", "security_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Get complete access history for a specific memory.
    
    **Required role:** org_admin, security_admin, or system_admin
    """
    result = await db.execute(
        select(MemoryAccessLog).where(
            MemoryAccessLog.memory_id == memory_id,
            MemoryAccessLog.organization_id == tenant.org_id,
        ).order_by(desc(MemoryAccessLog.timestamp)).limit(limit)
    )
    logs = result.scalars().all()
    
    return [AccessLogResponse.model_validate(l) for l in logs]


# =============================================================================
# Statistics Endpoints
# =============================================================================

@router.get("/stats", response_model=AuditStatsResponse)
async def get_audit_stats(
    start_date: Optional[datetime] = Query(None, description="Period start (defaults to 7 days ago)"),
    end_date: Optional[datetime] = Query(None, description="Period end (defaults to now)"),
    tenant: TenantContext = Depends(require_roles("org_admin", "security_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Get audit statistics for a time period.
    
    **Required role:** org_admin, security_admin, or system_admin
    """
    from datetime import timedelta

    def _to_utc_naive(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(timezone.utc).replace(tzinfo=None)

    # Postgres columns are TIMESTAMP WITHOUT TIME ZONE; asyncpg rejects tz-aware params.
    # Treat incoming datetimes as UTC, converting aware inputs to UTC then dropping tzinfo.
    if end_date is None:
        end_date = datetime.utcnow()
    else:
        end_date = _to_utc_naive(end_date)

    if start_date is None:
        start_date = end_date - timedelta(days=7)
    else:
        start_date = _to_utc_naive(start_date)

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date must be <= end_date",
        )

    
    # Base filter
    base_filter = and_(
        AuditEvent.organization_id == tenant.org_id,
        AuditEvent.timestamp >= start_date,
        AuditEvent.timestamp <= end_date,
    )
    
    # Total events
    total_result = await db.execute(
        select(func.count(AuditEvent.id)).where(base_filter)
    )
    total_events = total_result.scalar() or 0
    
    # Events by type
    type_result = await db.execute(
        select(
            AuditEvent.event_type,
            func.count(AuditEvent.id).label("count"),
        ).where(base_filter)
        .group_by(AuditEvent.event_type)
    )
    events_by_type = {row[0]: row[1] for row in type_result.all()}
    
    # Events by action
    action_result = await db.execute(
        select(
            AuditEvent.event_category,
            func.count(AuditEvent.id).label("count"),
        ).where(base_filter)
        .group_by(AuditEvent.event_category)
    )
    events_by_action = {row[0]: row[1] for row in action_result.all()}
    
    # Top actors
    actor_result = await db.execute(
        select(
            AuditEvent.actor_id,
            AuditEvent.actor_type,
            func.count(AuditEvent.id).label("count"),
        ).where(
            base_filter,
            AuditEvent.actor_id.is_not(None),
        )
        .group_by(AuditEvent.actor_id, AuditEvent.actor_type)
        .order_by(desc("count"))
        .limit(10)
    )
    top_actors = [
        {"actor_id": row[0], "actor_type": row[1], "count": row[2]}
        for row in actor_result.all()
    ]
    
    # Recent denials (from access logs)
    denial_filter = and_(
        MemoryAccessLog.organization_id == tenant.org_id,
        MemoryAccessLog.timestamp >= start_date,
        MemoryAccessLog.timestamp <= end_date,
        MemoryAccessLog.authorized == False,
    )
    denial_result = await db.execute(
        select(func.count(MemoryAccessLog.id)).where(denial_filter)
    )
    recent_denials = denial_result.scalar() or 0
    
    return AuditStatsResponse(
        total_events=total_events,
        events_by_type=events_by_type,
        events_by_action=events_by_action,
        top_actors=top_actors,
        recent_denials=recent_denials,
        period_start=start_date.replace(tzinfo=timezone.utc),
        period_end=end_date.replace(tzinfo=timezone.utc),
    )


@router.get("/stats/denials", response_model=List[AccessLogResponse])
async def get_recent_denials(
    limit: int = Query(50, ge=1, le=200),
    tenant: TenantContext = Depends(require_roles("org_admin", "security_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Get recent access denials for security review.
    
    **Required role:** org_admin, security_admin, or system_admin
    """
    result = await db.execute(
        select(MemoryAccessLog).where(
            MemoryAccessLog.organization_id == tenant.org_id,
            MemoryAccessLog.authorized == False,
        ).order_by(desc(MemoryAccessLog.timestamp)).limit(limit)
    )
    logs = result.scalars().all()
    
    return [AccessLogResponse.model_validate(l) for l in logs]
