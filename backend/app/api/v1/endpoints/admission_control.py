"""
Admission Control Management Endpoints

Provides API endpoints for managing admission control, quotas, and circuit breakers.
"""

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.admission_control import AdmissionControlService
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.middleware.tenant_context import get_tenant_context, TenantContext

router = APIRouter(prefix="/admission", tags=["admission-control"])


# Request/Response Models
class AdmissionCheckRequest(BaseModel):
    """Request to check if a request should be admitted."""
    request_type: str = Field(..., description="Type of request")
    user_id: Optional[uuid.UUID] = Field(None, description="User ID (uses current user if not provided)")
    priority: int = Field(5, ge=1, le=10, description="Request priority (1-10)")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class AdmissionResponse(BaseModel):
    """Response indicating admission decision."""
    admitted: bool = Field(..., description="Whether request was admitted")
    reason: str = Field(..., description="Reason for admission/rejection")
    priority: Optional[int] = Field(None, description="Request priority")
    metadata: Optional[dict] = Field(None, description="Additional response data")
    quota_remaining: Optional[int] = Field(None, description="Remaining quota")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")


class QuotaUpdateRequest(BaseModel):
    """Request to update quota for a user."""
    user_id: uuid.UUID = Field(..., description="User ID")
    request_type: str = Field(..., description="Request type")
    limit: int = Field(..., ge=0, description="Maximum requests allowed")
    window_seconds: int = Field(3600, ge=1, description="Time window in seconds")


class LoadThresholdUpdate(BaseModel):
    """Request to update system load threshold."""
    threshold: float = Field(..., ge=0.0, le=1.0, description="Load threshold (0-1)")


class CircuitBreakerReset(BaseModel):
    """Request to reset circuit breaker."""
    request_type: str = Field(..., description="Request type to reset")


# Endpoints
@router.post("/check", response_model=AdmissionResponse)
async def check_admission(
    request: AdmissionCheckRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """
    Check if a request should be admitted.
    
    Evaluates:
    - Circuit breaker state
    - Quota limits
    - System load
    - Rate limits
    """
    service = AdmissionControlService(db, tenant_context.org_id)
    
    user_id = request.user_id or current_user.id
    
    result = await service.should_admit_request(
        request_type=request.request_type,
        user_id=user_id,
        priority=request.priority,
        metadata=request.metadata
    )
    
    return AdmissionResponse(**result)


@router.post("/quota")
async def update_quota(
    request: QuotaUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """
    Update quota for a user/request type.
    
    Requires admin permissions.
    """
    # Check admin permission
    if not any(role in current_user.roles for role in ["admin", "org_admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required"
        )
    
    service = AdmissionControlService(db, tenant_context.org_id)
    
    result = await service.update_quota(
        user_id=request.user_id,
        request_type=request.request_type,
        limit=request.limit,
        window_seconds=request.window_seconds
    )
    
    return result


@router.get("/stats")
async def get_admission_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """
    Get admission control statistics.
    
    Returns:
    - Active quotas
    - Load thresholds
    - Admission/rejection counts
    """
    service = AdmissionControlService(db, tenant_context.org_id)
    
    stats = await service.get_admission_stats()
    
    return stats


@router.put("/load-threshold")
async def update_load_threshold(
    request: LoadThresholdUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """
    Update system load threshold for admission control.
    
    Requires admin permissions.
    """
    # Check admin permission
    if not any(role in current_user.roles for role in ["admin", "org_admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required"
        )
    
    service = AdmissionControlService(db, tenant_context.org_id)
    
    result = await service.set_load_threshold(request.threshold)
    
    return result


@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker(
    request: CircuitBreakerReset,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """
    Manually reset circuit breaker for a request type.
    
    Requires admin permissions.
    """
    # Check admin permission
    if not any(role in current_user.roles for role in ["admin", "org_admin"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required"
        )
    
    service = AdmissionControlService(db, tenant_context.org_id)
    
    result = await service.reset_circuit_breaker(request.request_type)
    
    return result
