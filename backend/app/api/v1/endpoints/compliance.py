"""GDPR / Compliance Endpoints.

Provides right-to-be-forgotten, data retention policy management,
data portability export, and user consent recording.

Routes:
  GET    /compliance/retention-policy
  PUT    /compliance/retention-policy
  POST   /compliance/deletion-request
  GET    /compliance/deletion-request/{request_id}
  GET    /compliance/deletion-requests
  GET    /compliance/data-export
  POST   /compliance/consent
  GET    /compliance/consent
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.services.compliance_service import ComplianceService
from app.services.tenant_offboarding_service import TenantOffboardingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["Compliance (GDPR)"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class RetentionPolicyResponse(BaseModel):
    id: str
    organization_id: str
    retention_days_memory: int
    retention_days_audit: int
    auto_archive_after_days: int
    is_active: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class RetentionPolicyUpdate(BaseModel):
    retention_days_memory: int = Field(365, ge=1, le=3650)
    retention_days_audit: int = Field(730, ge=1, le=3650)
    auto_archive_after_days: int = Field(90, ge=1, le=1825)


class DeletionRequestCreate(BaseModel):
    user_id: str
    reason: Optional[str] = None
    deletion_scope: str = Field("account", pattern="^(account|org_data|all)$")


class DeletionRequestResponse(BaseModel):
    id: str
    user_id: str
    organization_id: str
    deletion_scope: str
    status: str
    reason: Optional[str] = None
    deleted_at: Optional[datetime] = None
    error_detail: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ConsentCreate(BaseModel):
    consent_type: str = Field(..., pattern="^(data_processing|analytics|marketing)$")
    granted: bool
    expires_at: Optional[datetime] = None


class ConsentResponse(BaseModel):
    id: str
    user_id: str
    organization_id: str
    consent_type: str
    granted: bool
    granted_at: datetime
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ExportRequest(BaseModel):
    request_type: str = Field("data_export", pattern="^(data_export|full_deletion)$")
    user_id: Optional[str] = None
    export_first: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_admin(ctx: TenantContext) -> None:
    roles = set((ctx.roles or "").split(","))
    if "admin" not in roles and "org_admin" not in roles and "system_admin" not in roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")


# ---------------------------------------------------------------------------
# Retention Policy
# ---------------------------------------------------------------------------

@router.get("/retention-policy", response_model=RetentionPolicyResponse)
async def get_retention_policy(
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> RetentionPolicyResponse:
    """Return the data retention policy for the caller's organization."""
    svc = ComplianceService(db)
    policy = await svc.get_retention_policy(ctx.organization_id)
    if policy is None:
        # Return default policy values when not yet configured
        return RetentionPolicyResponse(
            id="default",
            organization_id=ctx.organization_id,
            retention_days_memory=365,
            retention_days_audit=730,
            auto_archive_after_days=90,
            is_active=False,
        )
    return RetentionPolicyResponse.model_validate(policy)


@router.put("/retention-policy", response_model=RetentionPolicyResponse)
async def upsert_retention_policy(
    body: RetentionPolicyUpdate,
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> RetentionPolicyResponse:
    """Create or update the retention policy. Admin only."""
    _require_admin(ctx)
    svc = ComplianceService(db)
    policy = await svc.upsert_retention_policy(
        ctx.organization_id,
        retention_days_memory=body.retention_days_memory,
        retention_days_audit=body.retention_days_audit,
        auto_archive_after_days=body.auto_archive_after_days,
        created_by_user_id=ctx.user_id,
    )
    return RetentionPolicyResponse.model_validate(policy)


# ---------------------------------------------------------------------------
# Deletion Requests
# ---------------------------------------------------------------------------

@router.post(
    "/deletion-request",
    response_model=DeletionRequestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_deletion_request(
    body: DeletionRequestCreate,
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> DeletionRequestResponse:
    """Initiate a GDPR right-to-be-forgotten request. Admin only."""
    _require_admin(ctx)
    svc = ComplianceService(db)
    try:
        req = await svc.create_deletion_request(
            user_id=body.user_id,
            org_id=ctx.organization_id,
            requested_by_user_id=ctx.user_id,
            reason=body.reason,
            deletion_scope=body.deletion_scope,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # Dispatch async deletion task
    try:
        from app.tasks.gdpr_pipeline import process_deletion_task
        process_deletion_task.delay(str(req.id))
    except Exception:
        logger.exception("Failed to enqueue deletion task", extra={"request_id": str(req.id)})
        # Request is still created; operator can retry manually

    return DeletionRequestResponse.model_validate(req)


@router.get("/deletion-request/{request_id}", response_model=DeletionRequestResponse)
async def get_deletion_request(
    request_id: str,
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> DeletionRequestResponse:
    """Poll the status of a deletion request. Admin only."""
    _require_admin(ctx)
    svc = ComplianceService(db)
    req = await svc.get_deletion_request(request_id)
    if req is None or req.organization_id != ctx.organization_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Deletion request not found")
    return DeletionRequestResponse.model_validate(req)


@router.get("/deletion-requests", response_model=List[DeletionRequestResponse])
async def list_deletion_requests(
    status_filter: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> List[DeletionRequestResponse]:
    """List deletion requests for the organization. Admin only."""
    _require_admin(ctx)
    svc = ComplianceService(db)
    requests = await svc.list_deletion_requests(
        ctx.organization_id, status=status_filter, limit=limit
    )
    return [DeletionRequestResponse.model_validate(r) for r in requests]


# ---------------------------------------------------------------------------
# Data Export (portability)
# ---------------------------------------------------------------------------

@router.get("/data-export")
async def request_data_export(
    user_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> dict:
    """Request a JSON export of a user's data. Returns an export job ID.

    Uses the existing ExportJobService. Admins may specify user_id; otherwise
    the export covers the caller's own data.
    """
    target_user = user_id if user_id else ctx.user_id
    if user_id and user_id != ctx.user_id:
        _require_admin(ctx)

    return await _create_gdpr_export_job(db=db, ctx=ctx, target_user=target_user)


async def _create_gdpr_export_job(db: AsyncSession, ctx: TenantContext, target_user: str) -> dict:
    try:
        from app.services.export_job_service import ExportJobService

        export_svc = ExportJobService(db)
        job = await export_svc.create_snapshot_job(
            org_id=ctx.organization_id,
            created_by_user_id=ctx.user_id,
            job_type="gdpr_export",
            metadata={"target_user_id": target_user},
        )
        return {
            "job_id": str(job.id),
            "status": job.status,
            "message": "Export job created. Poll /exports/jobs/{job_id} for status.",
        }
    except Exception:
        logger.exception("Failed to create GDPR export job")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate export",
        )


@router.post("/export/request")
async def request_export_or_deletion(
    body: ExportRequest,
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> dict:
    """Request GDPR export or full deletion for the current organization."""
    _require_admin(ctx)

    if body.request_type == "full_deletion":
        offboarding = TenantOffboardingService(db)
        report = await offboarding.offboard(
            ctx.organization_id,
            export_first=body.export_first,
        )
        await db.commit()
        return {
            "request_type": body.request_type,
            "status": "completed",
            "offboarding": {
                "org_id": report.org_id,
                "subscription_canceled": report.subscription_canceled,
                "memories_deleted": report.memories_deleted,
                "users_anonymized": report.users_anonymized,
                "export_path": report.export_path,
                "completed_at": report.completed_at,
            },
        }

    target_user = body.user_id or ctx.user_id
    if body.user_id and body.user_id != ctx.user_id:
        _require_admin(ctx)
    return await _create_gdpr_export_job(db=db, ctx=ctx, target_user=target_user)


# ---------------------------------------------------------------------------
# Consent Records
# ---------------------------------------------------------------------------

@router.post("/consent", response_model=ConsentResponse, status_code=status.HTTP_201_CREATED)
async def record_consent(
    body: ConsentCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> ConsentResponse:
    """Record a user consent grant or revoke."""
    svc = ComplianceService(db)
    try:
        record = await svc.record_consent(
            user_id=ctx.user_id,
            org_id=ctx.organization_id,
            consent_type=body.consent_type,
            granted=body.granted,
            ip_address=request.client.host if request.client else None,
            expires_at=body.expires_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return ConsentResponse.model_validate(record)


@router.get("/consent", response_model=List[ConsentResponse])
async def list_consents(
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> List[ConsentResponse]:
    """List all consent records for the calling user."""
    svc = ComplianceService(db)
    records = await svc.list_user_consents(ctx.user_id, ctx.organization_id)
    return [ConsentResponse.model_validate(r) for r in records]
