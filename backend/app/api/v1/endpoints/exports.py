from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.models.export_job import ExportJob
from app.schemas.base import PaginatedResponse
from app.schemas.export_job import (
    SnapshotExportRequest,
    ExportJobResponse,
    ExportJobDownloadTokenResponse,
)
from app.services.export_job_service import ExportJobService
from app.services.audit_service import AuditService


admin_router = APIRouter()
public_router = APIRouter()


@asynccontextmanager
async def _maybe_begin(session: AsyncSession):
    try:
        async with session.begin():
            yield
    except TypeError:
        yield


def _job_to_response(job) -> ExportJobResponse:
    return ExportJobResponse.model_validate(job)


@admin_router.get("/exports/jobs", response_model=PaginatedResponse)
async def list_export_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    async with _maybe_begin(db):
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        total = int(
            (await db.execute(select(func.count()).select_from(ExportJob).where(ExportJob.organization_id == tenant.org_id)))
            .scalar()
            or 0
        )

        offset = (page - 1) * page_size
        stmt = (
            select(ExportJob)
            .where(ExportJob.organization_id == tenant.org_id)
            .order_by(desc(ExportJob.created_at))
            .offset(offset)
            .limit(page_size)
        )
        rows = (await db.execute(stmt)).scalars().all()

    return PaginatedResponse(
        items=[_job_to_response(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size if total > 0 else 0,
    )


@admin_router.post("/exports/snapshots", response_model=ExportJobResponse, status_code=status.HTTP_201_CREATED)
async def create_snapshot_export_job(
    request: Request,
    body: SnapshotExportRequest,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    request_id = getattr(getattr(request, "state", None), "request_id", None)

    async with _maybe_begin(db):
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
        svc = ExportJobService(db)
        job = await svc.create_snapshot_job(
            organization_id=tenant.org_id,
            created_by_user_id=tenant.user_id,
            expires_in_seconds=body.expires_in_seconds,
        )
        audit = AuditService(db)
        await audit.log_event(
            event_type="export.snapshot_job_created",
            actor_id=tenant.user_id,
            organization_id=tenant.org_id,
            resource_type="export_job",
            resource_id=job.id,
            success=True,
            request_id=request_id,
            details={"expires_in_seconds": body.expires_in_seconds},
        )
        await db.commit()

    celery_app.send_task(
        "app.tasks.export_jobs.run_snapshot_export_job_task",
        kwargs={"org_id": tenant.org_id, "job_id": job.id, "initiator_user_id": tenant.user_id},
    )

    return _job_to_response(job)


@admin_router.get("/exports/jobs/{job_id}", response_model=ExportJobResponse)
async def get_export_job(
    job_id: str,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    async with _maybe_begin(db):
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
        svc = ExportJobService(db)
        job = await svc.get_job(organization_id=tenant.org_id, job_id=job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export job not found")
        return _job_to_response(job)


@admin_router.get("/exports/jobs/{job_id}/download-token", response_model=ExportJobDownloadTokenResponse)
async def create_export_job_download_token(
    job_id: str,
    expires_in_seconds: int = Query(900, ge=60, le=24 * 3600),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    async with _maybe_begin(db):
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
        svc = ExportJobService(db)
        job = await svc.get_job(organization_id=tenant.org_id, job_id=job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export job not found")

        token, exp_dt = ExportJobService.build_download_token(
            job_id=job.id,
            org_id=tenant.org_id,
            expires_in_seconds=expires_in_seconds,
        )

    download_url = f"/api/v1/exports/jobs/{job_id}/download?token={token}"
    return ExportJobDownloadTokenResponse(job_id=job_id, expires_at=exp_dt, token=token, download_url=download_url)


@public_router.get("/exports/jobs/{job_id}/download")
async def download_export_job_bundle(
    request: Request,
    job_id: str,
    token: str = Query(..., description="Signed download token"),
    db: AsyncSession = Depends(get_db),
):
    request_id = getattr(getattr(request, "state", None), "request_id", None)

    payload = ExportJobService.verify_download_token(token)
    if payload is None or payload.job_id != job_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    service_user_id = str(getattr(settings, "SYSTEM_TASK_USER_ID", None) or "")
    service_roles = "system_admin" if service_user_id else ""

    async with _maybe_begin(db):
        # Use token org_id for org scoping.
        await set_tenant_context(db, service_user_id, payload.org_id, roles=service_roles, clearance_level=4)
        svc = ExportJobService(db)
        job = await svc.get_job(organization_id=payload.org_id, job_id=job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export job not found")

    if job.status != "succeeded":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Export job not completed")

    if job.expires_at is not None and job.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Export job expired")

    if not job.file_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export artifact missing")

    path = Path(job.file_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export file not found")

    async with _maybe_begin(db):
        await set_tenant_context(db, service_user_id, payload.org_id, roles=service_roles, clearance_level=4)
        await AuditService(db).log_event(
            event_type="export.snapshot_downloaded",
            actor_id=None,
            organization_id=payload.org_id,
            resource_type="export_job",
            resource_id=job.id,
            success=True,
            request_id=request_id,
            details={"bytes": job.file_bytes, "sha256": job.file_sha256},
            actor_type="system",
        )
        await db.commit()

    filename = path.name
    return FileResponse(path=str(path), media_type="application/zip", filename=filename)
