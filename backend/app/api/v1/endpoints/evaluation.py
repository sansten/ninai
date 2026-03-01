"""Evaluation API endpoints (PR6: Eval Harness + Drift Detection)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.eval_suite import EvalSuite
from app.models.eval_run import EvalRun
from app.models.drift_report import DriftReport
from app.schemas.evaluation import (
    EvalSuiteCreate,
    EvalSuiteUpdate,
    EvalSuiteResponse,
    EvalRunTrigger,
    EvalRunResponse,
    DriftReportTrigger,
    DriftReportResponse,
)
from app.services.eval_run_service import EvalRunService
from app.services.drift_detection_service import DriftDetectionService
from app.tasks.eval_pipeline import enqueue_eval_suite, enqueue_drift_computation
from uuid import uuid4

router = APIRouter(prefix="/eval", tags=["evaluation"])


# Eval Suite endpoints

@router.post("/suites", response_model=EvalSuiteResponse, status_code=status.HTTP_201_CREATED)
async def create_eval_suite(
    data: EvalSuiteCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
) -> EvalSuite:
    """Create a new evaluation suite."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, roles=tenant.roles, clearance_level=tenant.clearance)

    suite = EvalSuite(
        id=str(uuid4()),
        organization_id=tenant.org_id,
        name=data.name,
        description=data.description,
        queries=data.queries,
        expected=data.expected,
        is_active=data.is_active,
    )
    db.add(suite)
    await db.commit()
    await db.refresh(suite)
    
    return suite


@router.get("/suites/{suite_id}", response_model=EvalSuiteResponse)
async def get_eval_suite(
    suite_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
) -> EvalSuite:
    """Get an evaluation suite by ID."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, roles=tenant.roles, clearance_level=tenant.clearance)

    result = await db.execute(
        select(EvalSuite).where(
            EvalSuite.id == suite_id,
            EvalSuite.organization_id == tenant.org_id,
        )
    )
    suite = result.scalar_one_or_none()
    
    if not suite:
        raise HTTPException(status_code=404, detail="Eval suite not found")
    
    return suite


@router.get("/suites", response_model=list[EvalSuiteResponse])
async def list_eval_suites(
    db: Annotated[AsyncSession, Depends(get_db)],
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
    is_active: bool | None = None,
    limit: int = 50,
) -> list[EvalSuite]:
    """List evaluation suites for the organization."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, roles=tenant.roles, clearance_level=tenant.clearance)

    query = select(EvalSuite).where(EvalSuite.organization_id == tenant.org_id)
    
    if is_active is not None:
        query = query.where(EvalSuite.is_active == is_active)
    
    query = query.order_by(EvalSuite.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    return list(result.scalars().all())


@router.patch("/suites/{suite_id}", response_model=EvalSuiteResponse)
async def update_eval_suite(
    suite_id: str,
    data: EvalSuiteUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
) -> EvalSuite:
    """Update an evaluation suite."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, roles=tenant.roles, clearance_level=tenant.clearance)

    result = await db.execute(
        select(EvalSuite).where(
            EvalSuite.id == suite_id,
            EvalSuite.organization_id == tenant.org_id,
        )
    )
    suite = result.scalar_one_or_none()
    
    if not suite:
        raise HTTPException(status_code=404, detail="Eval suite not found")
    
    if data.name is not None:
        suite.name = data.name
    if data.description is not None:
        suite.description = data.description
    if data.queries is not None:
        suite.queries = data.queries
    if data.expected is not None:
        suite.expected = data.expected
    if data.is_active is not None:
        suite.is_active = data.is_active
    
    await db.commit()
    await db.refresh(suite)
    
    return suite


@router.delete("/suites/{suite_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_eval_suite(
    suite_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
) -> None:
    """Delete an evaluation suite."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, roles=tenant.roles, clearance_level=tenant.clearance)

    result = await db.execute(
        select(EvalSuite).where(
            EvalSuite.id == suite_id,
            EvalSuite.organization_id == tenant.org_id,
        )
    )
    suite = result.scalar_one_or_none()
    
    if not suite:
        raise HTTPException(status_code=404, detail="Eval suite not found")
    
    await db.delete(suite)
    await db.commit()


# Eval Run endpoints

@router.post("/runs", response_model=dict[str, str], status_code=status.HTTP_202_ACCEPTED)
async def trigger_eval_run(
    data: EvalRunTrigger,
    db: Annotated[AsyncSession, Depends(get_db)],
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
) -> dict[str, str]:
    """Trigger an evaluation run (async)."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, roles=tenant.roles, clearance_level=tenant.clearance)

    # Verify suite exists
    result = await db.execute(
        select(EvalSuite).where(
            EvalSuite.id == data.suite_id,
            EvalSuite.organization_id == tenant.org_id,
        )
    )
    suite = result.scalar_one_or_none()
    
    if not suite:
        raise HTTPException(status_code=404, detail="Eval suite not found")
    
    # Enqueue eval run
    enqueue_eval_suite(
        organization_id=tenant.org_id,
        suite_id=data.suite_id,
        user_id=tenant.user_id,
        config=data.config,
    )
    
    return {"message": "Eval run enqueued", "suite_id": data.suite_id}


@router.get("/runs/{run_id}", response_model=EvalRunResponse)
async def get_eval_run(
    run_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
) -> EvalRun:
    """Get an evaluation run by ID."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, roles=tenant.roles, clearance_level=tenant.clearance)

    service = EvalRunService(db, tenant.org_id)
    eval_run = await service.get_eval_run(run_id)
    
    if not eval_run:
        raise HTTPException(status_code=404, detail="Eval run not found")
    
    return eval_run


@router.get("/runs", response_model=list[EvalRunResponse])
async def list_eval_runs(
    db: Annotated[AsyncSession, Depends(get_db)],
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
    suite_id: str | None = None,
    limit: int = 50,
) -> list[EvalRun]:
    """List evaluation runs for the organization."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, roles=tenant.roles, clearance_level=tenant.clearance)

    service = EvalRunService(db, tenant.org_id)
    return await service.list_eval_runs(suite_id=suite_id, limit=limit)


# Drift Report endpoints

@router.post("/drift", response_model=dict[str, str], status_code=status.HTTP_202_ACCEPTED)
async def trigger_drift_computation(
    data: DriftReportTrigger,
    db: Annotated[AsyncSession, Depends(get_db)],
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
) -> dict[str, str]:
    """Trigger drift computation between two eval runs (async)."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, roles=tenant.roles, clearance_level=tenant.clearance)

    # Verify both runs exist and belong to org
    result = await db.execute(
        select(EvalRun).where(
            EvalRun.id.in_([data.baseline_run_id, data.current_run_id]),
            EvalRun.organization_id == tenant.org_id,
        )
    )
    runs = list(result.scalars().all())
    
    if len(runs) != 2:
        raise HTTPException(status_code=404, detail="One or both eval runs not found")
    
    # Enqueue drift computation
    enqueue_drift_computation(
        organization_id=tenant.org_id,
        baseline_run_id=data.baseline_run_id,
        current_run_id=data.current_run_id,
        user_id=tenant.user_id,
    )
    
    return {"message": "Drift computation enqueued"}


@router.get("/drift/{drift_id}", response_model=DriftReportResponse)
async def get_drift_report(
    drift_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
) -> DriftReport:
    """Get a drift report by ID."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, roles=tenant.roles, clearance_level=tenant.clearance)

    service = DriftDetectionService(db, tenant.org_id)
    drift_report = await service.get_drift_report(drift_id)
    
    if not drift_report:
        raise HTTPException(status_code=404, detail="Drift report not found")
    
    return drift_report


@router.get("/drift", response_model=list[DriftReportResponse])
async def list_drift_reports(
    db: Annotated[AsyncSession, Depends(get_db)],
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
    severity: str | None = None,
    limit: int = 50,
) -> list[DriftReport]:
    """List drift reports for the organization."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, roles=tenant.roles, clearance_level=tenant.clearance)

    service = DriftDetectionService(db, tenant.org_id)
    return await service.list_drift_reports(severity=severity, limit=limit)
