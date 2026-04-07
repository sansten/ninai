"""Cognitive Schedules API endpoints — Phase 84.

Prefix: /cognitive/schedules
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import TenantContext
from app.services.cognitive_schedule_service import CognitiveScheduleService

router = APIRouter()


def _schedule_to_dict(sched: Any) -> dict:
    return {
        "id": sched.id,
        "organization_id": sched.organization_id,
        "cron_expression": sched.cron_expression,
        "cognitive_verb": sched.cognitive_verb,
        "payload": sched.payload,
        "label": sched.label,
        "is_active": sched.is_active,
        "next_run_at": sched.next_run_at.isoformat() if sched.next_run_at else None,
        "last_run_at": sched.last_run_at.isoformat() if sched.last_run_at else None,
        "created_at": sched.created_at.isoformat() if sched.created_at else None,
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_schedule(
    body: dict = Body(...),
    ctx: TenantContext = Depends(),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = CognitiveScheduleService(session=db, org_id=ctx.org_id)
    try:
        sched = await svc.create(
            cron_expression=body.get("cron_expression", ""),
            cognitive_verb=body.get("cognitive_verb", ""),
            payload=body.get("payload"),
            label=body.get("label"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    await db.commit()
    return _schedule_to_dict(sched)


@router.get("")
async def list_schedules(
    ctx: TenantContext = Depends(),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    svc = CognitiveScheduleService(session=db, org_id=ctx.org_id)
    schedules = await svc.list()
    return [_schedule_to_dict(s) for s in schedules]


@router.get("/{schedule_id}")
async def get_schedule(
    schedule_id: str = Path(...),
    ctx: TenantContext = Depends(),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = CognitiveScheduleService(session=db, org_id=ctx.org_id)
    sched = await svc.get(schedule_id)
    if sched is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="schedule not found")
    return _schedule_to_dict(sched)


@router.patch("/{schedule_id}")
async def update_schedule(
    schedule_id: str = Path(...),
    body: dict = Body(...),
    ctx: TenantContext = Depends(),
    db: AsyncSession = Depends(get_db),
) -> dict:
    svc = CognitiveScheduleService(session=db, org_id=ctx.org_id)
    try:
        sched = await svc.update(
            schedule_id,
            cron_expression=body.get("cron_expression"),
            cognitive_verb=body.get("cognitive_verb"),
            payload=body.get("payload"),
            label=body.get("label"),
            is_active=body.get("is_active"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if sched is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="schedule not found")
    await db.commit()
    return _schedule_to_dict(sched)


@router.delete("/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schedule(
    schedule_id: str = Path(...),
    ctx: TenantContext = Depends(),
    db: AsyncSession = Depends(get_db),
):
    svc = CognitiveScheduleService(session=db, org_id=ctx.org_id)
    deleted = await svc.delete(schedule_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="schedule not found")
    await db.commit()


@router.post("/{schedule_id}/run", status_code=status.HTTP_202_ACCEPTED)
async def trigger_schedule(
    schedule_id: str = Path(...),
    ctx: TenantContext = Depends(),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Manually trigger a schedule immediately (enqueues the Celery task)."""
    svc = CognitiveScheduleService(session=db, org_id=ctx.org_id)
    sched = await svc.get(schedule_id)
    if sched is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="schedule not found")
    # Enqueue via Celery (import deferred to avoid circular at module load)
    from app.tasks.cognitive_schedule_runner import run_cognitive_schedule_task
    run_cognitive_schedule_task.delay(schedule_id=schedule_id, org_id=ctx.org_id)
    return {"status": "queued", "schedule_id": schedule_id}
