"""Admin: benchmark run endpoints.

POST /admin/benchmarks  — persist a completed run (called by run_all.py --save-to-api)
GET  /admin/benchmarks  — list runs ordered newest-first (time-series feed for dashboard)
GET  /admin/benchmarks/latest — most recent run
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.api.v1.admin.dependencies import AdminUser, get_admin_user
from app.database import get_db
from app.models.benchmark_run import BenchmarkRun

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class BenchmarkRunCreate(BaseModel):
    run_at: Optional[str] = None
    mode: str
    strategy: str
    dataset: str
    ollama_model: Optional[str] = None
    duration_seconds: float
    composite_score: float
    results: list[Any]


class BenchmarkRunResponse(BaseModel):
    id: str
    run_at: str
    mode: str
    strategy: str
    dataset: str
    ollama_model: Optional[str]
    duration_seconds: float
    composite_score: float
    results: list[Any]

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/admin/benchmarks",
    response_model=BenchmarkRunResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Admin - Benchmarks"],
)
def create_benchmark_run(
    payload: BenchmarkRunCreate,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
) -> BenchmarkRunResponse:
    """Persist a completed benchmark run. Requires system:write permission."""
    if not admin.has_permission("system:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: system:write",
        )
    run_at = (
        datetime.fromisoformat(payload.run_at)
        if payload.run_at
        else datetime.now(timezone.utc)
    )
    run = BenchmarkRun(
        id=str(uuid4()),
        run_at=run_at,
        mode=payload.mode,
        strategy=payload.strategy,
        dataset=payload.dataset,
        ollama_model=payload.ollama_model,
        duration_seconds=payload.duration_seconds,
        composite_score=payload.composite_score,
        results=payload.results,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return _to_response(run)


@router.get(
    "/admin/benchmarks",
    response_model=list[BenchmarkRunResponse],
    tags=["Admin - Benchmarks"],
)
def list_benchmark_runs(
    limit: int = Query(default=50, ge=1, le=200),
    strategy: Optional[str] = Query(default=None),
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
) -> list[BenchmarkRunResponse]:
    """List benchmark runs ordered newest-first. Requires system:read permission."""
    if not admin.has_permission("system:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: system:read",
        )
    query = db.query(BenchmarkRun).order_by(desc(BenchmarkRun.run_at))
    if strategy:
        query = query.filter(BenchmarkRun.strategy == strategy)
    runs = query.limit(limit).all()
    return [_to_response(r) for r in runs]


@router.get(
    "/admin/benchmarks/latest",
    response_model=BenchmarkRunResponse,
    tags=["Admin - Benchmarks"],
)
def get_latest_benchmark_run(
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
) -> BenchmarkRunResponse:
    """Return the most recent benchmark run. Requires system:read permission."""
    if not admin.has_permission("system:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: system:read",
        )
    run = db.query(BenchmarkRun).order_by(desc(BenchmarkRun.run_at)).first()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No benchmark runs found")
    return _to_response(run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_response(run: BenchmarkRun) -> BenchmarkRunResponse:
    return BenchmarkRunResponse(
        id=str(run.id),
        run_at=run.run_at.isoformat(),
        mode=run.mode,
        strategy=run.strategy,
        dataset=run.dataset,
        ollama_model=run.ollama_model,
        duration_seconds=run.duration_seconds,
        composite_score=run.composite_score,
        results=run.results,
    )
