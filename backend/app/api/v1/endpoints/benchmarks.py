"""Admin: benchmark run endpoints.

POST /admin/benchmarks  — persist a completed run (called by run_all.py --save-to-api)
GET  /admin/benchmarks  — list runs ordered newest-first (time-series feed for dashboard)
GET  /admin/benchmarks/latest — most recent run
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status, Request
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_token
from app.database import get_db
from app.models.benchmark_run import BenchmarkRun
from app.models.user import User


async def get_admin_user_simple(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """Simple admin auth - extract user from JWT token in Authorization header"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )
    
    token = auth_header.split(" ")[1]
    try:
        payload = verify_token(token)
        if not payload:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        
        user_id = getattr(payload, "sub", None) or getattr(payload, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token error: {str(e)}")
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    
    return user

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class BenchmarkRunCreate(BaseModel):
    run_at: Optional[str] = None
    mode: str
    strategy: str
    dataset: str
    VLLM_MODEL: Optional[str] = None
    duration_seconds: float
    composite_score: float
    results: list[Any]


class BenchmarkRunResponse(BaseModel):
    id: str
    run_at: str
    mode: str
    strategy: str
    dataset: str
    VLLM_MODEL: Optional[str]
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
async def create_benchmark_run(
    payload: BenchmarkRunCreate,
    admin_user: User = Depends(get_admin_user_simple),
    db: AsyncSession = Depends(get_db),
) -> BenchmarkRunResponse:
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
        VLLM_MODEL=payload.VLLM_MODEL,
        duration_seconds=payload.duration_seconds,
        composite_score=payload.composite_score,
        results=payload.results,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return _to_response(run)


@router.get(
    "/admin/benchmarks",
    response_model=list[BenchmarkRunResponse],
    tags=["Admin - Benchmarks"],
)
async def list_benchmark_runs(
    limit: int = Query(default=50, ge=1, le=200),
    strategy: Optional[str] = Query(default=None),
    admin_user: User = Depends(get_admin_user_simple),
    db: AsyncSession = Depends(get_db),
) -> list[BenchmarkRunResponse]:
    """List benchmark runs ordered newest-first. Requires admin role."""
    result = await db.execute(
        select(BenchmarkRun).order_by(desc(BenchmarkRun.run_at)).limit(limit)
    )
    runs = result.scalars().all()
    if strategy:
        runs = [r for r in runs if r.strategy == strategy]
    return [_to_response(r) for r in runs]


@router.get(
    "/admin/benchmarks/latest",
    response_model=BenchmarkRunResponse,
    tags=["Admin - Benchmarks"],
)
async def get_latest_benchmark_run(
    admin_user: User = Depends(get_admin_user_simple),
    db: AsyncSession = Depends(get_db),
) -> BenchmarkRunResponse:
    """Return the most recent benchmark run. Requires admin role."""
    result = await db.execute(
        select(BenchmarkRun).order_by(desc(BenchmarkRun.run_at)).limit(1)
    )
    run = result.scalar_one_or_none()
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
        VLLM_MODEL=run.VLLM_MODEL,
        duration_seconds=run.duration_seconds,
        composite_score=run.composite_score,
        results=run.results,
    )
