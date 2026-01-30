"""Memory Activation API endpoints.

Provides read APIs for:
- Retrieval explanations (auditable scoring breakdown per query)
- Co-activation edges (admin/debug)

Write paths are intentionally async via Celery tasks.
"""

from __future__ import annotations

from typing import Optional, List
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import celery_app
from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context, require_org_admin
from app.models.memory_activation import (
    MemoryRetrievalExplanation,
    MemoryCoactivationEdge,
    CausalHypothesis,
    MemoryActivationState,
)
from app.models.memory import MemoryMetadata
from app.schemas.memory import MemoryResponse
from app.schemas.memory_activation import (
    MemoryRetrievalExplanationSchema,
    CoactivationEdgeSchema,
    CoactivatedNeighborSchema,
    CoactivatedNeighborDetailSchema,
    CausalHypothesisSchema,
)
from app.services.permission_checker import PermissionChecker


router = APIRouter()


def _broker_enabled() -> bool:
    broker = getattr(celery_app.conf, "broker_url", None)
    return bool(broker) and not str(broker).startswith("memory://")


class NightlyDecayRefreshRequest(BaseModel):
    org_id: Optional[str] = Field(default=None, description="If omitted, runs for all active orgs")
    prune_min_weight: float = Field(default=0.01, ge=0.0, le=1.0)
    prune_older_than_days: int = Field(default=90, ge=1, le=3650)


class NightlyDecayRefreshResponse(BaseModel):
    ok: bool
    enqueued: bool
    task_id: Optional[str] = None
    org_id: Optional[str] = None
    reason: Optional[str] = None


class CausalHypothesisRefreshRequest(BaseModel):
    org_id: Optional[str] = Field(default=None, description="Defaults to caller org (system_admin may override)")
    min_edge_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    limit: int = Field(default=200, ge=1, le=2000)


class CausalHypothesisRefreshResponse(BaseModel):
    ok: bool
    enqueued: bool
    task_id: Optional[str] = None
    org_id: Optional[str] = None
    reason: Optional[str] = None


class RetrievalExplanationQueryAggSchema(BaseModel):
    query_hash: str
    count: int
    first_retrieved_at: datetime
    last_retrieved_at: datetime


class MostAccessedMemorySchema(BaseModel):
    memory: MemoryResponse
    access_count: int
    last_accessed_at: Optional[datetime] = None


@router.get(
    "/retrieval-explanations",
    response_model=List[MemoryRetrievalExplanationSchema],
)
async def list_retrieval_explanations(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: Optional[str] = Query(None, description="Filter by user_id (org_admin only unless self)"),
    query_hash: Optional[str] = Query(None, description="Filter by query_hash"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """List retrieval explanation logs.

    Default: returns only the caller's logs.
    Org/system admins may query across the org.
    """

    await set_tenant_context(db, tenant.user_id, tenant.org_id, ",".join(tenant.roles or []), tenant.clearance_level)

    effective_user_id = user_id
    if effective_user_id and (effective_user_id != tenant.user_id) and not tenant.has_any_role("org_admin", "system_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view other users' explanations")

    stmt = select(MemoryRetrievalExplanation).where(MemoryRetrievalExplanation.organization_id == tenant.org_id)

    # Non-admin: force self
    if not tenant.has_any_role("org_admin", "system_admin"):
        stmt = stmt.where(MemoryRetrievalExplanation.user_id == tenant.user_id)
    elif effective_user_id:
        stmt = stmt.where(MemoryRetrievalExplanation.user_id == effective_user_id)

    if query_hash:
        stmt = stmt.where(MemoryRetrievalExplanation.query_hash == query_hash)

    stmt = stmt.order_by(desc(MemoryRetrievalExplanation.retrieved_at)).offset(offset).limit(limit)

    res = await db.execute(stmt)
    rows = res.scalars().all()

    return [MemoryRetrievalExplanationSchema.model_validate(r) for r in rows]


@router.get(
    "/retrieval-explanations/{explanation_id}",
    response_model=MemoryRetrievalExplanationSchema,
)
async def get_retrieval_explanation(
    explanation_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Get a single retrieval explanation by id.

    Non-admin users can only access their own logs.
    """

    await set_tenant_context(db, tenant.user_id, tenant.org_id, ",".join(tenant.roles or []), tenant.clearance_level)

    stmt = select(MemoryRetrievalExplanation).where(
        and_(
            MemoryRetrievalExplanation.organization_id == tenant.org_id,
            MemoryRetrievalExplanation.id == explanation_id,
        )
    )
    res = await db.execute(stmt)
    row = res.scalar_one_or_none()

    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Explanation not found")

    if (row.user_id != tenant.user_id) and not tenant.has_any_role("org_admin", "system_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    return MemoryRetrievalExplanationSchema.model_validate(row)


@router.get(
    "/causal-hypotheses",
    response_model=List[CausalHypothesisSchema],
)
async def list_causal_hypotheses(
    status: Optional[str] = Query(None, description="Filter by status (proposed|active|contested|rejected)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """List causal hypotheses (admin only)."""

    await set_tenant_context(db, tenant.user_id, tenant.org_id, ",".join(tenant.roles or []), tenant.clearance_level)

    stmt = select(CausalHypothesis).where(CausalHypothesis.organization_id == tenant.org_id)
    if status:
        stmt = stmt.where(CausalHypothesis.status == status)
    stmt = stmt.order_by(desc(CausalHypothesis.updated_at)).offset(offset).limit(limit)

    rows = (await db.execute(stmt)).scalars().all()
    return [CausalHypothesisSchema.model_validate(r) for r in rows]


@router.get(
    "/causal-hypotheses/{hypothesis_id}",
    response_model=CausalHypothesisSchema,
)
async def get_causal_hypothesis(
    hypothesis_id: str,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Get a causal hypothesis by id (admin only)."""

    await set_tenant_context(db, tenant.user_id, tenant.org_id, ",".join(tenant.roles or []), tenant.clearance_level)

    stmt = select(CausalHypothesis).where(
        and_(CausalHypothesis.organization_id == tenant.org_id, CausalHypothesis.id == hypothesis_id)
    )
    row = (await db.execute(stmt)).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Hypothesis not found")

    return CausalHypothesisSchema.model_validate(row)


@router.post(
    "/admin/causal-hypotheses/refresh",
    response_model=CausalHypothesisRefreshResponse,
)
async def enqueue_causal_hypothesis_refresh(
    req: CausalHypothesisRefreshRequest,
    tenant: TenantContext = Depends(require_org_admin()),
):
    """Enqueue causal hypothesis refresh for an org (admin only)."""

    if not _broker_enabled():
        return CausalHypothesisRefreshResponse(ok=True, enqueued=False, reason="broker_disabled")

    target_org_id = req.org_id or tenant.org_id
    if (target_org_id != tenant.org_id) and not tenant.has_any_role("system_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    task = celery_app.send_task(
        "app.services.memory_activation.tasks.causal_hypothesis_update_task",
        kwargs={
            "org_id": target_org_id,
            "min_edge_weight": req.min_edge_weight,
            "limit": req.limit,
        },
    )

    return CausalHypothesisRefreshResponse(
        ok=True,
        enqueued=True,
        task_id=str(getattr(task, "id", "")) or None,
        org_id=target_org_id,
    )


@router.get(
    "/coactivation/edges/{memory_id}",
    response_model=List[CoactivationEdgeSchema],
)
async def list_coactivation_edges(
    memory_id: str,
    limit: int = Query(50, ge=1, le=200),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """List co-activation edges for a memory.

    Admin-only for now (debug/ops). This avoids having to re-check
    per-edge RBAC for both endpoint memories.
    """

    await set_tenant_context(db, tenant.user_id, tenant.org_id, ",".join(tenant.roles or []), tenant.clearance_level)

    stmt = (
        select(MemoryCoactivationEdge)
        .where(MemoryCoactivationEdge.organization_id == tenant.org_id)
        .where(or_(MemoryCoactivationEdge.memory_id_a == memory_id, MemoryCoactivationEdge.memory_id_b == memory_id))
        .order_by(desc(MemoryCoactivationEdge.edge_weight))
        .limit(limit)
    )

    res = await db.execute(stmt)
    edges = res.scalars().all()
    return [CoactivationEdgeSchema.model_validate(e) for e in edges]


@router.get(
    "/coactivation/neighbors/{memory_id}",
    response_model=List[CoactivatedNeighborSchema],
)
async def list_coactivation_neighbors(
    memory_id: str,
    limit: int = Query(25, ge=1, le=100),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """List co-activated neighbor memories that the caller can read.

    This is the end-user safe variant of coactivation graph access:
    - requires read access to the requested memory
    - filters neighbor nodes by read access
    """

    await set_tenant_context(db, tenant.user_id, tenant.org_id, ",".join(tenant.roles or []), tenant.clearance_level)

    checker = PermissionChecker(db)
    base_access = await checker.check_memory_access(
        user_id=tenant.user_id,
        org_id=tenant.org_id,
        memory_id=memory_id,
        action="read",
        clearance_level=tenant.clearance_level,
    )
    if not base_access.allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=base_access.reason)

    stmt = (
        select(MemoryCoactivationEdge)
        .where(MemoryCoactivationEdge.organization_id == tenant.org_id)
        .where(or_(MemoryCoactivationEdge.memory_id_a == memory_id, MemoryCoactivationEdge.memory_id_b == memory_id))
        .order_by(desc(MemoryCoactivationEdge.edge_weight))
        .limit(limit * 3)  # over-fetch; we'll filter by permissions
    )

    res = await db.execute(stmt)
    edges = res.scalars().all()

    candidate_neighbor_ids: list[str] = []
    edge_info_by_neighbor: dict[str, tuple[float, int]] = {}
    seen: set[str] = set()

    for edge in edges:
        a_id = str(edge.memory_id_a)
        b_id = str(edge.memory_id_b)
        neighbor_id = b_id if a_id == memory_id else a_id

        if neighbor_id == memory_id or neighbor_id in seen:
            continue

        seen.add(neighbor_id)
        candidate_neighbor_ids.append(neighbor_id)
        edge_info_by_neighbor[neighbor_id] = (
            float(edge.edge_weight or 0.0),
            int(edge.coactivation_count or 0),
        )

    allowed_neighbor_ids = await checker.filter_memory_ids_with_access(
        user_id=tenant.user_id,
        org_id=tenant.org_id,
        memory_ids=candidate_neighbor_ids,
        action="read",
        clearance_level=tenant.clearance_level,
    )

    neighbors: list[CoactivatedNeighborSchema] = []
    for neighbor_id in allowed_neighbor_ids:
        edge_weight, coactivation_count = edge_info_by_neighbor.get(neighbor_id, (0.0, 0))
        neighbors.append(
            CoactivatedNeighborSchema(
                memory_id=neighbor_id,
                edge_weight=edge_weight,
                coactivation_count=coactivation_count,
            )
        )
        if len(neighbors) >= limit:
            break

    return neighbors


@router.get(
    "/coactivation/neighbors/{memory_id}/details",
    response_model=List[CoactivatedNeighborDetailSchema],
)
async def list_coactivation_neighbors_with_metadata(
    memory_id: str,
    limit: int = Query(25, ge=1, le=100),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """List co-activated neighbors, including each neighbor's memory metadata.

    Keeps the same permission semantics as /coactivation/neighbors/{memory_id}:
    - requires read access to the requested memory
    - filters neighbor nodes by read access
    """

    await set_tenant_context(db, tenant.user_id, tenant.org_id, ",".join(tenant.roles or []), tenant.clearance_level)

    checker = PermissionChecker(db)
    base_access = await checker.check_memory_access(
        user_id=tenant.user_id,
        org_id=tenant.org_id,
        memory_id=memory_id,
        action="read",
        clearance_level=tenant.clearance_level,
    )
    if not base_access.allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=base_access.reason)

    stmt = (
        select(MemoryCoactivationEdge)
        .where(MemoryCoactivationEdge.organization_id == tenant.org_id)
        .where(or_(MemoryCoactivationEdge.memory_id_a == memory_id, MemoryCoactivationEdge.memory_id_b == memory_id))
        .order_by(desc(MemoryCoactivationEdge.edge_weight))
        .limit(limit * 5)  # over-fetch; we'll filter by permissions
    )

    res = await db.execute(stmt)
    edges = res.scalars().all()

    candidate_neighbor_ids: list[str] = []
    edge_info_by_neighbor: dict[str, tuple[float, int]] = {}
    seen: set[str] = set()

    for edge in edges:
        a_id = str(edge.memory_id_a)
        b_id = str(edge.memory_id_b)
        neighbor_id = b_id if a_id == memory_id else a_id

        if neighbor_id == memory_id or neighbor_id in seen:
            continue

        seen.add(neighbor_id)
        candidate_neighbor_ids.append(neighbor_id)
        edge_info_by_neighbor[neighbor_id] = (
            float(edge.edge_weight or 0.0),
            int(edge.coactivation_count or 0),
        )

    allowed_neighbor_ids = await checker.filter_memory_ids_with_access(
        user_id=tenant.user_id,
        org_id=tenant.org_id,
        memory_ids=candidate_neighbor_ids,
        action="read",
        clearance_level=tenant.clearance_level,
    )

    allowed_neighbor_ids = allowed_neighbor_ids[:limit]

    if not allowed_neighbor_ids:
        return []

    # Batch load metadata for neighbors (RLS + org constraint)
    mem_stmt = (
        select(MemoryMetadata)
        .where(MemoryMetadata.organization_id == tenant.org_id)
        .where(MemoryMetadata.is_active.is_(True))
        .where(MemoryMetadata.id.in_(allowed_neighbor_ids))
    )
    mem_res = await db.execute(mem_stmt)
    memories = mem_res.scalars().all()
    mem_by_id = {str(m.id): m for m in memories}

    # Preserve edge ranking order
    results: list[CoactivatedNeighborDetailSchema] = []
    for nid in allowed_neighbor_ids:
        mem = mem_by_id.get(nid)
        if not mem:
            continue
        edge_weight, coactivation_count = edge_info_by_neighbor.get(nid, (0.0, 0))
        results.append(
            CoactivatedNeighborDetailSchema(
                edge_weight=edge_weight,
                coactivation_count=coactivation_count,
                memory=mem,
            )
        )

    return results


@router.post(
    "/admin/nightly-decay-refresh",
    response_model=NightlyDecayRefreshResponse,
)
async def enqueue_nightly_decay_refresh(
    req: NightlyDecayRefreshRequest,
    tenant: TenantContext = Depends(require_org_admin()),
):
    """Enqueue the nightly decay refresh task (admin only).

    This is an ops/backfill endpoint. It enqueues the task and returns the Celery task id.
    In unit-test mode (memory:// broker) it becomes a safe no-op.
    """

    if not _broker_enabled():
        return NightlyDecayRefreshResponse(ok=True, enqueued=False, reason="broker_disabled")

    task = celery_app.send_task(
        "app.services.memory_activation.tasks.nightly_decay_refresh_task",
        kwargs={
            "org_id": req.org_id,
            "prune_min_weight": req.prune_min_weight,
            "prune_older_than_days": req.prune_older_than_days,
        },
    )

    return NightlyDecayRefreshResponse(
        ok=True,
        enqueued=True,
        task_id=str(getattr(task, "id", "")) or None,
        org_id=req.org_id,
    )
