"""Memory consolidation endpoints (Feature 3).

This module exposes:
- Suggestions for in-place dedupe consolidation (backed by ConsolidationService)
- A durable consolidation record API for consolidated-summary memories
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.memory import MemoryMetadata
from app.models.memory_consolidation import MemoryConsolidation
from app.schemas.memory import MemoryCreate
from app.services.consolidation_service import ConsolidationService
from app.services.embedding_service import EmbeddingService
from app.services.memory_service import MemoryService


# Schemas
class ConsolidationCandidate(BaseModel):
    """Candidate group for consolidation."""
    primary_id: str
    duplicate_ids: List[str]
    similarity_scores: List[float]
    
    class Config:
        from_attributes = True


class FindCandidatesRequest(BaseModel):
    """Find consolidation candidates request."""
    memory_id: Optional[str] = Field(None, description="Specific memory to check")
    similarity_threshold: float = Field(0.85, ge=0.0, le=1.0)
    scope: Optional[str] = Field(None, description="Filter by memory scope")


class FindCandidatesResponse(BaseModel):
    """Find consolidation candidates response."""
    candidates: List[ConsolidationCandidate]
    total_candidates: int
    summary: Dict[str, Any]


class ConsolidationStatus(BaseModel):
    """Consolidation status of a memory."""
    memory_id: str
    is_consolidated: bool
    tags: List[str]
    entities_count: int
    relationships_count: int


class CreateConsolidationRequest(BaseModel):
    """Create a consolidation record + consolidated summary memory."""

    memory_ids: List[str] = Field(..., min_length=2, description="Source memory IDs")
    title: Optional[str] = Field(None, description="Optional title for consolidated memory")
    created_by: str = Field("manual", description="'manual' or 'system'")


class ConsolidationRecordResponse(BaseModel):
    id: str
    organization_id: str
    user_id: str
    consolidated_memory_id: Optional[str]
    title: Optional[str]
    summary: Optional[str]
    source_memory_ids: List[str]
    created_by: str
    status: str
    error_message: Optional[str]
    created_at: datetime


router = APIRouter(prefix="/consolidations", tags=["Consolidations"])


@router.get("/suggestions", response_model=FindCandidatesResponse)
async def get_consolidation_suggestions(
    memory_id: Optional[str] = Query(None, description="Specific memory to check"),
    similarity_threshold: float = Query(0.85, ge=0.0, le=1.0),
    scope: Optional[str] = Query(None, description="Filter by memory scope"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> FindCandidatesResponse:
    """
    Find memories that can be consolidated.
    
    Returns groups of similar memories that could be merged.
    """
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    service = ConsolidationService(db, tenant.org_id)

    candidates = await service.find_consolidation_candidates(
        memory_id=memory_id,
        similarity_threshold=similarity_threshold,
        scope=scope,
    )
        
    formatted_candidates: List[ConsolidationCandidate] = []
    for candidate in candidates:
        formatted_candidates.append(
            ConsolidationCandidate(
                primary_id=str(candidate["primary"].id),
                duplicate_ids=[str(d.id) for d in candidate["duplicates"]],
                similarity_scores=candidate["similarity_scores"],
            )
        )

    return FindCandidatesResponse(
        candidates=formatted_candidates,
        total_candidates=len(formatted_candidates),
        summary={
            "duplicates_found": sum(len(c["duplicates"]) for c in candidates),
            "similarity_threshold": similarity_threshold,
        },
    )


@router.post("", response_model=ConsolidationRecordResponse, status_code=status.HTTP_201_CREATED)
async def create_consolidation(
    request: CreateConsolidationRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ConsolidationRecordResponse:
    """Create a consolidated summary memory and persist a consolidation record."""

    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    # Load source memories (metadata only).
    src_stmt = select(MemoryMetadata).where(
        and_(
            MemoryMetadata.organization_id == tenant.org_id,
            MemoryMetadata.id.in_([str(mid) for mid in request.memory_ids]),
        )
    )
    sources = (await db.execute(src_stmt)).scalars().all()
    found_ids = {str(m.id) for m in sources}
    missing = [mid for mid in request.memory_ids if str(mid) not in found_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"Memories not found: {missing}")

    record_id = str(uuid.uuid4())

    # Best-effort summary: based on previews (full content may live in vector store).
    title = request.title or (sources[0].title and f"Consolidated: {sources[0].title}") or "Consolidated Summary"
    summary_lines = [f"# {title}", "", "## Sources", ""]
    for i, m in enumerate(sources, start=1):
        preview = (m.content_preview or "").strip()
        if len(preview) > 280:
            preview = preview[:280].rstrip() + "…"
        summary_lines.append(f"{i}. **{m.title or str(m.id)}** — {preview}")
    summary_text = "\n".join(summary_lines)

    # Union metadata.
    tags: List[str] = []
    seen_tags: set[str] = set()
    entities: Dict[str, List[str]] = {}
    for m in sources:
        for t in (m.tags or []):
            if t not in seen_tags:
                seen_tags.add(t)
                tags.append(t)
        for k, v in (m.entities or {}).items():
            values = v if isinstance(v, list) else [v]
            bucket = entities.setdefault(str(k), [])
            for item in values:
                s = str(item)
                if s not in bucket:
                    bucket.append(s)

    memory_payload = MemoryCreate(
        title=title,
        content=summary_text,
        scope=sources[0].scope,
        scope_id=sources[0].scope_id,
        memory_type=sources[0].memory_type,
        classification=sources[0].classification,
        required_clearance=sources[0].required_clearance,
        tags=tags,
        entities=entities,
        extra_metadata={
            "consolidated_from_ids": [str(mid) for mid in request.memory_ids],
            "consolidation_record_id": record_id,
        },
    )

    embedding = await EmbeddingService.embed(memory_payload.content)
    mem_svc = MemoryService(db, tenant.user_id, tenant.org_id, tenant.clearance_level)
    consolidated_memory = await mem_svc.create_memory(memory_payload, embedding=embedding)

    record = MemoryConsolidation(
        id=record_id,
        organization_id=str(tenant.org_id),
        user_id=str(tenant.user_id),
        consolidated_memory_id=str(consolidated_memory.id),
        title=title,
        summary=summary_text,
        source_memory_ids=[str(mid) for mid in request.memory_ids],
        created_by=request.created_by,
        status="completed",
        error_message=None,
    )
    db.add(record)

    await db.commit()
    await db.refresh(record)

    return ConsolidationRecordResponse(
        id=str(record.id),
        organization_id=str(record.organization_id),
        user_id=str(record.user_id),
        consolidated_memory_id=str(record.consolidated_memory_id) if record.consolidated_memory_id else None,
        title=record.title,
        summary=record.summary,
        source_memory_ids=list(record.source_memory_ids or []),
        created_by=record.created_by,
        status=record.status,
        error_message=record.error_message,
        created_at=record.created_at,
    )


@router.get("/memories/{memory_id}/status", response_model=ConsolidationStatus)
async def get_memory_consolidation_status(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ConsolidationStatus:
    """Get in-place consolidation status for a given memory."""

    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    service = ConsolidationService(db, tenant.org_id)
    status_obj = await service.get_consolidation_status(memory_id)
    return ConsolidationStatus(**status_obj)


@router.get("/{consolidation_id}", response_model=ConsolidationRecordResponse)
async def get_consolidation_record(
    consolidation_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ConsolidationRecordResponse:
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = select(MemoryConsolidation).where(
        and_(
            MemoryConsolidation.organization_id == str(tenant.org_id),
            MemoryConsolidation.id == str(consolidation_id),
        )
    )
    record = (await db.execute(stmt)).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Consolidation not found")

    return ConsolidationRecordResponse(
        id=str(record.id),
        organization_id=str(record.organization_id),
        user_id=str(record.user_id),
        consolidated_memory_id=str(record.consolidated_memory_id) if record.consolidated_memory_id else None,
        title=record.title,
        summary=record.summary,
        source_memory_ids=list(record.source_memory_ids or []),
        created_by=record.created_by,
        status=record.status,
        error_message=record.error_message,
        created_at=record.created_at,
    )


@router.delete(
    "/{consolidation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
)
async def delete_consolidation_record(
    consolidation_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> Response:
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = select(MemoryConsolidation).where(
        and_(
            MemoryConsolidation.organization_id == str(tenant.org_id),
            MemoryConsolidation.id == str(consolidation_id),
        )
    )
    record = (await db.execute(stmt)).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Consolidation not found")

    await db.delete(record)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
