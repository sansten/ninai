"""Knowledge Provenance & Citation API endpoints (Feature 16)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context, require_org_admin
from app.models.memory import MemoryMetadata
from app.models.provenance_edge import ProvenanceEdge
from app.services.memory_provenance_service import MemoryProvenanceService

router = APIRouter()


class ProvenanceAssertRequest(BaseModel):
    memory_id: str
    source: str = Field(min_length=1, max_length=255)
    source_id: str | None = Field(default=None, max_length=255)
    source_type: str | None = Field(default=None, max_length=64)
    edge_type: str = Field(default="manual_assert", max_length=64)
    author: str | None = Field(default=None, max_length=255)
    timestamp: datetime | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    verified_by: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _edge_to_dict(edge: ProvenanceEdge) -> dict[str, Any]:
    return {
        "id": str(edge.id),
        "source_id": str(edge.source_id),
        "target_id": str(edge.target_id),
        "edge_type": str(edge.edge_type),
        "agent_name": str(edge.agent_name),
        "created_at": edge.created_at,
        "metadata": dict(edge.edge_metadata or {}),
    }


def _citation_from_edge(edge: dict[str, Any], verified_by: list[str]) -> dict[str, Any]:
    metadata = dict(edge.get("metadata") or {})
    source = (
        metadata.get("source")
        or metadata.get("source_name")
        or metadata.get("source_system")
        or edge.get("source_id")
    )
    author = metadata.get("author") or metadata.get("user") or metadata.get("created_by")
    confidence = metadata.get("confidence")
    timestamp = metadata.get("timestamp") or edge.get("created_at")

    return {
        "source": source,
        "author": author,
        "timestamp": timestamp,
        "confidence": confidence,
        "verified_by": verified_by,
    }


@router.get("/{memory_id}/lineage")
async def get_memory_lineage(
    memory_id: str,
    max_depth: int = Query(default=10, ge=0, le=50),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Return full lineage graph to source for a memory."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    lineage = await MemoryProvenanceService().get_lineage(
        db=db,
        org_id=tenant.org_id,
        memory_id=memory_id,
        max_depth=max_depth,
    )

    return {
        "memory_id": memory_id,
        "lineage": lineage,
        "summary": MemoryProvenanceService().summarise_lineage(lineage),
    }


@router.get("/{memory_id}/citations")
async def get_memory_citations(
    memory_id: str,
    max_depth: int = Query(default=10, ge=0, le=50),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Return formatted citations for one memory."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    memory_res = await db.execute(
        select(MemoryMetadata).where(
            MemoryMetadata.id == memory_id,
            MemoryMetadata.organization_id == tenant.org_id,
        )
    )
    memory = memory_res.scalar_one_or_none()
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    lineage = await MemoryProvenanceService().get_lineage(
        db=db,
        org_id=tenant.org_id,
        memory_id=memory_id,
        max_depth=max_depth,
    )
    verified_by = list(lineage.get("agent_chain") or [])
    edges = list(lineage.get("edges") or [])

    citations = [_citation_from_edge(edge, verified_by) for edge in edges]
    if not citations:
        citations = [
            {
                "source": memory.source_id or "memory",
                "author": None,
                "timestamp": memory.updated_at or memory.created_at,
                "confidence": None,
                "verified_by": verified_by,
            }
        ]

    return {
        "memory_id": str(memory.id),
        "content": memory.content_preview,
        "citation": citations[0],
        "citations": citations,
    }


@router.get("/search")
async def search_provenance_by_source(
    source: str = Query(..., min_length=1, max_length=255),
    limit: int = Query(default=200, ge=1, le=1000),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Return memories that match a source system/value across provenance edges."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    res = await db.execute(select(ProvenanceEdge).where(ProvenanceEdge.org_id == tenant.org_id))
    rows = list(res.scalars().all())

    needle = source.strip().lower()
    matches: list[ProvenanceEdge] = []
    for edge in rows:
        metadata = dict(edge.edge_metadata or {})
        fields = [
            str(edge.source_id or ""),
            str(metadata.get("source") or ""),
            str(metadata.get("source_name") or ""),
            str(metadata.get("source_system") or ""),
            str(metadata.get("source_type") or ""),
        ]
        if any(needle in value.lower() for value in fields if value):
            matches.append(edge)

    trimmed = matches[:limit]
    memory_ids = sorted({str(edge.target_id) for edge in trimmed})

    return {
        "source": source,
        "count": len(trimmed),
        "memory_ids": memory_ids,
        "edges": [_edge_to_dict(edge) for edge in trimmed],
    }


@router.post("/assert")
async def assert_provenance(
    body: ProvenanceAssertRequest,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Manually assert provenance for imported data."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    memory_res = await db.execute(
        select(MemoryMetadata).where(
            MemoryMetadata.id == body.memory_id,
            MemoryMetadata.organization_id == tenant.org_id,
        )
    )
    memory = memory_res.scalar_one_or_none()
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    metadata = dict(body.metadata)
    metadata.update(
        {
            "source": body.source,
            "source_type": body.source_type,
            "author": body.author,
            "timestamp": body.timestamp.isoformat() if body.timestamp else None,
            "confidence": body.confidence,
            "verified_by": list(body.verified_by or []),
            "asserted_by": tenant.user_id,
        }
    )

    edge = await MemoryProvenanceService().record_edge(
        db=db,
        org_id=tenant.org_id,
        source_id=body.source_id or f"manual:{body.source}",
        target_id=body.memory_id,
        edge_type=body.edge_type,
        agent_name="ManualProvenanceAssertion",
        metadata=metadata,
    )
    await db.commit()

    return {
        "asserted": True,
        "memory_id": body.memory_id,
        "edge": _edge_to_dict(edge),
    }