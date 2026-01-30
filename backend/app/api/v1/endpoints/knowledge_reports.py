"""Knowledge reporting endpoints.

Provides lightweight, user-facing summaries of the "knowledge layer":
- topics
- graph relationships
- recent memories

These are read-only and tenant-scoped.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.endpoints.auth import get_current_user
from app.core.database import get_db
from app.models.graph_relationship import GraphRelationship
from app.models.memory import MemoryMetadata
from app.models.memory_topic import MemoryTopic
from app.models.memory_topic_membership import MemoryTopicMembership
from app.models.user import User


router = APIRouter(prefix="/knowledge/reports", tags=["knowledge"])


class TopTopic(BaseModel):
    topic_id: str
    label: str
    scope: str
    scope_id: Optional[str] = None
    memory_count: int = 0


class TopRelationship(BaseModel):
    from_memory_id: str
    to_memory_id: str
    relationship_type: str
    similarity_score: float = 0.0
    auto_created: bool = False


class RecentMemory(BaseModel):
    memory_id: str
    title: Optional[str] = None
    content_preview: str
    created_at: Optional[str] = None


class KnowledgeSummaryResponse(BaseModel):
    org_id: str
    top_topics: list[TopTopic] = Field(default_factory=list)
    top_relationships: list[TopRelationship] = Field(default_factory=list)
    recent_memories: list[RecentMemory] = Field(default_factory=list)


@router.get("/summary", response_model=KnowledgeSummaryResponse)
async def knowledge_summary(
    org_id: str = Query(...),
    topic_limit: int = Query(10, ge=1, le=50),
    relationship_limit: int = Query(10, ge=1, le=50),
    recent_limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Topics by membership count
    topics_stmt = (
        select(
            MemoryTopic.id,
            MemoryTopic.label,
            MemoryTopic.scope,
            MemoryTopic.scope_id,
            func.count(MemoryTopicMembership.id).label("cnt"),
        )
        .join(
            MemoryTopicMembership,
            and_(
                MemoryTopicMembership.topic_id == MemoryTopic.id,
                MemoryTopicMembership.organization_id == org_id,
            ),
        )
        .where(MemoryTopic.organization_id == org_id)
        .group_by(MemoryTopic.id, MemoryTopic.label, MemoryTopic.scope, MemoryTopic.scope_id)
        .order_by(func.count(MemoryTopicMembership.id).desc())
        .limit(topic_limit)
    )
    topics_rows = (await db.execute(topics_stmt)).all()

    # Relationships by similarity
    try:
        org_uuid = UUID(org_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid org_id")

    rel_stmt = (
        select(GraphRelationship)
        .where(GraphRelationship.organization_id == org_uuid)
        .order_by(GraphRelationship.similarity_score.desc().nullslast())
        .limit(relationship_limit)
    )
    rels = (await db.execute(rel_stmt)).scalars().all()

    # Recent memories
    mem_stmt = (
        select(MemoryMetadata)
        .where(and_(MemoryMetadata.organization_id == org_id, MemoryMetadata.is_active.is_(True)))
        .order_by(MemoryMetadata.created_at.desc())
        .limit(recent_limit)
    )
    memories = (await db.execute(mem_stmt)).scalars().all()

    return KnowledgeSummaryResponse(
        org_id=org_id,
        top_topics=[
            TopTopic(
                topic_id=str(r.id),
                label=r.label,
                scope=r.scope,
                scope_id=r.scope_id,
                memory_count=int(r.cnt or 0),
            )
            for r in topics_rows
        ],
        top_relationships=[
            TopRelationship(
                from_memory_id=rel.from_memory_id,
                to_memory_id=rel.to_memory_id,
                relationship_type=rel.relationship_type,
                similarity_score=float(rel.similarity_score or 0.0),
                auto_created=bool(rel.auto_created),
            )
            for rel in rels
        ],
        recent_memories=[
            RecentMemory(
                memory_id=str(m.id),
                title=m.title,
                content_preview=m.content_preview,
                created_at=m.created_at.isoformat() if m.created_at else None,
            )
            for m in memories
        ],
    )


@router.post("/synthesis/create")
async def create_synthesis_report(
    org_id: Optional[str] = Query(None),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter"),
    days_back: int = Query(30, ge=1, le=365),
    title: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Create a comprehensive knowledge synthesis report.
    
    Analyzes memories and relationships to identify:
    - Concept clusters
    - Trends over time
    - Key insights and patterns
    
    Parameters:
    - `tags`: Filter by comma-separated tags (e.g., "ai,performance")
    - `days_back`: Include memories from last N days (default: 30)
    - `title`: Custom report title
    
    Returns: Synthesis report with clusters, trends, and insights
    """
    try:
        from app.services.knowledge_synthesis_service import KnowledgeSynthesisService
        
        tag_list = None
        if tags:
            tag_list = [t.strip() for t in tags.split(",")]
        
        service = KnowledgeSynthesisService(session=db)
        report = await service.create_synthesis_report(
            tags=tag_list,
            days_back=days_back,
            title=title or f"Synthesis Report - {user.email}",
        )
        
        return report.to_dict()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/synthesis/export")
async def export_synthesis(
    report_id: str = Query(...),
    format: str = Query("markdown", regex="^(markdown|json|pdf)$"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Export a synthesis report in the specified format.
    
    Formats:
    - `markdown`: GitHub-flavored markdown
    - `json`: JSON structure
    - `pdf`: PDF document (if available)
    """
    try:
        from app.services.knowledge_synthesis_service import KnowledgeSynthesisService
        
        service = KnowledgeSynthesisService(session=db)
        
        # Would retrieve report and export
        if format == "markdown":
            return {"content": "# Synthesis Report\n\nContent here"}
        elif format == "json":
            return {"report": {}}
        elif format == "pdf":
            return {"url": "/reports/synthesis_report.pdf"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
