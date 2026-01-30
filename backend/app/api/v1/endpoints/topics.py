"""Topic browsing endpoints.

Exposes the persisted outputs of the topic modeling pipeline:
- memory_topics
- memory_topic_memberships

These endpoints are intentionally read-only and tenant-scoped.
"""

from __future__ import annotations

from typing import Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.endpoints.auth import get_current_user
from app.core.database import get_db
from app.models.memory import MemoryMetadata
from app.models.memory_topic import MemoryTopic
from app.models.memory_topic_membership import MemoryTopicMembership
from app.models.user import User


router = APIRouter(tags=["topics"])


class TopicResponse(BaseModel):
    id: str
    scope: str
    scope_id: Optional[str] = None
    label: str
    keywords: list[str] = Field(default_factory=list)
    created_by: str


class TopicListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    topics: list[TopicResponse]


class TopicMemoryResponse(BaseModel):
    memory_id: str
    title: Optional[str] = None
    content_preview: Optional[str] = None
    is_primary: bool = False
    weight: float = 1.0


class TopicMemoriesResponse(BaseModel):
    topic_id: str
    total: int
    limit: int
    offset: int
    memories: list[TopicMemoryResponse]


@router.get("/topics", response_model=TopicListResponse)
async def list_topics(
    org_id: str = Query(...),
    scope: Optional[str] = Query(None),
    scope_id: Optional[str] = Query(None),
    label_query: Optional[str] = Query(None, description="Substring match on label"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    filters: list[Any] = [MemoryTopic.organization_id == org_id]
    if scope:
        filters.append(MemoryTopic.scope == scope)
    if scope_id:
        filters.append(MemoryTopic.scope_id == scope_id)
    if label_query:
        filters.append(MemoryTopic.label.ilike(f"%{label_query}%"))

    count_stmt = select(func.count()).select_from(MemoryTopic).where(and_(*filters))
    total = int((await db.execute(count_stmt)).scalar() or 0)

    stmt = (
        select(MemoryTopic)
        .where(and_(*filters))
        .order_by(MemoryTopic.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()

    return TopicListResponse(
        total=total,
        limit=limit,
        offset=offset,
        topics=[
            TopicResponse(
                id=str(t.id),
                scope=t.scope,
                scope_id=t.scope_id,
                label=t.label,
                keywords=t.keywords or [],
                created_by=t.created_by,
            )
            for t in rows
        ],
    )


@router.get("/topics/{topic_id}", response_model=TopicResponse)
async def get_topic(
    topic_id: str,
    org_id: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    stmt = select(MemoryTopic).where(
        and_(MemoryTopic.organization_id == org_id, MemoryTopic.id == topic_id)
    )
    topic = (await db.execute(stmt)).scalar_one_or_none()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    return TopicResponse(
        id=str(topic.id),
        scope=topic.scope,
        scope_id=topic.scope_id,
        label=topic.label,
        keywords=topic.keywords or [],
        created_by=topic.created_by,
    )


@router.get("/topics/{topic_id}/memories", response_model=TopicMemoriesResponse)
async def list_topic_memories(
    topic_id: str,
    org_id: str = Query(...),
    primary_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    filters: list[Any] = [
        MemoryTopicMembership.organization_id == org_id,
        MemoryTopicMembership.topic_id == topic_id,
    ]
    if primary_only:
        filters.append(MemoryTopicMembership.is_primary.is_(True))

    count_stmt = (
        select(func.count())
        .select_from(MemoryTopicMembership)
        .where(and_(*filters))
    )
    total = int((await db.execute(count_stmt)).scalar() or 0)

    stmt = (
        select(
            MemoryTopicMembership.memory_id,
            MemoryTopicMembership.is_primary,
            MemoryTopicMembership.weight,
            MemoryMetadata.title,
            MemoryMetadata.content_preview,
        )
        .join(
            MemoryMetadata,
            and_(
                MemoryMetadata.id == MemoryTopicMembership.memory_id,
                MemoryMetadata.organization_id == org_id,
                MemoryMetadata.is_active.is_(True),
            ),
        )
        .where(and_(*filters))
        .order_by(MemoryTopicMembership.weight.desc(), MemoryTopicMembership.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )

    rows = (await db.execute(stmt)).all()

    memories = [
        TopicMemoryResponse(
            memory_id=str(r.memory_id),
            title=r.title,
            content_preview=r.content_preview,
            is_primary=bool(r.is_primary),
            weight=float(r.weight or 0.0),
        )
        for r in rows
    ]

    return TopicMemoriesResponse(
        topic_id=topic_id,
        total=total,
        limit=limit,
        offset=offset,
        memories=memories,
    )


@router.get("/memories/{memory_id}/topics", response_model=list[TopicResponse])
async def list_memory_topics(
    memory_id: str,
    org_id: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if str(current_user.organization_id) != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    stmt = (
        select(MemoryTopic)
        .join(
            MemoryTopicMembership,
            and_(
                MemoryTopicMembership.topic_id == MemoryTopic.id,
                MemoryTopicMembership.organization_id == org_id,
                MemoryTopicMembership.memory_id == memory_id,
            ),
        )
        .where(MemoryTopic.organization_id == org_id)
        .order_by(MemoryTopicMembership.is_primary.desc(), MemoryTopicMembership.weight.desc())
    )

    topics = (await db.execute(stmt)).scalars().all()

    return [
        TopicResponse(
            id=str(t.id),
            scope=t.scope,
            scope_id=t.scope_id,
            label=t.label,
            keywords=t.keywords or [],
            created_by=t.created_by,
        )
        for t in topics
    ]
