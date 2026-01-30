"""Reviewer knowledge queue endpoints (non-admin HITL).

These endpoints are for knowledge reviewers (role: knowledge_reviewer).
They intentionally do NOT live under /admin and do not grant admin portal access.

Admins (org_admin/system_admin) can still use these endpoints as well.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import TenantContext, require_knowledge_reviewer
from app.schemas.knowledge import (
    KnowledgeItemVersionsResponse,
    KnowledgeReviewDecision,
    KnowledgeReviewListResponse,
    KnowledgeReviewRequestResponse,
)
from app.services import knowledge_review_service


router = APIRouter()


@router.get("/review-requests", response_model=KnowledgeReviewListResponse)
async def list_review_requests(
    status: str | None = Query("pending", description="Filter by status (pending|approved|rejected)"),
    limit: int = Query(50, ge=1, le=200),
    tenant: TenantContext = Depends(require_knowledge_reviewer()),
    db: AsyncSession = Depends(get_db),
):
    return await knowledge_review_service.list_review_requests(db=db, tenant=tenant, status_filter=status, limit=limit)


@router.post("/review-requests/{request_id}/approve", response_model=KnowledgeReviewRequestResponse)
async def approve_review_request(
    request_id: str,
    body: KnowledgeReviewDecision,
    tenant: TenantContext = Depends(require_knowledge_reviewer()),
    db: AsyncSession = Depends(get_db),
):
    return await knowledge_review_service.approve_review_request(
        db=db,
        tenant=tenant,
        request_id=request_id,
        comment=body.comment,
        promote_to_memory=body.promote_to_memory,
        tags=body.tags,
        topics=body.topics,
        primary_topic=body.primary_topic,
        topic_confidence=body.topic_confidence,
        memory_scope=body.memory_scope,
        memory_type=body.memory_type,
        classification=body.classification,
    )


@router.post("/review-requests/{request_id}/reject", response_model=KnowledgeReviewRequestResponse)
async def reject_review_request(
    request_id: str,
    body: KnowledgeReviewDecision,
    tenant: TenantContext = Depends(require_knowledge_reviewer()),
    db: AsyncSession = Depends(get_db),
):
    return await knowledge_review_service.reject_review_request(
        db=db,
        tenant=tenant,
        request_id=request_id,
        comment=body.comment,
    )


@router.get("/items/{item_id}/versions", response_model=KnowledgeItemVersionsResponse)
async def list_versions(
    item_id: str,
    limit: int = Query(50, ge=1, le=200),
    tenant: TenantContext = Depends(require_knowledge_reviewer()),
    db: AsyncSession = Depends(get_db),
):
    return await knowledge_review_service.list_item_versions(db=db, tenant=tenant, item_id=item_id, limit=limit)
