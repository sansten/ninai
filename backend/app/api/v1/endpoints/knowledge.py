"""Knowledge submission endpoints (non-admin)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.schemas.knowledge import KnowledgeReviewRequestResponse, KnowledgeSubmissionCreate
from app.services import knowledge_review_service


router = APIRouter()


@router.post("/review-requests", response_model=KnowledgeReviewRequestResponse)
async def submit_knowledge_for_review(
    body: KnowledgeSubmissionCreate,
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    trace_id = getattr(request.state, "trace_id", None) or getattr(request.state, "request_id", None)
    return await knowledge_review_service.submit_for_review(db=db, tenant=tenant, body=body, trace_id=trace_id)
