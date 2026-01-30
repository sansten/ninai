"""Knowledge review (HITL) service.

This module owns the DB logic for:
- Submitting knowledge updates for review
- Listing and deciding pending review requests
- Publishing and rolling back versions

Endpoints can monkeypatch these functions in DB-less unit tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import set_tenant_context
from app.core.config import settings
from app.middleware.tenant_context import TenantContext
from app.models.knowledge_item import KnowledgeItem
from app.models.knowledge_item_version import KnowledgeItemVersion
from app.models.knowledge_review_request import KnowledgeReviewRequest, KnowledgeReviewStatus
from app.schemas.memory import MemoryCreate
from app.services.memory_service import MemoryService
from app.services.topic_service import TopicService
from app.schemas.knowledge import (
    KnowledgeItemVersionResponse,
    KnowledgeItemVersionsResponse,
    KnowledgeReviewRequestResponse,
    KnowledgeReviewListResponse,
    KnowledgeSubmissionCreate,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _review_to_schema(row: KnowledgeReviewRequest) -> KnowledgeReviewRequestResponse:
    return KnowledgeReviewRequestResponse(
        id=row.id,
        organization_id=row.organization_id,
        item_id=row.item_id,
        item_version_id=row.item_version_id,
        status=row.status,
        requested_by_user_id=row.requested_by_user_id,
        reviewed_by_user_id=row.reviewed_by_user_id,
        reviewed_at=row.reviewed_at,
        decision_comment=row.decision_comment,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _version_to_schema(row: KnowledgeItemVersion) -> KnowledgeItemVersionResponse:
    return KnowledgeItemVersionResponse(
        id=row.id,
        organization_id=row.organization_id,
        item_id=row.item_id,
        version_number=row.version_number,
        title=row.title,
        content=row.content,
        extra_metadata=row.extra_metadata,
        created_by_user_id=row.created_by_user_id,
        trace_id=row.trace_id,
        provenance=row.provenance or [],
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def submit_for_review(
    db: AsyncSession,
    tenant: TenantContext,
    body: KnowledgeSubmissionCreate,
    trace_id: Optional[str] = None,
) -> KnowledgeReviewRequestResponse:
    async with db.begin():
        await set_tenant_context(
            session=db,
            user_id=tenant.user_id,
            org_id=tenant.org_id,
            roles=tenant.roles_string,
            clearance_level=tenant.clearance_level,
        )
        if body.item_id:
            item = await db.scalar(
                select(KnowledgeItem).where(
                    KnowledgeItem.organization_id == tenant.org_id,
                    KnowledgeItem.id == body.item_id,
                )
            )
            if not item:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge item not found")
        else:
            item = KnowledgeItem(
                organization_id=tenant.org_id,
                title=body.title,
                key=body.key,
            )
            db.add(item)
            await db.flush()

        next_vn = await db.scalar(
            select(func.coalesce(func.max(KnowledgeItemVersion.version_number), 0) + 1).where(
                KnowledgeItemVersion.organization_id == tenant.org_id,
                KnowledgeItemVersion.item_id == item.id,
            )
        )
        version = KnowledgeItemVersion(
            organization_id=tenant.org_id,
            item_id=item.id,
            version_number=int(next_vn or 1),
            title=body.title,
            content=body.content,
            extra_metadata=body.extra_metadata,
            created_by_user_id=tenant.user_id,
            trace_id=trace_id,
            provenance=[p.model_dump() for p in body.provenance],
        )
        db.add(version)
        await db.flush()

        req = KnowledgeReviewRequest(
            organization_id=tenant.org_id,
            item_id=item.id,
            item_version_id=version.id,
            status=KnowledgeReviewStatus.PENDING,
            requested_by_user_id=tenant.user_id,
        )
        db.add(req)
        await db.flush()

        return _review_to_schema(req)


async def list_review_requests(
    db: AsyncSession,
    tenant: TenantContext,
    status_filter: Optional[str] = None,
    limit: int = 50,
) -> KnowledgeReviewListResponse:
    async with db.begin():
        await set_tenant_context(
            session=db,
            user_id=tenant.user_id,
            org_id=tenant.org_id,
            roles=tenant.roles_string,
            clearance_level=tenant.clearance_level,
        )

        q = select(KnowledgeReviewRequest).where(KnowledgeReviewRequest.organization_id == tenant.org_id)
        if status_filter:
            q = q.where(KnowledgeReviewRequest.status == status_filter)
        q = q.order_by(KnowledgeReviewRequest.created_at.desc()).limit(limit)

        result = await db.execute(q)
        rows = list(result.scalars().all())
        return KnowledgeReviewListResponse(items=[_review_to_schema(r) for r in rows])


async def approve_review_request(
    db: AsyncSession,
    tenant: TenantContext,
    request_id: str,
    comment: Optional[str] = None,
    promote_to_memory: bool = False,
    tags: Optional[list[str]] = None,
    topics: Optional[list[str]] = None,
    primary_topic: Optional[str] = None,
    topic_confidence: float = 0.8,
    memory_scope: str = "organization",
    memory_type: str = "procedural",
    classification: str = "internal",
) -> KnowledgeReviewRequestResponse:
    async with db.begin():
        await set_tenant_context(
            session=db,
            user_id=tenant.user_id,
            org_id=tenant.org_id,
            roles=tenant.roles_string,
            clearance_level=tenant.clearance_level,
        )
        req = await db.scalar(
            select(KnowledgeReviewRequest).where(
                KnowledgeReviewRequest.organization_id == tenant.org_id,
                KnowledgeReviewRequest.id == request_id,
            )
        )
        if not req:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review request not found")
        if req.status != KnowledgeReviewStatus.PENDING:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Review request is not pending")

        item = await db.scalar(
            select(KnowledgeItem).where(
                KnowledgeItem.organization_id == tenant.org_id,
                KnowledgeItem.id == req.item_id,
            )
        )
        if not item:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge item not found")

        version = await db.scalar(
            select(KnowledgeItemVersion).where(
                KnowledgeItemVersion.organization_id == tenant.org_id,
                KnowledgeItemVersion.id == req.item_version_id,
            )
        )
        if not version:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge item version not found")

        req.status = KnowledgeReviewStatus.APPROVED
        req.reviewed_by_user_id = tenant.user_id
        req.reviewed_at = _utc_now()
        req.decision_comment = comment

        item.is_published = True
        item.published_version_id = req.item_version_id
        item.published_at = _utc_now()

        # Optional: promote approved version into long-term memory with reviewer mapping.
        if promote_to_memory:
            mem_service = MemoryService(
                session=db,
                user_id=tenant.user_id,
                org_id=tenant.org_id,
                clearance_level=tenant.clearance_level,
            )

            memory_title = version.title or item.title
            create = MemoryCreate(
                content=version.content,
                title=memory_title,
                scope=memory_scope,
                scope_id=None,
                memory_type=memory_type,
                classification=classification,
                required_clearance=0,
                tags=tags or [],
                entities={},
                extra_metadata={
                    **(version.extra_metadata or {}),
                    "knowledge_item_id": item.id,
                    "knowledge_item_version_id": version.id,
                    "knowledge_review_request_id": req.id,
                },
                source_type="knowledge_review",
                source_id=req.id,
                retention_days=None,
                ttl=None,
            )

            embedding = [0.0] * int(settings.EMBEDDING_DIMENSIONS)
            promoted = await mem_service.create_memory(create, embedding=embedding, request_id=version.trace_id)

            if topics:
                outputs = {
                    "topics": topics,
                    "primary_topic": primary_topic or (topics[0] if topics else None),
                    "confidence": float(topic_confidence),
                }
                await TopicService(db).upsert_topics_for_memory(
                    organization_id=tenant.org_id,
                    memory_id=promoted.id,
                    scope=create.scope,
                    scope_id=create.scope_id,
                    outputs=outputs,
                    created_by="admin",
                )

        await db.flush()
        return _review_to_schema(req)


async def reject_review_request(
    db: AsyncSession,
    tenant: TenantContext,
    request_id: str,
    comment: Optional[str] = None,
) -> KnowledgeReviewRequestResponse:
    async with db.begin():
        await set_tenant_context(
            session=db,
            user_id=tenant.user_id,
            org_id=tenant.org_id,
            roles=tenant.roles_string,
            clearance_level=tenant.clearance_level,
        )
        req = await db.scalar(
            select(KnowledgeReviewRequest).where(
                KnowledgeReviewRequest.organization_id == tenant.org_id,
                KnowledgeReviewRequest.id == request_id,
            )
        )
        if not req:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review request not found")
        if req.status != KnowledgeReviewStatus.PENDING:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Review request is not pending")

        req.status = KnowledgeReviewStatus.REJECTED
        req.reviewed_by_user_id = tenant.user_id
        req.reviewed_at = _utc_now()
        req.decision_comment = comment

        await db.flush()
        return _review_to_schema(req)


async def list_item_versions(
    db: AsyncSession,
    tenant: TenantContext,
    item_id: str,
    limit: int = 50,
) -> KnowledgeItemVersionsResponse:
    async with db.begin():
        await set_tenant_context(
            session=db,
            user_id=tenant.user_id,
            org_id=tenant.org_id,
            roles=tenant.roles_string,
            clearance_level=tenant.clearance_level,
        )

        q = (
            select(KnowledgeItemVersion)
            .where(
                KnowledgeItemVersion.organization_id == tenant.org_id,
                KnowledgeItemVersion.item_id == item_id,
            )
            .order_by(KnowledgeItemVersion.version_number.desc())
            .limit(limit)
        )

        result = await db.execute(q)
        rows = list(result.scalars().all())
        return KnowledgeItemVersionsResponse(items=[_version_to_schema(r) for r in rows])


async def rollback_item(
    db: AsyncSession,
    tenant: TenantContext,
    item_id: str,
    target_version_id: str,
    comment: Optional[str] = None,
) -> None:
    # Minimal rollback: repoint the published version.
    async with db.begin():
        await set_tenant_context(
            session=db,
            user_id=tenant.user_id,
            org_id=tenant.org_id,
            roles=tenant.roles_string,
            clearance_level=tenant.clearance_level,
        )
        item = await db.scalar(
            select(KnowledgeItem).where(
                KnowledgeItem.organization_id == tenant.org_id,
                KnowledgeItem.id == item_id,
            )
        )
        if not item:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge item not found")

        version = await db.scalar(
            select(KnowledgeItemVersion).where(
                KnowledgeItemVersion.organization_id == tenant.org_id,
                KnowledgeItemVersion.id == target_version_id,
                KnowledgeItemVersion.item_id == item_id,
            )
        )
        if not version:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target version not found")

        item.is_published = True
        item.published_version_id = target_version_id
        item.published_at = _utc_now()

        # Optional: record comment later via audit table; kept for future extension.
        _ = comment
        await db.flush()
