"""
Memory Endpoints
================

CRUD operations for memories with search and sharing.
"""

import time
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.services.webhook_service import WebhookService
from app.schemas.memory import (
    MemoryCreate,
    MemoryUpdate,
    MemoryResponse,
    MemoryListResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    SearchHnmsMode,
    MemoryShareRequest,
    MemorySharingResponse,
    AccessExplanation,
)
from app.schemas.memory_batch import (
    MemoryBatchUpdateRequest,
    MemoryBatchUpdateResponse,
    MemoryBatchDeleteRequest,
    MemoryBatchDeleteResponse,
    MemoryBatchShareRequest,
    MemoryBatchShareResponse,
    MemoryBatchUpdateResult,
    MemoryBatchResult,
    MemoryBatchShareResult,
)
from app.services.memory_service import MemoryService
from app.services.embedding_service import EmbeddingService
from app.tasks.memory_pipeline import enqueue_memory_pipeline
from app.services.memory_attachment_service import (
    MemoryAttachmentService,
    AttachmentNotFoundError,
    AttachmentTooLargeError,
    AttachmentStorageError,
)
from app.schemas.memory_attachment import (
    MemoryAttachmentResponse,
    MemoryAttachmentListResponse,
)
from app.schemas.feedback import (
    MemoryFeedbackCreate,
    MemoryFeedbackListResponse,
    MemoryFeedbackResponse,
)
from app.schemas.base import BaseSchema
from app.services.memory_feedback_service import MemoryFeedbackService
from app.tasks.memory_pipeline import enqueue_feedback_learning


router = APIRouter()


class MemoryRelevanceFeedbackCreate(BaseSchema):
    relevant: bool
    query: Optional[str] = None
    hnms_mode: Optional[str] = None
    target_agent: Optional[str] = None
    trace_id: Optional[str] = None


def get_memory_service(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> MemoryService:
    """Dependency to get configured MemoryService."""
    return MemoryService(
        session=db,
        user_id=tenant.user_id,
        org_id=tenant.org_id,
        clearance_level=tenant.clearance_level,
    )


def get_attachment_service(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> MemoryAttachmentService:
    return MemoryAttachmentService(
        session=db,
        user_id=tenant.user_id,
        org_id=tenant.org_id,
        clearance_level=tenant.clearance_level,
    )


def get_feedback_service(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> MemoryFeedbackService:
    return MemoryFeedbackService(session=db, user_id=tenant.user_id, org_id=tenant.org_id)


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    request: Request,
    body: MemoryCreate,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Create a new memory.
    
    The content will be embedded and stored in both Postgres (metadata)
    and Qdrant (vector + payload).
    
    **Required permissions:** `memory:create:{scope}`
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    request_id = getattr(request.state, "request_id", None)

    embedding = await EmbeddingService.embed(body.content)
    
    try:
        memory = await service.create_memory(
            data=body,
            embedding=embedding,
            request_id=request_id,
        )
        await db.commit()

        # Emit webhook event (best-effort). In tests, avoid invoking webhook
        # persistence with mocked DB sessions to prevent un-awaited AsyncMock warnings.
        if settings.APP_ENV != "test":
            try:
                webhook_svc = WebhookService(db)

                # Handle created_at - could be datetime or already a string
                created_at_str = memory.created_at
                if hasattr(memory.created_at, "isoformat"):
                    created_at_str = memory.created_at.isoformat()

                await webhook_svc.emit_event(
                    organization_id=tenant.org_id,
                    event_type="memory.created",
                    payload={
                        "memory_id": memory.id,
                        "title": memory.title,
                        "scope": memory.scope,
                        "user_id": tenant.user_id,
                        "created_at": created_at_str,
                    },
                )
            except (AttributeError, TypeError):
                pass
            await db.commit()

        enqueue_memory_pipeline(
            org_id=tenant.org_id,
            memory_id=memory.id,
            initiator_user_id=tenant.user_id,
            trace_id=request_id,
            storage="long_term",
        )
        
        return MemoryResponse.model_validate(memory)
    
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.get("", response_model=MemoryListResponse, status_code=status.HTTP_200_OK)
async def list_memories(
    scope: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    memory_type: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    List long-term memories with optional filters.
    """
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    items, total, has_more = await service.list_memories(
        scope=scope,
        tags=tags,
        memory_type=memory_type,
        page=page,
        page_size=page_size,
    )

    return MemoryListResponse(
        items=[MemoryResponse.model_validate(m) for m in items],
        total=total,
        page=page,
        page_size=page_size,
        has_more=has_more,
    )


# =============================================================================
# Static Routes (must come BEFORE dynamic /{memory_id} routes)
# =============================================================================


@router.post("/batch/update", response_model=MemoryBatchUpdateResponse)
async def batch_update_memories(
    request: Request,
    body: MemoryBatchUpdateRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """Bulk update memories (partial success allowed).

    Returns per-item success/error. Uses a single commit if at least one update succeeds.
    """
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    request_id = getattr(request.state, "request_id", None)

    results: list[MemoryBatchUpdateResult] = []
    any_success = False

    for item in body.items:
        try:
            memory = await service.update_memory(
                memory_id=item.memory_id,
                data=item.update,
                new_embedding=None,
                request_id=request_id,
            )
            results.append(
                MemoryBatchUpdateResult(
                    memory_id=item.memory_id,
                    success=True,
                    memory=MemoryResponse.model_validate(memory),
                )
            )
            any_success = True
        except (PermissionError, ValueError) as e:
            results.append(MemoryBatchUpdateResult(memory_id=item.memory_id, success=False, error=str(e)))

    if any_success:
        await db.commit()
    else:
        await db.rollback()

    return MemoryBatchUpdateResponse(trace_id=request_id, results=results)


@router.post("/batch/delete", response_model=MemoryBatchDeleteResponse)
async def batch_delete_memories(
    request: Request,
    body: MemoryBatchDeleteRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """Bulk delete (soft-delete) memories (partial success allowed)."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    request_id = getattr(request.state, "request_id", None)

    results: list[MemoryBatchResult] = []
    any_success = False

    for memory_id in body.memory_ids:
        try:
            await service.delete_memory(memory_id=memory_id, request_id=request_id)
            results.append(MemoryBatchResult(memory_id=memory_id, success=True))
            any_success = True
        except (PermissionError, ValueError) as e:
            results.append(MemoryBatchResult(memory_id=memory_id, success=False, error=str(e)))

    if any_success:
        await db.commit()
    else:
        await db.rollback()

    return MemoryBatchDeleteResponse(trace_id=request_id, results=results)


@router.post("/batch/share", response_model=MemoryBatchShareResponse, status_code=status.HTTP_201_CREATED)
async def batch_share_memories(
    request: Request,
    body: MemoryBatchShareRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """Bulk share memories with a common share request (partial success allowed)."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    request_id = getattr(request.state, "request_id", None)

    results: list[MemoryBatchShareResult] = []
    any_success = False

    for memory_id in body.memory_ids:
        try:
            share = await service.share_memory(memory_id=memory_id, request=body.share, request_id=request_id)
            results.append(MemoryBatchShareResult(memory_id=memory_id, success=True, share_id=getattr(share, "id", None)))
            any_success = True
        except (PermissionError, ValueError) as e:
            results.append(MemoryBatchShareResult(memory_id=memory_id, success=False, error=str(e)))

    if any_success:
        await db.commit()
    else:
        await db.rollback()

    return MemoryBatchShareResponse(trace_id=request_id, results=results)

@router.get("/all", status_code=status.HTTP_200_OK)
async def list_all_memories(
    request: Request,
    include_short_term: bool = Query(True, description="Include short-term memories from Redis"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    List all memories for the current user from both short-term and long-term storage.
    
    Returns memories grouped by storage type:
    - `short_term`: Memories in Redis (temporary, may expire)
    - `long_term`: Memories in PostgreSQL (permanent)
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    result = await service.list_all_memories(include_short_term=include_short_term)
    
    return {
        "short_term": [
            {
                "id": m.id,
                "title": m.title,
                "content_preview": m.content[:200] if m.content else None,
                "scope": m.scope,
                "tags": m.tags,
                "access_count": m.access_count,
                "importance_score": m.importance_score,
                "promotion_eligible": m.promotion_eligible,
                "created_at": m.created_at,
            }
            for m in result["short_term"]
        ],
        "long_term": [
            {
                "id": m.id,
                "title": m.title,
                "content_preview": m.content_preview,
                "scope": m.scope,
                "tags": m.tags,
                "memory_type": m.memory_type,
                "is_promoted": m.is_promoted,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in result["long_term"]
        ],
        "summary": {
            "short_term_count": len(result["short_term"]),
            "long_term_count": len(result["long_term"]),
            "promotion_eligible_count": sum(1 for m in result["short_term"] if m.promotion_eligible),
        },
    }


@router.get("/stats", status_code=status.HTTP_200_OK)
async def get_memory_stats(
    tenant: TenantContext = Depends(get_tenant_context),
):
    """
    Get statistics about memory storage across short-term and long-term.
    """
    from app.services.short_term_memory import ShortTermMemoryStats
    
    stm_stats = await ShortTermMemoryStats.get_stats()
    
    return {
        "short_term": stm_stats,
        "architecture": {
            "short_term_storage": "Redis",
            "short_term_ttl_seconds": 3600,
            "long_term_storage": "PostgreSQL + Qdrant",
            "promotion_threshold_access_count": 3,
            "promotion_threshold_importance_score": 0.7,
        },
    }


# =============================================================================
# Attachments (Multimodal MVP)
# =============================================================================


@router.get(
    "/{memory_id}/attachments",
    response_model=MemoryAttachmentListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_memory_attachments(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryAttachmentService = Depends(get_attachment_service),
):
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    try:
        items = await service.list_attachments(memory_id)
        return MemoryAttachmentListResponse(
            items=[MemoryAttachmentResponse.model_validate(a) for a in items],
            total=len(items),
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.post(
    "/{memory_id}/attachments",
    response_model=MemoryAttachmentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_memory_attachment(
    request: Request,
    memory_id: str,
    file: UploadFile = File(...),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryAttachmentService = Depends(get_attachment_service),
):
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    request_id = getattr(request.state, "request_id", None)

    try:
        att = await service.create_attachment(memory_id=memory_id, file=file)
        await db.commit()
        return MemoryAttachmentResponse.model_validate(att)
    except AttachmentTooLargeError as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))
    except AttachmentStorageError as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except PermissionError as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(
    "/{memory_id}/attachments/{attachment_id}",
    status_code=status.HTTP_200_OK,
)
async def download_memory_attachment(
    memory_id: str,
    attachment_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryAttachmentService = Depends(get_attachment_service),
):
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    try:
        att, path = await service.get_attachment(memory_id=memory_id, attachment_id=attachment_id)
        return FileResponse(
            path=str(path),
            media_type=att.content_type or "application/octet-stream",
            filename=att.file_name,
        )
    except AttachmentNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.delete(
    "/{memory_id}/attachments/{attachment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_memory_attachment(
    request: Request,
    memory_id: str,
    attachment_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryAttachmentService = Depends(get_attachment_service),
):
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    request_id = getattr(request.state, "request_id", None)

    try:
        await service.delete_attachment(memory_id=memory_id, attachment_id=attachment_id)
        await db.commit()
        return None
    except AttachmentNotFoundError as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.post("/smart", status_code=status.HTTP_201_CREATED)
async def create_memory_smart(
    request: Request,
    body: MemoryCreate,
    force_long_term: bool = Query(False, description="Force immediate long-term storage"),
    ttl: Optional[int] = Query(None, description="Short-term memory TTL in seconds (overrides default if set)"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Create a memory using the hybrid architecture.
    
    By default, memories start in short-term storage (Redis) and are
    automatically promoted to long-term storage (PostgreSQL + Qdrant) when:
    - Accessed 3+ times (frequently used)
    - Content contains important keywords (orders, preferences, etc.)
    
    Set `force_long_term=true` to skip short-term and store directly in long-term.
    
    **This is the recommended way to create memories.**
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    request_id = getattr(request.state, "request_id", None)
    
    try:
        result = await service.create_memory_smart(
            data=body,
            embedding=None,
            request_id=request_id,
            force_long_term=force_long_term,
            ttl=ttl,
        )
        await db.commit()

        from app.services.short_term_memory import ShortTermMemory
        storage = "short_term" if isinstance(result, ShortTermMemory) else "long_term"
        enqueue_memory_pipeline(
            org_id=tenant.org_id,
            memory_id=result.id,
            initiator_user_id=tenant.user_id,
            trace_id=request_id,
            storage=storage,
        )

        from app.services.short_term_memory import ShortTermMemory
        if isinstance(result, ShortTermMemory):
            return {
                "storage": "short_term",
                "id": result.id,
                "title": result.title,
                "content_preview": result.content[:200] if result.content else None,
                "scope": result.scope,
                "tags": result.tags,
                "importance_score": result.importance_score,
                "promotion_eligible": result.promotion_eligible,
                "created_at": result.created_at,
                "message": "Memory stored in short-term (Redis). Will auto-promote if accessed frequently or contains important content.",
            }
        else:
            return {
                "storage": "long_term",
                "id": result.id,
                "title": result.title,
                "content_preview": result.content_preview,
                "scope": result.scope,
                "tags": result.tags,
                "memory_type": result.memory_type,
                "created_at": result.created_at.isoformat() if result.created_at else None,
                "message": "Memory stored directly in long-term storage (PostgreSQL + Qdrant).",
            }

    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.get("/search", response_model=MemorySearchResponse)
async def search_memories(
    request: Request,
    query: str = Query(..., min_length=1, max_length=1000, description="Search query text"),
    scope: Optional[str] = Query(None, description="Filter by scope: personal, team, department, organization"),
    team_id: Optional[str] = Query(None, description="Filter by team ID (requires scope=team)"),
    hybrid: bool = Query(
        True, 
        description="Enable hybrid search (BM25 + vector similarity). When true, combines full-text lexical search with semantic vector search for better recall."
    ),
    hnms_mode: Optional[SearchHnmsMode] = Query(
        None,
        description="HNMS-inspired ranking mode: balanced (default), performance (prefer recent), research (prefer historical)",
        examples={
            "performance": {
                "summary": "Prefer recency (strong temporal decay)",
                "value": "performance",
            },
            "research": {
                "summary": "Prefer older context (weak temporal decay)",
                "value": "research",
            },
            "balanced": {
                "summary": "Balanced temporal weighting (default)",
                "value": "balanced",
            },
        },
    ),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results to return"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Search memories using hybrid semantic + lexical search.
    
    ## Search Modes
    
    **Hybrid Search (default):**
    - Combines BM25-style full-text search with vector similarity
    - Better recall for exact terms (IDs, error codes, technical terms)
    - Maintains semantic understanding from vector search
    - Uses PostgreSQL's ts_rank_cd with weighted fields:
      - Title: 1.0x weight (highest priority)
      - Content: 0.4x weight
      - Tags: 0.1x weight (lowest priority)
    
    **Vector-Only Search:**
    - Set `hybrid=false` to use only semantic vector similarity
    - Best for conceptual/semantic queries
    - Uses embeddings from Qdrant vector store
    
    ## Security
    
    All searches are:
    - Scoped to your organization automatically
    - Filtered by Row-Level Security (RLS) in PostgreSQL
    - Verified by permission checker before returning results
    - Logged in audit trail
    
    ## Examples
    
    ```bash
    # Hybrid search for error codes (benefits from BM25 exact matching)
    GET /api/v1/memories/search?query=ERROR-404&hybrid=true
    
    # Semantic search for concepts
    GET /api/v1/memories/search?query=database performance issues&hybrid=true&hnms_mode=research
    
    # Recent memories only (performance mode)
    GET /api/v1/memories/search?query=sprint planning&hybrid=true&hnms_mode=performance&limit=20
    
    # Team-scoped search
    GET /api/v1/memories/search?query=deployment&scope=team&team_id=<uuid>&hybrid=true
    ```
    
    ## Response
    
    Returns ranked memories with:
    - Combined score from vector + lexical signals
    - Temporal decay weighting (based on hnms_mode)
    - Relevance feedback adjustments (if enabled)
    - Provenance metadata for citations
    """
    start_time = time.perf_counter()
    
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    request_id = getattr(request.state, "request_id", None)

    query_embedding = await EmbeddingService.embed(query)
    
    search_request = MemorySearchRequest(
        query=query,
        scope=scope,
        team_id=team_id,
        limit=limit,
        hybrid=hybrid,
        hnms_mode=hnms_mode,
    )
    
    try:
        results = await service.search_memories(
            query_embedding=query_embedding,
            request=search_request,
            request_id=request_id,
        )

        # Persist audit logs + retrieval explanations (both use flush-only writes).
        await db.commit()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return MemorySearchResponse(
            trace_id=request_id,
            query=query,
            results=[MemoryResponse.model_validate(m) for m in results],
            total=len(results),
            took_ms=round(elapsed_ms, 2),
            ranking_meta=service.get_search_ranking_meta(search_request),
        )
    
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    request: Request,
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Get a memory by ID.
    
    Returns the memory if the user has read permission.
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    request_id = getattr(request.state, "request_id", None)
    
    try:
        memory = await service.get_memory(
            memory_id=memory_id,
            request_id=request_id,
        )
        
        if not memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory not found",
            )
        
        await db.commit()
        
        return MemoryResponse.model_validate(memory)
    
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.post(
    "/{memory_id}/feedback",
    response_model=MemoryFeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_memory_feedback(
    request: Request,
    memory_id: str,
    body: MemoryFeedbackCreate,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
    feedback_service: MemoryFeedbackService = Depends(get_feedback_service),
):
    """Submit feedback signals about a memory.

    Requires the user to have read access to the memory.
    """
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    request_id = getattr(request.state, "request_id", None)

    try:
        memory = await service.get_memory(memory_id=memory_id, request_id=request_id)
        if not memory:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

        row = await feedback_service.create_feedback(
            memory_id=memory_id,
            feedback_type=body.feedback_type,
            payload=body.payload,
            target_agent=body.target_agent,
        )
        await db.commit()

        enqueue_feedback_learning(
            org_id=tenant.org_id,
            memory_id=memory_id,
            initiator_user_id=tenant.user_id,
            trace_id=request_id,
            storage="long_term",
        )

        return MemoryFeedbackResponse.model_validate(row)

    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.post(
    "/{memory_id}/relevance",
    response_model=MemoryFeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_memory_relevance_feedback(
    request: Request,
    memory_id: str,
    body: MemoryRelevanceFeedbackCreate,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
    feedback_service: MemoryFeedbackService = Depends(get_feedback_service),
):
    """Submit explicit relevance feedback (thumbs up/down) for a memory.

    This is a convenience wrapper around the generic feedback endpoint that stores
    a standardized payload for feedback_type="relevance".
    """

    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    request_id = getattr(request.state, "request_id", None)
    trace_id = body.trace_id or request_id

    try:
        memory = await service.get_memory(memory_id=memory_id, request_id=request_id)
        if not memory:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

        payload = {
            "relevant": bool(body.relevant),
            "value": 1 if body.relevant else -1,
            "trace_id": trace_id,
        }
        if body.query:
            payload["query"] = body.query
        if body.hnms_mode:
            payload["hnms_mode"] = body.hnms_mode

        row = await feedback_service.create_feedback(
            memory_id=memory_id,
            feedback_type="relevance",
            payload=payload,
            target_agent=body.target_agent,
        )
        await db.commit()

        enqueue_feedback_learning(
            org_id=tenant.org_id,
            memory_id=memory_id,
            initiator_user_id=tenant.user_id,
            trace_id=request_id,
            storage="long_term",
        )

        return MemoryFeedbackResponse.model_validate(row)

    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.get(
    "/{memory_id}/feedback",
    response_model=MemoryFeedbackListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_memory_feedback(
    request: Request,
    memory_id: str,
    include_applied: bool = Query(True),
    limit: int = Query(50, ge=1, le=200),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
    feedback_service: MemoryFeedbackService = Depends(get_feedback_service),
):
    """List feedback entries for a memory (requires read access)."""

    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    request_id = getattr(request.state, "request_id", None)

    try:
        memory = await service.get_memory(memory_id=memory_id, request_id=request_id)
        if not memory:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

        rows, total = await feedback_service.list_feedback(
            memory_id=memory_id,
            include_applied=include_applied,
            limit=limit,
        )
        await db.commit()

        return MemoryFeedbackListResponse(
            items=[MemoryFeedbackResponse.model_validate(r) for r in rows],
            total=total,
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.get("/{memory_id}/explain", response_model=AccessExplanation)
async def explain_memory_access(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Explain why user can or cannot access a memory.
    
    Returns detailed explanation of access decisions for all
    action types (read, write, share, delete).
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    explanation = await service.explain_access(memory_id)
    
    return AccessExplanation.model_validate(explanation)


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    request: Request,
    memory_id: str,
    body: MemoryUpdate,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Update a memory.
    
    **Required permissions:** Write access to the memory.
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    request_id = getattr(request.state, "request_id", None)
    
    try:
        async with db.begin():
            memory = await service.update_memory(
                memory_id=memory_id,
                data=body,
                request_id=request_id,
            )
            await db.commit()
            
            # Emit webhook event
            webhook_svc = WebhookService(db)
            await webhook_svc.emit_event(
                organization_id=tenant.org_id,
                event_type="memory.updated",
                payload={
                    "memory_id": memory.id,
                    "title": memory.title,
                    "user_id": tenant.user_id,
                    "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                },
            )
            await db.commit()
            
            return MemoryResponse.model_validate(memory)
    
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    request: Request,
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Delete a memory (soft delete).
    
    Memories under legal hold cannot be deleted.
    
    **Required permissions:** Delete access to the memory.
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    request_id = getattr(request.state, "request_id", None)
    
    try:
        await service.delete_memory(
            memory_id=memory_id,
            request_id=request_id,
        )
        await db.commit()
        
        # Emit webhook event
        webhook_svc = WebhookService(db)
        await webhook_svc.emit_event(
            organization_id=tenant.org_id,
            event_type="memory.deleted",
            payload={
                "memory_id": memory_id,
                "user_id": tenant.user_id,
                "deleted_at": datetime.utcnow().isoformat(),
            },
        )
        await db.commit()
    
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{memory_id}/share", response_model=MemorySharingResponse, status_code=status.HTTP_201_CREATED)
async def share_memory(
    request: Request,
    memory_id: str,
    body: MemoryShareRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Share a memory with a user, team, or other entity.
    
    **Required permissions:** Share access to the memory.
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    request_id = getattr(request.state, "request_id", None)
    
    try:
        async with db.begin():
            share = await service.share_memory(
                memory_id=memory_id,
                request=body,
                request_id=request_id,
            )
            await db.commit()
            
            return MemorySharingResponse.model_validate(share)
    
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


# =============================================================================
# Hybrid Memory Architecture - Dynamic Routes
# =============================================================================

@router.post("/{memory_id}/promote", status_code=status.HTTP_200_OK)
async def promote_memory(
    request: Request,
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Manually promote a short-term memory to long-term storage.
    
    Use this to immediately promote a memory from Redis to PostgreSQL + Qdrant
    without waiting for automatic promotion criteria to be met.
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    try:
        promoted = await service.promote_memory(memory_id)
        await db.commit()
        
        if not promoted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Short-term memory {memory_id} not found",
            )
        
        return {
            "success": True,
            "message": "Memory promoted to long-term storage",
            "long_term_id": promoted.id,
            "title": promoted.title,
            "scope": promoted.scope,
            "created_at": promoted.created_at.isoformat() if promoted.created_at else None,
        }
    
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.post("/{memory_id}/consolidate", status_code=status.HTTP_202_ACCEPTED)
async def consolidate_memory(
    request: Request,
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Consolidate a memory with related memories.
    
    This triggers the memory consolidation pipeline:
    1. Find related memories using semantic similarity
    2. Extract common themes and deduplicate information
    3. Create a consolidated memory entry
    4. Update references and maintain lineage
    
    Returns immediately with a task_id. Use task status endpoint to check progress.
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    try:
        # Verify memory exists and user has access
        memory = await service.get_memory(memory_id)
        if not memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found",
            )
        
        # Enqueue consolidation task
        from app.tasks.memory_consolidation import consolidate_memory_task
        task = consolidate_memory_task.apply_async(
            args=[memory_id, tenant.org_id, tenant.user_id],
            task_id=f"consolidate_{memory_id}_{int(time.time())}",
        )
        
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": f"Consolidation task queued for memory {memory_id}",
            "memory_id": memory_id,
        }
    
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


# =============================================================================
# Batch Operations
# =============================================================================

@router.post("/batch/update", response_model=MemoryBatchUpdateResponse)
async def batch_update_memories(
    request: Request,
    batch_request: MemoryBatchUpdateRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Update multiple memories in a single transaction.
    
    Each memory can have different updates. Use this for bulk metadata updates,
    tag additions, or scope changes.
    
    **Limits**: Max 200 memories per batch.
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    results: List[MemoryBatchUpdateResult] = []
    
    try:
        for item in batch_request.items:
            try:
                memory = await service.get_memory(item.memory_id)
                if not memory:
                    results.append(MemoryBatchUpdateResult(
                        memory_id=item.memory_id,
                        success=False,
                        error="Memory not found"
                    ))
                    continue
                
                updated_memory = await service.update_memory(item.memory_id, item.update)
                
                results.append(MemoryBatchUpdateResult(
                    memory_id=item.memory_id,
                    success=True,
                    memory=MemoryResponse.model_validate(updated_memory)
                ))
                
            except Exception as e:
                results.append(MemoryBatchUpdateResult(
                    memory_id=item.memory_id,
                    success=False,
                    error=str(e)
                ))
        
        await db.commit()
        
        return MemoryBatchUpdateResponse(results=results)
    
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch update operation failed: {str(e)}"
        )


@router.post("/batch/delete", response_model=MemoryBatchDeleteResponse)
async def batch_delete_memories(
    request: Request,
    batch_request: MemoryBatchDeleteRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Delete multiple memories in a single transaction.
    
    All deletions succeed or all fail (atomic operation).
    
    **Limits**: Max 500 memories per batch.
    **Warning**: This is a destructive operation and cannot be undone.
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    results: List[MemoryBatchResult] = []
    
    try:
        for memory_id in batch_request.memory_ids:
            try:
                memory = await service.get_memory(memory_id)
                if not memory:
                    results.append(MemoryBatchResult(
                        memory_id=memory_id,
                        success=False,
                        error="Memory not found"
                    ))
                    continue
                
                await service.delete_memory(memory_id)
                
                results.append(MemoryBatchResult(
                    memory_id=memory_id,
                    success=True
                ))
                
            except Exception as e:
                results.append(MemoryBatchResult(
                    memory_id=memory_id,
                    success=False,
                    error=str(e)
                ))
        
        await db.commit()
        
        return MemoryBatchDeleteResponse(results=results)
    
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch delete operation failed: {str(e)}"
        )


@router.post("/batch/share", response_model=MemoryBatchShareResponse)
async def batch_share_memories(
    request: Request,
    batch_request: MemoryBatchShareRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Share multiple memories with an agent in a single transaction.
    
    All shares succeed or all fail (atomic operation).
    
    **Limits**: Max 500 memories per batch.
    """
    # Set tenant context for RLS
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    
    results: List[MemoryBatchShareResult] = []
    
    try:
        for memory_id in batch_request.memory_ids:
            try:
                memory = await service.get_memory(memory_id)
                if not memory:
                    results.append(MemoryBatchShareResult(
                        memory_id=memory_id,
                        success=False,
                        error="Memory not found"
                    ))
                    continue
                
                share = await service.share_memory(memory_id, batch_request.share)
                
                results.append(MemoryBatchShareResult(
                    memory_id=memory_id,
                    success=True,
                    share_id=share.id if hasattr(share, 'id') else None
                ))
                
            except Exception as e:
                results.append(MemoryBatchShareResult(
                    memory_id=memory_id,
                    success=False,
                    error=str(e)
                ))
        
        await db.commit()
        
        return MemoryBatchShareResponse(results=results)
    
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch share operation failed: {str(e)}"
        )
