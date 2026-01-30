"""
Memory Syscall API Endpoints - Phase 2 Implementation

Exposes capability-scoped memory operations:
- read, append, search, upsert, consolidate
- capability token management (admin only)
"""

from typing import Optional, List
import uuid
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import get_tenant_context, TenantContext
from app.services.memory_syscall_service import (
    MemorySyscall,
    CapabilityDeniedException,
    TokenExpiredException,
    QuotaExceededException
)
from app.services.capability_token_service import CapabilityTokenService
from app.schemas.capability import (
    CapabilityTokenResponse,
    CapabilityTokenCreateRequest,
    MemoryReadRequest,
    MemoryAppendRequest,
    MemorySearchRequest,
    MemoryUpsertRequest,
    MemoryConsolidateRequest,
    MemorySyscallResponse
)

router = APIRouter(prefix="/memory", tags=["memory_syscall"])

# Extract bearer token from Authorization header
def get_bearer_token(authorization: Optional[str] = Header(None)) -> str:
    """Extract Bearer token from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    return authorization[7:]  # Remove "Bearer " prefix


# ============================================================================
# SYSCALL ENDPOINTS (capability-scoped)
# ============================================================================

@router.post("/syscall/read")
async def syscall_read(
    request: MemoryReadRequest,
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
):
    """
    Read knowledge item via capability token.
    
    Requires: Capability token with 'read' scope
    """
    try:
        token_str = authorization.replace("Bearer ", "")
        syscall = MemorySyscall(db, tenant.org_id)
        result = await syscall.read(
            token_str=token_str,
            knowledge_id=request.knowledge_id,
            user_id=tenant.user_id
        )
        return {"success": True, "data": result}
    except TokenExpiredException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except CapabilityDeniedException as e:
        raise HTTPException(status_code=403, detail=str(e))
    except QuotaExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/syscall/append")
async def syscall_append(
    request: MemoryAppendRequest,
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
):
    """
    Append new knowledge item via capability token.
    
    Requires: Capability token with 'append' scope
    """
    try:
        token_str = authorization.replace("Bearer ", "")
        syscall = MemorySyscall(db, tenant.org_id)
        result = await syscall.append(
            token_str=token_str,
            content=request.content,
            embedding=request.embedding,
            metadata=request.metadata,
            user_id=tenant.user_id
        )
        return {"success": True, "data": result}
    except TokenExpiredException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except CapabilityDeniedException as e:
        raise HTTPException(status_code=403, detail=str(e))
    except QuotaExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/syscall/search")
async def syscall_search(
    request: MemorySearchRequest,
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
):
    """
    Search knowledge via vector similarity with RLS verification.
    
    Requires: Capability token with 'search' scope
    Process: Qdrant vector search + Postgres RLS re-verification
    """
    try:
        token_str = authorization.replace("Bearer ", "")
        syscall = MemorySyscall(db, tenant.org_id)
        results = await syscall.search(
            token_str=token_str,
            query_embedding=request.embedding,
            limit=request.limit or 10,
            user_id=tenant.user_id
        )
        return {"success": True, "data": results, "count": len(results)}
    except TokenExpiredException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except CapabilityDeniedException as e:
        raise HTTPException(status_code=403, detail=str(e))
    except QuotaExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/syscall/upsert")
async def syscall_upsert(
    request: MemoryUpsertRequest,
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
):
    """
    Update or insert knowledge item via capability token.
    
    Requires: Capability token with 'upsert' scope
    """
    try:
        token_str = authorization.replace("Bearer ", "")
        syscall = MemorySyscall(db, tenant.org_id)
        result = await syscall.upsert(
            token_str=token_str,
            knowledge_id=request.knowledge_id,
            content=request.content,
            embedding=request.embedding,
            metadata=request.metadata,
            user_id=tenant.user_id
        )
        return {"success": True, "data": result}
    except TokenExpiredException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except CapabilityDeniedException as e:
        raise HTTPException(status_code=403, detail=str(e))
    except QuotaExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/syscall/consolidate")
async def syscall_consolidate(
    request: MemoryConsolidateRequest,
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
):
    """
    Consolidate (merge) multiple knowledge items.
    
    Requires: Capability token with 'consolidate' scope
    """
    try:
        token_str = authorization.replace("Bearer ", "")
        syscall = MemorySyscall(db, tenant.org_id)
        result = await syscall.consolidate(
            token_str=token_str,
            knowledge_ids=request.knowledge_ids,
            merged_content=request.merged_content,
            metadata=request.metadata,
            user_id=tenant.user_id
        )
        return {"success": True, "data": result}
    except TokenExpiredException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except CapabilityDeniedException as e:
        raise HTTPException(status_code=403, detail=str(e))
    except QuotaExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# CAPABILITY MANAGEMENT ENDPOINTS (admin only)
# ============================================================================

@router.post("/capabilities/issue")
async def issue_capability_token(
    request: CapabilityTokenCreateRequest,
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
):
    """
    Issue a new capability token (admin only).
    
    Returns: Token with .token field containing Bearer value
    """
    # Check admin
    if "admin" not in tenant.roles:
        raise HTTPException(status_code=403, detail="Admin only")

    try:
        svc = CapabilityTokenService(db, tenant.org_id)
        token = await svc.issue_token(
            name=request.name,
            scopes=request.scopes,
            session_id=request.session_id,
            agent_name=request.agent_name,
            issued_to_user_id=request.issued_to_user_id,
            ttl_seconds=request.ttl_seconds or 86400,
            max_tokens_per_month=request.max_tokens_per_month,
            max_storage_bytes=request.max_storage_bytes,
            max_requests_per_minute=request.max_requests_per_minute,
            created_by_user_id=tenant.user_id
        )
        await db.commit()

        return {
            "success": True,
            "data": {
                "id": str(token.id),
                "token": token.token,  # RETURN THE ACTUAL TOKEN VALUE
                "name": token.name,
                "scopes": token.scopes,
                "expires_at": token.expires_at.isoformat(),
                "created_at": token.issued_at.isoformat()
            }
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/capabilities/{token_id}/revoke")
async def revoke_capability_token(
    token_id: uuid.UUID,
    reason: str = Query(...),
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
):
    """Revoke a capability token (admin only)."""
    if "admin" not in tenant.roles:
        raise HTTPException(status_code=403, detail="Admin only")

    try:
        svc = CapabilityTokenService(db, tenant.org_id)
        await svc.revoke_token(
            token_id=token_id,
            reason=reason,
            revoked_by_user_id=tenant.user_id
        )
        await db.commit()
        return {"success": True, "message": "Token revoked"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/capabilities")
async def list_capability_tokens(
    agent_name: Optional[str] = Query(None),
    include_revoked: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
):
    """List organization's capability tokens (admin only)."""
    if "admin" not in tenant.roles:
        raise HTTPException(status_code=403, detail="Admin only")

    try:
        svc = CapabilityTokenService(db, tenant.org_id)
        tokens = await svc.list_tokens(
            agent_name=agent_name,
            include_revoked=include_revoked
        )

        return {
            "success": True,
            "data": [
                {
                    "id": str(t.id),
                    "name": t.name,
                    "scopes": t.scopes,
                    "agent_name": t.agent_name,
                    "expires_at": t.expires_at.isoformat(),
                    "revoked_at": t.revoked_at.isoformat() if t.revoked_at else None,
                    "tokens_used": t.tokens_used,
                    "storage_used_bytes": t.storage_used_bytes
                }
                for t in tokens
            ],
            "count": len(tokens)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/capabilities/{token_id}")
async def get_capability_token(
    token_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
):
    """Get capability token details (admin only)."""
    if "admin" not in tenant.roles:
        raise HTTPException(status_code=403, detail="Admin only")

    try:
        svc = CapabilityTokenService(db, tenant.org_id)
        token = await svc.get_token(token_id)

        if not token:
            raise HTTPException(status_code=404, detail="Token not found")

        return {
            "success": True,
            "data": {
                "id": str(token.id),
                "name": token.name,
                "scopes": token.scopes,
                "agent_name": token.agent_name,
                "max_tokens_per_month": token.max_tokens_per_month,
                "max_storage_bytes": token.max_storage_bytes,
                "max_requests_per_minute": token.max_requests_per_minute,
                "tokens_used": token.tokens_used,
                "storage_used_bytes": token.storage_used_bytes,
                "expires_at": token.expires_at.isoformat(),
                "revoked_at": token.revoked_at.isoformat() if token.revoked_at else None,
                "last_used_at": token.last_used_at.isoformat() if token.last_used_at else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
