"""
Capability Token Management Endpoints

Endpoints for issuing, revoking, and managing capability tokens for memory syscall access.
Admin-only operations - requires admin role.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.models.capability_token import CapabilityToken
from app.schemas.capability import (
    CapabilityTokenCreateRequest,
    CapabilityTokenResponse,
)
from app.services.capability_token_service import CapabilityTokenService
from app.services.audit_service import AuditService
from app.middleware.tenant_context import get_tenant_context, TenantContext
import logging

router = APIRouter(prefix="/capability-tokens", tags=["Capability Tokens"])
logger = logging.getLogger(__name__)


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Dependency to require admin role."""
    if not user.is_admin and not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required for capability token management"
        )
    return user


@router.post("/", response_model=CapabilityTokenResponse, status_code=status.HTTP_201_CREATED)
async def create_capability_token(
    request: CapabilityTokenCreateRequest,
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
    current_user: User = Depends(require_admin),
):
    """
    Issue a new capability token.
    
    **Admin only**. Creates a new capability token with specified scopes and quotas.
    
    Scopes:
    - `read`: Read/retrieve memory operations
    - `write`: Create/update memory operations  
    - `search`: Vector and BM25 search operations
    - `consolidate`: Short-term to long-term memory migration
    - `promote`: Auto-promotion of memories
    - `append`: Append-only memory operations
    - `upsert`: Update or insert operations
    
    The token is returned **only once** on creation. Store it securely.
    """
    try:
        service = CapabilityTokenService(db, tenant.org_id)
        
        token = await service.issue_token(
            name=request.name,
            scopes=request.scopes,
            session_id=request.session_id,
            agent_name=request.agent_name,
            issued_to_user_id=request.issued_to_user_id,
            ttl_seconds=request.ttl_seconds or 86400,
            max_tokens_per_month=request.max_tokens_per_month,
            max_storage_bytes=request.max_storage_bytes,
            max_requests_per_minute=request.max_requests_per_minute,
            created_by_user_id=current_user.id,
        )
        
        # Audit log
        audit = AuditService(db, tenant.user_id, tenant.org_id)
        await audit.log_memory_operation(
            actor_id=current_user.id,
            organization_id=tenant.org_id,
            memory_id=None,
            operation="capability_token.create",
            success=True,
            details={
                "token_id": token.id,
                "scopes": request.scopes,
                "agent_name": request.agent_name,
            }
        )
        
        await db.commit()
        
        return CapabilityTokenResponse(
            id=token.id,
            token=token.token,  # Only returned here
            name=request.name,
            scopes=token.scopes,
            agent_name=token.token_metadata.get("agent_name") if token.token_metadata else None,
            expires_at=token.expires_at.isoformat(),
            created_at=token.created_at.isoformat(),
            tokens_used=token.tokens_used_this_month,
            storage_used_bytes=token.storage_used_bytes,
            revoked_at=token.revoked_at.isoformat() if token.revoked_at else None,
        )
        
    except Exception as e:
        logger.error(f"Error creating capability token: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create capability token: {str(e)}"
        )


@router.get("/", response_model=list[CapabilityTokenResponse])
async def list_capability_tokens(
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
    current_user: User = Depends(require_admin),
    active_only: bool = True,
    limit: int = 50,
    offset: int = 0,
):
    """
    List all capability tokens for the organization.
    
    **Admin only**. Returns tokens without the actual token value (for security).
    """
    try:
        service = CapabilityTokenService(db, tenant.org_id)
        tokens = await service.list_tokens(active_only=active_only, limit=limit, offset=offset)
        
        return [
            CapabilityTokenResponse(
                id=token.id,
                token="[REDACTED]",  # Never expose in list
                name=token.token_metadata.get("name", "Unnamed") if token.token_metadata else "Unnamed",
                scopes=token.scopes,
                agent_name=token.token_metadata.get("agent_name") if token.token_metadata else None,
                expires_at=token.expires_at.isoformat(),
                created_at=token.created_at.isoformat(),
                tokens_used=token.tokens_used_this_month,
                storage_used_bytes=token.storage_used_bytes,
                revoked_at=token.revoked_at.isoformat() if token.revoked_at else None,
            )
            for token in tokens
        ]
        
    except Exception as e:
        logger.error(f"Error listing capability tokens: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list capability tokens: {str(e)}"
        )


@router.get("/{token_id}", response_model=CapabilityTokenResponse)
async def get_capability_token(
    token_id: str,
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
    current_user: User = Depends(require_admin),
):
    """
    Get details of a specific capability token.
    
    **Admin only**. Does not return the actual token value.
    """
    try:
        service = CapabilityTokenService(db, tenant.org_id)
        token = await service.get_token(token_id)
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Token {token_id} not found"
            )
        
        return CapabilityTokenResponse(
            id=token.id,
            token="[REDACTED]",
            name=token.token_metadata.get("name", "Unnamed") if token.token_metadata else "Unnamed",
            scopes=token.scopes,
            agent_name=token.token_metadata.get("agent_name") if token.token_metadata else None,
            expires_at=token.expires_at.isoformat(),
            created_at=token.created_at.isoformat(),
            tokens_used=token.tokens_used_this_month,
            storage_used_bytes=token.storage_used_bytes,
            revoked_at=token.revoked_at.isoformat() if token.revoked_at else None,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting capability token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get capability token: {str(e)}"
        )


@router.delete("/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_capability_token(
    token_id: str,
    reason: str = "Admin revocation",
    db: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
    current_user: User = Depends(require_admin),
):
    """
    Revoke a capability token.
    
    **Admin only**. Once revoked, the token cannot be used for any operations.
    """
    try:
        service = CapabilityTokenService(db, tenant.org_id)
        
        success = await service.revoke_token(
            token_id=token_id,
            reason=reason,
            revoked_by_user_id=current_user.id,
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Token {token_id} not found or already revoked"
            )
        
        # Audit log
        audit = AuditService(db, tenant.user_id, tenant.org_id)
        await audit.log_memory_operation(
            actor_id=current_user.id,
            organization_id=tenant.org_id,
            memory_id=None,
            operation="capability_token.revoke",
            success=True,
            details={"token_id": token_id, "reason": reason}
        )
        
        await db.commit()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking capability token: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke capability token: {str(e)}"
        )


@router.post("/validate", status_code=status.HTTP_200_OK)
async def validate_capability_token(
    authorization: str = Header(..., description="Bearer token"),
    required_scopes: list[str] = [],
    db: AsyncSession = Depends(get_db),
):
    """
    Validate a capability token and check scopes.
    
    Public endpoint for agents/services to validate their tokens.
    Returns 200 if valid, 401/403 if invalid or insufficient scopes.
    """
    try:
        # Extract token from Bearer header
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format"
            )
        
        token_value = authorization[7:]  # Remove "Bearer "
        
        service = CapabilityTokenService(db, None)  # org_id not needed for validation
        token = await service.validate_token(token_value)
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        # Check scopes
        token_scopes = set(token.scopes.split(","))
        if required_scopes and not set(required_scopes).issubset(token_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient scopes. Required: {required_scopes}, Have: {list(token_scopes)}"
            )
        
        return {
            "valid": True,
            "organization_id": token.organization_id,
            "scopes": list(token_scopes),
            "expires_at": token.expires_at.isoformat(),
            "quota_status": {
                "tokens_used": token.tokens_used_this_month,
                "tokens_quota": token.quota_tokens_per_month,
                "storage_used": token.storage_used_bytes,
                "storage_quota": token.quota_storage_bytes,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating capability token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate capability token: {str(e)}"
        )
