"""
Policy Versioning Endpoints

REST API for managing policy versions with canary rollouts and rollback support.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import uuid

from app.core.database import get_db
from app.api.v1.endpoints.auth import get_current_user
from app.models.user import User
from app.middleware.tenant_context import get_tenant_context, TenantContext
from app.services.policy_versioning import PolicyVersioningService
import logging

router = APIRouter(prefix="/policy-versions", tags=["Policy Versioning"])
logger = logging.getLogger(__name__)


# Schemas
class PolicyVersionCreateRequest(BaseModel):
    """Create policy version request."""
    name: str
    policy_type: str  # rbac, capability, rate_limit
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class PolicyVersionResponse(BaseModel):
    """Policy version response."""
    id: str
    name: str
    version: int
    policy_type: str
    status: str  # draft, active, canary, retired
    rollout_percentage: int
    created_at: str
    created_by: str
    activated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class PolicyActivateRequest(BaseModel):
    """Activate policy version request."""
    policy_id: str
    rollout_percentage: int = 100  # 0-100 for canary rollout


class PolicyRollbackRequest(BaseModel):
    """Rollback policy to previous version."""
    name: str
    policy_type: str
    target_version: int


class PolicyHistoryResponse(BaseModel):
    """Policy history entry."""
    event_type: str
    version: int
    status: str
    created_at: str
    created_by: Optional[str] = None
    rollout_percentage: Optional[int] = None


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Dependency to require admin role."""
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user


@router.post("/", response_model=PolicyVersionResponse)
async def create_policy_version(
    request: PolicyVersionCreateRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Create a new policy version.
    
    Admin only. Creates policy in 'draft' status. Use /activate to deploy.
    
    Policy types:
    - rbac: Role-based access control rules
    - capability: Memory syscall capabilities
    - rate_limit: API rate limiting rules
    """
    try:
        service = PolicyVersioningService(db, tenant.org_id)
        
        policy = await service.create_policy_version(
            name=request.name,
            policy_type=request.policy_type,
            content=request.content,
            created_by=user.id,
            metadata=request.metadata
        )
        
        logger.info(f"Created policy version: {request.name} v{policy['version']}")
        
        return PolicyVersionResponse(
            id=policy["id"],
            name=policy["name"],
            version=policy["version"],
            policy_type=policy["policy_type"],
            status=policy["status"],
            rollout_percentage=policy["rollout_percentage"],
            created_at=policy["created_at"],
            created_by=str(policy["created_by"]),
            metadata=policy.get("metadata")
        )
    except Exception as e:
        logger.error(f"Error creating policy version: {e}")
        raise HTTPException(status_code=500, detail="Failed to create policy version")


@router.post("/activate", response_model=Dict[str, Any])
async def activate_policy_version(
    request: PolicyActivateRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Activate a policy version with optional canary rollout.
    
    Admin only.
    
    Rollout strategy:
    - 0-50%: Canary rollout to subset of users
    - 50-99%: Gradual rollout
    - 100%: Full deployment
    
    Example: POST /activate with 50% to test new policy on 50% of users.
    """
    try:
        service = PolicyVersioningService(db, tenant.org_id)
        
        success = await service.activate_policy(
            policy_id=request.policy_id,
            rollout_percentage=request.rollout_percentage,
            activated_by=user.id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Policy not found")
        
        rollout_desc = (
            "canary rollout" if request.rollout_percentage < 100
            else "full deployment"
        )
        
        return {
            "policy_id": request.policy_id,
            "status": "activated",
            "rollout_percentage": request.rollout_percentage,
            "rollout_type": rollout_desc,
            "message": f"Policy activated ({rollout_desc})"
        }
    except Exception as e:
        logger.error(f"Error activating policy: {e}")
        raise HTTPException(status_code=500, detail="Failed to activate policy")


@router.post("/rollback", response_model=Dict[str, Any])
async def rollback_policy(
    request: PolicyRollbackRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Rollback policy to a specific previous version.
    
    Admin only. Useful for reverting problematic policy changes.
    """
    try:
        service = PolicyVersioningService(db, tenant.org_id)
        
        success = await service.rollback_policy(
            name=request.name,
            policy_type=request.policy_type,
            target_version=request.target_version,
            rolled_back_by=user.id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Target version not found")
        
        return {
            "name": request.name,
            "policy_type": request.policy_type,
            "target_version": request.target_version,
            "status": "rolled_back",
            "message": f"Policy rolled back to version {request.target_version}"
        }
    except Exception as e:
        logger.error(f"Error rolling back policy: {e}")
        raise HTTPException(status_code=500, detail="Failed to rollback policy")


@router.get("/history/{name}/{policy_type}", response_model=List[PolicyHistoryResponse])
async def get_policy_history(
    name: str,
    policy_type: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Get version history for a specific policy.
    
    Admin only. Returns all versions and events in descending order.
    """
    try:
        service = PolicyVersioningService(db, tenant.org_id)
        
        history = await service.get_policy_history(
            name=name,
            policy_type=policy_type
        )
        
        return [
            PolicyHistoryResponse(
                event_type=event["event_type"],
                version=event.get("version", 0),
                status=event.get("status", "unknown"),
                created_at=event.get("created_at", ""),
                created_by=event.get("created_by"),
                rollout_percentage=event.get("rollout_percentage")
            )
            for event in history
        ]
    except Exception as e:
        logger.error(f"Error getting policy history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get policy history")


@router.get("/compare/{policy_id_1}/{policy_id_2}", response_model=Dict[str, Any])
async def compare_policy_versions(
    policy_id_1: str,
    policy_id_2: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
    tenant: TenantContext = Depends(get_tenant_context)
):
    """
    Compare two policy versions and show differences.
    
    Admin only. Returns metadata differences and content change indicator.
    """
    try:
        service = PolicyVersioningService(db, tenant.org_id)
        
        diff = await service.compare_versions(policy_id_1, policy_id_2)
        
        if "error" in diff:
            raise HTTPException(status_code=404, detail=diff["error"])
        
        return {
            "policy_1": diff["policy_1"],
            "policy_2": diff["policy_2"],
            "content_changed": diff["content_changed"],
            "message": "Content differs between versions" if diff["content_changed"] else "Content is identical"
        }
    except Exception as e:
        logger.error(f"Error comparing policies: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare policies")
