"""
Permission Checker Service
==========================

Centralized permission checking with caching and explainability.
All access decisions should go through this service.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.redis import RedisClient
from app.models.user import User, UserRole, Role
from app.models.team import TeamMember
from app.models.memory import MemoryMetadata, MemorySharing


@dataclass
class AccessDecision:
    """
    Result of an access check with explanation.
    
    Attributes:
        allowed: Whether access is granted
        reason: Human-readable explanation
        method: How access was determined (own, team, share, policy)
        details: Additional details for audit
    """
    allowed: bool
    reason: str
    method: str
    details: dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "method": self.method,
            "details": self.details,
        }


class PermissionChecker:
    """
    Permission checking service with caching and explainability.
    
    This service is the central authority for all permission checks.
    It supports:
    - Role-based access control (RBAC)
    - Scope-based restrictions (team, department, etc.)
    - Explicit sharing
    - Classification/clearance checks
    - Cached permission lookups
    
    All methods return AccessDecision with explanation strings
    for audit logging and user feedback.
    """
    
    # Cache key prefixes
    CACHE_PREFIX_PERMISSIONS = "perms"
    CACHE_PREFIX_ROLES = "roles"
    
    def __init__(self, session: AsyncSession):
        """
        Initialize permission checker.
        
        Args:
            session: Database session with tenant context set
        """
        self.session = session
    
    # =========================================================================
    # Role & Permission Loading
    # =========================================================================
    
    async def get_effective_permissions(
        self,
        user_id: str,
        org_id: str,
    ) -> List[str]:
        """
        Get all effective permissions for a user in an organization.
        
        Aggregates permissions from all active, non-expired roles.
        Results are cached in Redis for performance.
        
        Args:
            user_id: User UUID
            org_id: Organization UUID
        
        Returns:
            List of permission strings
        """
        # Check cache first
        cache_key = f"{self.CACHE_PREFIX_PERMISSIONS}:{user_id}:{org_id}"
        cached = await RedisClient.get_json(cache_key)
        if cached is not None:
            return cached
        
        # Load from database
        permissions = await self._load_permissions_from_db(user_id, org_id)
        
        # Cache result
        await RedisClient.set_json(
            cache_key,
            permissions,
            ttl=settings.PERMISSION_CACHE_TTL,
        )
        
        return permissions
    
    async def _load_permissions_from_db(
        self,
        user_id: str,
        org_id: str,
    ) -> List[str]:
        """Load permissions from database (not cached)."""
        now = datetime.utcnow()
        
        # Build organization filter - if no org_id, get all user's roles
        if org_id:
            org_filter = UserRole.organization_id == org_id
        else:
            # When no org context, get permissions from any organization the user belongs to
            org_filter = UserRole.organization_id.isnot(None)
        
        # Query all active, non-expired role assignments
        query = (
            select(Role.permissions)
            .join(UserRole, UserRole.role_id == Role.id)
            .where(
                and_(
                    UserRole.user_id == user_id,
                    org_filter,
                    or_(
                        UserRole.expires_at.is_(None),
                        UserRole.expires_at > now,
                    ),
                )
            )
        )
        
        result = await self.session.execute(query)
        
        # Aggregate all permissions
        all_permissions = set()
        for (perms,) in result:
            if perms:
                all_permissions.update(perms)
        
        return list(all_permissions)
    
    async def get_user_roles(
        self,
        user_id: str,
        org_id: str,
    ) -> List[dict]:
        """
        Get all active roles for a user in an organization.
        
        Args:
            user_id: User UUID
            org_id: Organization UUID
        
        Returns:
            List of role dictionaries with scope info
        """
        now = datetime.utcnow()
        
        query = (
            select(Role, UserRole)
            .join(UserRole, UserRole.role_id == Role.id)
            .where(
                and_(
                    UserRole.user_id == user_id,
                    UserRole.organization_id == org_id,
                    or_(
                        UserRole.expires_at.is_(None),
                        UserRole.expires_at > now,
                    ),
                )
            )
        )
        
        result = await self.session.execute(query)
        
        roles = []
        for role, user_role in result:
            roles.append({
                "role_id": role.id,
                "role_name": role.name,
                "display_name": role.display_name,
                "scope_type": user_role.scope_type,
                "scope_id": user_role.scope_id,
                "expires_at": user_role.expires_at.isoformat() if user_role.expires_at else None,
            })
        
        return roles
    
    # =========================================================================
    # Permission Checking
    # =========================================================================
    
    async def check_permission(
        self,
        user_id: str,
        org_id: str,
        permission: str,
        resource_id: Optional[str] = None,
    ) -> AccessDecision:
        """
        Check if a user has a specific permission.
        
        Args:
            user_id: User UUID
            org_id: Organization UUID
            permission: Permission string (e.g., "memory:read:own")
            resource_id: Optional resource ID for resource-specific checks
        
        Returns:
            AccessDecision with allowed status and explanation
        """
        permissions = await self.get_effective_permissions(user_id, org_id)
        
        # Check for exact match
        if permission in permissions:
            return AccessDecision(
                allowed=True,
                reason=f"User has permission '{permission}'",
                method="direct_permission",
            )
        
        # Check for wildcard permissions
        parts = permission.split(":")
        if len(parts) >= 2:
            # Check for resource:* wildcard
            wildcard = f"{parts[0]}:*"
            if wildcard in permissions:
                return AccessDecision(
                    allowed=True,
                    reason=f"User has wildcard permission '{wildcard}'",
                    method="wildcard_permission",
                )
            
            # Check for resource:action:* wildcard
            if len(parts) >= 3:
                wildcard = f"{parts[0]}:{parts[1]}:*"
                if wildcard in permissions:
                    return AccessDecision(
                        allowed=True,
                        reason=f"User has scope wildcard permission '{wildcard}'",
                        method="wildcard_permission",
                    )
        
        # Check for super admin
        if "*:*" in permissions or "admin:*" in permissions:
            return AccessDecision(
                allowed=True,
                reason="User has admin privileges",
                method="admin",
            )
        
        return AccessDecision(
            allowed=False,
            reason=f"User lacks permission '{permission}'",
            method="none",
        )
    
    # =========================================================================
    # Memory Access Checking
    # =========================================================================
    
    async def check_memory_access(
        self,
        user_id: str,
        org_id: str,
        memory_id: str,
        action: str,
        clearance_level: int = 0,
    ) -> AccessDecision:
        """
        Check if a user can perform an action on a memory.
        
        This is the primary method for memory access control.
        It checks (in order):
        1. Own memory access
        2. Team membership access
        3. Explicit sharing
        4. Scope-based access (department/division/org)
        5. Classification/clearance requirements
        
        Args:
            user_id: User UUID
            org_id: Organization UUID
            memory_id: Memory UUID
            action: Action type (read, write, delete, share, export)
            clearance_level: User's security clearance level
        
        Returns:
            AccessDecision with allowed status and detailed explanation
        """
        # Load the memory
        memory = await self.session.get(MemoryMetadata, memory_id)
        
        if not memory:
            return AccessDecision(
                allowed=False,
                reason="Memory not found",
                method="not_found",
            )
        
        # Check organization isolation
        if memory.organization_id != org_id:
            return AccessDecision(
                allowed=False,
                reason="Memory belongs to different organization",
                method="org_isolation",
            )
        
        # Check classification/clearance
        if memory.required_clearance > clearance_level:
            return AccessDecision(
                allowed=False,
                reason=f"Requires clearance level {memory.required_clearance}, user has {clearance_level}",
                method="clearance",
                details={
                    "required_clearance": memory.required_clearance,
                    "user_clearance": clearance_level,
                    "classification": memory.classification,
                },
            )
        
        # Check if user owns the memory
        if memory.owner_id == user_id:
            return AccessDecision(
                allowed=True,
                reason="User owns this memory",
                method="own",
                details={"owner_id": user_id},
            )
        
        # Check team membership for team-scoped memories
        if memory.scope == "team" and memory.scope_id:
            team_access = await self._check_team_access(
                user_id, org_id, memory.scope_id, action
            )
            if team_access.allowed:
                return team_access
        
        # Check explicit sharing
        share_access = await self._check_share_access(
            user_id, org_id, memory_id, action
        )
        if share_access.allowed:
            return share_access
        
        # Check scope-based access (org-wide memories, etc.)
        scope_access = await self._check_scope_access(
            user_id, org_id, memory, action
        )
        if scope_access.allowed:
            return scope_access
        
        # Default deny
        return AccessDecision(
            allowed=False,
            reason=f"No access granted for {action} on this memory",
            method="none",
            details={
                "memory_scope": memory.scope,
                "memory_owner": memory.owner_id,
            },
        )

    async def filter_memory_ids_with_access(
        self,
        user_id: str,
        org_id: str,
        memory_ids: List[str],
        action: str,
        clearance_level: int = 0,
    ) -> List[str]:
        """Filter a list of memory IDs to those the user can access.

        This is a batched variant of `check_memory_access` intended for hot paths.
        It mirrors the same permission semantics currently implemented:
        - organization isolation
        - clearance
        - owner access
        - team membership for team-scoped memories
        - explicit user shares
        - organization/global scope reads

        Returns:
            Memory IDs in the same order as input, filtered to allowed.
        """

        if not memory_ids:
            return []

        # Load required fields for all candidate memories (single query)
        mem_stmt = (
            select(
                MemoryMetadata.id,
                MemoryMetadata.owner_id,
                MemoryMetadata.scope,
                MemoryMetadata.scope_id,
                MemoryMetadata.required_clearance,
                MemoryMetadata.organization_id,
            )
            .where(MemoryMetadata.id.in_(memory_ids))
        )
        mem_rows = (await self.session.execute(mem_stmt)).all()

        mem_by_id: dict[str, tuple[str, str, str | None, int, str]] = {}
        for row in mem_rows:
            mem_id, owner_id, scope, scope_id, required_clearance, organization_id = row
            mem_by_id[str(mem_id)] = (
                str(owner_id) if owner_id is not None else "",
                str(scope) if scope is not None else "",
                str(scope_id) if scope_id is not None else None,
                int(required_clearance or 0),
                str(organization_id) if organization_id is not None else "",
            )

        # Candidate IDs that exist and meet org + clearance requirements
        eligible_ids: list[str] = []
        for mem_id in memory_ids:
            meta = mem_by_id.get(mem_id)
            if not meta:
                continue
            owner_id, scope, _scope_id, required_clearance, organization_id = meta
            if organization_id != org_id:
                continue
            if required_clearance > clearance_level:
                continue
            eligible_ids.append(mem_id)

        if not eligible_ids:
            return []

        allowed_set: set[str] = set()

        # Owner access
        for mem_id in eligible_ids:
            owner_id, _scope, _scope_id, _required_clearance, _organization_id = mem_by_id[mem_id]
            if owner_id == user_id:
                allowed_set.add(mem_id)

        # Scope-based access (only what `check_memory_access` currently allows)
        if action == "read":
            for mem_id in eligible_ids:
                _owner_id, scope, _scope_id, _required_clearance, _organization_id = mem_by_id[mem_id]
                if scope in {"organization", "global"}:
                    allowed_set.add(mem_id)

        # Team membership access for team-scoped memories
        if action in {"read", "comment"}:
            team_ids: set[str] = set()
            mem_ids_by_team: dict[str, list[str]] = {}
            for mem_id in eligible_ids:
                _owner_id, scope, scope_id, _required_clearance, _organization_id = mem_by_id[mem_id]
                if scope == "team" and scope_id:
                    team_ids.add(scope_id)
                    mem_ids_by_team.setdefault(scope_id, []).append(mem_id)

            if team_ids:
                team_stmt = select(TeamMember.team_id).where(
                    and_(
                        TeamMember.user_id == user_id,
                        TeamMember.organization_id == org_id,
                        TeamMember.is_active == True,
                        TeamMember.team_id.in_(list(team_ids)),
                    )
                )
                team_rows = (await self.session.execute(team_stmt)).all()
                member_team_ids = {str(r[0]) for r in team_rows}
                for team_id in member_team_ids:
                    for mem_id in mem_ids_by_team.get(team_id, []):
                        allowed_set.add(mem_id)

        # Explicit user share access
        permission_map = {
            "read": ["read", "comment", "edit"],
            "comment": ["comment", "edit"],
            "write": ["edit"],
            "update": ["edit"],
        }
        allowed_permissions = permission_map.get(action, [])
        if allowed_permissions:
            now = datetime.now(timezone.utc)
            share_stmt = select(MemorySharing.memory_id).where(
                and_(
                    MemorySharing.memory_id.in_(eligible_ids),
                    MemorySharing.share_type == "user",
                    MemorySharing.target_id == user_id,
                    MemorySharing.organization_id == org_id,
                    MemorySharing.is_active == True,
                    MemorySharing.permission.in_(allowed_permissions),
                    or_(MemorySharing.expires_at.is_(None), MemorySharing.expires_at > now),
                )
            )
            share_rows = (await self.session.execute(share_stmt)).all()
            for row in share_rows:
                allowed_set.add(str(row[0]))

        # Preserve input order
        return [mem_id for mem_id in memory_ids if mem_id in allowed_set]
    
    async def _check_team_access(
        self,
        user_id: str,
        org_id: str,
        team_id: str,
        action: str,
    ) -> AccessDecision:
        """Check if user has team-based access."""
        query = select(TeamMember).where(
            and_(
                TeamMember.user_id == user_id,
                TeamMember.team_id == team_id,
                TeamMember.organization_id == org_id,
                TeamMember.is_active == True,
            )
        )
        
        result = await self.session.execute(query)
        member = result.scalar_one_or_none()
        
        if not member:
            return AccessDecision(
                allowed=False,
                reason="User is not a member of the team",
                method="team",
            )
        
        # Check action-specific permissions based on team role
        if action in ("read", "comment"):
            return AccessDecision(
                allowed=True,
                reason=f"User is a {member.role} of the team",
                method="team",
                details={"team_role": member.role},
            )
        
        if action in ("write", "update", "share"):
            if member.role in ("lead", "admin"):
                return AccessDecision(
                    allowed=True,
                    reason=f"User is a {member.role} of the team",
                    method="team",
                    details={"team_role": member.role},
                )
            return AccessDecision(
                allowed=False,
                reason=f"Team members cannot {action} (requires lead/admin)",
                method="team",
            )
        
        if action == "delete":
            if member.role == "admin":
                return AccessDecision(
                    allowed=True,
                    reason="User is admin of the team",
                    method="team",
                    details={"team_role": member.role},
                )
            return AccessDecision(
                allowed=False,
                reason="Only team admins can delete",
                method="team",
            )
        
        return AccessDecision(
            allowed=False,
            reason=f"Unknown action: {action}",
            method="team",
        )
    
    async def _check_share_access(
        self,
        user_id: str,
        org_id: str,
        memory_id: str,
        action: str,
    ) -> AccessDecision:
        """Check if user has explicit share-based access."""
        now = datetime.now(timezone.utc)
        
        # Check for direct user share
        query = select(MemorySharing).where(
            and_(
                MemorySharing.memory_id == memory_id,
                MemorySharing.share_type == "user",
                MemorySharing.target_id == user_id,
                MemorySharing.organization_id == org_id,
                MemorySharing.is_active == True,
                or_(
                    MemorySharing.expires_at.is_(None),
                    MemorySharing.expires_at > now,
                ),
            )
        )
        
        result = await self.session.execute(query)
        share = result.scalar_one_or_none()
        
        if not share:
            return AccessDecision(
                allowed=False,
                reason="No active share found for user",
                method="share",
            )
        
        # Check permission level
        permission_map = {
            "read": ["read", "comment", "edit"],
            "comment": ["comment", "edit"],
            "write": ["edit"],
            "update": ["edit"],
        }
        
        allowed_permissions = permission_map.get(action, [])
        
        if share.permission in allowed_permissions:
            return AccessDecision(
                allowed=True,
                reason=f"Memory shared with user ({share.permission} access)",
                method="share",
                details={
                    "share_permission": share.permission,
                    "shared_by": share.shared_by,
                },
            )
        
        return AccessDecision(
            allowed=False,
            reason=f"Share permission '{share.permission}' insufficient for '{action}'",
            method="share",
        )
    
    async def _check_scope_access(
        self,
        user_id: str,
        org_id: str,
        memory: MemoryMetadata,
        action: str,
    ) -> AccessDecision:
        """Check scope-based access (org-wide, etc.)."""
        # Organization-scoped memories are readable by all org members
        if memory.scope == "organization":
            if action == "read":
                return AccessDecision(
                    allowed=True,
                    reason="Organization-scoped memory is readable by all org members",
                    method="scope",
                )
        
        # Global-scoped memories (public within platform)
        if memory.scope == "global":
            if action == "read":
                return AccessDecision(
                    allowed=True,
                    reason="Global-scoped memory is readable by all users",
                    method="scope",
                )
        
        return AccessDecision(
            allowed=False,
            reason=f"Scope '{memory.scope}' does not grant {action} access",
            method="scope",
        )
    
    # =========================================================================
    # Explanation & Audit
    # =========================================================================
    
    async def explain_access(
        self,
        user_id: str,
        org_id: str,
        memory_id: str,
        clearance_level: int = 0,
    ) -> dict:
        """
        Generate a detailed explanation of why a user can/cannot access a memory.
        
        This is used for:
        - UI "Why can I see this?" feature
        - Audit logging
        - Debugging access issues
        
        Args:
            user_id: User UUID
            org_id: Organization UUID
            memory_id: Memory UUID
            clearance_level: User's security clearance
        
        Returns:
            Dictionary with detailed access explanation
        """
        # Check all access methods
        read_decision = await self.check_memory_access(
            user_id, org_id, memory_id, "read", clearance_level
        )
        write_decision = await self.check_memory_access(
            user_id, org_id, memory_id, "write", clearance_level
        )
        share_decision = await self.check_memory_access(
            user_id, org_id, memory_id, "share", clearance_level
        )
        delete_decision = await self.check_memory_access(
            user_id, org_id, memory_id, "delete", clearance_level
        )
        
        # Get user's roles
        roles = await self.get_user_roles(user_id, org_id)
        
        return {
            "memory_id": memory_id,
            "user_id": user_id,
            "organization_id": org_id,
            "clearance_level": clearance_level,
            "access": {
                "read": read_decision.to_dict(),
                "write": write_decision.to_dict(),
                "share": share_decision.to_dict(),
                "delete": delete_decision.to_dict(),
            },
            "user_roles": roles,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
    
    # =========================================================================
    # Cache Invalidation
    # =========================================================================
    
    async def invalidate_user_cache(
        self,
        user_id: str,
        org_id: Optional[str] = None,
    ) -> None:
        """
        Invalidate cached permissions for a user.
        
        Call this when:
        - User's roles change
        - User is added/removed from teams
        - Shares are created/revoked
        
        Args:
            user_id: User UUID
            org_id: Optional org UUID (invalidates all orgs if not provided)
        """
        if org_id:
            cache_key = f"{self.CACHE_PREFIX_PERMISSIONS}:{user_id}:{org_id}"
            await RedisClient.delete(cache_key)
        else:
            pattern = f"{self.CACHE_PREFIX_PERMISSIONS}:{user_id}:*"
            await RedisClient.delete_pattern(pattern)
    
    async def invalidate_org_cache(self, org_id: str) -> None:
        """
        Invalidate all cached permissions for an organization.
        
        Call this when:
        - Roles are modified
        - Organization-wide policy changes
        
        Args:
            org_id: Organization UUID
        """
        pattern = f"{self.CACHE_PREFIX_PERMISSIONS}:*:{org_id}"
        await RedisClient.delete_pattern(pattern)
