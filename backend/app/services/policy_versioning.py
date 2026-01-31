"""
Policy Versioning Service

Version management for RBAC rules with history tracking and rollback capability.
Supports canary rollouts and A/B testing.
"""

import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PolicyVersion(BaseModel):
    """Policy version model."""
    id: uuid.UUID
    name: str
    version: int
    policy_type: str  # "rbac", "capability", "rate_limit"
    content: Dict[str, Any]
    status: str  # "draft", "active", "canary", "retired"
    rollout_percentage: int  # 0-100 for canary
    created_at: datetime
    created_by: uuid.UUID
    activated_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class PolicyVersioningService:
    """Service for managing policy versions."""
    
    def __init__(self, db: AsyncSession, org_id: uuid.UUID):
        self.db = db
        self.org_id = org_id
    
    async def create_policy_version(
        self,
        name: str,
        policy_type: str,
        content: Dict[str, Any],
        created_by: uuid.UUID,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new policy version.
        
        Args:
            name: Policy name
            policy_type: Type (rbac, capability, rate_limit)
            content: Policy content/rules
            created_by: User creating the policy
            metadata: Additional metadata
            
        Returns:
            Created policy version
        """
        try:
            # Get latest version number
            latest_version = await self._get_latest_version(name, policy_type)
            new_version = (latest_version or 0) + 1
            
            # Create policy record
            # In production, this would use a PolicyVersion model
            # For now, store in metadata table or similar
            
            policy = {
                "id": str(uuid.uuid4()),
                "name": name,
                "version": new_version,
                "policy_type": policy_type,
                "content": content,
                "status": "draft",
                "rollout_percentage": 0,
                "created_at": datetime.utcnow().isoformat(),
                "created_by": str(created_by),
                "metadata": metadata or {}
            }
            
            # Store in audit log or dedicated table
            from app.models.audit import AuditEvent
            
            event = AuditEvent(
                id=uuid.uuid4(),
                organization_id=self.org_id,
                user_id=created_by,
                event_type="policy.version.created",
                resource_type="policy_version",
                resource_id=policy["id"],
                success=True,
                metadata=policy
            )
            
            self.db.add(event)
            await self.db.commit()
            
            logger.info(f"Created policy version: {name} v{new_version}")
            return policy
        
        except Exception as e:
            logger.error(f"Error creating policy version: {e}")
            await self.db.rollback()
            raise
    
    async def _get_latest_version(self, name: str, policy_type: str) -> Optional[int]:
        """Get latest version number for policy."""
        from app.models.audit import AuditEvent
        
        stmt = select(AuditEvent).where(
            and_(
                AuditEvent.organization_id == self.org_id,
                AuditEvent.event_type == "policy.version.created",
                AuditEvent.metadata["name"].astext == name,
                AuditEvent.metadata["policy_type"].astext == policy_type
            )
        ).order_by(desc(AuditEvent.created_at))
        
        result = await self.db.execute(stmt)
        events = result.scalars().all()
        
        if not events:
            return None
        
        versions = [
            e.metadata.get("version", 0)
            for e in events
            if e.metadata and "version" in e.metadata
        ]
        
        return max(versions) if versions else None
    
    async def activate_policy(
        self,
        policy_id: str,
        rollout_percentage: int = 100,
        activated_by: uuid.UUID = None
    ) -> bool:
        """
        Activate a policy version.
        
        Args:
            policy_id: Policy version ID
            rollout_percentage: Percentage to roll out (0-100)
            activated_by: User activating the policy
            
        Returns:
            True if activated successfully
        """
        try:
            from app.models.audit import AuditEvent
            
            # Get policy
            stmt = select(AuditEvent).where(
                and_(
                    AuditEvent.organization_id == self.org_id,
                    AuditEvent.event_type == "policy.version.created",
                    AuditEvent.resource_id == policy_id
                )
            )
            result = await self.db.execute(stmt)
            event = result.scalar_one_or_none()
            
            if not event or not event.metadata:
                logger.error(f"Policy {policy_id} not found")
                return False
            
            # Update status
            policy = event.metadata
            
            # Create activation event
            activation_event = AuditEvent(
                id=uuid.uuid4(),
                organization_id=self.org_id,
                user_id=activated_by,
                event_type="policy.version.activated",
                resource_type="policy_version",
                resource_id=policy_id,
                success=True,
                metadata={
                    "policy_name": policy.get("name"),
                    "version": policy.get("version"),
                    "rollout_percentage": rollout_percentage,
                    "previous_status": policy.get("status"),
                    "new_status": "canary" if rollout_percentage < 100 else "active",
                    "activated_at": datetime.utcnow().isoformat()
                }
            )
            
            self.db.add(activation_event)
            await self.db.commit()
            
            logger.info(
                f"Activated policy {policy.get('name')} v{policy.get('version')} "
                f"with {rollout_percentage}% rollout"
            )
            return True
        
        except Exception as e:
            logger.error(f"Error activating policy: {e}")
            await self.db.rollback()
            return False
    
    async def rollback_policy(
        self,
        name: str,
        policy_type: str,
        target_version: int,
        rolled_back_by: uuid.UUID
    ) -> bool:
        """
        Rollback to a previous policy version.
        
        Args:
            name: Policy name
            policy_type: Policy type
            target_version: Version to rollback to
            rolled_back_by: User performing rollback
            
        Returns:
            True if rollback successful
        """
        try:
            from app.models.audit import AuditEvent
            
            # Find target version
            stmt = select(AuditEvent).where(
                and_(
                    AuditEvent.organization_id == self.org_id,
                    AuditEvent.event_type == "policy.version.created",
                    AuditEvent.metadata["name"].astext == name,
                    AuditEvent.metadata["policy_type"].astext == policy_type,
                    AuditEvent.metadata["version"].astext == str(target_version)
                )
            )
            result = await self.db.execute(stmt)
            target_event = result.scalar_one_or_none()
            
            if not target_event or not target_event.metadata:
                logger.error(f"Target version {target_version} not found")
                return False
            
            # Create rollback event
            rollback_event = AuditEvent(
                id=uuid.uuid4(),
                organization_id=self.org_id,
                user_id=rolled_back_by,
                event_type="policy.version.rollback",
                resource_type="policy_version",
                resource_id=target_event.resource_id,
                success=True,
                metadata={
                    "policy_name": name,
                    "policy_type": policy_type,
                    "target_version": target_version,
                    "rolled_back_at": datetime.utcnow().isoformat(),
                    "reason": "Manual rollback"
                }
            )
            
            self.db.add(rollback_event)
            await self.db.commit()
            
            # Re-activate the target version
            await self.activate_policy(
                policy_id=target_event.resource_id,
                rollout_percentage=100,
                activated_by=rolled_back_by
            )
            
            logger.info(f"Rolled back {name} to version {target_version}")
            return True
        
        except Exception as e:
            logger.error(f"Error rolling back policy: {e}")
            await self.db.rollback()
            return False
    
    async def get_policy_history(
        self,
        name: str,
        policy_type: str
    ) -> List[Dict[str, Any]]:
        """Get version history for a policy."""
        try:
            from app.models.audit import AuditEvent
            
            stmt = select(AuditEvent).where(
                and_(
                    AuditEvent.organization_id == self.org_id,
                    AuditEvent.event_type.in_([
                        "policy.version.created",
                        "policy.version.activated",
                        "policy.version.rollback"
                    ]),
                    AuditEvent.metadata["name"].astext == name,
                    AuditEvent.metadata["policy_type"].astext == policy_type
                )
            ).order_by(desc(AuditEvent.created_at))
            
            result = await self.db.execute(stmt)
            events = result.scalars().all()
            
            return [
                {
                    "event_type": e.event_type,
                    "created_at": e.created_at.isoformat(),
                    "user_id": str(e.user_id) if e.user_id else None,
                    **e.metadata
                }
                for e in events
                if e.metadata
            ]
        
        except Exception as e:
            logger.error(f"Error getting policy history: {e}")
            return []
    
    async def compare_versions(
        self,
        policy_id_1: str,
        policy_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two policy versions.
        
        Returns diff information.
        """
        try:
            from app.models.audit import AuditEvent
            
            # Get both policies
            stmt = select(AuditEvent).where(
                and_(
                    AuditEvent.organization_id == self.org_id,
                    AuditEvent.event_type == "policy.version.created",
                    AuditEvent.resource_id.in_([policy_id_1, policy_id_2])
                )
            )
            result = await self.db.execute(stmt)
            events = result.scalars().all()
            
            if len(events) != 2:
                return {"error": "Could not find both policies"}
            
            policy1 = events[0].metadata if events[0].metadata else {}
            policy2 = events[1].metadata if events[1].metadata else {}
            
            # Simple diff (in production, use proper diff library)
            return {
                "policy_1": {
                    "id": policy_id_1,
                    "version": policy1.get("version"),
                    "created_at": policy1.get("created_at")
                },
                "policy_2": {
                    "id": policy_id_2,
                    "version": policy2.get("version"),
                    "created_at": policy2.get("created_at")
                },
                "content_changed": policy1.get("content") != policy2.get("content")
            }
        
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return {"error": str(e)}
