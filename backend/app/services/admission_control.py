"""
Admission Control Service

Manages request admission based on system load, quotas, and circuit breaker state.
Prevents system overload by rejecting requests when necessary.
"""

import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class AdmissionControlService:
    """
    Service for admission control and load shedding.
    
    Features:
    - Request rate limiting per tenant
    - System load-based admission
    - Circuit breaker integration
    - Quota enforcement
    - Priority-based admission
    """
    
    def __init__(self, db: AsyncSession, org_id: uuid.UUID):
        self.db = db
        self.org_id = org_id
        self._load_threshold = 0.8  # 80% capacity
        self._circuit_breaker_threshold = 5  # Consecutive failures
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
    
    async def should_admit_request(
        self,
        request_type: str,
        user_id: uuid.UUID,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determine if a request should be admitted.
        
        Args:
            request_type: Type of request (e.g., 'memory_create', 'search')
            user_id: User making the request
            priority: Request priority (1-10, higher = more important)
            metadata: Additional request metadata
        
        Returns:
            Dict with admission decision and reason
        """
        try:
            # Check circuit breaker state
            circuit_check = await self._check_circuit_breaker(request_type)
            if not circuit_check["admitted"]:
                return circuit_check
            
            # Check quota limits
            quota_check = await self._check_quota(user_id, request_type)
            if not quota_check["admitted"]:
                return quota_check
            
            # Check system load
            load_check = await self._check_system_load(priority)
            if not load_check["admitted"]:
                return load_check
            
            # Check rate limits
            rate_check = await self._check_rate_limit(user_id, request_type)
            if not rate_check["admitted"]:
                return rate_check
            
            # Admit the request
            await self._record_admission(user_id, request_type, metadata)
            
            return {
                "admitted": True,
                "reason": "Request admitted",
                "priority": priority,
                "metadata": {
                    "org_id": str(self.org_id),
                    "user_id": str(user_id),
                    "request_type": request_type
                }
            }
            
        except Exception as e:
            logger.error(f"Admission control error: {e}")
            # Fail open - admit on error
            return {
                "admitted": True,
                "reason": f"Admission control error, failing open: {str(e)}",
                "priority": priority,
                "error": str(e)
            }
    
    async def _check_circuit_breaker(self, request_type: str) -> Dict[str, Any]:
        """Check if circuit breaker is open for this request type."""
        # Simplified circuit breaker check
        # In production, integrate with actual circuit breaker service
        
        circuit_key = f"circuit:{self.org_id}:{request_type}"
        # Check if circuit is open (would query from cache/db)
        is_open = False  # Placeholder
        
        if is_open:
            return {
                "admitted": False,
                "reason": "Circuit breaker is open",
                "retry_after": 60  # seconds
            }
        
        return {"admitted": True, "reason": "Circuit breaker closed"}
    
    async def _check_quota(
        self,
        user_id: uuid.UUID,
        request_type: str
    ) -> Dict[str, Any]:
        """Check if user has quota remaining."""
        # Check cached quota
        quota_key = f"{user_id}:{request_type}"
        
        if quota_key in self._quota_cache:
            quota_data = self._quota_cache[quota_key]
            if quota_data["remaining"] <= 0:
                reset_time = quota_data.get("reset_at", datetime.utcnow())
                if datetime.utcnow() < reset_time:
                    return {
                        "admitted": False,
                        "reason": "Quota exceeded",
                        "quota_remaining": 0,
                        "reset_at": reset_time.isoformat()
                    }
        
        # Default: admit (quota check passed)
        return {
            "admitted": True,
            "reason": "Quota available",
            "quota_remaining": 1000  # Placeholder
        }
    
    async def _check_system_load(self, priority: int) -> Dict[str, Any]:
        """Check system load and apply priority-based admission."""
        # Get current system load (simplified)
        current_load = 0.5  # Placeholder - would get from monitoring
        
        if current_load > self._load_threshold:
            # System is under load, only admit high priority
            if priority < 7:  # Only priority 7+ admitted
                return {
                    "admitted": False,
                    "reason": "System under load, low priority request rejected",
                    "current_load": current_load,
                    "min_priority_required": 7
                }
        
        return {
            "admitted": True,
            "reason": "System load acceptable",
            "current_load": current_load
        }
    
    async def _check_rate_limit(
        self,
        user_id: uuid.UUID,
        request_type: str
    ) -> Dict[str, Any]:
        """Check rate limit for user."""
        # Simplified rate limit check
        # In production, integrate with Redis-based rate limiter
        
        return {
            "admitted": True,
            "reason": "Rate limit not exceeded"
        }
    
    async def _record_admission(
        self,
        user_id: uuid.UUID,
        request_type: str,
        metadata: Optional[Dict[str, Any]]
    ):
        """Record that a request was admitted (for metrics/audit)."""
        # Log admission for monitoring
        logger.info(
            f"Request admitted",
            extra={
                "org_id": str(self.org_id),
                "user_id": str(user_id),
                "request_type": request_type,
                "metadata": metadata
            }
        )
    
    async def update_quota(
        self,
        user_id: uuid.UUID,
        request_type: str,
        limit: int,
        window_seconds: int = 3600
    ) -> Dict[str, Any]:
        """
        Update quota for a user/request type.
        
        Args:
            user_id: User ID
            request_type: Type of request
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            Updated quota information
        """
        quota_key = f"{user_id}:{request_type}"
        reset_at = datetime.utcnow() + timedelta(seconds=window_seconds)
        
        self._quota_cache[quota_key] = {
            "limit": limit,
            "remaining": limit,
            "reset_at": reset_at,
            "window_seconds": window_seconds
        }
        
        return {
            "user_id": str(user_id),
            "request_type": request_type,
            "limit": limit,
            "remaining": limit,
            "reset_at": reset_at.isoformat()
        }
    
    async def get_admission_stats(self) -> Dict[str, Any]:
        """Get admission control statistics."""
        return {
            "org_id": str(self.org_id),
            "active_quotas": len(self._quota_cache),
            "load_threshold": self._load_threshold,
            "circuit_breaker_threshold": self._circuit_breaker_threshold,
            "stats": {
                "total_admitted": 0,  # Placeholder
                "total_rejected": 0,  # Placeholder
                "rejection_reasons": {}  # Placeholder
            }
        }
    
    async def set_load_threshold(self, threshold: float) -> Dict[str, Any]:
        """Update the system load threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        old_threshold = self._load_threshold
        self._load_threshold = threshold
        
        return {
            "old_threshold": old_threshold,
            "new_threshold": threshold,
            "updated_at": datetime.utcnow().isoformat()
        }
    
    async def reset_circuit_breaker(self, request_type: str) -> Dict[str, Any]:
        """Manually reset circuit breaker for a request type."""
        circuit_key = f"circuit:{self.org_id}:{request_type}"
        
        # Reset circuit breaker state
        logger.info(f"Circuit breaker reset for {request_type}")
        
        return {
            "circuit_key": circuit_key,
            "status": "reset",
            "reset_at": datetime.utcnow().isoformat()
        }
