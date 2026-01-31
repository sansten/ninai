"""
Resource Accounting Service

Tracks resource usage per tenant: tokens, storage, latency, request rates.
Implements admission control with graceful degradation.
"""

import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.models.memory import Memory
from app.models.audit import AuditEvent
from app.services.rate_limiter import rate_limiter, RateLimitExceeded

logger = logging.getLogger(__name__)


class ResourceAccountingService:
    """
    Service for tracking and managing resource usage.
    
    Tracks:
    - Token usage (LLM API calls)
    - Storage usage (DB + vector store)
    - Request rates
    - Latency metrics (p50, p95, p99)
    """
    
    def __init__(self, db: AsyncSession, org_id: uuid.UUID):
        self.db = db
        self.org_id = org_id
    
    async def track_request(
        self,
        endpoint: str,
        method: str,
        duration_ms: float,
        status_code: int,
        user_id: Optional[uuid.UUID] = None
    ) -> None:
        """
        Track API request metrics.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            duration_ms: Request duration in milliseconds
            status_code: HTTP status code
            user_id: User who made request (optional)
        """
        try:
            # Create audit event
            event = AuditEvent(
                id=uuid.uuid4(),
                organization_id=self.org_id,
                user_id=user_id,
                event_type="api.request",
                resource_type="api",
                resource_id=endpoint,
                success=200 <= status_code < 400,
                metadata={
                    "endpoint": endpoint,
                    "method": method,
                    "duration_ms": duration_ms,
                    "status_code": status_code,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            self.db.add(event)
            await self.db.flush()
            
            # Track rate limit
            key = f"org:{self.org_id}:api:{endpoint}"
            try:
                await rate_limiter.check_rate_limit(key, limit=1000, window=60)
            except RateLimitExceeded:
                logger.warning(f"Rate limit exceeded for {key}")
        
        except Exception as e:
            logger.error(f"Error tracking request: {e}")
    
    async def track_token_usage(
        self,
        tokens: int,
        model: str,
        operation: str
    ) -> None:
        """
        Track LLM token usage.
        
        Args:
            tokens: Number of tokens used
            model: Model name (e.g., 'gpt-4')
            operation: Operation type (e.g., 'embedding', 'completion')
        """
        try:
            # Check monthly quota
            await rate_limiter.check_quota(
                org_id=self.org_id,
                resource="tokens",
                amount=tokens,
                monthly_limit=1000000  # 1M tokens/month default
            )
            
            # Record usage event
            event = AuditEvent(
                id=uuid.uuid4(),
                organization_id=self.org_id,
                event_type="llm.token_usage",
                resource_type="tokens",
                resource_id=model,
                success=True,
                metadata={
                    "tokens": tokens,
                    "model": model,
                    "operation": operation,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            self.db.add(event)
            await self.db.flush()
        
        except RateLimitExceeded as e:
            logger.error(f"Token quota exceeded: {e}")
            raise
        except Exception as e:
            logger.error(f"Error tracking token usage: {e}")
    
    async def get_storage_usage(self) -> Dict[str, Any]:
        """
        Calculate storage usage for organization.
        
        Returns:
            Dict with storage metrics (bytes, memory count)
        """
        try:
            # Count memories
            count_stmt = select(func.count(Memory.id)).where(
                Memory.organization_id == self.org_id
            )
            result = await self.db.execute(count_stmt)
            memory_count = result.scalar_one()
            
            # Estimate storage (simplified - actual would query content length)
            # Assume average of 1KB per memory
            estimated_bytes = memory_count * 1024
            
            return {
                "memory_count": memory_count,
                "estimated_bytes": estimated_bytes,
                "estimated_mb": estimated_bytes / (1024 * 1024),
                "organization_id": str(self.org_id)
            }
        
        except Exception as e:
            logger.error(f"Error calculating storage: {e}")
            return {"memory_count": 0, "estimated_bytes": 0}
    
    async def get_request_metrics(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get request metrics for time window.
        
        Args:
            time_window_hours: Hours to look back
            
        Returns:
            Dict with request stats
        """
        try:
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Get request events
            stmt = select(AuditEvent).where(
                and_(
                    AuditEvent.organization_id == self.org_id,
                    AuditEvent.event_type == "api.request",
                    AuditEvent.created_at >= cutoff
                )
            )
            result = await self.db.execute(stmt)
            events = result.scalars().all()
            
            if not events:
                return {
                    "total_requests": 0,
                    "time_window_hours": time_window_hours
                }
            
            # Calculate metrics
            durations = [
                e.metadata.get("duration_ms", 0)
                for e in events
                if e.metadata and "duration_ms" in e.metadata
            ]
            durations.sort()
            
            def percentile(data, p):
                if not data:
                    return 0
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = int(k) + 1 if k < len(data) - 1 else f
                return data[f] + (k - f) * (data[c] - data[f])
            
            success_count = sum(1 for e in events if e.success)
            error_count = len(events) - success_count
            
            # Group by endpoint
            by_endpoint = {}
            for e in events:
                endpoint = e.metadata.get("endpoint", "unknown") if e.metadata else "unknown"
                by_endpoint[endpoint] = by_endpoint.get(endpoint, 0) + 1
            
            return {
                "total_requests": len(events),
                "success_count": success_count,
                "error_count": error_count,
                "success_rate": success_count / len(events) if events else 0,
                "latency_p50_ms": percentile(durations, 50),
                "latency_p95_ms": percentile(durations, 95),
                "latency_p99_ms": percentile(durations, 99),
                "avg_latency_ms": sum(durations) / len(durations) if durations else 0,
                "by_endpoint": by_endpoint,
                "time_window_hours": time_window_hours,
                "requests_per_hour": len(events) / time_window_hours
            }
        
        except Exception as e:
            logger.error(f"Error getting request metrics: {e}")
            return {"total_requests": 0}
    
    async def get_token_usage_stats(
        self,
        time_window_days: int = 30
    ) -> Dict[str, Any]:
        """Get token usage statistics."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=time_window_days)
            
            stmt = select(AuditEvent).where(
                and_(
                    AuditEvent.organization_id == self.org_id,
                    AuditEvent.event_type == "llm.token_usage",
                    AuditEvent.created_at >= cutoff
                )
            )
            result = await self.db.execute(stmt)
            events = result.scalars().all()
            
            total_tokens = sum(
                e.metadata.get("tokens", 0)
                for e in events
                if e.metadata
            )
            
            # Group by model
            by_model = {}
            by_operation = {}
            
            for e in events:
                if not e.metadata:
                    continue
                
                tokens = e.metadata.get("tokens", 0)
                model = e.metadata.get("model", "unknown")
                operation = e.metadata.get("operation", "unknown")
                
                by_model[model] = by_model.get(model, 0) + tokens
                by_operation[operation] = by_operation.get(operation, 0) + tokens
            
            # Get current month quota status
            quota_stats = await rate_limiter.get_usage_stats(
                org_id=self.org_id,
                resource="tokens"
            )
            
            return {
                "total_tokens": total_tokens,
                "time_window_days": time_window_days,
                "tokens_per_day": total_tokens / time_window_days if time_window_days > 0 else 0,
                "by_model": by_model,
                "by_operation": by_operation,
                "current_month_usage": quota_stats.get("used", 0),
                "current_month": quota_stats.get("month")
            }
        
        except Exception as e:
            logger.error(f"Error getting token stats: {e}")
            return {"total_tokens": 0}
    
    async def check_admission(
        self,
        operation: str,
        estimated_cost: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Check if operation should be admitted (admission control).
        
        Args:
            operation: Operation name
            estimated_cost: Estimated resource costs
                {
                    "tokens": int,
                    "storage_bytes": int,
                    "requests": int
                }
                
        Returns:
            Dict with admission decision:
            {
                "admitted": bool,
                "reason": str (if rejected),
                "degraded": bool (if should use degraded mode),
                "quotas": dict (current quota status)
            }
        """
        try:
            result = {
                "admitted": True,
                "degraded": False,
                "quotas": {}
            }
            
            estimated_cost = estimated_cost or {}
            
            # Check token quota
            if estimated_cost.get("tokens"):
                try:
                    token_status = await rate_limiter.check_quota(
                        org_id=self.org_id,
                        resource="tokens",
                        amount=estimated_cost["tokens"],
                        monthly_limit=1000000
                    )
                    result["quotas"]["tokens"] = token_status
                    
                    # Warn if approaching limit (>80%)
                    if token_status["remaining"] < token_status["limit"] * 0.2:
                        result["degraded"] = True
                
                except RateLimitExceeded as e:
                    result["admitted"] = False
                    result["reason"] = str(e)
                    return result
            
            # Check request rate
            try:
                rate_status = await rate_limiter.check_rate_limit(
                    key=f"org:{self.org_id}:api:{operation}",
                    limit=100,
                    window=60
                )
                result["quotas"]["rate_limit"] = rate_status
            
            except RateLimitExceeded as e:
                result["admitted"] = False
                result["reason"] = str(e)
                return result
            
            # Check storage (simplified - would check against org limits)
            storage = await self.get_storage_usage()
            max_storage_mb = 10000  # 10GB default
            
            if storage["estimated_mb"] > max_storage_mb * 0.9:
                result["degraded"] = True
            
            if storage["estimated_mb"] > max_storage_mb:
                result["admitted"] = False
                result["reason"] = f"Storage quota exceeded: {storage['estimated_mb']:.2f}MB / {max_storage_mb}MB"
            
            result["quotas"]["storage"] = storage
            
            return result
        
        except Exception as e:
            logger.error(f"Error checking admission: {e}")
            # Fail open - admit request
            return {"admitted": True, "degraded": False, "quotas": {}}
