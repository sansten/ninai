"""
Rate Limiting Service

Redis-based rate limiting with per-tenant quotas and backpressure.
Implements sliding window rate limiting and quota enforcement.
"""

import time
from typing import Optional, Dict, Any
import logging
import uuid
from datetime import datetime, timedelta

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


class RateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm.
    
    Features:
    - Per-tenant rate limits
    - Per-endpoint rate limits
    - Sliding window (more accurate than fixed window)
    - Graceful degradation when Redis unavailable
    - Backpressure signals
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_limit: int = 100,
        default_window: int = 60
    ):
        """
        Initialize rate limiter.
        
        Args:
            redis_url: Redis connection URL
            default_limit: Default requests per window
            default_window: Default window in seconds
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, rate limiting disabled")
            self.redis = None
        else:
            try:
                self.redis = aioredis.from_url(
                    redis_url,
                    decode_responses=False
                )
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis = None
        
        self.default_limit = default_limit
        self.default_window = default_window
    
    async def check_rate_limit(
        self,
        key: str,
        limit: Optional[int] = None,
        window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check if request is within rate limit.
        
        Args:
            key: Rate limit key (e.g., "org:123:api:memories")
            limit: Max requests per window (uses default if None)
            window: Window size in seconds (uses default if None)
            
        Returns:
            Dict with rate limit status:
            {
                "allowed": bool,
                "limit": int,
                "remaining": int,
                "reset_at": float (timestamp),
                "retry_after": int (seconds until can retry)
            }
            
        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        if not self.redis:
            # Graceful degradation - allow all requests if Redis unavailable
            return {
                "allowed": True,
                "limit": limit or self.default_limit,
                "remaining": -1,
                "reset_at": time.time() + (window or self.default_window),
                "retry_after": 0
            }
        
        limit = limit or self.default_limit
        window = window or self.default_window
        
        try:
            now = time.time()
            window_start = now - window
            
            # Use sorted set for sliding window
            redis_key = f"ratelimit:{key}"
            
            # Remove old entries
            await self.redis.zremrangebyscore(redis_key, 0, window_start)
            
            # Count requests in current window
            count = await self.redis.zcard(redis_key)
            
            if count >= limit:
                # Rate limit exceeded
                # Get oldest entry to calculate reset time
                oldest = await self.redis.zrange(redis_key, 0, 0, withscores=True)
                if oldest:
                    reset_at = oldest[0][1] + window
                    retry_after = max(0, int(reset_at - now))
                else:
                    reset_at = now + window
                    retry_after = window
                
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {key}. Limit: {limit}/{window}s. "
                    f"Retry after {retry_after}s"
                )
            
            # Add current request
            await self.redis.zadd(redis_key, {str(uuid.uuid4()): now})
            
            # Set expiration (window + buffer)
            await self.redis.expire(redis_key, window + 60)
            
            return {
                "allowed": True,
                "limit": limit,
                "remaining": limit - count - 1,
                "reset_at": now + window,
                "retry_after": 0
            }
        
        except RateLimitExceeded:
            raise
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # Fail open - allow request
            return {
                "allowed": True,
                "limit": limit,
                "remaining": -1,
                "reset_at": time.time() + window,
                "retry_after": 0
            }
    
    async def check_quota(
        self,
        org_id: uuid.UUID,
        resource: str,
        amount: int = 1,
        monthly_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check monthly quota for organization.
        
        Args:
            org_id: Organization ID
            resource: Resource type (e.g., 'tokens', 'storage', 'requests')
            amount: Amount to consume
            monthly_limit: Monthly limit (None = no limit)
            
        Returns:
            Dict with quota status
            
        Raises:
            RateLimitExceeded: If quota exceeded
        """
        if not self.redis or monthly_limit is None:
            return {
                "allowed": True,
                "used": 0,
                "limit": monthly_limit,
                "remaining": -1
            }
        
        try:
            # Use current month as key
            month_key = datetime.utcnow().strftime("%Y-%m")
            redis_key = f"quota:{org_id}:{resource}:{month_key}"
            
            # Get current usage
            current = await self.redis.get(redis_key)
            current_usage = int(current) if current else 0
            
            new_usage = current_usage + amount
            
            if new_usage > monthly_limit:
                raise RateLimitExceeded(
                    f"Monthly quota exceeded for {resource}. "
                    f"Used: {current_usage}, Limit: {monthly_limit}, Requesting: {amount}"
                )
            
            # Increment usage
            await self.redis.set(redis_key, new_usage)
            
            # Set expiration (60 days to handle month boundaries)
            await self.redis.expire(redis_key, 60 * 24 * 60 * 60)
            
            return {
                "allowed": True,
                "used": new_usage,
                "limit": monthly_limit,
                "remaining": monthly_limit - new_usage
            }
        
        except RateLimitExceeded:
            raise
        except Exception as e:
            logger.error(f"Error checking quota: {e}")
            return {
                "allowed": True,
                "used": 0,
                "limit": monthly_limit,
                "remaining": -1
            }
    
    async def get_usage_stats(
        self,
        org_id: uuid.UUID,
        resource: str
    ) -> Dict[str, Any]:
        """Get current month usage stats."""
        if not self.redis:
            return {"used": 0, "month": datetime.utcnow().strftime("%Y-%m")}
        
        try:
            month_key = datetime.utcnow().strftime("%Y-%m")
            redis_key = f"quota:{org_id}:{resource}:{month_key}"
            
            current = await self.redis.get(redis_key)
            current_usage = int(current) if current else 0
            
            return {
                "used": current_usage,
                "month": month_key,
                "resource": resource,
                "organization_id": str(org_id)
            }
        
        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return {"used": 0, "month": datetime.utcnow().strftime("%Y-%m")}
    
    async def reset_usage(
        self,
        org_id: uuid.UUID,
        resource: str,
        month: Optional[str] = None
    ) -> bool:
        """Reset usage for organization/resource (admin operation)."""
        if not self.redis:
            return False
        
        try:
            month_key = month or datetime.utcnow().strftime("%Y-%m")
            redis_key = f"quota:{org_id}:{resource}:{month_key}"
            
            await self.redis.delete(redis_key)
            
            logger.info(f"Reset usage for org {org_id}, resource {resource}, month {month_key}")
            return True
        
        except Exception as e:
            logger.error(f"Error resetting usage: {e}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()


# Global rate limiter instance
rate_limiter = RateLimiter()
