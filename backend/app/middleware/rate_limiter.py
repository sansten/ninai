"""Rate limiting for admin endpoints."""

from functools import wraps
from typing import Callable
import time
import asyncio

from fastapi import HTTPException, Request, status
from redis import Redis

from app.core.config import settings


class RateLimiter:
    """Redis-backed rate limiter for admin endpoints."""

    def __init__(self, redis_client: Redis | None = None):
        self.redis = redis_client
        # In production we prefer Redis. In tests we enable an in-memory limiter
        # so integration tests can validate 429 behavior without requiring Redis.
        self._memory: dict[str, tuple[int, float]] | None = None
        self._memory_lock: asyncio.Lock | None = None

        if redis_client is not None:
            self.enabled = True
        elif settings.APP_ENV == "test":
            self.enabled = True
            self._memory = {}
            self._memory_lock = asyncio.Lock()
        else:
            self.enabled = False

    def limit(
        self,
        key_prefix: str,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> Callable:
        """
        Rate limit decorator for FastAPI endpoints.

        Args:
            key_prefix: Prefix for rate limit key (e.g., "admin_ops")
            max_requests: Max requests allowed in window
            window_seconds: Time window in seconds

        Raises:
            HTTPException: 429 if rate limit exceeded
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.enabled:
                    return await func(*args, **kwargs)

                # Extract request from kwargs
                request: Request | None = kwargs.get("request")
                if not request:
                    # Try to find Request in args
                    for arg in args:
                        if isinstance(arg, Request):
                            request = arg
                            break

                if not request:
                    # Many endpoints don't accept Request as a parameter.
                    # Fall back to using tenant/user identity when present.
                    tenant = kwargs.get("tenant")
                    if tenant is None:
                        for arg in args:
                            if hasattr(arg, "user_id") and hasattr(arg, "org_id"):
                                tenant = arg
                                break

                    subject = getattr(tenant, "user_id", None) or "anonymous"
                    endpoint_id = key_prefix
                    rate_key = f"ratelimit:{key_prefix}:{subject}:{endpoint_id}"
                else:
                    # Build rate limit key from IP + endpoint
                    client_ip = request.client.host if request.client else "unknown"
                    endpoint = request.url.path
                    rate_key = f"ratelimit:{key_prefix}:{client_ip}:{endpoint}"


                # Check rate limit
                try:
                    if self.redis is not None:
                        current = self.redis.incr(rate_key)
                        if current == 1:
                            self.redis.expire(rate_key, window_seconds)

                        if current > max_requests:
                            ttl = None
                            try:
                                ttl_val = self.redis.ttl(rate_key)
                                if isinstance(ttl_val, int) and ttl_val > 0:
                                    ttl = ttl_val
                            except Exception:
                                ttl = None

                            headers = {"Retry-After": str(ttl)} if ttl is not None else None
                            raise HTTPException(
                                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail=f"Rate limit exceeded. Max {max_requests} requests per {window_seconds}s.",
                                headers=headers,
                            )

                    # In-memory fallback (tests)
                    elif self._memory is not None and self._memory_lock is not None:
                        now = time.monotonic()
                        async with self._memory_lock:
                            count, expires_at = self._memory.get(rate_key, (0, now + window_seconds))
                            if now >= expires_at:
                                count = 0
                                expires_at = now + window_seconds
                            count += 1
                            self._memory[rate_key] = (count, expires_at)

                        if count > max_requests:
                            retry_after = max(0, int(expires_at - now))
                            raise HTTPException(
                                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail=f"Rate limit exceeded. Max {max_requests} requests per {window_seconds}s.",
                                headers={"Retry-After": str(retry_after)},
                            )

                except HTTPException:
                    # Do not swallow intentional 429s.
                    raise
                except Exception:
                    # Fail open on backend errors.
                    pass

                return await func(*args, **kwargs)

            return wrapper

        return decorator


# Global rate limiter instance (initialized in main.py with Redis client)
rate_limiter = RateLimiter()


def init_rate_limiter(redis_client: Redis) -> None:
    """Initialize global rate limiter with Redis client."""
    global rate_limiter
    rate_limiter = RateLimiter(redis_client)
