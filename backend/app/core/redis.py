"""
Redis Cache Client
==================

Redis connection and caching utilities for permission caching
and other application-wide caching needs.
"""

from typing import Optional, Any
import json

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from app.core.config import settings


class RedisClient:
    """
    Async Redis client wrapper with convenience methods.
    
    Provides type-safe caching operations with automatic JSON
    serialization/deserialization.
    """
    
    _pool: Optional[ConnectionPool] = None
    _client: Optional[redis.Redis] = None
    
    @classmethod
    async def get_client(cls) -> redis.Redis:
        """
        Get or create Redis client.
        
        Uses connection pooling for efficient connection reuse.
        
        Returns:
            redis.Redis: Async Redis client
        """
        if cls._client is None:
            cls._pool = ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=20,
                decode_responses=True,
            )
            cls._client = redis.Redis(connection_pool=cls._pool)
        return cls._client
    
    @classmethod
    async def close(cls) -> None:
        """Close Redis connections gracefully."""
        if cls._client:
            await cls._client.close()
            cls._client = None
        if cls._pool:
            await cls._pool.disconnect()
            cls._pool = None
    
    @classmethod
    async def get(cls, key: str) -> Optional[str]:
        """
        Get a value from Redis.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        client = await cls.get_client()
        return await client.get(key)
    
    @classmethod
    async def get_json(cls, key: str) -> Optional[Any]:
        """
        Get a JSON value from Redis.
        
        Args:
            key: Cache key
        
        Returns:
            Deserialized JSON value or None
        """
        value = await cls.get(key)
        if value:
            return json.loads(value)
        return None
    
    @classmethod
    async def set(
        cls,
        key: str,
        value: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set a value in Redis.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        
        Returns:
            bool: True if successful
        """
        client = await cls.get_client()
        if ttl:
            return await client.setex(key, ttl, value)
        return await client.set(key, value)
    
    @classmethod
    async def set_json(
        cls,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set a JSON value in Redis.
        
        Args:
            key: Cache key
            value: Value to serialize and cache
            ttl: Time-to-live in seconds
        
        Returns:
            bool: True if successful
        """
        return await cls.set(key, json.dumps(value), ttl)
    
    @classmethod
    async def delete(cls, key: str) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: Cache key to delete
        
        Returns:
            bool: True if key was deleted
        """
        client = await cls.get_client()
        return await client.delete(key) > 0
    
    @classmethod
    async def delete_pattern(cls, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "user:*:permissions")
        
        Returns:
            int: Number of keys deleted
        """
        client = await cls.get_client()
        keys = []
        async for key in client.scan_iter(pattern):
            keys.append(key)
        
        if keys:
            return await client.delete(*keys)
        return 0
    
    @classmethod
    async def exists(cls, key: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: Cache key
        
        Returns:
            bool: True if key exists
        """
        client = await cls.get_client()
        return await client.exists(key) > 0


# Convenience function for dependency injection
async def get_redis() -> redis.Redis:
    """Redis dependency for FastAPI."""
    return await RedisClient.get_client()
