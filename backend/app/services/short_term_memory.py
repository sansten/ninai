"""
Short-Term Memory Service
=========================

Redis-based storage for short-term memories with automatic TTL,
access tracking, and promotion eligibility scoring.

Short-term memories are stored in Redis for fast access and
automatically expire after a configurable period. Frequently
accessed or important memories are flagged for promotion to
long-term storage (PostgreSQL + Qdrant).
"""

import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4
from dataclasses import dataclass, asdict

from app.core.redis import RedisClient
from app.core.config import settings


# Redis key prefixes
SHORT_TERM_PREFIX = "stm:"  # Short-term memory
STM_INDEX_PREFIX = "stm_idx:"  # Index for user's memories
STM_ACCESS_PREFIX = "stm_access:"  # Access count tracking


@dataclass
class ShortTermMemory:
    """Short-term memory data structure."""
    id: str
    organization_id: str
    owner_id: str
    content: str
    title: Optional[str] = None
    scope: str = "personal"
    tags: List[str] = None
    entities: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    created_at: str = None
    access_count: int = 0
    importance_score: float = 0.0
    promotion_eligible: bool = False
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.entities is None:
            self.entities = {}
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShortTermMemory":
        return cls(**data)


class ShortTermMemoryService:
    """
    Service for managing short-term memories in Redis.
    
    Features:
    - Fast in-memory storage with TTL
    - Access count tracking
    - Importance scoring for promotion decisions
    - Automatic promotion eligibility detection
    """
    
    # Default TTL for short-term memories (from config, default 7 days)
    DEFAULT_TTL = settings.SHORT_TERM_TTL
    
    # Promotion thresholds
    ACCESS_COUNT_THRESHOLD = 3  # Accessed 3+ times = promotion eligible
    IMPORTANCE_SCORE_THRESHOLD = 0.7  # High importance = promotion eligible
    
    # Keywords that indicate importance (boost importance score)
    IMPORTANCE_KEYWORDS = [
        "order", "purchase", "payment", "confirmation",
        "contract", "agreement", "important", "critical",
        "preference", "setting", "configuration",
        "complaint", "issue", "resolved", "escalation"
    ]
    
    def __init__(self, user_id: str, org_id: str):
        self.user_id = user_id
        self.org_id = org_id
    
    async def store(
        self,
        content: str,
        title: Optional[str] = None,
        scope: str = "personal",
        tags: Optional[List[str]] = None,
        entities: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        ttl: Optional[int] = None,
    ) -> ShortTermMemory:
        """
        Store a new short-term memory in Redis.
        
        Args:
            content: Memory content
            title: Optional title
            scope: Memory scope (personal, team, etc.)
            tags: Optional tags for categorization
            entities: Optional extracted entities
            metadata: Optional additional metadata
            ttl: Time-to-live in seconds (default: 1 hour)
        
        Returns:
            Created ShortTermMemory
        """
        memory_id = str(uuid4())
        
        # Calculate initial importance score
        importance_score = self._calculate_importance(content, title, tags)
        
        memory = ShortTermMemory(
            id=memory_id,
            organization_id=self.org_id,
            owner_id=self.user_id,
            content=content,
            title=title,
            scope=scope,
            tags=tags or [],
            entities=entities or {},
            metadata=metadata or {},
            importance_score=importance_score,
            promotion_eligible=importance_score >= self.IMPORTANCE_SCORE_THRESHOLD,
        )
        
        # Store in Redis with TTL
        client = await RedisClient.get_client()
        key = f"{SHORT_TERM_PREFIX}{memory_id}"
        effective_ttl = ttl or self.DEFAULT_TTL
        
        await client.setex(key, effective_ttl, json.dumps(memory.to_dict()))
        
        # Add to user's memory index
        index_key = f"{STM_INDEX_PREFIX}{self.user_id}"
        await client.sadd(index_key, memory_id)
        await client.expire(index_key, effective_ttl * 2)  # Index lives longer
        
        # Initialize access count
        access_key = f"{STM_ACCESS_PREFIX}{memory_id}"
        await client.setex(access_key, effective_ttl, "0")
        
        return memory
    
    async def get(self, memory_id: str) -> Optional[ShortTermMemory]:
        """
        Retrieve a short-term memory and increment access count.
        
        Args:
            memory_id: Memory UUID
        
        Returns:
            ShortTermMemory or None if not found/expired
        """
        client = await RedisClient.get_client()
        key = f"{SHORT_TERM_PREFIX}{memory_id}"
        
        data = await client.get(key)
        if not data:
            return None
        
        memory = ShortTermMemory.from_dict(json.loads(data))
        
        # Increment access count
        access_key = f"{STM_ACCESS_PREFIX}{memory_id}"
        access_count = await client.incr(access_key)
        memory.access_count = int(access_count)
        
        # Check if now eligible for promotion
        if access_count >= self.ACCESS_COUNT_THRESHOLD:
            memory.promotion_eligible = True
            # Update the stored memory with new eligibility
            await client.set(key, json.dumps(memory.to_dict()), keepttl=True)
        
        return memory
    
    async def list_user_memories(self) -> List[ShortTermMemory]:
        """
        List all short-term memories for the current user.
        
        Returns:
            List of ShortTermMemory objects
        """
        client = await RedisClient.get_client()
        index_key = f"{STM_INDEX_PREFIX}{self.user_id}"
        
        memory_ids = await client.smembers(index_key)
        if not memory_ids:
            return []
        
        memories = []
        for memory_id in memory_ids:
            key = f"{SHORT_TERM_PREFIX}{memory_id}"
            data = await client.get(key)
            if data:
                memory = ShortTermMemory.from_dict(json.loads(data))
                # Get current access count
                access_key = f"{STM_ACCESS_PREFIX}{memory_id}"
                access_count = await client.get(access_key)
                memory.access_count = int(access_count) if access_count else 0
                memories.append(memory)
            else:
                # Memory expired, remove from index
                await client.srem(index_key, memory_id)
        
        return memories
    
    async def get_promotion_candidates(self) -> List[ShortTermMemory]:
        """
        Get all memories eligible for promotion to long-term storage.
        
        Returns:
            List of promotion-eligible ShortTermMemory objects
        """
        memories = await self.list_user_memories()
        return [m for m in memories if m.promotion_eligible]
    
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a short-term memory.
        
        Args:
            memory_id: Memory UUID
        
        Returns:
            True if deleted, False if not found
        """
        client = await RedisClient.get_client()
        key = f"{SHORT_TERM_PREFIX}{memory_id}"
        access_key = f"{STM_ACCESS_PREFIX}{memory_id}"
        index_key = f"{STM_INDEX_PREFIX}{self.user_id}"
        
        deleted = await client.delete(key)
        await client.delete(access_key)
        await client.srem(index_key, memory_id)
        
        return deleted > 0
    
    async def extend_ttl(self, memory_id: str, additional_seconds: int) -> bool:
        """
        Extend the TTL of a short-term memory.
        
        Args:
            memory_id: Memory UUID
            additional_seconds: Seconds to add to current TTL
        
        Returns:
            True if extended, False if not found
        """
        client = await RedisClient.get_client()
        key = f"{SHORT_TERM_PREFIX}{memory_id}"
        
        current_ttl = await client.ttl(key)
        if current_ttl < 0:
            return False
        
        new_ttl = current_ttl + additional_seconds
        await client.expire(key, new_ttl)
        
        # Also extend access count key
        access_key = f"{STM_ACCESS_PREFIX}{memory_id}"
        await client.expire(access_key, new_ttl)
        
        return True
    
    def _calculate_importance(
        self,
        content: str,
        title: Optional[str],
        tags: Optional[List[str]],
    ) -> float:
        """
        Calculate importance score based on content analysis.
        
        Uses keyword matching and heuristics. In production,
        this would use an LLM for better classification.
        
        Args:
            content: Memory content
            title: Optional title
            tags: Optional tags
        
        Returns:
            Importance score between 0.0 and 1.0
        """
        score = 0.0
        text = f"{content} {title or ''} {' '.join(tags or [])}".lower()
        
        # Check for importance keywords
        keyword_matches = sum(1 for kw in self.IMPORTANCE_KEYWORDS if kw in text)
        if keyword_matches > 0:
            score += min(keyword_matches * 0.15, 0.6)
        
        # Longer content tends to be more important
        content_length = len(content)
        if content_length > 500:
            score += 0.1
        if content_length > 1000:
            score += 0.1
        
        # Having a title indicates more structured/important content
        if title:
            score += 0.1
        
        # More tags = more categorization = likely more important
        if tags and len(tags) >= 3:
            score += 0.1
        
        return min(score, 1.0)


class ShortTermMemoryStats:
    """Utility class for getting short-term memory statistics."""
    
    @staticmethod
    async def get_stats() -> Dict[str, Any]:
        """Get overall short-term memory statistics."""
        client = await RedisClient.get_client()
        
        # Count all short-term memory keys
        stm_keys = []
        async for key in client.scan_iter(match=f"{SHORT_TERM_PREFIX}*"):
            stm_keys.append(key)
        
        # Count promotion eligible
        promotion_count = 0
        for key in stm_keys:
            data = await client.get(key)
            if data:
                memory = json.loads(data)
                if memory.get("promotion_eligible"):
                    promotion_count += 1
        
        return {
            "total_short_term_memories": len(stm_keys),
            "promotion_eligible_count": promotion_count,
            "storage": "redis",
        }
