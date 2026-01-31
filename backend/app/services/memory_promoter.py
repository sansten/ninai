"""
Memory Promoter Service
=======================

Handles the automatic and manual promotion of short-term memories
to long-term storage. Implements the hybrid memory architecture
where memories start in Redis and are promoted to PostgreSQL + Qdrant
based on access patterns and importance.

Promotion Criteria:
1. Access count exceeds threshold (frequently accessed)
2. Importance score above threshold (contains important keywords)
3. Manual promotion by user/system
4. Memory survives near full TTL (persistent relevance)
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.qdrant import QdrantService
from app.models.memory import MemoryMetadata
from app.models.memory_promotion_history import MemoryPromotionHistory
from app.services.short_term_memory import ShortTermMemory, ShortTermMemoryService
from app.services.audit_service import AuditService
from app.services.simulation_service import SimulationService


class MemoryPromoter:
    """
    Service for promoting short-term memories to long-term storage.

    The promotion process:
    1. Retrieve memory from Redis
    2. Generate embedding (if not already done)
    3. Store metadata in PostgreSQL
    4. Store vector in Qdrant
    5. Delete from Redis (optional - can keep for cache)
    6. Log the promotion event
    """

    def __init__(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
    ):
        self.session = session
        self.user_id = user_id
        self.org_id = org_id
        self.stm_service = ShortTermMemoryService(user_id, org_id)
        self.audit_service = AuditService(session)
        self.simulation_service = SimulationService()

    async def summarize_and_promote_all(self) -> Optional[MemoryMetadata]:
        """
        Summarize all promotion-eligible short-term memories and promote the summary as a long-term memory.
        Returns the created summary MemoryMetadata, or None if no candidates.
        """
        from app.services.summarization_service import summarize_short_term_memories
        candidates = await self.stm_service.get_promotion_candidates()
        if not candidates:
            return None
        # Gather contents for summarization
        contents = [m.content for m in candidates]
        summary = await summarize_short_term_memories(contents)
        if not summary:
            return None
        # Create a synthetic STM for the summary
        from app.services.short_term_memory import ShortTermMemory
        summary_stm = ShortTermMemory(
            id="summary-" + str(uuid4()),
            organization_id=self.org_id,
            owner_id=self.user_id,
            content=summary,
            title="Summary of recent short-term memories",
            scope="personal",
            tags=["summary"],
            entities={},
            metadata={"summarized_from": [m.id for m in candidates]},
            importance_score=1.0,
            promotion_eligible=True,
        )
        # Promote the summary as a long-term memory
        return await self.promote_memory(summary_stm, promotion_reason="llm_summary")
    
    async def promote_memory(
        self,
        short_term_memory: ShortTermMemory,
        embedding: Optional[List[float]] = None,
        keep_in_cache: bool = False,
        promotion_reason: str = "auto",
    ) -> MemoryMetadata:
        """
        Promote a short-term memory to long-term storage.
        
        Args:
            short_term_memory: The short-term memory to promote
            embedding: Pre-computed embedding (generates placeholder if None)
            keep_in_cache: Whether to keep in Redis as cache after promotion
            promotion_reason: Reason for promotion (auto, manual, access_count, importance)
        
        Returns:
            Created MemoryMetadata (long-term memory)
        """
        stm = short_term_memory
        
        # Simulate promotion decision for risk assessment
        simulation_report = self.simulation_service.simulate_memory_promotion(
            memory_content=stm.content,
            access_count=stm.access_count,
            importance_score=stm.importance_score,
            memory_scope=stm.scope,
            tags=stm.tags,
        )
        
        # Log simulation results (don't block on low confidence unless it's a hard "no")
        # This is fail-open: we warn but still promote unless simulation explicitly rejects
        if not simulation_report["should_promote"]:
            # Log warning but proceed (fail-open behavior)
            await self.audit_service.log_memory_operation(
                actor_id=self.user_id,
                organization_id=self.org_id,
                memory_id=stm.id,
                operation="promote_simulation_warning",
                success=True,
                details={
                    "simulation_report": simulation_report,
                    "decision": "promoted_despite_warning",
                    "reason": "fail_open_policy",
                },
            )
        
        # Generate IDs
        memory_id = str(uuid4())
        vector_id = str(uuid4())
        
        # Compute content hash
        content_hash = hashlib.sha256(stm.content.encode("utf-8")).hexdigest()
        
        # Use provided embedding or placeholder
        if embedding is None:
            embedding = [0.0] * settings.EMBEDDING_DIMENSIONS
        
        # Create long-term memory record
        memory = MemoryMetadata(
            id=memory_id,
            organization_id=self.org_id,
            owner_id=self.user_id,
            scope=stm.scope,
            scope_id=None,
            memory_type="long_term",  # Promoted memories are always long-term
            classification="internal",
            required_clearance=0,
            title=stm.title,
            content_preview=stm.content[:500],
            content_hash=content_hash,
            tags=stm.tags,
            entities=stm.entities,
            extra_metadata={
                **stm.metadata,
                "promoted_from": "short_term",
                "original_stm_id": stm.id,
                "promotion_reason": promotion_reason,
                "access_count_at_promotion": stm.access_count,
                "importance_score": stm.importance_score,
            },
            source_type="promotion",
            source_id=stm.id,
            vector_id=vector_id,
            embedding_model=settings.EMBEDDING_MODEL,
            is_promoted=True,
        )
        
        # Save to PostgreSQL
        self.session.add(memory)
        await self.session.flush()

        # Record promotion history (required for observability/provenance).
        # Idempotency is enforced by a unique index on (organization_id, from_stm_id).
        history = MemoryPromotionHistory(
            organization_id=self.org_id,
            from_stm_id=stm.id,
            to_memory_id=memory_id,
            from_type="short_term",
            to_type="long_term",
            promotion_reason=promotion_reason,
            actor_id=self.user_id or None,
            trace_id=None,
            details={
                "source": "memory_promoter",
                "access_count": stm.access_count,
                "importance_score": stm.importance_score,
                "keep_in_cache": bool(keep_in_cache),
                "simulation_report": simulation_report,  # Store simulation audit trail
            },
        )
        self.session.add(history)
        await self.session.flush()
        
        # Save to Qdrant
        await QdrantService.upsert_memory(
            memory_id=vector_id,
            org_id=self.org_id,
            vector=embedding,
            payload={
                "memory_id": memory_id,
                "scope": stm.scope,
                "scope_id": None,
                    "team_id": stm.scope_id if str(stm.scope) == "team" else None,
                "owner_id": self.user_id,
                "tags": stm.tags,
                "classification": "internal",
                "memory_type": "long_term",
                "promoted": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        
        # Log the promotion
        await self.audit_service.log_memory_operation(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            operation="promote",
            success=True,
            details={
                "from_stm_id": stm.id,
                "reason": promotion_reason,
                "access_count": stm.access_count,
                "importance_score": stm.importance_score,
            },
        )
        
        # Remove from Redis unless keeping as cache
        if not keep_in_cache:
            await self.stm_service.delete(stm.id)
        
        return memory
    
    async def promote_by_id(
        self,
        stm_id: str,
        embedding: Optional[List[float]] = None,
        reason: str = "manual",
    ) -> Optional[MemoryMetadata]:
        """
        Promote a specific short-term memory by its ID.
        
        Args:
            stm_id: Short-term memory ID
            embedding: Optional pre-computed embedding
            reason: Promotion reason
        
        Returns:
            Created MemoryMetadata or None if STM not found
        """
        stm = await self.stm_service.get(stm_id)
        if not stm:
            return None
        
        return await self.promote_memory(stm, embedding, promotion_reason=reason)
    
    async def promote_all_eligible(self) -> List[MemoryMetadata]:
        """
        Promote all promotion-eligible short-term memories.
        First, summarize and promote the summary as a long-term memory (if any).
        Then, promote each eligible memory individually as before.
        
        Returns:
            List of promoted MemoryMetadata objects (summary first if created)
        """
        promoted = []
        # Summarize and promote summary first
        summary = await self.summarize_and_promote_all()
        if summary:
            promoted.append(summary)
        # Then promote each eligible memory individually
        candidates = await self.stm_service.get_promotion_candidates()
        for stm in candidates:
            # Determine promotion reason
            if stm.access_count >= ShortTermMemoryService.ACCESS_COUNT_THRESHOLD:
                reason = "access_count"
            elif stm.importance_score >= ShortTermMemoryService.IMPORTANCE_SCORE_THRESHOLD:
                reason = "importance"
            else:
                reason = "auto"
            
            memory = await self.promote_memory(stm, promotion_reason=reason)
            promoted.append(memory)
        
        return promoted
    
    async def check_and_promote(
        self,
        stm: ShortTermMemory,
    ) -> Optional[MemoryMetadata]:
        """
        Check if a memory should be promoted and do so if eligible.
        
        This method is called on every memory access to implement
        automatic promotion based on access patterns.
        
        Args:
            stm: Short-term memory to check
        
        Returns:
            MemoryMetadata if promoted, None otherwise
        """
        if not stm.promotion_eligible:
            return None
        
        # Determine reason
        if stm.access_count >= ShortTermMemoryService.ACCESS_COUNT_THRESHOLD:
            reason = "access_count"
        elif stm.importance_score >= ShortTermMemoryService.IMPORTANCE_SCORE_THRESHOLD:
            reason = "importance"
        else:
            return None  # Not actually eligible
        
        # Promote with cache kept for fast subsequent access
        return await self.promote_memory(stm, keep_in_cache=True, promotion_reason=reason)


class PromotionScheduler:
    """
    Background task for periodic promotion of eligible memories.
    
    In production, this would be run by a task scheduler like
    Celery or APScheduler. For now, provides manual invocation.
    """
    
    @staticmethod
    async def run_promotion_cycle(
        session: AsyncSession,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Run a full promotion cycle for an organization.
        
        This scans all short-term memories in Redis and promotes
        those that meet the criteria.
        
        Args:
            session: Database session
            org_id: Organization ID to process
        
        Returns:
            Statistics about the promotion cycle
        """
        from app.core.redis import RedisClient
        import json
        
        client = await RedisClient.get_client()
        
        # Find all short-term memories for this org
        promoted_count = 0
        checked_count = 0
        errors = []
        
        # Scan for all short-term memory keys
        async for key in client.scan_iter(match="stm:*"):
            try:
                data = await client.get(key)
                if not data:
                    continue
                
                memory_data = json.loads(data)
                if memory_data.get("organization_id") != org_id:
                    continue
                
                checked_count += 1
                stm = ShortTermMemory.from_dict(memory_data)
                
                if stm.promotion_eligible:
                    promoter = MemoryPromoter(
                        session=session,
                        user_id=stm.owner_id,
                        org_id=org_id,
                    )
                    await promoter.promote_memory(stm, promotion_reason="scheduled")
                    promoted_count += 1
                    
            except Exception as e:
                errors.append({"key": key, "error": str(e)})
        
        await session.commit()
        
        return {
            "organization_id": org_id,
            "memories_checked": checked_count,
            "memories_promoted": promoted_count,
            "errors": errors,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
