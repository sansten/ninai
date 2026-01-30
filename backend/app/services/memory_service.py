"""
Memory Service
==============

Service for memory operations with integrated permission checking,
vector storage, and audit logging.

HYBRID MEMORY ARCHITECTURE:
- By default, memories start as short-term in Redis
- Frequently accessed or important memories are auto-promoted to long-term
- Use create_memory() for direct long-term storage (explicit)
- Use create_memory_smart() for hybrid auto-classification (recommended)
"""

import hashlib
import math
from typing import Optional, List, Union
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.qdrant import QdrantService
from app.models.memory import MemoryMetadata, MemorySharing
from app.models.memory_feedback import MemoryFeedback
from app.services.permission_checker import PermissionChecker, AccessDecision
from app.services.audit_service import AuditService
from app.services.short_term_memory import ShortTermMemory, ShortTermMemoryService
from app.services.memory_promoter import MemoryPromoter
from app.schemas.memory import (
    MemoryCreate,
    MemoryUpdate,
    MemorySearchRequest,
    MemoryShareRequest,
)


class MemoryService:
    """
    Memory operations service.
    
    Handles all memory CRUD operations with:
    - Permission checking before every operation
    - Dual write to Postgres (metadata) and Qdrant (vectors)
    - Audit logging for all operations
    - RLS-verified search results
    
    SECURITY: All vector search results are re-verified against
    Postgres RLS before being returned to the user.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        clearance_level: int = 0,
    ):
        """
        Initialize memory service.
        
        Args:
            session: Database session with tenant context set
            user_id: Current user's UUID
            org_id: Current organization's UUID
            clearance_level: User's security clearance level
        """
        self.session = session
        self.user_id = user_id
        self.org_id = org_id
        self.clearance_level = clearance_level
        
        self.permission_checker = PermissionChecker(session)
        self.audit_service = AuditService(session)
    
    # =========================================================================
    # Create
    # =========================================================================
    
    async def create_memory(
        self,
        data: MemoryCreate,
        embedding: List[float],
        request_id: Optional[str] = None,
    ) -> MemoryMetadata:
        """
            # Determine TTL: query param > body.ttl > default
            effective_ttl = ttl if ttl is not None else getattr(data, 'ttl', None)
        
        Writes metadata to Postgres and embedding to Qdrant.
        
        Args:
            stm = await stm_service.store(
                content=data.content,
                title=data.title,
                scope=data.scope,
                tags=data.tags,
                entities=data.entities,
                metadata=data.extra_metadata,
                ttl=effective_ttl,
            )
            PermissionError: If user lacks permission to create
        """
        # Check permission
        permission_check = await self.permission_checker.check_permission(
            self.user_id, self.org_id, f"memory:create:{data.scope}"
        )
        
        if not permission_check.allowed:
            await self.audit_service.log_memory_operation(
                actor_id=self.user_id,
                organization_id=self.org_id,
                memory_id="",
                operation="create",
                success=False,
                error_message=permission_check.reason,
            )
            raise PermissionError(permission_check.reason)
        
        # Generate IDs
        memory_id = str(uuid4())
        vector_id = str(uuid4())
        
        # Compute content hash for deduplication
        content_hash = hashlib.sha256(
            data.content.encode("utf-8")
        ).hexdigest()
        
        # Create metadata record
        memory = MemoryMetadata(
            id=memory_id,
            organization_id=self.org_id,
            owner_id=self.user_id,
            scope=data.scope,
            scope_id=data.scope_id,
            memory_type=data.memory_type,
            classification=data.classification,
            required_clearance=data.required_clearance or 0,
            title=data.title,
            content_preview=data.content[:500],
            content_hash=content_hash,
            tags=data.tags or [],
            entities=data.entities or {},
            extra_metadata=data.extra_metadata or {},
            source_type=data.source_type,
            source_id=data.source_id,
            vector_id=vector_id,
            embedding_model=settings.EMBEDDING_MODEL or "text-embedding-3-small",
            retention_days=data.retention_days,
        )
        
        # Save to Postgres
        self.session.add(memory)
        await self.session.flush()
        
        # Save to Qdrant
        await QdrantService.upsert_memory(
            memory_id=vector_id,
            org_id=self.org_id,
            vector=embedding,
            payload={
                "memory_id": memory_id,
                "scope": data.scope,
                "scope_id": data.scope_id,
                # Denormalized for Qdrant filtering convenience
                "team_id": data.scope_id if str(data.scope) == "team" else None,
                "owner_id": self.user_id,
                "tags": data.tags or [],
                "classification": data.classification,
                "memory_type": data.memory_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        
        # Audit log
        await self.audit_service.log_memory_operation(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            operation="create",
            success=True,
            details={
                "scope": data.scope,
                "classification": data.classification,
            },
        )
        
        return memory
    
    async def create_memory_smart(
        self,
        data: MemoryCreate,
        embedding: Optional[List[float]] = None,
        request_id: Optional[str] = None,
        force_long_term: bool = False,
        ttl: Optional[int] = None,
    ) -> Union[ShortTermMemory, MemoryMetadata]:
        """
        Create a memory using the hybrid architecture.
        
        By default, memories start as short-term in Redis and are
        automatically promoted to long-term when:
        - Accessed frequently (3+ times)
        - Content is detected as important (orders, preferences, etc.)
        
        Args:
            data: Memory creation data
            embedding: Optional pre-computed embedding vector
            request_id: Request ID for audit correlation
            force_long_term: If True, skip short-term and store directly in long-term
        
        Returns:
            ShortTermMemory (in Redis) or MemoryMetadata (if force_long_term or auto-promoted)
        """
        # Check permission
        permission_check = await self.permission_checker.check_permission(
            self.user_id, self.org_id, f"memory:create:{data.scope}"
        )
        
        if not permission_check.allowed:
            await self.audit_service.log_memory_operation(
                actor_id=self.user_id,
                organization_id=self.org_id,
                memory_id="",
                operation="create_smart",
                success=False,
                error_message=permission_check.reason,
            )
            raise PermissionError(permission_check.reason)
        
        # If explicitly requesting long-term via force flag, use traditional create
        # The smart endpoint defaults to short-term unless force_long_term=True
        if force_long_term:
            if embedding is None:
                embedding = [0.0] * settings.EMBEDDING_DIMENSIONS
            return await self.create_memory(data, embedding, request_id)
        
        # Create short-term memory in Redis
        stm_service = ShortTermMemoryService(self.user_id, self.org_id)
        
        stm = await stm_service.store(
            content=data.content,
            title=data.title,
            scope=data.scope,
            tags=data.tags,
            entities=data.entities,
            metadata=data.extra_metadata,
            ttl=ttl if ttl is not None else getattr(data, "ttl", None),
        )
        
        # Log creation
        await self.audit_service.log_memory_operation(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=stm.id,
            operation="create_short_term",
            success=True,
            details={
                "scope": data.scope,
                "importance_score": stm.importance_score,
                "promotion_eligible": stm.promotion_eligible,
            },
        )
        
        # If immediately eligible for promotion, promote now
        if stm.promotion_eligible:
            promoter = MemoryPromoter(self.session, self.user_id, self.org_id)
            if embedding is None:
                embedding = [0.0] * settings.EMBEDDING_DIMENSIONS
            promoted = await promoter.promote_memory(
                stm, 
                embedding=embedding,
                keep_in_cache=True,  # Keep in Redis for fast access
                promotion_reason="importance",
            )
            return promoted
        
        return stm
    
    async def get_memory_smart(
        self,
        memory_id: str,
        request_id: Optional[str] = None,
    ) -> Optional[Union[ShortTermMemory, MemoryMetadata]]:
        """
        Get a memory from either short-term (Redis) or long-term (PostgreSQL).
        
        Checks Redis first, then PostgreSQL. Accessing short-term memories
        may trigger automatic promotion if access count threshold is met.
        
        Args:
            memory_id: Memory UUID
            request_id: Request ID for audit correlation
        
        Returns:
            ShortTermMemory, MemoryMetadata, or None if not found
        """
        # Try short-term first (faster)
        stm_service = ShortTermMemoryService(self.user_id, self.org_id)
        stm = await stm_service.get(memory_id)
        
        if stm:
            # Check if now eligible for promotion due to access count
            if stm.promotion_eligible and stm.access_count >= ShortTermMemoryService.ACCESS_COUNT_THRESHOLD:
                promoter = MemoryPromoter(self.session, self.user_id, self.org_id)
                promoted = await promoter.check_and_promote(stm)
                if promoted:
                    return promoted  # Return the long-term version
            return stm
        
        # Fall back to long-term storage
        return await self.get_memory(memory_id, request_id)
    
    async def list_all_memories(
        self,
        include_short_term: bool = True,
        request_id: Optional[str] = None,
    ) -> dict:
        """
        List all memories for the user, from both short-term and long-term storage.
        
        Args:
            include_short_term: Whether to include short-term memories from Redis
            request_id: Request ID for audit correlation
        
        Returns:
            Dict with 'short_term' and 'long_term' memory lists
        """
        from sqlalchemy import select
        
        result = {
            "short_term": [],
            "long_term": [],
        }
        
        # Get short-term memories from Redis
        if include_short_term:
            stm_service = ShortTermMemoryService(self.user_id, self.org_id)
            result["short_term"] = await stm_service.list_user_memories()
        
        # Get long-term memories from PostgreSQL
        query = (
            select(MemoryMetadata)
            .where(MemoryMetadata.organization_id == self.org_id)
            .where(MemoryMetadata.is_active == True)
            .order_by(MemoryMetadata.created_at.desc())
            .limit(100)
        )
        db_result = await self.session.execute(query)
        result["long_term"] = list(db_result.scalars().all())
        
        return result

    async def list_memories(
        self,
        scope: Optional[str] = None,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[MemoryMetadata], int, bool]:
        """
        List long-term memories with optional filters.

        Args:
            scope: Filter by scope
            tags: Filter by tags (must contain all)
            memory_type: Filter by memory_type
            page: Page number (1-indexed)
            page_size: Items per page

        Returns:
            (items, total, has_more)
        """
        from sqlalchemy import func

        base_query = (
            select(MemoryMetadata)
            .where(MemoryMetadata.organization_id == self.org_id)
            .where(MemoryMetadata.is_active == True)
        )

        if scope:
            base_query = base_query.where(MemoryMetadata.scope == scope)

        if memory_type:
            base_query = base_query.where(MemoryMetadata.memory_type == memory_type)

        if tags:
            base_query = base_query.where(MemoryMetadata.tags.contains(tags))

        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await self.session.execute(count_query)
        total = total_result.scalar() or 0

        offset = max(page - 1, 0) * page_size
        query = (
            base_query
            .order_by(MemoryMetadata.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )

        db_result = await self.session.execute(query)
        items = list(db_result.scalars().all())

        has_more = total > (offset + page_size)

        return items, total, has_more
    
    async def promote_memory(
        self,
        stm_id: str,
        embedding: Optional[List[float]] = None,
    ) -> Optional[MemoryMetadata]:
        """
        Manually promote a short-term memory to long-term storage.
        
        Args:
            stm_id: Short-term memory ID
            embedding: Optional embedding vector
        
        Returns:
            Created MemoryMetadata or None if STM not found
        """
        promoter = MemoryPromoter(self.session, self.user_id, self.org_id)
        return await promoter.promote_by_id(stm_id, embedding, reason="manual")

    # =========================================================================
    # Read
    # =========================================================================
    
    async def get_memory(
        self,
        memory_id: str,
        request_id: Optional[str] = None,
    ) -> Optional[MemoryMetadata]:
        """
        Get a memory by ID with permission checking.
        
        Args:
            memory_id: Memory UUID
            request_id: Request ID for audit correlation
        
        Returns:
            MemoryMetadata if found and authorized, None otherwise
        
        Raises:
            PermissionError: If user lacks read permission
        """
        # Check permission
        access = await self.permission_checker.check_memory_access(
            self.user_id, self.org_id, memory_id, "read", self.clearance_level
        )
        
        # Log access attempt
        await self.audit_service.log_memory_access(
            user_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            action="read",
            authorized=access.allowed,
            authorization_method=access.method,
            denial_reason=access.reason if not access.allowed else None,
            request_id=request_id,
        )
        
        if not access.allowed:
            raise PermissionError(access.reason)
        
        # Get memory (RLS will filter)
        memory = await self.session.get(MemoryMetadata, memory_id)
        
        if memory:
            # Update access tracking
            memory.access_count += 1
            memory.last_accessed_at = datetime.now(timezone.utc)
        
        return memory
    
    async def explain_access(
        self,
        memory_id: str,
    ) -> dict:
        """
        Explain why user can or cannot access a memory.
        
        Args:
            memory_id: Memory UUID
        
        Returns:
            Detailed access explanation
        """
        return await self.permission_checker.explain_access(
            self.user_id, self.org_id, memory_id, self.clearance_level
        )
    
    # =========================================================================
    # Search
    # =========================================================================
    
    async def search_memories(
        self,
        query_embedding: List[float],
        request: MemorySearchRequest,
        request_id: Optional[str] = None,
    ) -> List[MemoryMetadata]:
        """
        Search memories using vector similarity.
        
        SECURITY: This method:
        1. Searches Qdrant with org filter
        2. Fetches results from Postgres (RLS filtered)
        3. Verifies each result against permission checker
        
        Args:
            query_embedding: Query vector
            request: Search parameters
            request_id: Request ID for audit
        
        Returns:
            List of authorized MemoryMetadata results
        """
        qdrant_results = []
        lexical_scores: dict[str, float] = {}

        ranking_meta = self.get_search_ranking_meta(request)
        decay_enabled = bool(ranking_meta.get("temporal_decay_enabled"))
        half_life_days = float(ranking_meta.get("temporal_decay_half_life_days") or 0.0)

        # Vector leg (Qdrant)
        scope_val = request.scope.value if hasattr(request.scope, "value") else request.scope

        qdrant_results = await QdrantService.search(
            org_id=self.org_id,
            query_vector=query_embedding,
            limit=request.limit * 2,  # Over-fetch to account for RLS filtering
            score_threshold=request.score_threshold or 0.0,
            scope_filter=scope_val,
            team_id=request.team_id,
        )

        # Lexical leg (Postgres FTS) - opt-in via request.hybrid
        if getattr(request, "hybrid", False):
            # Full-text search using pre-computed search_vector column with GIN index.
            # Uses BM25-style ranking via ts_rank_cd with normalization.
            # 
            # Normalization flags (see PostgreSQL docs):
            # 0 = default (ignores document length)
            # 1 = divides rank by 1 + log(document length)
            # 2 = divides rank by document length
            # 4 = divides rank by mean harmonic distance between extents
            # 8 = divides rank by number of unique words
            # 16 = divides rank by 1 + log(number of unique words)
            # 32 = divides rank by rank + 1
            # 
            # We use normalization=1 (BM25-like length normalization)
            normalization = 1
            
            # Build query using plainto_tsquery for user-friendly parsing
            # Alternatively: websearch_to_tsquery for more advanced queries
            tsq = func.plainto_tsquery("simple", request.query)
            
            # Rank using ts_rank_cd (Cover Density ranking)
            # This is more sophisticated than ts_rank and similar to BM25
            # Weights: {D, C, B, A} = {0.1, 0.2, 0.4, 1.0}
            # A (title) gets highest weight, D (tags) gets lowest
            rank = func.ts_rank_cd(
                "{0.1, 0.2, 0.4, 1.0}",  # Weights for D, C, B, A
                MemoryMetadata.search_vector,
                tsq,
                normalization
            )

            stmt = (
                select(MemoryMetadata.id, rank.label("rank"))
                .where(
                    MemoryMetadata.organization_id == self.org_id,
                    MemoryMetadata.is_active.is_(True),
                    MemoryMetadata.search_vector.op("@@")(tsq),
                )
                .order_by(rank.desc())
                .limit(request.limit * 2)
            )

            if scope_val:
                stmt = stmt.where(MemoryMetadata.scope == scope_val)
            if request.team_id:
                stmt = stmt.where(MemoryMetadata.scope == "team", MemoryMetadata.scope_id == request.team_id)

            lex_res = await self.session.execute(stmt)
            for row in lex_res.all():
                memory_id = str(row[0])
                lexical_scores[memory_id] = float(row[1] or 0.0)

        # Candidate IDs from both legs
        vector_scores = {r["payload"]["memory_id"]: float(r.get("score") or 0.0) for r in qdrant_results}
        candidate_ids = list({*vector_scores.keys(), *lexical_scores.keys()})
        if not candidate_ids:
            return []

        # Fetch from Postgres (RLS will filter unauthorized)
        query = select(MemoryMetadata).where(
            and_(
                MemoryMetadata.id.in_(candidate_ids),
                MemoryMetadata.is_active.is_(True),
                # Defense-in-depth: even with RLS, constrain by org_id explicitly.
                MemoryMetadata.organization_id == self.org_id,
            )
        )

        result = await self.session.execute(query)
        memories = result.scalars().all()

        # Optional: feedback-driven reranking (closed-loop retrieval).
        # Uses most recent per-user feedback of type "relevance" for each memory.
        feedback_payloads: dict[str, dict] = {}
        if bool(ranking_meta.get("feedback_rerank_enabled")) and candidate_ids:
            window_days = float(ranking_meta.get("feedback_rerank_window_days") or 90.0)
            pos_mult = float(ranking_meta.get("feedback_rerank_positive_multiplier") or 1.15)
            neg_mult = float(ranking_meta.get("feedback_rerank_negative_multiplier") or 0.5)
            if pos_mult <= 0:
                pos_mult = 1.15
            if neg_mult <= 0:
                neg_mult = 0.5

            stmt = select(
                MemoryFeedback.memory_id.label("memory_id"),
                MemoryFeedback.payload.label("payload"),
                func.row_number()
                .over(partition_by=MemoryFeedback.memory_id, order_by=MemoryFeedback.created_at.desc())
                .label("rn"),
            ).where(
                MemoryFeedback.organization_id == self.org_id,
                MemoryFeedback.actor_id == self.user_id,
                MemoryFeedback.feedback_type == "relevance",
                MemoryFeedback.memory_id.in_(candidate_ids),
            )
            if window_days > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
                stmt = stmt.where(MemoryFeedback.created_at >= cutoff)

            subq = stmt.subquery()
            latest_stmt = select(subq.c.memory_id, subq.c.payload).where(subq.c.rn == 1)
            fb_res = await self.session.execute(latest_stmt)
            for mid, payload in fb_res.all():
                if isinstance(payload, dict):
                    feedback_payloads[str(mid)] = payload

        # Normalize scores and compute a combined score
        max_vec = max(vector_scores.values(), default=0.0)
        max_lex = max(lexical_scores.values(), default=0.0)

        vec_weight = 0.7
        lex_weight = 0.3

        # HNMS-inspired ranking mode selector.
        # Mode influences temporal decay weighting (recency bias).
        # (Computed once via get_search_ranking_meta.)
        
        # Verify each memory and collect authorized ones
        authorized_memories: list[MemoryMetadata] = []
        normalized_similarities: dict[str, float] = {}
        for memory in memories:
            access = await self.permission_checker.check_memory_access(
                self.user_id, self.org_id, memory.id, "read", self.clearance_level
            )
            
            if access.allowed:
                vec = vector_scores.get(memory.id, 0.0)
                lex = lexical_scores.get(memory.id, 0.0)

                vec_norm = (vec / max_vec) if max_vec > 0 else 0.0
                lex_norm = (lex / max_lex) if max_lex > 0 else 0.0

                normalized_similarities[str(memory.id)] = float(vec_norm)

                if getattr(request, "hybrid", False):
                    memory.score = (vec_weight * vec_norm) + (lex_weight * lex_norm)
                else:
                    memory.score = vec

                # Optional: temporal decay weighting (HNMS-inspired).
                # Uses last_accessed_at if present, else updated_at, else created_at.
                if decay_enabled and half_life_days > 0:
                    anchor = (
                        getattr(memory, "last_accessed_at", None)
                        or getattr(memory, "updated_at", None)
                        or getattr(memory, "created_at", None)
                    )
                    if anchor is not None:
                        now = datetime.now(timezone.utc)
                        if getattr(anchor, "tzinfo", None) is None:
                            anchor = anchor.replace(tzinfo=timezone.utc)
                        age_days = max(0.0, (now - anchor).total_seconds() / 86400.0)
                        decay = math.pow(0.5, age_days / half_life_days)
                        memory.score = float(memory.score or 0.0) * float(decay)

                # Optional: per-user relevance feedback reranking.
                # Expected payload shapes:
                # - {"value": 1} / {"value": -1}
                # - {"relevant": true} / {"relevant": false}
                # - {"relevance": 1.0} / {"relevance": -1.0}
                if feedback_payloads:
                    payload = feedback_payloads.get(str(memory.id))
                    if isinstance(payload, dict):
                        raw = payload.get("value")
                        if raw is None:
                            raw = payload.get("relevance")
                        if raw is None:
                            raw = payload.get("relevant")

                        v: float | None = None
                        if isinstance(raw, bool):
                            v = 1.0 if raw else -1.0
                        else:
                            try:
                                v = float(raw)  # type: ignore[arg-type]
                            except Exception:
                                v = None

                        if v is not None:
                            if v > 0:
                                memory.score = float(memory.score or 0.0) * pos_mult
                            elif v < 0:
                                memory.score = float(memory.score or 0.0) * neg_mult

                # Attach best-effort provenance for citations.
                # For now, the "source" is the memory itself (future: attachments/docs).
                # We use updated_at + content_hash as a stable version hint.
                memory.provenance = [
                    {
                        "kind": "memory",
                        "source_type": getattr(memory, "source_type", None),
                        "source_id": getattr(memory, "source_id", None),
                        "source_version": (
                            memory.updated_at.isoformat() if getattr(memory, "updated_at", None) else None
                        ),
                        "content_hash": getattr(memory, "content_hash", None),
                        "title": getattr(memory, "title", None),
                        "excerpt": getattr(memory, "content_preview", None),
                        "score": memory.score,
                        "meta": {
                            "memory_id": memory.id,
                            "vector_id": getattr(memory, "vector_id", None),
                            "created_at": (
                                memory.created_at.isoformat() if getattr(memory, "created_at", None) else None
                            ),
                            "updated_at": (
                                memory.updated_at.isoformat() if getattr(memory, "updated_at", None) else None
                            ),
                            "embedding_model": getattr(memory, "embedding_model", None),
                            "scope": getattr(memory, "scope", None),
                            "scope_id": getattr(memory, "scope_id", None),
                            "classification": getattr(memory, "classification", None),
                        },
                    }
                ]

                authorized_memories.append(memory)
                
                # Log authorized access
                await self.audit_service.log_memory_access(
                    user_id=self.user_id,
                    organization_id=self.org_id,
                    memory_id=memory.id,
                    action="search_read",
                    authorized=True,
                    authorization_method=access.method,
                    request_id=request_id,
                    access_context={"search_query": request.query},
                )
        
        # Activation scoring + explanation logging + async update tasks.
        # This keeps the synchronous request path fast (math + batched reads) and
        # pushes counter/coactivation writes into Celery.
        if authorized_memories:
            try:
                from app.services.memory_activation.retrieval import MemoryRetrievalService

                retrieval = MemoryRetrievalService(
                    session=self.session,
                    org_id=str(self.org_id),
                    user_id=str(self.user_id),
                )

                authorized_ids = [str(m.id) for m in authorized_memories]

                ranked_dicts, explanation_results = await retrieval.score_and_rank_results(
                    memory_ids=authorized_ids,
                    query=request.query,
                    similarities=normalized_similarities,
                    scope=scope_val,
                )

                activation_by_id = {str(r["id"]): float(r["activation_score"]) for r in ranked_dicts}
                ranked_ids = [str(r["id"]) for r in ranked_dicts]

                mem_by_id = {str(m.id): m for m in authorized_memories}
                authorized_memories = [mem_by_id[mid] for mid in ranked_ids if mid in mem_by_id]

                for mem in authorized_memories:
                    mem.score = activation_by_id.get(str(mem.id), 0.0)

                # Write explanation log (append-only). Commit happens at request boundary.
                explanation_id = await retrieval.write_retrieval_explanation(
                    query=request.query,
                    results=explanation_results,
                    top_k=request.limit,
                )

                # Best-effort enqueue of background updates (no-op in unit tests).
                try:
                    from app.core.celery_app import celery_app

                    broker = celery_app.conf.broker_url
                    if broker and not str(broker).startswith("memory://"):
                        from app.services.memory_activation.tasks import (
                            memory_access_update_task,
                            coactivation_update_task,
                        )

                        top_ids = [str(m.id) for m in authorized_memories[: request.limit]]
                        for mid in top_ids:
                            memory_access_update_task.apply_async(
                                kwargs={
                                    "memory_id": mid,
                                    "org_id": str(self.org_id),
                                    "user_id": str(self.user_id),
                                    "retrieval_explanation_id": explanation_id,
                                },
                                countdown=2,
                            )

                        if len(top_ids) > 1:
                            coactivation_update_task.apply_async(
                                kwargs={
                                    "primary_memory_id": top_ids[0],
                                    "coactivated_memory_ids": top_ids[1:],
                                    "org_id": str(self.org_id),
                                },
                                countdown=2,
                            )
                except Exception:
                    # Celery isn't required for serving search.
                    pass

            except Exception:
                # If activation scoring fails, fall back to the legacy combined score.
                pass

        # Sort by score and limit
        authorized_memories.sort(key=lambda m: float(m.score or 0.0), reverse=True)
        return authorized_memories[: request.limit]

    def get_search_ranking_meta(self, request: MemorySearchRequest) -> dict[str, object]:
        """Compute effective ranking parameters for a search request.

        This is used for both search scoring and API response observability.
        """

        req_mode = getattr(request, "hnms_mode", None)
        allow_override = bool(getattr(settings, "SEARCH_HNMS_MODE_ALLOW_REQUEST_OVERRIDE", True))
        mode_source = "config"
        mode = req_mode if (allow_override and req_mode) else getattr(settings, "SEARCH_HNMS_MODE_DEFAULT", "balanced")
        if not isinstance(mode, str) and hasattr(mode, "value"):
            mode = mode.value
        if isinstance(req_mode, str) or hasattr(req_mode, "value"):
            if allow_override and req_mode:
                mode_source = "request"

        mode_str = str(mode or "balanced").lower()

        if mode_str == "performance":
            temporal_decay_enabled = True
            temporal_decay_half_life_days = float(
                getattr(settings, "SEARCH_HNMS_MODE_PERFORMANCE_HALF_LIFE_DAYS", 7.0) or 7.0
            )
        elif mode_str == "research":
            temporal_decay_enabled = True
            temporal_decay_half_life_days = float(
                getattr(settings, "SEARCH_HNMS_MODE_RESEARCH_HALF_LIFE_DAYS", 90.0) or 90.0
            )
        else:
            # balanced (or unknown) falls back to the base temporal decay knobs
            temporal_decay_enabled = bool(getattr(settings, "SEARCH_TEMPORAL_DECAY_ENABLED", False))
            temporal_decay_half_life_days = float(
                getattr(settings, "SEARCH_TEMPORAL_DECAY_HALF_LIFE_DAYS", 30.0) or 30.0
            )
            mode_str = "balanced"

        feedback_rerank_enabled = bool(getattr(settings, "SEARCH_FEEDBACK_RERANK_ENABLED", False))
        feedback_rerank_window_days = float(getattr(settings, "SEARCH_FEEDBACK_RERANK_WINDOW_DAYS", 90.0) or 90.0)
        feedback_rerank_positive_multiplier = float(
            getattr(settings, "SEARCH_FEEDBACK_RERANK_POSITIVE_MULTIPLIER", 1.15) or 1.15
        )
        feedback_rerank_negative_multiplier = float(
            getattr(settings, "SEARCH_FEEDBACK_RERANK_NEGATIVE_MULTIPLIER", 0.5) or 0.5
        )

        return {
            "hnms_mode_effective": mode_str,
            "hnms_mode_source": mode_source,
            "temporal_decay_enabled": temporal_decay_enabled,
            "temporal_decay_half_life_days": temporal_decay_half_life_days,
            "feedback_rerank_enabled": feedback_rerank_enabled,
            "feedback_rerank_window_days": feedback_rerank_window_days,
            "feedback_rerank_positive_multiplier": feedback_rerank_positive_multiplier,
            "feedback_rerank_negative_multiplier": feedback_rerank_negative_multiplier,
        }
    
    # =========================================================================
    # Update
    # =========================================================================
    
    async def update_memory(
        self,
        memory_id: str,
        data: MemoryUpdate,
        new_embedding: Optional[List[float]] = None,
        request_id: Optional[str] = None,
    ) -> MemoryMetadata:
        """
        Update a memory.
        
        Args:
            memory_id: Memory UUID
            data: Update data
            new_embedding: New embedding if content changed
            request_id: Request ID for audit
        
        Returns:
            Updated MemoryMetadata
        
        Raises:
            PermissionError: If user lacks write permission
        """
        # Check permission
        access = await self.permission_checker.check_memory_access(
            self.user_id, self.org_id, memory_id, "write", self.clearance_level
        )
        
        if not access.allowed:
            await self.audit_service.log_memory_operation(
                actor_id=self.user_id,
                organization_id=self.org_id,
                memory_id=memory_id,
                operation="update",
                success=False,
                error_message=access.reason,
            )
            raise PermissionError(access.reason)
        
        # Get memory
        memory = await self.session.get(MemoryMetadata, memory_id)
        if not memory:
            raise ValueError("Memory not found")
        
        # Track changes
        changes = {}
        
        # Update fields
        if data.title is not None:
            changes["title"] = {"old": memory.title, "new": data.title}
            memory.title = data.title
        
        if data.tags is not None:
            changes["tags"] = {"old": memory.tags, "new": data.tags}
            memory.tags = data.tags
        
        if data.classification is not None:
            changes["classification"] = {"old": memory.classification, "new": data.classification}
            memory.classification = data.classification
        
        if data.metadata is not None:
            changes["metadata"] = {"old": memory.metadata, "new": data.metadata}
            memory.metadata = data.metadata
        
        # Update embedding in Qdrant if provided
        if new_embedding:
            await QdrantService.upsert_memory(
                memory_id=memory.vector_id,
                org_id=self.org_id,
                vector=new_embedding,
                payload={
                    "memory_id": memory_id,
                    "scope": memory.scope,
                    "scope_id": memory.scope_id,
                    "team_id": memory.scope_id if str(memory.scope) == "team" else None,
                    "owner_id": memory.owner_id,
                    "tags": memory.tags,
                    "classification": memory.classification,
                    "memory_type": memory.memory_type,
                    "created_at": memory.created_at.isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        
        # Audit log
        await self.audit_service.log_memory_operation(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            operation="update",
            success=True,
            details={"changes": changes},
        )
        
        return memory
    
    # =========================================================================
    # Delete
    # =========================================================================
    
    async def delete_memory(
        self,
        memory_id: str,
        request_id: Optional[str] = None,
    ) -> bool:
        """
        Soft-delete a memory.
        
        Args:
            memory_id: Memory UUID
            request_id: Request ID for audit
        
        Returns:
            True if deleted
        
        Raises:
            PermissionError: If user lacks delete permission
        """
        # Check permission
        access = await self.permission_checker.check_memory_access(
            self.user_id, self.org_id, memory_id, "delete", self.clearance_level
        )
        
        if not access.allowed:
            await self.audit_service.log_memory_operation(
                actor_id=self.user_id,
                organization_id=self.org_id,
                memory_id=memory_id,
                operation="delete",
                success=False,
                error_message=access.reason,
            )
            raise PermissionError(access.reason)
        
        # Get memory
        memory = await self.session.get(MemoryMetadata, memory_id)
        if not memory:
            raise ValueError("Memory not found")
        
        # Check legal hold
        if memory.legal_hold:
            raise PermissionError("Memory is under legal hold and cannot be deleted")
        
        # Soft delete
        memory.is_active = False
        
        # Remove from Qdrant
        await QdrantService.delete_memory(memory.vector_id, self.org_id)
        
        # Audit log
        await self.audit_service.log_memory_operation(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            operation="delete",
            success=True,
        )
        
        return True
    
    # =========================================================================
    # Sharing
    # =========================================================================
    
    async def share_memory(
        self,
        memory_id: str,
        request: MemoryShareRequest,
        request_id: Optional[str] = None,
    ) -> MemorySharing:
        """
        Share a memory with a user or group.
        
        Args:
            memory_id: Memory UUID
            request: Share request with target and permission
            request_id: Request ID for audit
        
        Returns:
            Created MemorySharing record
        
        Raises:
            PermissionError: If user lacks share permission
        """
        # Check permission
        access = await self.permission_checker.check_memory_access(
            self.user_id, self.org_id, memory_id, "share", self.clearance_level
        )
        
        if not access.allowed:
            raise PermissionError(access.reason)
        
        # Create share record
        share = MemorySharing(
            memory_id=memory_id,
            organization_id=self.org_id,
            share_type=request.share_type,
            target_id=request.target_id,
            permission=request.permission,
            expires_at=request.expires_at,
            shared_by=self.user_id,
            share_reason=request.reason,
        )
        
        self.session.add(share)
        await self.session.flush()
        
        # Invalidate target's permission cache
        if request.share_type == "user":
            await self.permission_checker.invalidate_user_cache(
                request.target_id, self.org_id
            )
        
        # Audit log
        await self.audit_service.log_memory_share(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            share_type=request.share_type,
            target_id=request.target_id,
            permission=request.permission,
            expires_at=request.expires_at,
        )
        
        return share
