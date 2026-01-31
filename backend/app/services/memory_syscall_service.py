"""
Memory Syscall Service - Capability-Scoped Memory Operations

Implements core memory operations (read/append/search/upsert/consolidate)
with capability token validation and audit logging.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import logging

from app.core.database import get_db
from app.models.capability_token import CapabilityToken, CapabilityScope
from app.models.knowledge import Knowledge
from app.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class CapabilityDeniedException(Exception):
    """Raised when token lacks required capability."""
    pass


class QuotaExceededException(Exception):
    """Raised when token quota is exhausted."""
    pass


class TokenExpiredException(Exception):
    """Raised when token is expired or revoked."""
    pass


class MemorySyscall:
    """
    Core memory syscall operations with capability validation.
    
    Each operation checks:
    1. Token validity (not expired, not revoked)
    2. Capability scope (read, append, search, etc.)
    3. Quota limits (tokens, storage, rate)
    4. Organization isolation (RLS context)
    5. Audit logging
    """

    def __init__(self, db: AsyncSession, organization_id: uuid.UUID | str):
        self.db = db
        # Most models store UUIDs as strings (SQLA UUID with as_uuid=False).
        # Normalize to string to avoid UUID-vs-str mismatches in queries.
        self.organization_id = str(organization_id)

    async def _validate_token(
        self,
        token_str: str,
        required_scope: str | CapabilityScope,
        user_id: Optional[uuid.UUID | str] = None,
    ) -> CapabilityToken:
        """
        Validate token and check capability.
        
        Raises:
            TokenExpiredException: Token is revoked or expired
            CapabilityDeniedException: Token lacks required scope
            QuotaExceededException: Quota exceeded
        """
        required_scope_value = (
            required_scope.value if isinstance(required_scope, CapabilityScope) else required_scope
        )

        async def _safe_audit_denied(
            *,
            event_type: str,
            token: Optional[CapabilityToken],
            error_message: str,
            details: Optional[dict] = None,
        ) -> None:
            try:
                audit_svc = AuditService(self.db)
                await audit_svc.log_event(
                    event_type=event_type,
                    actor_id=str(user_id) if user_id else None,
                    organization_id=str(self.organization_id),
                    resource_type="capability_token",
                    resource_id=str(token.id) if token is not None else None,
                    success=False,
                    error_message=error_message,
                    details=details or {},
                    severity="warning",
                )
            except Exception:
                # Audit logging must not break syscall paths.
                pass

        # Fetch token
        stmt = select(CapabilityToken).where(
            CapabilityToken.token == token_str,
            CapabilityToken.organization_id == self.organization_id
        )
        result = await self.db.execute(stmt)
        token = result.scalar_one_or_none()

        if not token:
            await _safe_audit_denied(
                event_type="memory.token_invalid",
                token=None,
                error_message="Token not found or invalid",
                details={"required_scope": required_scope_value},
            )
            raise TokenExpiredException("Token not found or invalid")

        # Check validity
        if token.revoked_at is not None:
            await _safe_audit_denied(
                event_type="memory.token_revoked",
                token=token,
                error_message="Token revoked",
                details={
                    "required_scope": required_scope_value,
                    "revocation_reason": token.revocation_reason,
                },
            )
            raise TokenExpiredException(f"Token revoked: {token.revocation_reason}")

        if token.expires_at is not None and datetime.utcnow() > token.expires_at:
            await _safe_audit_denied(
                event_type="memory.token_expired",
                token=token,
                error_message="Token expired",
                details={"required_scope": required_scope_value},
            )
            raise TokenExpiredException("Token expired")

        # Check scope
        if not token.has_scope(required_scope_value):
            await _safe_audit_denied(
                event_type="memory.capability_denied",
                token=token,
                error_message="Token lacks required scope",
                details={
                    "required_scope": required_scope_value,
                    "allowed_scopes": token.scopes,
                },
            )
            raise CapabilityDeniedException(
                f"Token lacks scope '{required_scope_value}'. Allowed: {token.scopes}"
            )

        # Check quota (simple in-DB counters; production would likely use Redis for rate limiting)
        if token.max_tokens_per_month and token.tokens_used >= token.max_tokens_per_month:
            await _safe_audit_denied(
                event_type="memory.quota_exceeded",
                token=token,
                error_message="Monthly token quota exceeded",
                details={
                    "required_scope": required_scope_value,
                    "tokens_used": token.tokens_used,
                    "tokens_quota": token.max_tokens_per_month,
                },
            )
            raise QuotaExceededException("Token quota exceeded for this month")

        now = datetime.utcnow()
        if token.last_request_at is None or (now - token.last_request_at).total_seconds() >= 60:
            token.requests_this_minute = 0
            token.last_request_at = now

        if token.max_requests_per_minute and token.requests_this_minute >= token.max_requests_per_minute:
            await _safe_audit_denied(
                event_type="memory.rate_limited",
                token=token,
                error_message="Rate limit exceeded",
                details={
                    "required_scope": required_scope_value,
                    "requests_this_minute": token.requests_this_minute,
                    "requests_per_minute_quota": token.max_requests_per_minute,
                },
            )
            raise QuotaExceededException("Rate limit exceeded")

        return token

    async def read(
        self,
        token_str: str,
        knowledge_id: uuid.UUID,
        user_id: Optional[uuid.UUID | str] = None,
        test_user_id: Optional[uuid.UUID | str] = None,
    ) -> Dict[str, Any]:
        """
        Read a knowledge item (vector + metadata).
        
        Requires: read capability
        Audit: logged as READ operation
        """
        if user_id is None and test_user_id is not None:
            user_id = test_user_id

        # Validate token
        token = await self._validate_token(token_str, CapabilityScope.READ, user_id=user_id)

        # Fetch knowledge (RLS enforced at query level)
        stmt = select(Knowledge).where(
            Knowledge.id == knowledge_id,
            Knowledge.organization_id == self.organization_id
        )
        result = await self.db.execute(stmt)
        knowledge = result.scalar_one_or_none()

        if not knowledge:
            raise ValueError("Knowledge not found")

        # Audit log
        audit_svc = AuditService(self.db)
        await audit_svc.log_event(
            event_type="memory.read",
            actor_id=str(user_id) if user_id else None,
            organization_id=str(self.organization_id),
            resource_type="knowledge",
            resource_id=str(knowledge_id),
            success=True,
            details={"token_id": str(token.id)}
        )

        # Update token metrics
        await self._update_token_usage(token, tokens_used=1, storage_bytes=len(knowledge.content or ""))

        return {
            "id": str(knowledge.id),
            "content": knowledge.content,
            "embedding": knowledge.embedding,
            "metadata": knowledge.knowledge_metadata,
            "created_at": knowledge.created_at.isoformat(),
            "organization_id": str(knowledge.organization_id)
        }

    async def append(
        self,
        token_str: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[uuid.UUID | str] = None,
        test_user_id: Optional[uuid.UUID | str] = None,
    ) -> Dict[str, Any]:
        """
        Append new knowledge item.
        
        Requires: append capability
        Audit: logged as APPEND operation
        """
        if user_id is None and test_user_id is not None:
            user_id = test_user_id

        # Validate token
        token = await self._validate_token(token_str, CapabilityScope.APPEND, user_id=user_id)

        # Check storage quota
        content_size = len(content or "")
        if token.max_storage_bytes and (token.storage_used_bytes + content_size) > token.max_storage_bytes:
            raise QuotaExceededException("Storage quota would be exceeded")

        # Create knowledge
        new_knowledge = Knowledge(
            id=uuid.uuid4(),
            organization_id=self.organization_id,
            content=content,
            embedding=embedding,
            knowledge_metadata=metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by_user_id=str(user_id) if user_id else None,
            is_published=False,
            version=1
        )
        self.db.add(new_knowledge)
        await self.db.flush()

        # Audit log
        audit_svc = AuditService(self.db)
        await audit_svc.log_event(
            event_type="memory.append",
            actor_id=str(user_id) if user_id else None,
            organization_id=str(self.organization_id),
            resource_type="knowledge",
            resource_id=str(new_knowledge.id),
            success=True,
            details={
                "token_id": str(token.id),
                "content_length": content_size
            }
        )

        # Update token metrics
        await self._update_token_usage(token, tokens_used=1, storage_bytes=content_size)

        return {
            "id": str(new_knowledge.id),
            "content": new_knowledge.content,
            "created_at": new_knowledge.created_at.isoformat()
        }

    async def search(
        self,
        token_str: str,
        query_embedding: List[float],
        limit: int = 10,
        user_id: Optional[uuid.UUID | str] = None,
        test_user_id: Optional[uuid.UUID | str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge via vector similarity + RLS verification.
        
        Requires: search capability
        Audit: logged as SEARCH operation
        Process: vector search in Qdrant + RLS re-verification in Postgres
        """
        if user_id is None and test_user_id is not None:
            user_id = test_user_id

        # Validate token
        token = await self._validate_token(token_str, CapabilityScope.SEARCH, user_id=user_id)

        # Vector search via Qdrant
        # (Assumes Qdrant service is available)
        from app.services.knowledge_service import KnowledgeService
        knowledge_svc = KnowledgeService(self.db, self.organization_id)
        results = await knowledge_svc.search_by_embedding(
            embedding=query_embedding,
            limit=limit
        )

        # RLS re-verification: ensure all results belong to org
        verified_results = []
        for result in results:
            stmt = select(Knowledge).where(
                Knowledge.id == result["id"],
                Knowledge.organization_id == self.organization_id
            )
            check = await self.db.execute(stmt)
            if check.scalar_one_or_none():
                verified_results.append(result)

        # Audit log
        audit_svc = AuditService(self.db)
        await audit_svc.log_event(
            event_type="memory.search",
            actor_id=str(user_id) if user_id else None,
            organization_id=str(self.organization_id),
            resource_type="knowledge",
            success=True,
            details={
                "token_id": str(token.id),
                "results_count": len(verified_results)
            }
        )

        # Update token metrics
        await self._update_token_usage(token, tokens_used=1)

        return verified_results

    async def upsert(
        self,
        token_str: str,
        knowledge_id: uuid.UUID,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[uuid.UUID | str] = None,
        test_user_id: Optional[uuid.UUID | str] = None,
    ) -> Dict[str, Any]:
        """
        Update or insert knowledge item.
        
        Requires: upsert capability
        Audit: logged as UPSERT operation
        """
        if user_id is None and test_user_id is not None:
            user_id = test_user_id

        # Validate token
        token = await self._validate_token(token_str, CapabilityScope.UPSERT, user_id=user_id)

        # Fetch existing knowledge
        stmt = select(Knowledge).where(
            Knowledge.id == knowledge_id,
            Knowledge.organization_id == self.organization_id
        )
        result = await self.db.execute(stmt)
        knowledge = result.scalar_one_or_none()

        if knowledge is None:
            # Create new
            return await self.append(
                token_str=token_str,
                content=content or "",
                embedding=embedding,
                metadata=metadata,
                user_id=user_id
            )
        else:
            # Update existing
            if content is not None:
                knowledge.content = content
            if embedding is not None:
                knowledge.embedding = embedding
            if metadata is not None:
                knowledge.knowledge_metadata = metadata

            knowledge.updated_at = datetime.utcnow()
            knowledge.version = (knowledge.version or 1) + 1

            await self.db.flush()

            # Audit log
            audit_svc = AuditService(self.db)
            await audit_svc.log_event(
                event_type="memory.upsert",
                actor_id=str(user_id) if user_id else None,
                organization_id=str(self.organization_id),
                resource_type="knowledge",
                resource_id=str(knowledge.id),
                success=True,
                details={
                    "token_id": str(token.id),
                    "new_version": knowledge.version
                }
            )

            # Update token metrics
            await self._update_token_usage(token, tokens_used=1)

            return {
                "id": str(knowledge.id),
                "version": knowledge.version,
                "updated_at": knowledge.updated_at.isoformat()
            }

    async def consolidate(
        self,
        token_str: str,
        knowledge_ids: List[uuid.UUID],
        merged_content: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[uuid.UUID | str] = None,
        test_user_id: Optional[uuid.UUID | str] = None,
    ) -> Dict[str, Any]:
        """
        Consolidate (merge/deduplicate) multiple knowledge items into one.
        
        Requires: consolidate capability
        Audit: logged as CONSOLIDATE operation
        Process:
          1. Validate all source items belong to org
          2. Create merged knowledge item
          3. Mark sources as consolidated
          4. Record audit trail
        """
        if user_id is None and test_user_id is not None:
            user_id = test_user_id

        # Validate token
        token = await self._validate_token(token_str, CapabilityScope.CONSOLIDATE, user_id=user_id)

        # Verify all source knowledge items exist in org
        stmt = select(Knowledge).where(
            Knowledge.id.in_(knowledge_ids),
            Knowledge.organization_id == self.organization_id
        )
        result = await self.db.execute(stmt)
        sources = result.scalars().all()

        if len(sources) != len(knowledge_ids):
            raise ValueError("Not all knowledge items found in this organization")

        # Create merged knowledge
        merged_id = uuid.uuid4()
        merged = Knowledge(
            id=merged_id,
            organization_id=self.organization_id,
            content=merged_content,
            knowledge_metadata=metadata or {
                "consolidated_from": [str(kid) for kid in knowledge_ids],
                "consolidated_at": datetime.utcnow().isoformat()
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by_user_id=str(user_id) if user_id else None,
            is_published=False,
            version=1
        )
        self.db.add(merged)

        # Mark sources as consolidated
        for source in sources:
            source.is_consolidated = True
            source.consolidated_into_id = merged_id
            source.updated_at = datetime.utcnow()

        await self.db.flush()

        # Audit log
        audit_svc = AuditService(self.db)
        await audit_svc.log_event(
            event_type="memory.consolidate",
            actor_id=str(user_id) if user_id else None,
            organization_id=str(self.organization_id),
            resource_type="knowledge",
            resource_id=str(merged_id),
            success=True,
            details={
                "token_id": str(token.id),
                "source_count": len(sources),
                "source_ids": [str(k.id) for k in sources]
            }
        )

        # Update token metrics
        await self._update_token_usage(token, tokens_used=len(sources))

        return {
            "merged_id": str(merged_id),
            "source_count": len(sources),
            "created_at": merged.created_at.isoformat(),
            "metadata": merged.knowledge_metadata,
        }

    async def _update_token_usage(
        self,
        token: CapabilityToken,
        tokens_used: int = 0,
        storage_bytes: int = 0
    ) -> None:
        """Update token usage metrics."""
        now = datetime.utcnow()
        if token.last_request_at is None or (now - token.last_request_at).total_seconds() >= 60:
            token.requests_this_minute = 0
        token.requests_this_minute += 1
        token.last_request_at = now
        token.tokens_used += tokens_used
        token.storage_used_bytes += storage_bytes
        token.last_used_at = now
        await self.db.flush()
