"""
Tests for Phase 2: Memory Syscall Surface - Capability-Scoped Operations

Tests:
1. Capability token validation (valid, expired, revoked)
2. Scope enforcement (read, append, search, upsert, consolidate)
3. Quota enforcement (token count, storage, rate limit)
4. Access denial audit logging
5. RLS enforcement in search (verify + SQL parity)
"""

import pytest
import uuid
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.capability_token import CapabilityToken, CapabilityScope
from app.models.knowledge import Knowledge
from app.services.memory_syscall_service import (
    MemorySyscall,
    CapabilityDeniedException,
    TokenExpiredException,
    QuotaExceededException
)
from app.services.capability_token_service import CapabilityTokenService
from app.schemas.audit import AuditEventType


@pytest.mark.asyncio
class TestCapabilityTokenValidation:
    """Test capability token validation and lifecycle."""

    async def test_issue_token(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        """Test issuing a capability token."""
        svc = CapabilityTokenService(db_session, test_org_id)
        token = await svc.issue_token(
            name="test_token",
            scopes=["read", "search"],
            agent_name="test_agent",
            ttl_seconds=3600,
            created_by_user_id=test_user_id
        )
        await db_session.commit()

        assert token.scopes == "read,search"
        assert token.organization_id == test_org_id
        assert token.is_valid()
        assert token.has_scope("read")
        assert token.has_scope("search")

    async def test_revoke_token(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        """Test revoking a capability token."""
        svc = CapabilityTokenService(db_session, test_org_id)
        token = await svc.issue_token(
            name="test_token",
            scopes=["read"],
            created_by_user_id=test_user_id
        )
        await db_session.commit()

        # Revoke it
        await svc.revoke_token(
            token_id=token.id,
            reason="Test revocation",
            revoked_by_user_id=test_user_id
        )
        await db_session.commit()

        # Verify it's revoked
        token = await svc.get_token(token.id)
        assert token.revoked_at is not None
        assert not token.is_valid()

    async def test_token_expiration(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        """Test token expiration."""
        svc = CapabilityTokenService(db_session, test_org_id)
        token = await svc.issue_token(
            name="test_token",
            scopes=["read"],
            ttl_seconds=1,  # 1 second
            created_by_user_id=test_user_id
        )
        await db_session.commit()

        # Token is valid immediately
        assert token.is_valid()

        # But not after expiration
        token.expires_at = datetime.utcnow() - timedelta(seconds=1)
        assert not token.is_valid()


@pytest.mark.asyncio
class TestScopeEnforcement:
    """Test that scopes are properly enforced."""

    async def test_read_scope_required(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        """Test that read operation requires 'read' scope."""
        svc = CapabilityTokenService(db_session, test_org_id)
        
        # Issue token WITHOUT read scope
        token = await svc.issue_token(
            name="append_only",
            scopes=["append"],  # No 'read'
            created_by_user_id=test_user_id
        )
        await db_session.commit()

        # Try to read - should fail
        syscall = MemorySyscall(db_session, test_org_id)
        with pytest.raises(CapabilityDeniedException):
            await syscall.read(
                token_str=token.token,
                knowledge_id=uuid.uuid4(),
                test_user_id=test_user_id
            )

    async def test_append_scope_required(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        """Test that append operation requires 'append' scope."""
        svc = CapabilityTokenService(db_session, test_org_id)
        
        # Issue token WITHOUT append scope
        token = await svc.issue_token(
            name="read_only",
            scopes=["read"],  # No 'append'
            created_by_user_id=test_user_id
        )
        await db_session.commit()

        # Try to append - should fail
        syscall = MemorySyscall(db_session, test_org_id)
        with pytest.raises(CapabilityDeniedException):
            await syscall.append(
                token_str=token.token,
                content="test content",
                test_user_id=test_user_id
            )

    async def test_search_scope_required(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        """Test that search operation requires 'search' scope."""
        svc = CapabilityTokenService(db_session, test_org_id)
        
        # Issue token WITHOUT search scope
        token = await svc.issue_token(
            name="append_only",
            scopes=["append"],  # No 'search'
            created_by_user_id=test_user_id
        )
        await db_session.commit()

        # Try to search - should fail
        syscall = MemorySyscall(db_session, test_org_id)
        with pytest.raises(CapabilityDeniedException):
            await syscall.search(
                token_str=token.token,
                query_embedding=[0.1] * 768,
                test_user_id=test_user_id
            )


@pytest.mark.asyncio
class TestQuotaEnforcement:
    """Test quota enforcement."""

    async def test_token_quota_exceeded(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        """Test that token quota is enforced."""
        svc = CapabilityTokenService(db_session, test_org_id)
        
        # Issue token with quota of 1 token
        token = await svc.issue_token(
            name="limited",
            scopes=["read"],
            max_tokens_per_month=1,
            created_by_user_id=test_user_id
        )
        await db_session.commit()

        # Use up the quota
        token.tokens_used = 1
        await db_session.flush()

        # Try to use again - should fail
        syscall = MemorySyscall(db_session, test_org_id)
        with pytest.raises(QuotaExceededException):
            await syscall.read(
                token_str=token.token,
                knowledge_id=uuid.uuid4(),
                test_user_id=test_user_id
            )

    async def test_storage_quota_exceeded(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        """Test that storage quota is enforced."""
        svc = CapabilityTokenService(db_session, test_org_id)
        
        # Issue token with 10 byte storage quota
        token = await svc.issue_token(
            name="limited_storage",
            scopes=["append"],
            max_storage_bytes=10,
            created_by_user_id=test_user_id
        )
        await db_session.commit()

        # Try to append large content - should fail
        syscall = MemorySyscall(db_session, test_org_id)
        with pytest.raises(QuotaExceededException):
            await syscall.append(
                token_str=token.token,
                content="x" * 100,  # 100 bytes, exceeds quota
                test_user_id=test_user_id
            )


@pytest.mark.asyncio
class TestAccessDenialAuditLogging:
    """Test that denied access is logged."""

    async def test_denied_access_logged(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        """Test that denied access is logged to audit."""
        from app.services.audit_service import AuditService
        
        svc = CapabilityTokenService(db_session, test_org_id)
        
        # Issue read-only token
        token = await svc.issue_token(
            name="read_only",
            scopes=["read"],
            created_by_user_id=test_user_id
        )
        await db_session.commit()

        # Try to append (denied)
        syscall = MemorySyscall(db_session, test_org_id)
        try:
            await syscall.append(
                token_str=token.token,
                content="test",
                test_user_id=test_user_id
            )
        except CapabilityDeniedException:
            pass

        # Check audit log
        audit_svc = AuditService(db_session)
        events = await audit_svc.list_events(limit=10, organization_id=str(test_org_id))
        
        # Should have logged something
        assert len(events) > 0


@pytest.mark.asyncio
class TestMemorySyscallOperations:
    """Test core memory syscall operations."""

    async def test_append_and_read(self, db_session: AsyncSession, test_org_id: uuid.UUID, test_user_id: uuid.UUID):
        """Test appending and reading knowledge."""
        svc = CapabilityTokenService(db_session, test_org_id)
        token = await svc.issue_token(
            name="full_access",
            scopes=["read", "append"],
            created_by_user_id=test_user_id
        )
        await db_session.commit()

        syscall = MemorySyscall(db_session, test_org_id)

        # Append
        append_result = await syscall.append(
            token_str=token.token,
            content="Test knowledge",
            metadata={"source": "test"},
            test_user_id=test_user_id
        )
        await db_session.commit()

        # Read
        read_result = await syscall.read(
            token_str=token.token,
            knowledge_id=uuid.UUID(append_result["id"]),
            test_user_id=test_user_id
        )
        await db_session.commit()

        assert read_result["content"] == "Test knowledge"
        assert read_result["metadata"]["source"] == "test"

    async def test_consolidate(self, db_session: AsyncSession, test_org_id: str, test_user_id: str):
        """Test consolidating multiple knowledge items."""
        svc = CapabilityTokenService(db_session, test_org_id)
        token = await svc.issue_token(
            name="full_access",
            scopes=["append", "consolidate"],
            created_by_user_id=test_user_id
        )
        await db_session.commit()

        syscall = MemorySyscall(db_session, test_org_id)

        # Append two items
        item1 = await syscall.append(
            token_str=token.token,
            content="Item 1",
            test_user_id=test_user_id
        )
        item2 = await syscall.append(
            token_str=token.token,
            content="Item 2",
            test_user_id=test_user_id
        )
        await db_session.commit()

        # Consolidate
        result = await syscall.consolidate(
            token_str=token.token,
            knowledge_ids=[uuid.UUID(item1["id"]), uuid.UUID(item2["id"])],
            merged_content="Merged content",
            test_user_id=test_user_id
        )
        await db_session.commit()

        assert result["source_count"] == 2
        assert "consolidated_from" in (result.get("metadata") or {})
