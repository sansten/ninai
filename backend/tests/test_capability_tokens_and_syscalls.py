"""Tests for capability tokens and memory syscalls."""

import pytest
from datetime import datetime, timedelta, timezone

from app.services.capability_token import (
    CapabilityToken,
    CapabilityTokenIssuer,
    MemorySyscallScope,
)
from app.services.memory_syscall_api import MemorySyscallAPI


@pytest.mark.asyncio
async def test_capability_token_issuance(test_org_id, test_user_id):
    """Test issuing tokens with appropriate scopes."""
    
    # Read token has read+search
    read_token = CapabilityTokenIssuer.issue_read_token(
        organization_id=test_org_id,
        actor_user_id=test_user_id,
    )
    assert read_token.has_scope(MemorySyscallScope.READ)
    assert read_token.has_scope(MemorySyscallScope.SEARCH)
    assert not read_token.has_scope(MemorySyscallScope.APPEND)
    
    # Write token has append+upsert+feedback
    write_token = CapabilityTokenIssuer.issue_write_token(
        organization_id=test_org_id,
        actor_user_id=test_user_id,
    )
    assert write_token.has_scope(MemorySyscallScope.APPEND)
    assert write_token.has_scope(MemorySyscallScope.UPSERT)
    assert write_token.has_scope(MemorySyscallScope.FEEDBACK)
    assert not write_token.has_scope(MemorySyscallScope.READ)
    
    # Admin token has all scopes
    admin_token = CapabilityTokenIssuer.issue_admin_token(
        organization_id=test_org_id,
        actor_user_id=test_user_id,
    )
    assert admin_token.has_scope(MemorySyscallScope.READ)
    assert admin_token.has_scope(MemorySyscallScope.APPEND)
    assert admin_token.has_scope(MemorySyscallScope.CONSOLIDATE)


@pytest.mark.asyncio
async def test_capability_token_expiration(test_org_id, test_user_id):
    """Test token expiration."""
    
    # Create expired token
    expired_token = CapabilityToken(
        organization_id=test_org_id,
        actor_user_id=test_user_id,
        scopes={MemorySyscallScope.READ},
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    
    assert expired_token.is_expired()
    
    with pytest.raises(PermissionError, match="expired"):
        expired_token.validate(MemorySyscallScope.READ)


@pytest.mark.asyncio
async def test_capability_token_scope_validation(test_org_id, test_user_id):
    """Test that tokens deny access to missing scopes."""
    
    read_token = CapabilityTokenIssuer.issue_read_token(
        organization_id=test_org_id,
        actor_user_id=test_user_id,
    )
    
    # Valid: read token has READ scope
    read_token.validate(MemorySyscallScope.READ)
    
    # Invalid: read token lacks APPEND scope
    with pytest.raises(PermissionError, match="missing scope"):
        read_token.validate(MemorySyscallScope.APPEND)


@pytest.mark.asyncio
async def test_memory_syscall_read_requires_scope(db_session, test_org_id, test_user_id):
    """Test that read operation requires read capability."""
    
    api = MemorySyscallAPI(db_session)
    
    # Write token cannot read
    write_token = CapabilityTokenIssuer.issue_write_token(
        organization_id=test_org_id,
        actor_user_id=test_user_id,
    )
    
    with pytest.raises(PermissionError, match="missing scope"):
        await api.read(memory_id="mem1", token=write_token)


@pytest.mark.asyncio
async def test_memory_syscall_append_requires_scope(db_session, test_org_id, test_user_id):
    """Test that append operation requires append capability."""
    
    api = MemorySyscallAPI(db_session)
    
    # Read token cannot append
    read_token = CapabilityTokenIssuer.issue_read_token(
        organization_id=test_org_id,
        actor_user_id=test_user_id,
    )
    
    with pytest.raises(PermissionError, match="missing scope"):
        await api.append(memory_id="mem1", content={"text": "hello"}, token=read_token)


@pytest.mark.asyncio
async def test_memory_syscall_search_requires_scope(db_session, test_org_id, test_user_id):
    """Test that search operation requires search capability."""
    
    api = MemorySyscallAPI(db_session)
    
    # Write token can search (includes search scope)
    write_token = CapabilityTokenIssuer.issue_write_token(
        organization_id=test_org_id,
        actor_user_id=test_user_id,
    )
    
    # Should fail because write_token lacks SEARCH (has append/upsert/feedback)
    with pytest.raises(PermissionError, match="missing scope"):
        await api.search(query="test", token=write_token)


@pytest.mark.asyncio
async def test_memory_syscall_success_with_valid_token(db_session, test_org_id, test_user_id):
    """Test that operations succeed with valid tokens."""
    
    api = MemorySyscallAPI(db_session)
    
    read_token = CapabilityTokenIssuer.issue_read_token(
        organization_id=test_org_id,
        actor_user_id=test_user_id,
    )
    
    # Should succeed
    result = await api.read(memory_id="mem1", token=read_token)
    assert result["id"] == "mem1"


@pytest.mark.asyncio
async def test_agent_token_scoped_access(test_org_id, test_user_id):
    """Test that agent tokens have limited scopes."""
    
    agent_token = CapabilityTokenIssuer.issue_agent_token(
        organization_id=test_org_id,
        session_id="sess1",
        agent_id="agent1",
        actor_user_id=test_user_id,
    )
    
    # Agents get search+append by default
    assert agent_token.has_scope(MemorySyscallScope.SEARCH)
    assert agent_token.has_scope(MemorySyscallScope.APPEND)
    
    # But not consolidate (admin only)
    assert not agent_token.has_scope(MemorySyscallScope.CONSOLIDATE)


@pytest.mark.asyncio
async def test_token_serialization(test_org_id, test_user_id):
    """Test token serialization for audit logging."""
    
    token = CapabilityTokenIssuer.issue_read_token(
        organization_id=test_org_id,
        session_id="sess1",
        actor_user_id=test_user_id,
    )
    
    token_dict = token.to_dict()
    assert token_dict["organization_id"] == test_org_id
    assert token_dict["session_id"] == "sess1"
    assert "memory.read" in token_dict["scopes"]
    assert "memory.search" in token_dict["scopes"]
    assert not token_dict["is_expired"]
