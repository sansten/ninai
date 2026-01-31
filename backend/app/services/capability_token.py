"""Capability token system for memory OS syscalls.

Issues least-privilege scoped tokens for read/append/search/upsert/consolidate operations.
Tokens are bound to: tenant/org, session/context, and agent/actor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from app.models.base import utc_now


class MemorySyscallScope(str, Enum):
    """Memory syscall operation scopes."""
    READ = "memory.read"
    APPEND = "memory.append"
    SEARCH = "memory.search"
    UPSERT = "memory.upsert"
    CONSOLIDATE = "memory.consolidate"
    PROMOTE = "memory.promote"
    FEEDBACK = "memory.feedback"


class CapabilityToken:
    """Capability token for memory syscalls.
    
    Issued per tenant/session/agent with explicit scopes and TTL.
    Validates that requestor has permission for requested operation.
    """

    def __init__(
        self,
        *,
        token_id: str = "",
        organization_id: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        actor_user_id: str,
        scopes: set[MemorySyscallScope] | set[str],
        issued_at: datetime | None = None,
        expires_at: datetime | None = None,
        ttl_seconds: int = 3600,
    ):
        self.token_id = token_id or str(uuid4())
        self.organization_id = organization_id
        self.session_id = session_id
        self.agent_id = agent_id
        self.actor_user_id = actor_user_id
        self.scopes = {MemorySyscallScope(s) if isinstance(s, str) else s for s in scopes}
        self.issued_at = issued_at or utc_now()
        self.expires_at = expires_at or (self.issued_at + timedelta(seconds=ttl_seconds))

    def has_scope(self, required: MemorySyscallScope | str) -> bool:
        """Check if token has required scope."""
        req = MemorySyscallScope(required) if isinstance(required, str) else required
        return req in self.scopes

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return utc_now() > self.expires_at

    def validate(self, required_scope: MemorySyscallScope | str) -> None:
        """Raise PermissionError if token is invalid or missing scope."""
        if self.is_expired():
            raise PermissionError(f"Capability token expired: {self.token_id}")
        
        if not self.has_scope(required_scope):
            req = MemorySyscallScope(required_scope) if isinstance(required_scope, str) else required_scope
            raise PermissionError(
                f"Token {self.token_id} missing scope {req.value}. Has: {','.join(s.value for s in self.scopes)}"
            )

    def to_dict(self) -> dict:
        """Serialize token (safe for logging/audit, no secrets)."""
        return {
            "token_id": self.token_id,
            "organization_id": self.organization_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "actor_user_id": self.actor_user_id,
            "scopes": sorted([s.value for s in self.scopes]),
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "is_expired": self.is_expired(),
        }


class CapabilityTokenIssuer:
    """Issues capability tokens with least-privilege scopes."""

    @staticmethod
    def issue_read_token(
        *,
        organization_id: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        actor_user_id: str,
        ttl_seconds: int = 3600,
    ) -> CapabilityToken:
        """Issue token for memory read operations."""
        return CapabilityToken(
            organization_id=organization_id,
            session_id=session_id,
            agent_id=agent_id,
            actor_user_id=actor_user_id,
            scopes={MemorySyscallScope.READ, MemorySyscallScope.SEARCH},
            ttl_seconds=ttl_seconds,
        )

    @staticmethod
    def issue_write_token(
        *,
        organization_id: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        actor_user_id: str,
        ttl_seconds: int = 3600,
    ) -> CapabilityToken:
        """Issue token for memory write operations (append, upsert)."""
        return CapabilityToken(
            organization_id=organization_id,
            session_id=session_id,
            agent_id=agent_id,
            actor_user_id=actor_user_id,
            scopes={MemorySyscallScope.APPEND, MemorySyscallScope.UPSERT, MemorySyscallScope.FEEDBACK},
            ttl_seconds=ttl_seconds,
        )

    @staticmethod
    def issue_admin_token(
        *,
        organization_id: str,
        actor_user_id: str,
        ttl_seconds: int = 3600,
    ) -> CapabilityToken:
        """Issue full-capability token (org admin only)."""
        return CapabilityToken(
            organization_id=organization_id,
            actor_user_id=actor_user_id,
            scopes=set(MemorySyscallScope),
            ttl_seconds=ttl_seconds,
        )

    @staticmethod
    def issue_agent_token(
        *,
        organization_id: str,
        session_id: str,
        agent_id: str,
        actor_user_id: str,
        scopes: set[MemorySyscallScope] | set[str] | None = None,
        ttl_seconds: int = 300,  # Shorter TTL for agent tokens
    ) -> CapabilityToken:
        """Issue scoped token for agent execution."""
        if scopes is None:
            scopes = {MemorySyscallScope.SEARCH, MemorySyscallScope.APPEND}
        
        return CapabilityToken(
            organization_id=organization_id,
            session_id=session_id,
            agent_id=agent_id,
            actor_user_id=actor_user_id,
            scopes=scopes,
            ttl_seconds=ttl_seconds,
        )
