"""
Capability Token Model - Capability-scoped access tokens for the memory syscall surface

Tokens enable fine-grained, quota-bounded access to memory operations (read, append, search, upsert, consolidate).
"""

import uuid
import enum
from datetime import datetime, timedelta
from sqlalchemy import String, Text, Boolean, Integer, DateTime, ForeignKey, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as SQLA_UUID
from app.core.database import Base


class CapabilityScope(str, enum.Enum):
    """Capability operation scopes."""
    READ = "read"  # vector search, get knowledge
    APPEND = "append"  # add knowledge items
    SEARCH = "search"  # vector + SQL search
    UPSERT = "upsert"  # update or create knowledge
    CONSOLIDATE = "consolidate"  # merge/deduplicate knowledge
    PROMOTE = "promote"  # promote to long-term memory


class CapabilityToken(Base):
    """
    Capability token - enables quota-bounded, scoped access to memory operations.
    
    Tokens are:
    - Org-scoped (tied to organization)
    - Optionally session-scoped (tied to user session for temporary access)
    - Quota-bounded (monthly token count, storage, requests/minute)
    - Scope-limited (can restrict to read, append, search, upsert, consolidate)
    
    Lifecycle:
    - Created (active=True)
    - Revoked (active=False, revocation_reason set)
    - Quota exhausted (quota_exceeded=True)
    - Expired (now > expires_at)
    """
    __tablename__ = "capability_tokens"

    # Identification
    id: Mapped[str] = mapped_column(SQLA_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Token value (Bearer token format: cap_<urlsafe>)
    # Alembic migration uses VARCHAR(60); keep in sync.
    token: Mapped[str] = mapped_column(String(60), unique=True, nullable=False, index=True)
    
    # Scoping
    organization_id: Mapped[str] = mapped_column(SQLA_UUID(as_uuid=False), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(SQLA_UUID(as_uuid=False), nullable=True, index=True)  # Optional: limits token to session
    
    # Capabilities (CSV string: "read,append,search")
    scopes: Mapped[str] = mapped_column(String(256), nullable=False, default="read")  # CSV of capabilities
    
    # Quotas (per-token limits)
    quota_tokens_per_month: Mapped[int] = mapped_column(Integer, default=1_000_000, nullable=False)  # LLM token limit
    quota_storage_bytes: Mapped[int] = mapped_column(Integer, default=104_857_600, nullable=False)  # 100MB default
    quota_requests_per_minute: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    
    # Usage tracking
    tokens_used_this_month: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    storage_used_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    requests_this_minute: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_request_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Status flags
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    quota_exceeded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Revocation
    revocation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_by_user_id: Mapped[str | None] = mapped_column(SQLA_UUID(as_uuid=False), nullable=True)
    
    # Lifecycle
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.utcnow() + timedelta(days=365), nullable=False, index=True)
    
    # Audit metadata
    token_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # Custom metadata
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_by_user_id: Mapped[str | None] = mapped_column(SQLA_UUID(as_uuid=False), nullable=True)
    
    # Indices for efficient queries
    __table_args__ = (
        Index("idx_capability_token_org_active", "organization_id", "active"),
        Index("idx_capability_token_agent", "organization_id", "session_id"),
        Index("idx_capability_token_expires", "expires_at"),
    )

    def is_valid(self) -> bool:
        """Check if token is valid (active, not expired, not revoked)."""
        return (
            self.active
            and not self.quota_exceeded
            and (self.expires_at is None or datetime.utcnow() < self.expires_at)
            and self.revoked_at is None
        )

    def has_scope(self, scope: str) -> bool:
        """Check if token has required scope."""
        if not self.scopes:
            return False
        allowed_scopes = [s.strip() for s in self.scopes.split(",")]
        return scope in allowed_scopes

    def is_quota_exceeded(self) -> bool:
        """Check if any quota is exceeded."""
        return (
            self.tokens_used_this_month >= self.quota_tokens_per_month
            or self.storage_used_bytes >= self.quota_storage_bytes
        )

    @property
    def name(self) -> str | None:
        md = self.token_metadata
        if isinstance(md, dict):
            return md.get("name")
        return None

    @property
    def agent_name(self) -> str | None:
        md = self.token_metadata
        if isinstance(md, dict):
            return md.get("agent_name")
        return None

    # Backwards/alternate attribute names used by services/tests
    @property
    def max_tokens_per_month(self) -> int:
        return self.quota_tokens_per_month

    @max_tokens_per_month.setter
    def max_tokens_per_month(self, value: int) -> None:
        self.quota_tokens_per_month = value

    @property
    def max_storage_bytes(self) -> int:
        return self.quota_storage_bytes

    @max_storage_bytes.setter
    def max_storage_bytes(self, value: int) -> None:
        self.quota_storage_bytes = value

    @property
    def max_requests_per_minute(self) -> int:
        return self.quota_requests_per_minute

    @max_requests_per_minute.setter
    def max_requests_per_minute(self, value: int) -> None:
        self.quota_requests_per_minute = value

    @property
    def tokens_used(self) -> int:
        return self.tokens_used_this_month

    @tokens_used.setter
    def tokens_used(self, value: int) -> None:
        self.tokens_used_this_month = value

    def __repr__(self):
        return f"<CapabilityToken id={self.id} org={self.organization_id} active={self.active}>"
