"""Agent result cache.

Stores cached agent outputs keyed by a stable input hash.

Purpose:
- Reduce repeated LLM calls for identical inputs (content/enrichment)
- Keep multi-tenant isolation via PostgreSQL RLS on organization_id

Notes:
- Cache key is computed in the agent runner and intentionally excludes memory_id.
- Callers should bump agent versions when prompt logic changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class AgentResultCache(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "agent_result_cache"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this cache row belongs to (tenant isolation)",
    )

    agent_name: Mapped[str] = mapped_column(
        String(length=100),
        nullable=False,
        doc="Agent name (e.g. MetadataExtractionAgent)",
    )

    agent_version: Mapped[str] = mapped_column(
        String(length=50),
        nullable=False,
        doc="Agent version used to compute outputs",
    )

    strategy: Mapped[str] = mapped_column(
        String(length=20),
        nullable=False,
        doc="Execution strategy (llm|heuristic)",
    )

    model: Mapped[str] = mapped_column(
        String(length=100),
        nullable=False,
        doc="LLM model identifier (e.g. llama3:latest)",
    )

    cache_key: Mapped[str] = mapped_column(
        String(length=64),
        nullable=False,
        doc="Stable input hash (sha256 hex)",
    )

    outputs: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Cached agent outputs JSON",
    )

    confidence: Mapped[float] = mapped_column(
        nullable=False,
        default=0.0,
        doc="Cached confidence score",
    )

    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last access timestamp (nullable)",
    )

    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Optional expiry timestamp (nullable)",
    )

    __table_args__ = (
        Index(
            "ux_agent_result_cache_key",
            "organization_id",
            "agent_name",
            "agent_version",
            "strategy",
            "model",
            "cache_key",
            unique=True,
        ),
        Index(
            "ix_agent_result_cache_lookup",
            "organization_id",
            "agent_name",
            "agent_version",
            "updated_at",
            unique=False,
        ),
    )
