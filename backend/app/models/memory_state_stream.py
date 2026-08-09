"""State-space event stream for incremental write-time memory updates."""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryStateStream(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_state_stream"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scope_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    scope_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, default="memory_write")
    event_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    event_features: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    fact_delta: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    __table_args__ = (
        Index(
            "ix_memory_state_stream_org_scope_created",
            "organization_id",
            "scope_type",
            "scope_key",
            "created_at",
        ),
    )
