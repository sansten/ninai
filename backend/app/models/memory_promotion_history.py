"""Memory promotion history.

Records promotions from short-term (Redis) to long-term (Postgres/Qdrant).

This is required by AGENT_IMPLEMENTATION_GUIDE.md (PromotionAgent section).
Tenant isolation is enforced via PostgreSQL RLS on organization_id.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryPromotionHistory(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_promotion_history"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this promotion belongs to (tenant isolation)",
    )

    from_stm_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Source short-term memory id (Redis id; may not be UUID)",
    )

    to_memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Created long-term memory id",
    )

    from_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="short_term",
        doc="Source memory type (short_term)",
    )

    to_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="long_term",
        doc="Destination memory type (long_term/semantic/procedural)",
    )

    promotion_reason: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Reason for promotion (auto/manual/agent:...)",
    )

    actor_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=True,
        doc="User/system actor that initiated the promotion",
    )

    trace_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Trace/job correlation id",
    )

    details: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Additional details (access_count, importance, source)",
    )

    __table_args__ = (
        Index("ux_promotion_once", "organization_id", "from_stm_id", unique=True),
        Index("ix_promotion_lookup", "organization_id", "to_memory_id"),
    )
