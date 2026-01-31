"""Materialized graph edges.

Stores graph edges derived from agent outputs (GraphLinkingAgent v1).

This table is tenant-scoped (organization_id) and protected by PostgreSQL RLS.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryEdge(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_edges"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this edge belongs to (tenant isolation)",
    )

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Source memory this edge was derived from",
    )

    from_node: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        doc="Graph node identifier (string; e.g. memory:<uuid>, tag:<tag>)",
    )

    to_node: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        doc="Graph node identifier (string)",
    )

    relation: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="Relationship type (e.g. about/tagged/mentions:email)",
    )

    weight: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        doc="Edge weight (0..1), if available",
    )

    explanation: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
        doc="Optional human-readable explanation for why this edge exists",
    )

    created_by: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="agent",
        doc="Creator identifier (agent/system/user)",
    )

    __table_args__ = (
        Index(
            "ux_memory_edges_dedupe",
            "organization_id",
            "memory_id",
            "from_node",
            "to_node",
            "relation",
            unique=True,
        ),
        Index("ix_memory_edges_lookup", "organization_id", "memory_id", "relation"),
    )
