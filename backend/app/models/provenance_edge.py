"""Provenance edge model for memory lineage graph (Phase 77)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProvenanceEdge(Base, UUIDMixin):
    __tablename__ = "provenance_edges"

    org_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    target_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    edge_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )
    edge_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_provenance_edges_org_target_created", "org_id", "target_id", "created_at"),
        Index("ix_provenance_edges_org_source_created", "org_id", "source_id", "created_at"),
    )
