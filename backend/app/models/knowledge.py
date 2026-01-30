"""
Knowledge Model - Represents knowledge items in the memory system

Knowledge items are versioned, reviewable, and promotable to long-term memory.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, Boolean, Integer, ForeignKey, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as SQLA_UUID
from app.core.database import Base


class Knowledge(Base):
    """
    Knowledge item - atomic unit of information in the memory system.
    
    Lifecycle:
    - Created (draft, not_published)
    - Reviewed (pending review)
    - Published (approved, is_published=True)
    - Promoted to Memory (optional, moved to long-term memory)
    """
    __tablename__ = "knowledge"

    # Identification
    id: Mapped[str] = mapped_column(SQLA_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id: Mapped[str] = mapped_column(SQLA_UUID(as_uuid=False), nullable=False, index=True)

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    knowledge_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Versioning
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_published: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    # Consolidation tracking
    is_consolidated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    consolidated_into_id: Mapped[str | None] = mapped_column(SQLA_UUID(as_uuid=False), nullable=True)

    # Provenance
    created_by_user_id: Mapped[str | None] = mapped_column(SQLA_UUID(as_uuid=False), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Audit & Compliance
    source_trace_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    review_status: Mapped[str] = mapped_column(String(64), default="draft")

    # Indices
    __table_args__ = (
        Index("idx_knowledge_org", "organization_id"),
        Index("idx_knowledge_published", "organization_id", "is_published"),
        Index("idx_knowledge_created", "created_at"),
    )

    def __repr__(self):
        return f"<Knowledge id={self.id} org={self.organization_id} version={self.version}>"
