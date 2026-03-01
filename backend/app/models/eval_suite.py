"""Evaluation Suite model for memory quality testing."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import ForeignKey, Index, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.eval_run import EvalRun
    from app.models.organization import Organization


class EvalSuite(Base):
    """Test suite for evaluating memory retrieval quality.
    
    Contains query-expected pairs for measuring precision, recall, MRR, NDCG,
    cross-tenant leaks, policy violations, and other memory quality metrics.
    """

    __tablename__ = "eval_suites"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # JSONB structure: [{"query": "...", "expected_ids": [...], "filters": {...}, "metadata": {...}}]
    queries: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    
    # Expected results structure: {"query_0": {"ids": [...], "min_score": 0.8, ...}, ...}
    expected: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    organization: Mapped[Organization] = relationship("Organization", back_populates="eval_suites")
    eval_runs: Mapped[list[EvalRun]] = relationship(
        "EvalRun", back_populates="suite", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_eval_suites_org_active", "organization_id", "is_active"),
        Index("ix_eval_suites_org_created", "organization_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<EvalSuite(id={self.id}, name={self.name}, org={self.organization_id})>"
