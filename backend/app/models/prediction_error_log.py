"""PredictionErrorLog — Phase 85.

Records instances where the system's pre-inference expectation diverged
significantly from the actual retrieved/inferred result.  High-divergence
events are the primary signal for priority memory consolidation: the system
learns hardest from what surprised it most.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class PredictionErrorLog(Base, UUIDMixin):
    __tablename__ = "prediction_error_logs"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Short hash of the original question for deduplication lookups
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # First 300 chars of the question (debug / dashboard display)
    query_snippet: Mapped[str] = mapped_column(String(300), nullable=False)

    # What the system expected before the LLM call
    expected_category: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # "entity" | "date" | "boolean" | "narrative"
    expected_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # What the system actually got
    actual_category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    actual_answer_snippet: Mapped[str | None] = mapped_column(String(300), nullable=True)

    # Divergence score in [0, 1]; higher = more surprising
    divergence_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Reasoning strategy that was active at the time
    strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # IDs of retrieval chunks linked to this event (for importance boosting)
    chunk_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Whether a nightly consolidation pass has already processed this event
    consolidated: Mapped[bool] = mapped_column(nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
