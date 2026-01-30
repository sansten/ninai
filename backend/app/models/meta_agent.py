from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class MetaAgentRun(Base, UUIDMixin):
    __tablename__ = "meta_agent_runs"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    resource_type: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    resource_id: Mapped[str] = mapped_column(UUID(as_uuid=False), index=True, nullable=False)

    supervision_type: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)

    final_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class MetaConflictRegistry(Base, UUIDMixin):
    __tablename__ = "meta_conflict_registry"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[str] = mapped_column(UUID(as_uuid=False), index=True, nullable=False)

    conflict_type: Mapped[str] = mapped_column(String(50), nullable=False)
    candidates: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    resolution: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    resolved_by: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BeliefStore(Base, UUIDMixin):
    __tablename__ = "belief_store"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    memory_id: Mapped[str] = mapped_column(UUID(as_uuid=False), index=True, nullable=False)
    belief_key: Mapped[str] = mapped_column(String(200), nullable=False)
    belief_value: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    evidence_memory_ids: Mapped[list[str]] = mapped_column(ARRAY(UUID(as_uuid=False)), nullable=False, default=list)
    contradiction_ids: Mapped[list[str]] = mapped_column(ARRAY(UUID(as_uuid=False)), nullable=False, default=list)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("organization_id", "memory_id", "belief_key", name="uq_belief_store_org_memory_key"),
    )


class CalibrationProfile(Base):
    __tablename__ = "calibration_profiles"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    promotion_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.75)
    conflict_escalation_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.60)
    drift_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.20)
    signal_weights: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


