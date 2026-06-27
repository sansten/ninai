"""SchemaFrame — Phase 87.

A schema is an abstract knowledge template crystallized from recurring episodic
patterns.  Each frame has typed slots (roles) and a trigger signature.

Example: schema "project_kickoff" has slots {owner, deadline, goal, team}.
When Ninai encounters a new kickoff event, it fills the frame and can infer
missing slots from prior instances (frame-completion inference).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class SchemaFrame(Base, UUIDMixin):
    """A named abstract template for a recurring event/situation type."""

    __tablename__ = "schema_frames"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Human-readable name inferred from the triggering verb/noun cluster
    name: Mapped[str] = mapped_column(String(120), nullable=False)

    # Fingerprint of the event-type cluster that generated this schema
    trigger_signature: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Ordered list of slot definitions: [{name, type, required, fill_rate}]
    slots: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Typical slot values observed across instances (for frame-completion)
    # {slot_name: {value: count, ...}}
    slot_distributions: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Number of episodic instances that contributed to this schema
    instance_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Average confidence across contributing instances
    avg_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    # IDs of memory records that are instances of this schema
    instance_memory_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(), onupdate=func.now(),
    )
