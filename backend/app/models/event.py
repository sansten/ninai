"""
Event Model - System events for memory operations

Events are published when memory items are created, updated, reviewed, etc.
Used for webhooks, event streaming, and audit trails.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, DateTime, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as SQLA_UUID
from app.core.database import Base


class Event(Base):
    """
    Event - represents a system event for webhooks and event streaming.
    
    Events are:
    - Org-scoped (tied to organization)
    - Typed (memory.created, memory.updated, knowledge.reviewed, etc.)
    - Timestamped for ordering
    - Versioned for schema evolution
    - Traced for debugging
    """
    __tablename__ = "events"

    # Identification
    id: Mapped[str] = mapped_column(SQLA_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Event classification
    event_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)  # memory.created, memory.updated, knowledge.reviewed
    event_version: Mapped[int] = mapped_column(default=1, nullable=False)  # Schema versioning
    
    # Scoping
    organization_id: Mapped[str] = mapped_column(SQLA_UUID(as_uuid=False), nullable=False, index=True)
    
    # Resource information
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)  # memory, knowledge, agent, etc.
    resource_id: Mapped[str] = mapped_column(SQLA_UUID(as_uuid=False), nullable=False, index=True)  # The affected resource
    
    # Event payload
    payload: Mapped[dict] = mapped_column(JSON, nullable=True)  # Event-specific data
    
    # Actor and context
    actor_user_id: Mapped[str | None] = mapped_column(SQLA_UUID(as_uuid=False), nullable=True)  # Who triggered it
    actor_agent_id: Mapped[str | None] = mapped_column(SQLA_UUID(as_uuid=False), nullable=True)  # Or which agent
    
    # Tracing and debugging
    trace_id: Mapped[str | None] = mapped_column(String(256), nullable=True)  # Request trace ID
    request_id: Mapped[str | None] = mapped_column(String(256), nullable=True)  # Unique request ID
    
    # Lifecycle
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # When webhook was sent
    
    # Indices for efficient querying
    __table_args__ = (
        Index("idx_event_org_type", "organization_id", "event_type"),
        Index("idx_event_resource", "organization_id", "resource_type", "resource_id"),
        Index("idx_event_created", "created_at"),
    )

    def __repr__(self):
        return f"<Event id={self.id} type={self.event_type} org={self.organization_id}>"
