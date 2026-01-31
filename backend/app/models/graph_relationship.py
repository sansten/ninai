"""
GraphRelationship model - Stores relationships between memories in the knowledge graph.

Tracks both manual and auto-generated relationships with similarity scores.
Metadata includes algorithm info and source for auditing.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, JSON, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base


class GraphRelationship(Base):
    """
    Represents a relationship between two memories in the knowledge graph.
    
    Can be:
    - Auto-created via similarity detection
    - Manually created by user
    - With optional similarity score from embeddings
    """

    __tablename__ = "graph_relationships"

    # Core fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Relationship participants
    from_memory_id = Column(String(36), nullable=False, index=True)
    to_memory_id = Column(String(36), nullable=False, index=True)
    
    # Relationship properties
    relationship_type = Column(String(50), default="RELATES_TO", nullable=False)  # RELATES_TO, DEPENDS_ON, CONTRADICTS, REFINES, REFERENCES
    similarity_score = Column(Float, nullable=True)  # 0.0-1.0 for auto-generated
    
    # Origin tracking
    auto_created = Column(Boolean, default=False, index=True)
    created_by_user_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Metadata
    metadata_ = Column(JSON, default=dict)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes for common queries
    __table_args__ = (
        Index("idx_graph_rel_org_from", "organization_id", "from_memory_id"),
        Index("idx_graph_rel_org_to", "organization_id", "to_memory_id"),
        Index("idx_graph_rel_type", "organization_id", "relationship_type"),
        Index("idx_graph_rel_similarity", "similarity_score"),
        UniqueConstraint("organization_id", "from_memory_id", "to_memory_id", "relationship_type", name="uq_graph_rel_unique"),
    )

    def __repr__(self) -> str:
        return f"<GraphRelationship {self.from_memory_id} -[{self.relationship_type}]-> {self.to_memory_id} (similarity={self.similarity_score})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "organization_id": str(self.organization_id),
            "from_memory_id": self.from_memory_id,
            "to_memory_id": self.to_memory_id,
            "relationship_type": self.relationship_type,
            "similarity_score": self.similarity_score,
            "auto_created": self.auto_created,
            "created_by_user_id": str(self.created_by_user_id) if self.created_by_user_id else None,
            "metadata": self.metadata_,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
