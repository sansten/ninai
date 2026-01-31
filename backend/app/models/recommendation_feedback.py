"""
RecommendationFeedback model - Tracks user feedback on recommendations

Stores helpful/not-helpful votes to improve recommendation algorithm over time.
Used for ML training and algorithm tuning.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base


class RecommendationFeedback(Base):
    """
    User feedback on memory recommendations.
    
    Tracks:
    - Which recommendations user found helpful
    - Reasons for feedback
    - Temporal information for weighting recent feedback higher
    """

    __tablename__ = "recommendation_feedback"

    # Core fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Relationship
    base_memory_id = Column(String(36), nullable=False, index=True)  # Memory being viewed
    recommended_memory_id = Column(String(36), nullable=False, index=True)  # Recommendation
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)  # Who gave feedback
    
    # Feedback
    helpful = Column(Boolean, nullable=False, index=True)  # True/False vote
    reason = Column(Text, nullable=True)  # Optional text reason (e.g., "not relevant", "very helpful")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Indexes for queries
    __table_args__ = (
        Index("idx_rec_feedback_org_base", "organization_id", "base_memory_id"),
        Index("idx_rec_feedback_user_org", "user_id", "organization_id"),
        Index("idx_rec_feedback_helpful", "helpful"),
        Index("idx_rec_feedback_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<RecommendationFeedback {self.base_memory_id} -> {self.recommended_memory_id} (helpful={self.helpful})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "organization_id": str(self.organization_id),
            "base_memory_id": self.base_memory_id,
            "recommended_memory_id": self.recommended_memory_id,
            "user_id": str(self.user_id),
            "helpful": self.helpful,
            "reason": self.reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
