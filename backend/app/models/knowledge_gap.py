"""
Knowledge Gap Model

Represents detected gaps in the agent's knowledge or understanding.
These gaps trigger autonomous goal generation in the IntrinsicMotivationService.
"""

from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime, ARRAY, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.database import Base


class KnowledgeGap(Base):
    """
    Detected gap in agent's knowledge or reasoning.
    
    A knowledge gap is a systematic deficiency the agent identifies in its own
    understanding. It can be a missing fact, unresolved contradiction, outdated
    information, or low-confidence belief. Detection of gaps triggers autonomous
    goal generation with curiosity as the initiator.
    """
    __tablename__ = "knowledge_gaps"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Legacy + Phase 63 org references (both kept for compatibility).
    organization_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    org_id = Column(UUID(as_uuid=False), nullable=True, index=True)

    # Phase 63 references.
    goal_id = Column(UUID(as_uuid=False), nullable=True, index=True)
    
    # Gap characterization
    gap_type = Column(String, nullable=False)  # "missing_fact" | "contradiction" | "outdated" | "low_confidence"
    domain = Column(String, nullable=False)  # e.g. "customer_preferences", "system_behavior"
    description = Column(String, nullable=False)

    # Phase 63 fields.
    gap_description = Column(String, nullable=True)
    question_to_ask = Column(String, nullable=True)
    required_for = Column(String, nullable=True)
    priority = Column(String, nullable=False, default="high", index=True)  # critical|high|medium|low
    status = Column(String, nullable=False, default="open", index=True)  # open|resolved|waived
    
    # Validation
    confidence_in_gap = Column(Float, nullable=False, default=0.7)  # How sure we have a gap?
    related_memories = Column(ARRAY(String), nullable=False, default=[])  # Memory IDs related to gap
    
    # Learning approach
    suggested_learning_approach = Column(String, nullable=True)  # "search" | "ask_user" | "experiment" | "observation"
    
    # Lifecycle
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    discovered_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True, index=True)
    
    # Metadata
    meta = Column(JSON, nullable=True, default={})  # Additional context
    
    def __repr__(self) -> str:
        return f"<KnowledgeGap(id={self.id}, gap_type={self.gap_type}, domain={self.domain})>"
