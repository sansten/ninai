"""
Agent Model
===========

Model for AI agents that can access and create memories.
"""

from typing import Optional, List
from datetime import datetime

from sqlalchemy import String, Text, Boolean, Integer, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.models.base import Base, UUIDMixin, TimestampMixin


class Agent(Base, UUIDMixin, TimestampMixin):
    """
    AI Agent model.
    
    Agents operate on behalf of users and inherit the creating
    user's permissions. Agents can be scoped to different levels.
    
    SECURITY: Agent execution uses the caller's permissions;
    agents cannot exceed the user's access level.
    
    Attributes:
        organization_id: Organization the agent belongs to
        name: Agent display name
        owner_id: User who created/owns the agent
        scope: Agent visibility (personal/team/department/org)
        config: LLM provider, model, and settings
    """
    
    __tablename__ = "agents"
    
    # Organization (tenant)
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this agent belongs to",
    )
    
    # Basic info
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Agent display name",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Agent description",
    )
    
    # Ownership
    owner_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
        doc="User who owns this agent",
    )
    
    # Scope
    scope: Mapped[str] = mapped_column(
        String(50),
        default="personal",
        nullable=False,
        doc="Visibility: personal, team, department, division, organization",
    )
    scope_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Scope entity ID (e.g., team_id if scope is 'team')",
    )
    
    # Configuration
    config: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc="Agent configuration (LLM provider, model, etc.)",
    )
    
    # LLM settings
    llm_provider: Mapped[str] = mapped_column(
        String(50),
        default="openai",
        nullable=False,
        doc="LLM provider (openai, anthropic, etc.)",
    )
    llm_model: Mapped[str] = mapped_column(
        String(100),
        default="gpt-4",
        nullable=False,
        doc="LLM model identifier",
    )
    
    # Memory strategy
    memory_strategy: Mapped[str] = mapped_column(
        String(50),
        default="default",
        nullable=False,
        doc="Memory retrieval strategy",
    )
    
    # Limits
    max_tokens_per_request: Mapped[int] = mapped_column(
        Integer,
        default=4000,
        nullable=False,
        doc="Maximum tokens per LLM request",
    )
    max_memory_results: Mapped[int] = mapped_column(
        Integer,
        default=10,
        nullable=False,
        doc="Maximum memories to retrieve per query",
    )
    
    # System prompt
    system_prompt: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Custom system prompt for the agent",
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether agent is active",
    )
    
    # Usage tracking
    total_requests: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Total requests made by this agent",
    )
    total_tokens_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Total tokens consumed by this agent",
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="Last time agent was used",
    )
    
    __table_args__ = (
        # Index for owner lookups
        Index("ix_agents_owner", "owner_id", "organization_id"),
        # Index for scope lookups
        Index("ix_agents_scope", "organization_id", "scope", "scope_id"),
    )
    
    def __repr__(self) -> str:
        return f"<Agent {self.name}>"
