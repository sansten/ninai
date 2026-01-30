"""Schemas for Agent Process management."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AgentProcessResponse(BaseModel):
    id: str
    organization_id: str
    session_id: Optional[str] = None
    agent_run_id: Optional[str] = None
    agent_name: str
    priority: int
    status: str
    attempts: int
    max_attempts: int
    quota_tokens: int
    quota_storage_mb: int
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    trace_id: Optional[str] = None
    last_error: str
    process_metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentProcessListResponse(BaseModel):
    total: int
    items: list[AgentProcessResponse]
    status_summary: dict[str, int] = Field(default_factory=dict)
