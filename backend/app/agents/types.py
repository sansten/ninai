from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


AgentStatus = Literal["success", "retry", "failed", "skipped"]


class AgentContextTenant(TypedDict):
    org_id: str
    org_slug: str | None


class AgentContextActor(TypedDict, total=False):
    user_id: str
    roles: list[str]
    scopes: list[str]
    clearance_level: int


class AgentContextMemory(TypedDict, total=False):
    id: str
    storage: str
    content: str
    enrichment: dict
    memory_type: str
    scope: str
    scope_id: str | None
    classification: str
    is_sensitive: bool


class AgentContextRuntime(TypedDict, total=False):
    job_id: str
    attempt: int
    max_attempts: int
    deadline: str


class AgentContextConfig(TypedDict, total=False):
    prompts_version: str
    model_provider: str
    model_name: str
    temperature: float
    max_tokens: int


class AgentContext(TypedDict, total=False):
    tenant: AgentContextTenant
    actor: AgentContextActor
    memory: AgentContextMemory
    runtime: AgentContextRuntime
    config: AgentContextConfig


class AgentResult(BaseModel):
    agent_name: str
    agent_version: str
    memory_id: str

    status: AgentStatus
    confidence: float = Field(ge=0.0, le=1.0)

    outputs: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    started_at: datetime
    finished_at: datetime
    trace_id: str | None = None

    # Provenance/citations used to produce outputs (best-effort)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
