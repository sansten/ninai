"""Schemas for the public cognitive manifest."""

from __future__ import annotations

from pydantic import Field

from app.schemas.base import BaseSchema


class CognitiveManifestAgent(BaseSchema):
    name: str
    identifier: str
    version: str
    status: str
    capability: str
    phase: int | None = None
    summary: str | None = None
    dependencies: list[str] = Field(default_factory=list)


class CognitiveManifestIntegrations(BaseSchema):
    mcp: bool
    a2a: bool
    langchain: bool
    llamaindex: bool
    crewai: bool
    openai_tools: bool


class CognitiveManifestEventStream(BaseSchema):
    websocket: str
    sse: str
    webhooks: bool


class CognitiveManifestResponse(BaseSchema):
    name: str
    version: str
    deployed_phases: list[int] = Field(default_factory=list)
    active_agents: list[CognitiveManifestAgent] = Field(default_factory=list)
    cognitive_capabilities: list[str] = Field(default_factory=list)
    integrations: CognitiveManifestIntegrations
    event_stream: CognitiveManifestEventStream
