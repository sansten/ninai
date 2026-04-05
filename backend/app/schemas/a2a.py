"""Agent-to-Agent (A2A) protocol schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.base import BaseSchema


class A2AMessageRequest(BaseSchema):
    sender_agent_id: str = Field(..., description="Agent sending the message")
    target_agent_id: str = Field(..., description="Target recipient agent")
    message_type: str = Field(default="signal", description="Message type/category")
    payload: dict[str, Any] = Field(default_factory=dict)


class A2AMessageResponse(BaseSchema):
    status: str
    message_id: str
    sender_agent_id: str
    target_agent_id: str
    message_type: str
    received_at: datetime
    routed_via: str
    payload: dict[str, Any] = Field(default_factory=dict)


class A2ACapabilitiesResponse(BaseSchema):
    protocol: str
    version: str
    supported_message_types: list[str]
    delivery_guarantee: str
    max_payload_bytes: int
