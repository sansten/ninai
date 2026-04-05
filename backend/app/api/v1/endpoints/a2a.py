"""Agent-to-Agent (A2A) protocol endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, status

from app.schemas.a2a import (
    A2ACapabilitiesResponse,
    A2AMessageRequest,
    A2AMessageResponse,
)

router = APIRouter()


@router.get(
    "/capabilities",
    response_model=A2ACapabilitiesResponse,
    status_code=status.HTTP_200_OK,
    tags=["A2A Protocol"],
    operation_id="a2aCapabilities",
    summary="Get Ninai A2A protocol capabilities",
)
async def get_a2a_capabilities() -> A2ACapabilitiesResponse:
    return A2ACapabilitiesResponse(
        protocol="ninai-a2a",
        version="v1",
        supported_message_types=[
            "signal",
            "goal_update",
            "insight",
            "handoff",
            "coordination",
        ],
        delivery_guarantee="at_least_once",
        max_payload_bytes=65536,
    )


@router.post(
    "/messages",
    response_model=A2AMessageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["A2A Protocol"],
    operation_id="a2aSendMessage",
    summary="Send an A2A message to another agent",
)
async def send_a2a_message(body: A2AMessageRequest) -> A2AMessageResponse:
    # Route placeholder for initial protocol surface. Queue-backed delivery can
    # be wired in a follow-up without changing client contract.
    return A2AMessageResponse(
        status="accepted",
        message_id=str(uuid.uuid4()),
        sender_agent_id=body.sender_agent_id,
        target_agent_id=body.target_agent_id,
        message_type=body.message_type,
        received_at=datetime.now(timezone.utc),
        routed_via="a2a-router",
        payload=body.payload,
    )
