"""Closed-loop sync tests (action -> outbound payload -> inbound reflection)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.autonomous_action_agent import AutonomousActionAgent
from app.agents.types import AgentContext
from app.services.environment_sync_service import EnvironmentSyncService
from app.services.external_connector_service import ExternalConnectorService
from app.services.inbound_event_service import event_to_memory_fields, parse_inbound_event


def _urgent_enrichment() -> dict:
    return {
        "anomaly_score": 0.95,
        "org_attention_level": "high",
        "risk_level": "high",
        "org_tier": "enterprise",
    }


def _ctx(memory_id: str = "mem-1") -> AgentContext:
    return {
        "memory": {
            "id": memory_id,
            "content": "CPU saturation crossed threshold",
            "enrichment": _urgent_enrichment(),
        },
        "tenant": {"org_id": "org-1"},
        "runtime": {},
    }


@pytest.mark.asyncio
async def test_closed_loop_outbound_to_inbound_reflection_applies_state():
    # Capture outbound action payload.
    response = MagicMock()
    response.status_code = 200
    response.text = "ok"
    client = MagicMock()
    client.post = AsyncMock(return_value=response)

    connector = ExternalConnectorService(
        backoff_base=0.001,
        _http_client_factory=lambda: client,
    )
    agent = AutonomousActionAgent(connector_service=connector)
    heuristic_outputs = {
        "action_dispatched": True,
        "action_type": "webhook",
        "action_status": "success",
        "policy_decision": "auto_approved",
        "dispatch_confidence": 0.95,
        "action_target": "https://hooks.example.internal/ninai/action",
        "rationale": "test forced auto-approved",
    }

    with patch("app.agents.autonomous_action_agent.settings") as mock_settings, \
         patch("app.agents.autonomous_action_agent.run_heuristic", return_value=heuristic_outputs):
        mock_settings.AGENT_STRATEGY = "heuristic"
        result = await agent.run("mem-loop-1", _ctx("mem-loop-1"))

    assert result.outputs["action_status"] == "success"
    assert client.post.await_count == 1

    outbound_payload = client.post.await_args.kwargs["json"]

    # External reflects webhook event back inbound.
    inbound_payload = {
        "event_id": "evt-1",
        "id": "obj-1",
        "timestamp": "2026-03-31T12:00:00Z",
        "title": outbound_payload.get("title", "autonomous_action"),
        "summary": outbound_payload.get("summary", "dispatch"),
        "severity": outbound_payload.get("risk_level", "high"),
        "actor": outbound_payload.get("actor", "system"),
        "url": result.outputs.get("action_target"),
    }

    sync = EnvironmentSyncService()
    normalized = sync.normalize_inbound(
        org_id="org-1",
        connector_type="webhook",
        payload=inbound_payload,
    )
    applied = sync.apply_inbound(normalized)

    assert applied.status == "applied"
    assert applied.state is not None
    assert applied.state.external_object_id == "obj-1"
    assert applied.state.title

    event = parse_inbound_event("webhook", inbound_payload)
    fields = event_to_memory_fields(event)
    assert fields["memory_type"] == "episodic"
    assert fields["source_type"] == "integration"


@pytest.mark.asyncio
async def test_closed_loop_replay_then_out_of_order_behavior():
    sync = EnvironmentSyncService()

    baseline = {
        "event_id": "evt-1",
        "id": "obj-42",
        "timestamp": "2026-03-31T12:00:00Z",
        "title": "CPU high",
        "summary": "host overloaded",
    }

    # First event applies.
    first = sync.normalize_inbound(org_id="org-1", connector_type="webhook", payload=baseline)
    assert sync.apply_inbound(first).status == "applied"

    # Exact replay dedupes.
    replay = sync.normalize_inbound(org_id="org-1", connector_type="webhook", payload=baseline)
    assert sync.apply_inbound(replay).status == "duplicate"

    # Different event id but older timestamp is out_of_order.
    old = sync.normalize_inbound(
        org_id="org-1",
        connector_type="webhook",
        payload={
            "event_id": "evt-old",
            "id": "obj-42",
            "timestamp": "2026-03-31T11:59:00Z",
            "title": "CPU high (old)",
            "summary": "stale",
        },
    )
    assert sync.apply_inbound(old).status == "out_of_order"

    # Newer event should apply.
    new = sync.normalize_inbound(
        org_id="org-1",
        connector_type="webhook",
        payload={
            "event_id": "evt-new",
            "id": "obj-42",
            "timestamp": "2026-03-31T12:01:00Z",
            "title": "CPU recovered",
            "summary": "healthy",
        },
    )
    assert sync.apply_inbound(new).status == "applied"

    summary = sync.summary("org-1")
    assert summary.applied_events == 2
    assert summary.duplicate_events == 1
    assert summary.out_of_order_events == 1
