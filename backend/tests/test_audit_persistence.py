"""Tests for audit event persistence."""

import pytest
import uuid
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_audit_events_persist(pg_client, auth_headers):
    """Test that audit events are stored in database."""
    # Log an audit event
    event_body = {
        "category": "policy",
        "action": "policy_rollout",
        "severity": "info",
        "resource_type": "policy",
        "resource_id": str(uuid.uuid4()),  # Must be valid UUID
        "details": {"message": "Test rollout"},
    }
    res = await pg_client.post(
        "/api/v1/admin/audit", json=event_body, headers=auth_headers
    )
    assert res.status_code == 200
    event = res.json()
    assert event["category"] == "policy"
    assert event["action"] == "policy_rollout"

    # Retrieve audit trail
    res = await pg_client.get("/api/v1/admin/audit", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data["events"]) > 0
    assert any(e["id"] == event["id"] for e in data["events"])


@pytest.mark.asyncio
async def test_audit_trail_category_filter(pg_client, auth_headers):
    """Test category filtering on audit trail."""
    # Log events in different categories
    await pg_client.post(
        "/api/v1/admin/audit",
        json={"category": "security", "action": "token_revoke", "severity": "warning"},
        headers=auth_headers,
    )
    await pg_client.post(
        "/api/v1/admin/audit",
        json={"category": "config", "action": "setting_update", "severity": "info"},
        headers=auth_headers,
    )

    # Filter by category
    res = await pg_client.get(
        "/api/v1/admin/audit?category=security", headers=auth_headers
    )
    assert res.status_code == 200
    events = res.json()["events"]
    assert all(e["category"] == "security" for e in events)


@pytest.mark.asyncio
async def test_audit_event_includes_actor(pg_client, auth_headers, test_user_id):
    """Test that audit events record actor information."""
    res = await pg_client.post(
        "/api/v1/admin/audit",
        json={"category": "system", "action": "test_event", "severity": "info"},
        headers=auth_headers,
    )
    assert res.status_code == 200

    res = await pg_client.get("/api/v1/admin/audit", headers=auth_headers)
    events = res.json()["events"]
    assert len(events) > 0
    # At least one event should have the test user as actor
    assert any(e.get("actor") == test_user_id for e in events)
