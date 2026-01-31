from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient

from app.main import app
from app.api.v1.endpoints import memories as memories_endpoints


@dataclass
class _FakeFeedbackRow:
    id: str
    organization_id: str
    memory_id: str
    actor_id: str
    feedback_type: str
    target_agent: str | None
    payload: dict
    is_applied: bool
    applied_at: datetime | None
    applied_by: str | None
    created_at: datetime
    updated_at: datetime


@pytest.mark.asyncio
async def test_submit_feedback_enqueues_learning(client: AsyncClient, auth_headers: dict, monkeypatch):
    memory_id = str(uuid4())
    org_id = "org"
    user_id = "user"

    # Tenant context is derived from the JWT; we only need DB context setter to be a no-op.
    monkeypatch.setattr(memories_endpoints, "set_tenant_context", AsyncMock())

    # Stub services
    mock_memory_service = SimpleNamespace(get_memory=AsyncMock(return_value=SimpleNamespace(id=memory_id)))
    now = datetime.now(timezone.utc)
    fake_row = _FakeFeedbackRow(
        id=str(uuid4()),
        organization_id=org_id,
        memory_id=memory_id,
        actor_id=user_id,
        feedback_type="tag_add",
        target_agent=None,
        payload={"tag": "urgent"},
        is_applied=False,
        applied_at=None,
        applied_by=None,
        created_at=now,
        updated_at=now,
    )
    mock_feedback_service = SimpleNamespace(create_feedback=AsyncMock(return_value=fake_row))

    # Override dependencies
    app.dependency_overrides[memories_endpoints.get_memory_service] = lambda: mock_memory_service
    app.dependency_overrides[memories_endpoints.get_feedback_service] = lambda: mock_feedback_service

    enqueue_mock = MagicMock()
    monkeypatch.setattr(memories_endpoints, "enqueue_feedback_learning", enqueue_mock)

    try:
        resp = await client.post(
            f"/api/v1/memories/{memory_id}/feedback",
            headers=auth_headers,
            json={"feedback_type": "tag_add", "payload": {"tag": "urgent"}},
        )
    finally:
        app.dependency_overrides.pop(memories_endpoints.get_memory_service, None)
        app.dependency_overrides.pop(memories_endpoints.get_feedback_service, None)

    assert resp.status_code == 201
    data = resp.json()
    assert data["memory_id"] == memory_id
    assert data["feedback_type"] == "tag_add"
    assert data["payload"] == {"tag": "urgent"}

    assert enqueue_mock.call_count == 1


@pytest.mark.asyncio
async def test_list_feedback_returns_items(client: AsyncClient, auth_headers: dict, monkeypatch):
    memory_id = str(uuid4())

    monkeypatch.setattr(memories_endpoints, "set_tenant_context", AsyncMock())

    mock_memory_service = SimpleNamespace(get_memory=AsyncMock(return_value=SimpleNamespace(id=memory_id)))
    now = datetime.now(timezone.utc)
    rows = [
        _FakeFeedbackRow(
            id=str(uuid4()),
            organization_id="org",
            memory_id=memory_id,
            actor_id="u",
            feedback_type="note",
            target_agent="feedback",
            payload={"note": "please redact PII"},
            is_applied=False,
            applied_at=None,
            applied_by=None,
            created_at=now,
            updated_at=now,
        )
    ]
    mock_feedback_service = SimpleNamespace(list_feedback=AsyncMock(return_value=(rows, 1)))

    app.dependency_overrides[memories_endpoints.get_memory_service] = lambda: mock_memory_service
    app.dependency_overrides[memories_endpoints.get_feedback_service] = lambda: mock_feedback_service

    try:
        resp = await client.get(
            f"/api/v1/memories/{memory_id}/feedback?include_applied=true&limit=50",
            headers=auth_headers,
        )
    finally:
        app.dependency_overrides.pop(memories_endpoints.get_memory_service, None)
        app.dependency_overrides.pop(memories_endpoints.get_feedback_service, None)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["feedback_type"] == "note"


@pytest.mark.asyncio
async def test_feedback_requires_read_access(client: AsyncClient, auth_headers: dict, monkeypatch):
    memory_id = str(uuid4())

    monkeypatch.setattr(memories_endpoints, "set_tenant_context", AsyncMock())
    mock_memory_service = SimpleNamespace(get_memory=AsyncMock(side_effect=PermissionError("denied")))
    mock_feedback_service = SimpleNamespace(create_feedback=AsyncMock())

    app.dependency_overrides[memories_endpoints.get_memory_service] = lambda: mock_memory_service
    app.dependency_overrides[memories_endpoints.get_feedback_service] = lambda: mock_feedback_service

    try:
        resp = await client.post(
            f"/api/v1/memories/{memory_id}/feedback",
            headers=auth_headers,
            json={"feedback_type": "note", "payload": {"note": "x"}},
        )
    finally:
        app.dependency_overrides.pop(memories_endpoints.get_memory_service, None)
        app.dependency_overrides.pop(memories_endpoints.get_feedback_service, None)

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_submit_relevance_feedback_records_standard_payload(client: AsyncClient, auth_headers: dict, monkeypatch):
    memory_id = str(uuid4())
    org_id = "org"
    user_id = "user"

    monkeypatch.setattr(memories_endpoints, "set_tenant_context", AsyncMock())

    # Stub services
    mock_memory_service = SimpleNamespace(get_memory=AsyncMock(return_value=SimpleNamespace(id=memory_id)))
    now = datetime.now(timezone.utc)
    fake_row = _FakeFeedbackRow(
        id=str(uuid4()),
        organization_id=org_id,
        memory_id=memory_id,
        actor_id=user_id,
        feedback_type="relevance",
        target_agent=None,
        payload={"relevant": True, "value": 1},
        is_applied=False,
        applied_at=None,
        applied_by=None,
        created_at=now,
        updated_at=now,
    )

    create_feedback_mock = AsyncMock(return_value=fake_row)
    mock_feedback_service = SimpleNamespace(create_feedback=create_feedback_mock)

    app.dependency_overrides[memories_endpoints.get_memory_service] = lambda: mock_memory_service
    app.dependency_overrides[memories_endpoints.get_feedback_service] = lambda: mock_feedback_service

    enqueue_mock = MagicMock()
    monkeypatch.setattr(memories_endpoints, "enqueue_feedback_learning", enqueue_mock)

    try:
        resp = await client.post(
            f"/api/v1/memories/{memory_id}/relevance",
            headers=auth_headers,
            json={"relevant": True, "query": "hello", "hnms_mode": "performance"},
        )
    finally:
        app.dependency_overrides.pop(memories_endpoints.get_memory_service, None)
        app.dependency_overrides.pop(memories_endpoints.get_feedback_service, None)

    assert resp.status_code == 201

    # Ensure the feedback service got a standardized payload
    assert create_feedback_mock.call_count == 1
    kwargs = create_feedback_mock.call_args.kwargs
    assert kwargs["memory_id"] == memory_id
    assert kwargs["feedback_type"] == "relevance"
    assert kwargs["payload"]["relevant"] is True
    assert kwargs["payload"]["value"] == 1
    assert kwargs["payload"]["query"] == "hello"
    assert kwargs["payload"]["hnms_mode"] == "performance"

    assert enqueue_mock.call_count == 1
