from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


@dataclass
class _FakeActionRecord:
    id: str
    organization_id: str
    session_id: str | None
    action_type: str
    connector_id: str | None
    target_url: str | None
    payload_summary: dict[str, Any]
    status: str
    attempt_count: int
    http_status_code: int | None
    error_message: str | None
    policy_decision: str
    confidence_at_dispatch: float | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class _Scalars:
    def __init__(self, items: list[Any]):
        self._items = items

    def all(self):
        return self._items


class _ListResult:
    def __init__(self, items: list[Any]):
        self._items = items

    def scalars(self):
        return _Scalars(self._items)


class _ScalarOneOrNoneResult:
    def __init__(self, item: Any | None):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["org_admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_list_pending_actions():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    pending = _FakeActionRecord(
        id="a1",
        organization_id="o1",
        session_id="s1",
        action_type="pagerduty",
        connector_id=None,
        target_url="https://hooks.example",
        payload_summary={},
        status="pending_review",
        attempt_count=0,
        http_status_code=None,
        error_message=None,
        policy_decision="human_review_required",
        confidence_at_dispatch=0.82,
        created_at=now,
        updated_at=now,
        completed_at=None,
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM action_execution_records" in sql:
            return _ListResult([pending])
        return _ListResult([])

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/actions/pending", headers=_auth_headers())

        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 1
        assert rows[0]["id"] == "a1"
        assert rows[0]["policy_decision"] == "human_review_required"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_action_details():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    action = _FakeActionRecord(
        id="a2",
        organization_id="o1",
        session_id="s2",
        action_type="webhook",
        connector_id=None,
        target_url="https://hooks.example",
        payload_summary={},
        status="pending",
        attempt_count=1,
        http_status_code=None,
        error_message=None,
        policy_decision="human_review_required",
        confidence_at_dispatch=0.9,
        created_at=now,
        updated_at=now,
        completed_at=None,
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM action_execution_records" in sql:
            return _ScalarOneOrNoneResult(action)
        return _ScalarOneOrNoneResult(None)

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/actions/a2", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "a2"
        assert body["action_type"] == "webhook"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_approve_action_updates_policy_decision():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    action = _FakeActionRecord(
        id="a3",
        organization_id="o1",
        session_id="s3",
        action_type="pagerduty",
        connector_id=None,
        target_url="https://hooks.example",
        payload_summary={},
        status="pending_review",
        attempt_count=0,
        http_status_code=None,
        error_message=None,
        policy_decision="human_review_required",
        confidence_at_dispatch=0.87,
        created_at=now,
        updated_at=now,
        completed_at=None,
    )

    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM action_execution_records" in sql:
            return _ScalarOneOrNoneResult(action)
        return _ScalarOneOrNoneResult(None)

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/actions/a3/approve",
                headers=_auth_headers(),
                json={"comment": "approved by ops"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] == "approved"
        assert body["action"]["policy_decision"] == "auto_approved"
        assert action.policy_decision == "auto_approved"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_reject_action_marks_denied():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    action = _FakeActionRecord(
        id="a4",
        organization_id="o1",
        session_id="s4",
        action_type="jira",
        connector_id=None,
        target_url="https://jira.example",
        payload_summary={},
        status="pending",
        attempt_count=0,
        http_status_code=None,
        error_message=None,
        policy_decision="human_review_required",
        confidence_at_dispatch=0.79,
        created_at=now,
        updated_at=now,
        completed_at=None,
    )

    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM action_execution_records" in sql:
            return _ScalarOneOrNoneResult(action)
        return _ScalarOneOrNoneResult(None)

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/actions/a4/reject",
                headers=_auth_headers(),
                json={"reason": "insufficient evidence"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] == "rejected"
        assert body["action"]["policy_decision"] == "denied"
        assert body["action"]["status"] == "denied"
        assert action.completed_at is not None
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_action_history():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    history_row = _FakeActionRecord(
        id="a5",
        organization_id="o1",
        session_id="s5",
        action_type="webhook",
        connector_id=None,
        target_url="https://hooks.example",
        payload_summary={},
        status="success",
        attempt_count=1,
        http_status_code=200,
        error_message=None,
        policy_decision="auto_approved",
        confidence_at_dispatch=0.94,
        created_at=now,
        updated_at=now,
        completed_at=now,
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM action_execution_records" in sql:
            return _ListResult([history_row])
        return _ListResult([])

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/actions/history", headers=_auth_headers())

        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 1
        assert rows[0]["id"] == "a5"
        assert rows[0]["status"] == "success"
    finally:
        app.dependency_overrides.clear()
