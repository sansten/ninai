from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app
from app.services.cognitive_autonomy_control_service import get_cognitive_autonomy_control_service
from app.tasks.cognitive_heartbeat import cognitive_heartbeat_task
from app.tasks.cognitive_loop import cognitive_loop_task


class _ScalarOneOrNoneResult:
    def __init__(self, item):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


class _FakeSession:
    def __init__(
        self,
        *,
        session_id: str,
        org_id: str,
        user_id: str,
        agent_id: str = "cognitive_loop",
        trace_id: str | None = None,
    ):
        now = datetime(2026, 1, 24, tzinfo=timezone.utc)
        self.id = session_id
        self.organization_id = org_id
        self.user_id = user_id
        self.agent_id = agent_id
        self.status = "running"
        self.goal = "g"
        self.goal_id = None
        self.context_snapshot = {}
        self.created_at = now
        self.updated_at = now
        self.trace_id = trace_id


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["org_admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_admin_cognitive_autonomy_settings_roundtrip():
    svc = get_cognitive_autonomy_control_service()
    svc.reset()

    session = AsyncMock(spec=AsyncSession)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp0 = await ac.get(
                "/api/v1/admin/cognitive-autonomy",
                headers=_auth_headers(org_id="org-a", user_id="admin-a"),
            )
            assert resp0.status_code == 200
            assert resp0.json()["effective"]["enabled"] is True

            resp1 = await ac.put(
                "/api/v1/admin/cognitive-autonomy",
                headers=_auth_headers(org_id="org-a", user_id="admin-a"),
                json={"enabled": False, "reason": "maintenance_window"},
            )
            assert resp1.status_code == 200
            body1 = resp1.json()
            assert body1["org_config"]["enabled"] is False
            assert body1["effective"]["enabled"] is False
            assert body1["effective"]["reason"] == "maintenance_window"

            resp2 = await ac.put(
                "/api/v1/admin/cognitive-autonomy",
                headers=_auth_headers(org_id="org-a", user_id="admin-a"),
                json={"global_enabled": False, "global_reason": "incident_freeze"},
            )
            assert resp2.status_code == 200
            body2 = resp2.json()
            assert body2["global_config"]["enabled"] is False
            assert body2["effective"]["enabled"] is False
            assert body2["effective"]["reason"] == "incident_freeze"
    finally:
        app.dependency_overrides.clear()
        svc.reset()


@pytest.mark.asyncio
async def test_run_cognitive_session_denied_when_autonomy_disabled():
    svc = get_cognitive_autonomy_control_service()
    svc.reset()
    svc.set_org("o1", enabled=False, reason="manual_gate")

    fake = _FakeSession(session_id="s1", org_id="o1", user_id="u1")
    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM cognitive_sessions" in sql:
            return _ScalarOneOrNoneResult(fake)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/cognitive/sessions/s1/run",
                headers=_auth_headers(org_id="o1", user_id="u1", roles=["org_admin"]),
            )
        assert resp.status_code == 403
        detail = resp.json()["detail"]
        assert detail["error"] == "cognitive_autonomy_disabled"
        assert detail["reason"] == "manual_gate"
    finally:
        app.dependency_overrides.clear()
        svc.reset()


def test_cognitive_loop_task_blocked_by_autonomy_control():
    svc = get_cognitive_autonomy_control_service()
    svc.reset()
    svc.set_org("org-task", enabled=False, reason="ops_freeze")

    try:
        status = cognitive_loop_task(
            org_id="org-task",
            session_id="s-test",
            initiator_user_id="u-test",
        )
        assert status == "blocked_by_autonomy_control"
    finally:
        svc.reset()


def test_cognitive_heartbeat_skips_when_autonomy_disabled():
    svc = get_cognitive_autonomy_control_service()
    svc.reset()
    svc.set_org("org-heartbeat", enabled=False, reason="org_freeze")

    try:
        result = cognitive_heartbeat_task(org_id="org-heartbeat")
        assert result["status"] == "ok"
        assert result["processed_orgs"] == 1
        first = result["results"][0]
        assert first["status"] == "skipped"
        assert first["reason"] == "org_freeze"
    finally:
        svc.reset()


@pytest.mark.asyncio
async def test_org_level_autonomy_toggle_is_tenant_isolated():
    svc = get_cognitive_autonomy_control_service()
    svc.reset()

    session = AsyncMock(spec=AsyncSession)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp1 = await ac.put(
                "/api/v1/admin/cognitive-autonomy",
                headers=_auth_headers(org_id="org-a", user_id="admin-a"),
                json={"enabled": False, "reason": "maintenance_org_a"},
            )
            assert resp1.status_code == 200
            assert resp1.json()["effective"]["enabled"] is False

            # Different org should remain enabled by default.
            resp2 = await ac.get(
                "/api/v1/admin/cognitive-autonomy",
                headers=_auth_headers(org_id="org-b", user_id="admin-b"),
            )
            assert resp2.status_code == 200
            body2 = resp2.json()
            assert body2["effective"]["enabled"] is True
            assert body2["org_config"] is None
    finally:
        app.dependency_overrides.clear()
        svc.reset()


@pytest.mark.asyncio
async def test_cognitive_session_run_block_applies_only_to_target_org(monkeypatch):
    svc = get_cognitive_autonomy_control_service()
    svc.reset()
    svc.set_org("org-a", enabled=False, reason="org_a_only")

    fake_b = _FakeSession(session_id="s2", org_id="org-b", user_id="u2")
    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM cognitive_sessions" in sql:
            return _ScalarOneOrNoneResult(fake_b)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    fake_proc = MagicMock()
    fake_proc.id = "proc-123"
    monkeypatch.setattr(
        "app.api.v1.endpoints.cognitive_loop.AgentSchedulerService.enqueue",
        AsyncMock(return_value=fake_proc),
    )
    delay_mock = MagicMock()
    monkeypatch.setattr("app.api.v1.endpoints.cognitive_loop.cognitive_loop_task.delay", delay_mock)

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # org-b should not be blocked by org-a toggle
            resp = await ac.post(
                "/api/v1/cognitive/sessions/s2/run",
                headers=_auth_headers(org_id="org-b", user_id="u2", roles=["org_admin"]),
            )
        assert resp.status_code == 202
        assert resp.json()["queued"] is True
        delay_mock.assert_called_once()
    finally:
        app.dependency_overrides.clear()
        svc.reset()
