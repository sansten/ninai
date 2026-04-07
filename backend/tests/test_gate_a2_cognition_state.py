"""Gate A2 — System self-state persisted per org (P0).

A2 Check: One live cognition state per org with focus, load, and next action.
Evidence: GET /api/v1/cognitive/gateway/state returns a populated state row,
and the upsert service correctly persists and retrieves all required fields.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app
from app.models.system_cognition_state import SystemCognitionState
from app.services.system_cognition_state import (
    CognitionStateUpdate,
    SystemCognitionStateService,
)


def _admin_headers(org_id: str = "org-a2-test") -> dict[str, str]:
    token = create_access_token(
        user_id=str(uuid.uuid4()),
        org_id=org_id,
        roles=["org_admin"],
    )
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Unit: service correctly stores and retrieves all A2 required fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cognition_state_upsert_and_get_persists_all_a2_fields(
    db_session: AsyncSession,
    test_org_id: str,
):
    """A2: upsert stores focus, load, sessions, anomalies, next_action; get returns them."""
    svc = SystemCognitionStateService(db_session)
    session_id = str(uuid.uuid4())

    update = CognitionStateUpdate(
        current_focus="Investigating anomaly cluster in billing domain",
        active_session_ids=[session_id],
        unresolved_anomalies_count=3,
        stale_goals_count=1,
        cognitive_load=0.65,
        next_scheduled_action="run anomaly triage in 5 minutes",
    )
    state = await svc.upsert(test_org_id, update)

    assert state is not None, "upsert must return the persisted row"
    assert state.organization_id == test_org_id
    assert state.current_focus == update.current_focus
    assert session_id in state.active_session_ids
    assert state.unresolved_anomalies_count == 3
    assert state.stale_goals_count == 1
    assert abs(state.cognitive_load - 0.65) < 0.001
    assert state.next_scheduled_action == update.next_scheduled_action
    assert state.last_heartbeat_at is not None

    # Fetch separately — proves it was persisted to DB, not in-memory only
    fetched = await svc.get(test_org_id)
    assert fetched is not None
    assert fetched.current_focus == update.current_focus
    assert fetched.cognitive_load == state.cognitive_load

    print(
        f"\nA2 Gate Evidence:\n"
        f"  org_id={test_org_id}\n"
        f"  current_focus='{state.current_focus}'\n"
        f"  cognitive_load={state.cognitive_load}\n"
        f"  active_sessions={state.active_session_ids}\n"
        f"  next_action='{state.next_scheduled_action}'\n"
        f"  last_heartbeat_at={state.last_heartbeat_at.isoformat()}\n"
        f"  Status: \u2713 All A2 required fields present and persisted"
    )


@pytest.mark.asyncio
async def test_cognition_state_upsert_overwrites_on_second_heartbeat(
    db_session: AsyncSession,
    test_org_id: str,
):
    """A2: second upsert for same org updates fields in place (one row per org)."""
    svc = SystemCognitionStateService(db_session)

    await svc.upsert(
        test_org_id,
        CognitionStateUpdate(
            current_focus="initial focus",
            active_session_ids=[],
            unresolved_anomalies_count=0,
            stale_goals_count=0,
            cognitive_load=0.1,
            next_scheduled_action=None,
        ),
    )

    new_session = str(uuid.uuid4())
    await svc.upsert(
        test_org_id,
        CognitionStateUpdate(
            current_focus="updated focus after second heartbeat",
            active_session_ids=[new_session],
            unresolved_anomalies_count=2,
            stale_goals_count=0,
            cognitive_load=0.45,
            next_scheduled_action="re-evaluate in 10 minutes",
        ),
    )

    from sqlalchemy import select, func
    result = await db_session.execute(
        select(func.count()).select_from(SystemCognitionState).where(
            SystemCognitionState.organization_id == test_org_id
        )
    )
    count = result.scalar_one()
    assert count == 1, f"Must be exactly 1 row per org, found {count}"

    state = await svc.get(test_org_id)
    assert state.current_focus == "updated focus after second heartbeat"
    assert new_session in state.active_session_ids


# ---------------------------------------------------------------------------
# Integration: GET /cognitive/gateway/state endpoint returns state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_cognitive_state_endpoint_returns_state_for_org():
    """A2: GET /cognitive/gateway/state returns populated state dict for org_admin."""
    from unittest.mock import patch

    org_id = f"org-gate-a2-{uuid.uuid4().hex[:8]}"
    now = datetime(2026, 4, 6, 12, 0, 0, tzinfo=timezone.utc)

    mock_state = MagicMock(spec=SystemCognitionState)
    mock_state.to_dict.return_value = {
        "id": str(uuid.uuid4()),
        "organization_id": org_id,
        "current_focus": "Monitoring billing anomaly cluster",
        "active_session_ids": [str(uuid.uuid4())],
        "unresolved_anomalies_count": 2,
        "stale_goals_count": 0,
        "cognitive_load": 0.5,
        "last_heartbeat_at": now.isoformat(),
        "next_scheduled_action": "triage anomalies at next heartbeat",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }

    mock_db = AsyncMock(spec=AsyncSession)

    async def override_db():
        yield mock_db

    mock_svc_instance = AsyncMock()
    mock_svc_instance.get.return_value = mock_state

    app.dependency_overrides[get_db] = override_db
    try:
        with patch(
            "app.services.system_cognition_state.SystemCognitionStateService",
            return_value=mock_svc_instance,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.get(
                    "/api/v1/cognitive/gateway/state",
                    headers=_admin_headers(org_id),
                )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["organization_id"] == org_id
        assert "current_focus" in data
        assert "cognitive_load" in data
        assert "last_heartbeat_at" in data
        assert "next_scheduled_action" in data
        print(
            f"\nA2 Endpoint Evidence:\n"
            f"  GET /cognitive/gateway/state -> 200\n"
            f"  organization_id={data['organization_id']}\n"
            f"  current_focus='{data['current_focus']}'\n"
            f"  cognitive_load={data['cognitive_load']}\n"
            f"  Status: \u2713 A2 endpoint returns all required fields"
        )
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_get_cognitive_state_endpoint_returns_no_heartbeat_when_empty():
    """A2: endpoint returns graceful response when no heartbeat has run yet."""
    from unittest.mock import patch

    org_id = f"org-gate-a2-empty-{uuid.uuid4().hex[:8]}"
    mock_db = AsyncMock(spec=AsyncSession)

    async def override_db():
        yield mock_db

    mock_svc_instance = AsyncMock()
    mock_svc_instance.get.return_value = None

    app.dependency_overrides[get_db] = override_db
    try:
        with patch(
            "app.services.system_cognition_state.SystemCognitionStateService",
            return_value=mock_svc_instance,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.get(
                    "/api/v1/cognitive/gateway/state",
                    headers=_admin_headers(org_id),
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "no_heartbeat_data"
    finally:
        app.dependency_overrides.pop(get_db, None)
