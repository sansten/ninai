from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.main import app
from app.api.v1.endpoints import meta_agent as meta_endpoints


@pytest.mark.asyncio
async def test_enqueue_meta_review_memory_calls_celery_delay(client: AsyncClient, auth_headers: dict, monkeypatch):
    memory_id = str(uuid4())

    monkeypatch.setattr(meta_endpoints, "set_tenant_context", AsyncMock())

    delay_mock = MagicMock(return_value=SimpleNamespace(id="job-1"))
    monkeypatch.setattr(meta_endpoints.meta_review_memory_task, "delay", delay_mock)

    resp = await client.post(f"/api/v1/meta/review/memories/{memory_id}", headers=auth_headers)
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "queued"
    assert data["task_id"] == "job-1"

    delay_mock.assert_called_once()
    kwargs = delay_mock.call_args.kwargs
    assert kwargs["memory_id"] == memory_id


@pytest.mark.asyncio
async def test_update_calibration_profile_requires_admin(client: AsyncClient, test_org_id: str, test_user_id: str):
    from app.core.security import create_access_token

    token = create_access_token(user_id=test_user_id, org_id=test_org_id, roles=["viewer"])
    headers = {"Authorization": f"Bearer {token}"}

    resp = await client.put(
        "/api/v1/meta/calibration-profile",
        headers=headers,
        json={"signal_weights": {"w_agent_confidence": 1.0, "w_evidence_strength": 0.0, "w_historical_accuracy": 0.0, "w_consistency_score": 0.0, "w_contradiction_penalty": 0.0}},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_meta_metrics_uses_counts(client: AsyncClient, auth_headers: dict, monkeypatch):
    monkeypatch.setattr(meta_endpoints, "set_tenant_context", AsyncMock())

    # db is provided via dependency override in conftest (AsyncMock spec=AsyncSession)
    # Endpoint executes: open_conflicts, runs_total, escalations, drift_incidents, avg_conf
    res1 = MagicMock(); res1.scalar_one.return_value = 2
    res2 = MagicMock(); res2.scalar_one.return_value = 8
    res3 = MagicMock(); res3.scalar_one.return_value = 1
    res4 = MagicMock(); res4.scalar_one.return_value = 3
    res5 = MagicMock(); res5.scalar_one.return_value = 0.5

    # Patch the injected db execute behavior via dependency override function closure
    # We patch get_db in endpoint module scope by overriding dependency.
    async def override_get_db():
        db = AsyncMock()
        db.execute = AsyncMock(side_effect=[res1, res2, res3, res4, res5])
        yield db

    app.dependency_overrides[meta_endpoints.get_db] = override_get_db

    try:
        resp = await client.get("/api/v1/meta/metrics", headers=auth_headers)
    finally:
        app.dependency_overrides.pop(meta_endpoints.get_db, None)

    assert resp.status_code == 200
    data = resp.json()
    assert "org_id" in data
    assert data["avg_confidence_by_org"] == 0.5
    assert data["drift_incidents"] == 3
    assert data["conflict_rate"] == 2 / 8
    assert data["escalation_rate"] == 1 / 8
