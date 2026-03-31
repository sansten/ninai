from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["member"])
    return {"Authorization": f"Bearer {token}"}


def _mock_db_session() -> AsyncMock:
    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)
    return session


@pytest.mark.asyncio
async def test_compute_proof_scorecard_endpoint_happy_path():
    session = _mock_db_session()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    payload = {
        "baseline": {
            "lead_time_hours": 10.0,
            "mttr_hours": 8.0,
            "false_escalation_rate": 0.2,
        },
        "records": [
            {
                "incident_id": "inc-1",
                "lead_time_hours": 7.0,
                "mttr_hours": 6.0,
                "avoided_sla_breach": True,
                "false_escalation": False,
            },
            {
                "incident_id": "inc-2",
                "lead_time_hours": 8.0,
                "mttr_hours": 6.5,
                "avoided_sla_breach": False,
                "false_escalation": False,
            },
        ],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/v1/proof/scorecard", json=payload, headers=_auth_headers())

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["incidents_count"] == 2
    assert data["score"] >= 0
    assert len(data["reproducibility_hash"]) == 32


@pytest.mark.asyncio
async def test_compute_monthly_impact_endpoint_happy_path():
    session = _mock_db_session()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    payload = {
        "month": "2026-03",
        "baseline": {
            "lead_time_hours": 10.0,
            "mttr_hours": 8.0,
            "false_escalation_rate": 0.25,
            "sla_penalty_per_breach": 1000.0,
        },
        "monthly_operating_cost": 1000.0,
        "records": [
            {
                "incident_id": "inc-1",
                "lead_time_hours": 7.0,
                "mttr_hours": 5.0,
                "avoided_sla_breach": True,
                "false_escalation": False,
            },
            {
                "incident_id": "inc-2",
                "lead_time_hours": 8.0,
                "mttr_hours": 6.0,
                "avoided_sla_breach": True,
                "false_escalation": False,
            },
        ],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/v1/proof/monthly-impact", json=payload, headers=_auth_headers(org_id="o-proof"))

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["tenant_id"] == "o-proof"
    assert data["month"] == "2026-03"
    assert data["net_impact"] > 0
    assert len(data["reproducibility_hash"]) == 32


@pytest.mark.asyncio
async def test_proof_endpoints_require_authentication():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/v1/proof/scorecard", json={"baseline": {}, "records": []})

    assert resp.status_code == 401