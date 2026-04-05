from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.main import app


def _auth_headers(*, org_id: str, user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["org_admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_cognitive_namespace_rejects_x_org_id_mismatch():
    headers = _auth_headers(org_id="org-token")
    headers["X-Org-Id"] = "org-header"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/v1/cognitive/gateway/write", headers=headers, json={"content": "test"})

    assert resp.status_code == 403
    assert resp.json()["detail"] == "Org namespace mismatch"


@pytest.mark.asyncio
async def test_cognitive_namespace_sets_response_header_for_cognitive_paths():
    headers = _auth_headers(org_id="org-abc")

    with patch("app.api.v1.endpoints.cognitive_gateway.CognitiveGatewayService.write", AsyncMock(return_value=type("R", (), {
        "memory_id": "m1",
        "enriched": True,
        "enrichment_summary": {},
        "tags": [],
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    })())):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/v1/cognitive/gateway/write", headers=headers, json={"content": "test"})

    assert resp.status_code == 200
    assert resp.headers.get("X-Ninai-Org-Id") == "org-abc"
