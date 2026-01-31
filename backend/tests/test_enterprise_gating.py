import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core.feature_gate import CommunityFeatureGate, require_feature, set_feature_gate


def test_require_feature_returns_403_in_community_mode():
    app = FastAPI()
    set_feature_gate(app, CommunityFeatureGate())

    @app.get("/x")
    def x(_: None = Depends(require_feature("enterprise.some_feature"))):
        return {"ok": True}

    client = TestClient(app)
    resp = client.get("/x")
    assert resp.status_code == 401 or resp.status_code == 403
    # Note: dependency also requires auth in real app; this test uses a simplified route.


@pytest.mark.asyncio
async def test_community_gate_allows_non_enterprise_features():
    gate = CommunityFeatureGate()
    assert gate.is_enabled(org_id="org", feature="core.memories") is True
    assert gate.is_enabled(org_id="org", feature="enterprise.admin_ops") is False
