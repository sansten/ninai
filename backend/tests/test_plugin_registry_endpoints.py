from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.security import create_access_token
from app.main import app


def _admin_headers(*, org_id: str = "o1", user_id: str = "u_admin") -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=["org_admin"])
    return {"Authorization": f"Bearer {token}"}


def _member_headers(*, org_id: str = "o1", user_id: str = "u_member") -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=["member"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_plugin_registry_requires_org_admin():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/plugins", headers=_member_headers())
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_install_list_patch_logs_delete_plugin_happy_path():
    transport = ASGITransport(app=app)

    payload = {
        "name": "ninai-jira-connector",
        "version": "1.0.0",
        "description": "Sync Jira issues into Ninai cognitive memory in real time",
        "capabilities": ["inbound_connector", "entity_enricher"],
        "entrypoint": "ninai_jira.plugin:JiraPlugin",
        "config_schema": {
            "jira_url": {"type": "string", "required": True},
            "jira_token": {"type": "string", "required": True, "secret": True},
        },
        "events_emitted": ["memory.created", "memory.updated"],
        "events_consumed": ["goal.created"],
    }

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        install_resp = await ac.post("/api/v1/plugins/install", headers=_admin_headers(), json=payload)
        assert install_resp.status_code == 201
        body = install_resp.json()
        assert body["installed"] is True
        assert body["plugin"]["name"] == "ninai-jira-connector"
        assert body["plugin"]["entrypoint"] == "ninai_jira.plugin:JiraPlugin"

        list_resp = await ac.get("/api/v1/plugins", headers=_admin_headers())
        assert list_resp.status_code == 200
        listed = list_resp.json()
        assert listed["total"] >= 1
        assert any(p["name"] == "ninai-jira-connector" for p in listed["plugins"])

        patch_resp = await ac.patch(
            "/api/v1/plugins/ninai-jira-connector/config",
            headers=_admin_headers(),
            json={"config": {"jira_url": "https://jira.company.com", "project": "OPS"}},
        )
        assert patch_resp.status_code == 200
        patched = patch_resp.json()
        assert patched["updated"] is True
        assert patched["plugin"]["config"]["project"] == "OPS"

        logs_resp = await ac.get("/api/v1/plugins/ninai-jira-connector/logs", headers=_admin_headers())
        assert logs_resp.status_code == 200
        logs = logs_resp.json()
        assert logs["name"] == "ninai-jira-connector"
        assert logs["total"] >= 2

        delete_resp = await ac.delete("/api/v1/plugins/ninai-jira-connector", headers=_admin_headers())
        assert delete_resp.status_code == 200
        deleted = delete_resp.json()
        assert deleted["deleted"] is True
        assert deleted["name"] == "ninai-jira-connector"


@pytest.mark.asyncio
async def test_install_requires_name_version_entrypoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/v1/plugins/install",
            headers=_admin_headers(),
            json={"name": "", "version": "", "entrypoint": ""},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_patch_config_requires_object_and_404_for_missing_plugin():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        bad_resp = await ac.patch(
            "/api/v1/plugins/missing/config",
            headers=_admin_headers(),
            json={"config": "not-an-object"},
        )
        assert bad_resp.status_code == 422

        missing_resp = await ac.patch(
            "/api/v1/plugins/missing/config",
            headers=_admin_headers(),
            json={"config": {"x": 1}},
        )
        assert missing_resp.status_code == 404


@pytest.mark.asyncio
async def test_logs_and_delete_return_404_for_missing_plugin():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        logs_resp = await ac.get("/api/v1/plugins/does-not-exist/logs", headers=_admin_headers())
        assert logs_resp.status_code == 404

        delete_resp = await ac.delete("/api/v1/plugins/does-not-exist", headers=_admin_headers())
        assert delete_resp.status_code == 404
