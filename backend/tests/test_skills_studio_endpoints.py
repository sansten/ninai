import pytest

from app.core.security import create_access_token


def _headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["org_admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_skills_studio_get_update_publish_flow(pg_client, auth_headers):
    # Initial fetch
    res = await pg_client.get("/api/v1/admin/skills-studio", headers=auth_headers)
    assert res.status_code == 200
    body = res.json()

    assert body["total_agents"] >= 10
    assert len(body["core_agents"]) >= 1
    assert len(body["non_core_agents"]) >= 1

    target = body["non_core_agents"][0]["agent_name"]

    # Update draft skill for a non-core agent
    update_res = await pg_client.put(
        f"/api/v1/admin/skills-studio/{target}",
        headers=auth_headers,
        json={
            "enabled": True,
            "instructions": "Use concise domain language for support analysts.",
            "parameters": {"tone": "technical", "max_examples": 2},
        },
    )
    assert update_res.status_code == 200
    update_body = update_res.json()
    assert update_body["agent_name"] == target
    assert update_body["skill"]["enabled"] is True
    assert "support analysts" in update_body["skill"]["instructions"]

    # Publish all drafts
    publish_res = await pg_client.post(
        "/api/v1/admin/skills-studio/publish",
        headers=auth_headers,
    )
    assert publish_res.status_code == 200
    publish_body = publish_res.json()
    assert publish_body["status"] == "published"
    assert publish_body["published_count"] >= 1

    # Verify published snapshot is visible
    final_res = await pg_client.get("/api/v1/admin/skills-studio", headers=auth_headers)
    assert final_res.status_code == 200
    final_body = final_res.json()
    row = next((r for r in final_body["non_core_agents"] if r["agent_name"] == target), None)
    assert row is not None
    assert row["published_skill"]["enabled"] is True
    assert row["published_skill"]["parameters"]["tone"] == "technical"


@pytest.mark.asyncio
async def test_skills_studio_rejects_core_agent_edit(pg_client, auth_headers):
    res = await pg_client.get("/api/v1/admin/skills-studio", headers=auth_headers)
    assert res.status_code == 200
    core_agent = res.json()["core_agents"][0]["agent_name"]

    update_res = await pg_client.put(
        f"/api/v1/admin/skills-studio/{core_agent}",
        headers=auth_headers,
        json={"enabled": True, "instructions": "should fail"},
    )
    assert update_res.status_code == 400
    assert "Core agents" in update_res.json()["detail"]


@pytest.mark.asyncio
async def test_skills_studio_developer_submit_admin_approve_and_rollback(pg_client):
    admin_headers = _headers(org_id="o1", user_id="00000000-0000-0000-0000-000000000101", roles=["org_admin"])
    developer_headers = _headers(org_id="o1", user_id="00000000-0000-0000-0000-000000000102", roles=["developer"])

    initial = await pg_client.get("/api/v1/admin/skills-studio", headers=developer_headers)
    assert initial.status_code == 200
    row = initial.json()["non_core_agents"][0]
    target = row["agent_name"]

    # Developer updates and submits draft.
    update_res = await pg_client.put(
        f"/api/v1/admin/skills-studio/{target}",
        headers=developer_headers,
        json={
            "enabled": True,
            "instructions": "Developer-authored instructions",
            "parameters": {"mode": "developer"},
        },
    )
    assert update_res.status_code == 200

    submit_res = await pg_client.post(
        f"/api/v1/admin/skills-studio/{target}/submit",
        headers=developer_headers,
    )
    assert submit_res.status_code == 200
    assert submit_res.json()["status"] == "submitted"

    # Developer cannot approve.
    dev_approve = await pg_client.post(
        f"/api/v1/admin/skills-studio/{target}/approve",
        headers=developer_headers,
    )
    assert dev_approve.status_code == 403

    # Admin approves submission -> next version should be v2.
    approve_res = await pg_client.post(
        f"/api/v1/admin/skills-studio/{target}/approve",
        headers=admin_headers,
    )
    assert approve_res.status_code == 200
    assert approve_res.json()["status"] == "approved"
    assert approve_res.json()["version"] == "v2"

    post_approve = await pg_client.get("/api/v1/admin/skills-studio", headers=admin_headers)
    assert post_approve.status_code == 200
    approved_row = next(
        (r for r in post_approve.json()["non_core_agents"] if r["agent_name"] == target),
        None,
    )
    assert approved_row is not None
    assert approved_row["active_version"] == "v2"
    assert approved_row["published_skill"]["parameters"]["mode"] == "developer"

    # Admin can rollback to baseline v1 and gets new active version label.
    rollback_res = await pg_client.post(
        f"/api/v1/admin/skills-studio/{target}/rollback",
        headers=admin_headers,
        json={"target_version": "v1"},
    )
    assert rollback_res.status_code == 200
    assert rollback_res.json()["status"] == "rolled_back"
    assert rollback_res.json()["target_version"] == "v1"
    assert rollback_res.json()["active_version"] == "v3"

    final_res = await pg_client.get("/api/v1/admin/skills-studio", headers=admin_headers)
    assert final_res.status_code == 200
    final_row = next(
        (r for r in final_res.json()["non_core_agents"] if r["agent_name"] == target),
        None,
    )
    assert final_row is not None
    assert final_row["active_version"] == "v3"
    assert final_row["published_skill"]["instructions"] == ""
    assert final_row["published_skill"]["parameters"] == {}
