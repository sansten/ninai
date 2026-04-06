import pytest


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
