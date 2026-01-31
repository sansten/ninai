import pytest


@pytest.mark.asyncio
async def test_observability_persists(pg_client, auth_headers):
    # Fetch defaults
    res = await pg_client.get("/api/v1/admin/observability", headers=auth_headers)
    assert res.status_code == 200

    # Update log config and ensure persistence
    update_body = {
        "log_config": {
            "services": [
                {"service": "api", "module": "handlers", "level": "ERROR"},
            ]
        }
    }
    res = await pg_client.put(
        "/api/v1/admin/observability/log-config",
        json=update_body,
        headers=auth_headers,
    )
    assert res.status_code == 200

    res = await pg_client.get("/api/v1/admin/observability", headers=auth_headers)
    assert res.status_code == 200
    payload = res.json()
    assert payload["log_config"]["services"][0]["level"] == "ERROR"


@pytest.mark.asyncio
async def test_alert_notifications_persist(pg_client, auth_headers):
    create_body = {
        "name": "Policy rollout alert",
        "enabled": True,
        "channel": "email",
        "recipients": ["ops@example.com"],
        "conditions": {"event_type": "policy_rollout", "severity": "error"},
    }
    res = await pg_client.post(
        "/api/v1/admin/alerts/notifications", json=create_body, headers=auth_headers
    )
    assert res.status_code == 200
    alert = res.json()

    res = await pg_client.get(
        "/api/v1/admin/alerts/notifications", headers=auth_headers
    )
    assert res.status_code == 200
    items = res.json()["notifications"]
    assert any(item["id"] == alert["id"] for item in items)

    res = await pg_client.put(
        f"/api/v1/admin/alerts/notifications/{alert['id']}",
        json={"enabled": False},
        headers=auth_headers,
    )
    assert res.status_code == 200

    res = await pg_client.get(
        "/api/v1/admin/alerts/notifications", headers=auth_headers
    )
    assert res.status_code == 200
    updated = next(item for item in res.json()["notifications"] if item["id"] == alert["id"])
    assert updated["enabled"] is False


@pytest.mark.asyncio
async def test_alert_notification_test_endpoint_logs_mode(pg_client, auth_headers):
    res = await pg_client.post(
        "/api/v1/admin/alerts/notifications/test",
        json={"channel": "email", "recipients": ["dev@example.com"]},
        headers=auth_headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["delivery"]["mode"] == "log"
