from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_cognitive_manifest_returns_live_deployment_document():
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/.well-known/cognitive-manifest.json")

    assert resp.status_code == 200
    payload = resp.json()

    assert payload["name"]
    assert payload["version"] == "1.0.0"
    assert 20 in payload["deployed_phases"]
    assert payload["event_stream"]["websocket"] == "/ws/stream"
    assert payload["event_stream"]["sse"] == "/sse/events"
    assert payload["integrations"]["mcp"] is True
    assert payload["integrations"]["a2a"] is True
    assert payload["integrations"]["langchain"] is True
    assert payload["integrations"]["llamaindex"] is True
    assert payload["integrations"]["crewai"] is True
    assert payload["integrations"]["openai_tools"] is True
    assert "semantic_search" in payload["cognitive_capabilities"]
    assert any(agent["name"] == "ClassificationAgent" for agent in payload["active_agents"])
    assert any(agent["name"] == "ConflictDetectionAgent" for agent in payload["active_agents"])


@pytest.mark.asyncio
async def test_cognitive_manifest_agent_entries_include_runtime_metadata():
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/.well-known/cognitive-manifest.json")

    assert resp.status_code == 200
    agents = resp.json()["active_agents"]

    conflict_agent = next(agent for agent in agents if agent["name"] == "ConflictDetectionAgent")
    assert conflict_agent["status"] == "active"
    assert conflict_agent["capability"] == "conflict_detection"
    assert conflict_agent["phase"] == 13
    assert isinstance(conflict_agent["dependencies"], list)