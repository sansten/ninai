"""Tests for APIVersionHeaderMiddleware."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_x_api_version_header_present(client: AsyncClient):
    r = await client.get("/api/v1/health")
    assert "x-api-version" in r.headers


@pytest.mark.asyncio
async def test_x_api_version_value_is_current(client: AsyncClient):
    from app.core.config import settings
    r = await client.get("/api/v1/health")
    assert r.headers["x-api-version"] == settings.CURRENT_API_VERSION
