from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


@dataclass
class _FakeMemory:
    id: str
    title: str
    content_preview: str
    business_domain: str
    extra_metadata: dict[str, Any]
    tags: list[str]
    created_at: datetime
    is_active: bool = True


class _Scalars:
    def __init__(self, items: list[Any]):
        self._items = items

    def all(self):
        return self._items


class _ListResult:
    def __init__(self, items: list[Any]):
        self._items = items

    def scalars(self):
        return _Scalars(self._items)


def _admin_headers(*, org_id: str = "o1", user_id: str = "u_admin") -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=["org_admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_graphql_requires_authentication():
    transport = ASGITransport(app=app)
    query = {"query": "query { goals { id title } }"}

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/graphql", json=query)

    assert resp.status_code in (200, 401)
    body = resp.json()
    assert "errors" in body or body.get("detail") == "Authentication required"


@pytest.mark.asyncio
async def test_graphql_search_memory_happy_path():
    now = datetime(2026, 4, 4, tzinfo=timezone.utc)
    fake_memory = _FakeMemory(
        id="m1",
        title="Auth outage",
        content_preview="Auth service returned 500 after deploy.",
        business_domain="incident",
        extra_metadata={"credibility_score": 0.88, "freshness_score": 0.73, "conflict_count": 2},
        tags=["incident", "auth"],
        created_at=now,
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM memory_metadata" in sql:
            return _ListResult([fake_memory])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    gql_query = {
        "query": """
            query SearchMemory($q: String!) {
              searchMemory(query: $q, limit: 5) {
                id
                content
                domain
                credibilityScore
                decayScore
                tags
                conflicts {
                  conflictType
                  severity
                }
              }
            }
        """,
        "variables": {"q": "auth"},
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/graphql", headers=_admin_headers(), json=gql_query)

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert "errors" not in payload

    items = payload["data"]["searchMemory"]
    assert len(items) == 1
    assert items[0]["id"] == "m1"
    assert items[0]["domain"] == "incident"
    assert items[0]["credibilityScore"] == pytest.approx(0.88)
    assert items[0]["decayScore"] == pytest.approx(0.73)
    assert len(items[0]["conflicts"]) == 1
