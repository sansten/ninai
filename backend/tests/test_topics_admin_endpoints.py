from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.security import create_access_token
from app.core.database import get_db
from app.main import app
from app.models.organization import Organization
from app.models.user import User
from app.models.memory_semantic_node import MemorySemanticNode
from app.models.memory_semantic_node_topic_history import MemorySemanticNodeTopicHistory
from app.models.memory_topic import MemoryTopic


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _auth_headers(*, org_id: str, user_id: str, roles: list[str]) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles)
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def topic_admin_session(test_engine) -> AsyncSession:
    session_factory = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def topic_admin_client(test_engine) -> AsyncClient:
    session_factory = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.pop(get_db, None)


async def _seed_org_user(session: AsyncSession, *, org_id: str, user_id: str) -> None:
    await session.execute(
        insert(Organization),
        {
            "id": org_id,
            "name": "Test Organization",
            "slug": f"test-org-{org_id[:8]}",
            "is_active": True,
        },
    )
    await session.execute(
        insert(User),
        {
            "id": user_id,
            "email": f"test-{user_id[:8]}@example.com",
            "hashed_password": "$2b$12$placeholder",
            "full_name": "Test User",
            "is_active": True,
            "role": "org_admin",
        },
    )


@pytest.mark.asyncio
async def test_topic_reassignment_ratio_admin(topic_admin_client, topic_admin_session):
    org_id = str(uuid4())
    user_id = str(uuid4())

    topic_id_1 = str(uuid4())
    topic_id_2 = str(uuid4())
    node_id_1 = str(uuid4())
    node_id_2 = str(uuid4())
    node_id_3 = str(uuid4())

    async with topic_admin_session.begin():
        await _seed_org_user(topic_admin_session, org_id=org_id, user_id=user_id)
        await topic_admin_session.execute(
            insert(MemoryTopic),
            [
                {
                    "id": topic_id_1,
                    "organization_id": org_id,
                    "scope": "personal",
                    "scope_id": None,
                    "scope_key": "personal:",
                    "label": "topic_a",
                    "label_normalized": "topic_a",
                    "keywords": ["alpha"],
                    "created_by": "system",
                },
                {
                    "id": topic_id_2,
                    "organization_id": org_id,
                    "scope": "personal",
                    "scope_id": None,
                    "scope_key": "personal:",
                    "label": "topic_b",
                    "label_normalized": "topic_b",
                    "keywords": ["beta"],
                    "created_by": "system",
                },
            ],
        )

        await topic_admin_session.execute(
            insert(MemorySemanticNode),
            [
                {
                    "id": node_id_1,
                    "organization_id": org_id,
                    "owner_id": user_id,
                    "scope": "personal",
                    "scope_id": None,
                    "content": "Node one",
                    "content_hash": _hash_content("Node one"),
                    "topic_id": topic_id_2,
                    "created_by": "system",
                },
                {
                    "id": node_id_2,
                    "organization_id": org_id,
                    "owner_id": user_id,
                    "scope": "personal",
                    "scope_id": None,
                    "content": "Node two",
                    "content_hash": _hash_content("Node two"),
                    "topic_id": topic_id_2,
                    "created_by": "system",
                },
                {
                    "id": node_id_3,
                    "organization_id": org_id,
                    "owner_id": user_id,
                    "scope": "personal",
                    "scope_id": None,
                    "content": "Node three",
                    "content_hash": _hash_content("Node three"),
                    "topic_id": topic_id_1,
                    "created_by": "system",
                },
            ],
        )

        await topic_admin_session.execute(
            insert(MemorySemanticNodeTopicHistory),
            [
                {
                    "id": str(uuid4()),
                    "organization_id": org_id,
                    "semantic_node_id": node_id_1,
                    "topic_id": topic_id_1,
                    "previous_topic_id": None,
                    "reason": "initial_attach",
                },
                {
                    "id": str(uuid4()),
                    "organization_id": org_id,
                    "semantic_node_id": node_id_1,
                    "topic_id": topic_id_2,
                    "previous_topic_id": topic_id_1,
                    "reason": "periodic_restructure",
                },
                {
                    "id": str(uuid4()),
                    "organization_id": org_id,
                    "semantic_node_id": node_id_2,
                    "topic_id": topic_id_1,
                    "previous_topic_id": None,
                    "reason": "initial_attach",
                },
                {
                    "id": str(uuid4()),
                    "organization_id": org_id,
                    "semantic_node_id": node_id_2,
                    "topic_id": topic_id_2,
                    "previous_topic_id": topic_id_1,
                    "reason": "periodic_restructure",
                },
                {
                    "id": str(uuid4()),
                    "organization_id": org_id,
                    "semantic_node_id": node_id_3,
                    "topic_id": topic_id_1,
                    "previous_topic_id": None,
                    "reason": "initial_attach",
                },
            ],
        )

    headers = _auth_headers(org_id=org_id, user_id=user_id, roles=["org_admin"])
    response = await topic_admin_client.get(
        "/api/v1/topics/admin/reassignment-ratio",
        headers=headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_nodes"] == 3
    assert payload["nodes_with_history"] == 3
    assert payload["nodes_reassigned"] == 2
    assert payload["reassignment_ratio"] == pytest.approx(2 / 3)
    assert payload["initial_only_ratio"] == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_topic_reassignment_ratio_requires_admin(topic_admin_client, topic_admin_session):
    org_id = str(uuid4())
    user_id = str(uuid4())
    async with topic_admin_session.begin():
        await _seed_org_user(topic_admin_session, org_id=org_id, user_id=user_id)
    headers = _auth_headers(org_id=org_id, user_id=user_id, roles=["member"])

    response = await topic_admin_client.get(
        "/api/v1/topics/admin/reassignment-ratio",
        headers=headers,
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_topic_restructure_admin(topic_admin_client, topic_admin_session, monkeypatch):
    org_id = str(uuid4())
    user_id = str(uuid4())
    async with topic_admin_session.begin():
        await _seed_org_user(topic_admin_session, org_id=org_id, user_id=user_id)

    async def _mock_restructure(self, *, organization_id: str, scope: str | None = None, scope_id: str | None = None):
        return {
            "reassignments": 2,
            "splits": 1,
            "merges": 0,
            "guidance_score_before": 0.4,
            "guidance_score_after": 0.7,
            "reassignment_ratio": 0.5,
        }

    monkeypatch.setattr(
        "app.services.topic_structure_service.TopicStructureService.periodic_restructure",
        _mock_restructure,
    )

    headers = _auth_headers(org_id=org_id, user_id=user_id, roles=["org_admin"])
    response = await topic_admin_client.post(
        "/api/v1/topics/admin/restructure",
        headers=headers,
        json={},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reassignments"] == 2
    assert payload["splits"] == 1
    assert payload["guidance_score_after"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_topic_restructure_requires_admin(topic_admin_client, topic_admin_session):
    org_id = str(uuid4())
    user_id = str(uuid4())
    async with topic_admin_session.begin():
        await _seed_org_user(topic_admin_session, org_id=org_id, user_id=user_id)
    headers = _auth_headers(org_id=org_id, user_id=user_id, roles=["member"])

    response = await topic_admin_client.post(
        "/api/v1/topics/admin/restructure",
        headers=headers,
        json={},
    )

    assert response.status_code == 403
