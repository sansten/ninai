from __future__ import annotations

import os
from uuid import uuid4

import pytest

from app.core.database import set_tenant_context
from app.models.memory import MemoryMetadata
from app.models.organization import Organization
from app.models.user import User
from app.services.goal_service import GoalService


RUN_POSTGRES_TESTS = os.environ.get("RUN_POSTGRES_TESTS", "").lower() in {"1", "true", "yes"}
requires_postgres = pytest.mark.skipif(not RUN_POSTGRES_TESTS, reason="Set RUN_POSTGRES_TESTS=1 to run")


@requires_postgres
@pytest.mark.asyncio
async def test_goalgraph_create_node_link_and_idempotent_link(db_session, test_org_id: str, test_user_id: str):
    memory_id = str(uuid4())

    db_session.add(
        Organization(
            id=test_org_id,
            name="Test Org",
            slug=f"test-org-{uuid4().hex[:10]}",
            is_active=True,
            settings={},
        )
    )
    db_session.add(
        User(
            id=test_user_id,
            email=f"user-{uuid4().hex[:10]}@example.com",
            hashed_password="not-a-real-hash",
            full_name="Test User",
            is_active=True,
            is_superuser=False,
            clearance_level=0,
            preferences={},
        )
    )
    await db_session.flush()

    db_session.add(
        MemoryMetadata(
            id=memory_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            scope_id=None,
            memory_type="long_term",
            classification="internal",
            required_clearance=0,
            title="t",
            content_preview="preview",
            content_hash="0" * 64,
            tags=["progress"],
            entities={},
            extra_metadata={},
            source_type="manual",
            source_id=None,
            vector_id=f"vec-{uuid4().hex}",
            embedding_model="test",
        )
    )
    await db_session.flush()

    await set_tenant_context(db_session, test_user_id, test_org_id, roles="org_admin", clearance_level=0)

    svc = GoalService(db_session)
    goal = await svc.create_goal(
        org_id=test_org_id,
        created_by_user_id=test_user_id,
        payload={
            "owner_type": "user",
            "owner_id": test_user_id,
            "title": "Ship X",
            "description": None,
            "goal_type": "project",
            "status": "active",
            "priority": 2,
            "due_at": None,
            "confidence": 0.5,
            "visibility_scope": "personal",
            "scope_id": None,
            "tags": ["progress"],
            "metadata": {},
        },
    )

    node = await svc.add_node(
        org_id=test_org_id,
        goal_id=goal.id,
        actor_user_id=test_user_id,
        payload={
            "parent_node_id": None,
            "node_type": "task",
            "title": "Do thing",
            "description": None,
            "status": "todo",
            "priority": 1,
            "assigned_to_user_id": test_user_id,
            "assigned_to_team_id": None,
            "ordering": 0,
            "expected_outputs": None,
            "success_criteria": None,
            "blockers": None,
            "confidence": 0.6,
        },
    )
    assert node.goal_id == goal.id

    link1 = await svc.link_memory(
        org_id=test_org_id,
        actor_user_id=test_user_id,
        goal_id=goal.id,
        memory_id=memory_id,
        link_type="progress",
        confidence=0.7,
        node_id=None,
        linked_by="user",
    )

    link2 = await svc.link_memory(
        org_id=test_org_id,
        actor_user_id=test_user_id,
        goal_id=goal.id,
        memory_id=memory_id,
        link_type="evidence",
        confidence=0.9,
        node_id=link1.node_id,
        linked_by="agent",
    )

    assert link1.id == link2.id
    assert float(link2.confidence) == 0.9
    assert link2.link_type == "evidence"

    _goal, nodes, _edges, links, progress = await svc.get_goal_detail_bundle(goal_id=goal.id, org_id=test_org_id)
    assert len(nodes) == 1
    assert len(links) == 1
    assert progress.total_nodes == 1
