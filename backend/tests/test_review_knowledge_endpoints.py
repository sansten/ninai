import pytest
from datetime import datetime, timezone
from uuid import uuid4

from unittest.mock import AsyncMock

from app.core.security import create_access_token
from app.schemas.knowledge import (
    KnowledgeReviewListResponse,
    KnowledgeReviewRequestResponse,
    KnowledgeItemVersionResponse,
    KnowledgeItemVersionsResponse,
)


@pytest.mark.asyncio
async def test_reviewer_knowledge_list_requires_reviewer_role(client, test_org_id, test_user_id):
    token = create_access_token(user_id=test_user_id, org_id=test_org_id, roles=["viewer"])
    res = await client.get(
        "/api/v1/review/knowledge/review-requests",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_reviewer_knowledge_list_returns_items(client, monkeypatch, test_org_id, test_user_id):
    from app.services import knowledge_review_service

    token = create_access_token(user_id=test_user_id, org_id=test_org_id, roles=["knowledge_reviewer"])

    item = KnowledgeReviewRequestResponse(
        id=str(uuid4()),
        organization_id=test_org_id,
        item_id=str(uuid4()),
        item_version_id=str(uuid4()),
        status="pending",
        requested_by_user_id=str(uuid4()),
        reviewed_by_user_id=None,
        reviewed_at=None,
        decision_comment=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    monkeypatch.setattr(
        knowledge_review_service,
        "list_review_requests",
        AsyncMock(return_value=KnowledgeReviewListResponse(items=[item])),
    )

    res = await client.get(
        "/api/v1/review/knowledge/review-requests?status=pending&limit=10",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["status"] == "pending"


@pytest.mark.asyncio
async def test_reviewer_knowledge_approve_reject_wired(client, monkeypatch, test_org_id, test_user_id):
    from app.services import knowledge_review_service

    token = create_access_token(user_id=test_user_id, org_id=test_org_id, roles=["knowledge_reviewer"])

    approved = KnowledgeReviewRequestResponse(
        id=str(uuid4()),
        organization_id=test_org_id,
        item_id=str(uuid4()),
        item_version_id=str(uuid4()),
        status="approved",
        requested_by_user_id=str(uuid4()),
        reviewed_by_user_id=test_user_id,
        reviewed_at=datetime.now(timezone.utc),
        decision_comment="ok",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    rejected = KnowledgeReviewRequestResponse(
        id=str(uuid4()),
        organization_id=test_org_id,
        item_id=str(uuid4()),
        item_version_id=str(uuid4()),
        status="rejected",
        requested_by_user_id=str(uuid4()),
        reviewed_by_user_id=test_user_id,
        reviewed_at=datetime.now(timezone.utc),
        decision_comment="no",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    monkeypatch.setattr(knowledge_review_service, "approve_review_request", AsyncMock(return_value=approved))
    monkeypatch.setattr(knowledge_review_service, "reject_review_request", AsyncMock(return_value=rejected))

    res1 = await client.post(
        f"/api/v1/review/knowledge/review-requests/{approved.id}/approve",
        headers={"Authorization": f"Bearer {token}"},
        json={"comment": "ok", "promote_to_memory": False},
    )
    assert res1.status_code == 200
    assert res1.json()["status"] == "approved"

    res2 = await client.post(
        f"/api/v1/review/knowledge/review-requests/{rejected.id}/reject",
        headers={"Authorization": f"Bearer {token}"},
        json={"comment": "no"},
    )
    assert res2.status_code == 200
    assert res2.json()["status"] == "rejected"


@pytest.mark.asyncio
async def test_reviewer_knowledge_versions_wired(client, monkeypatch, test_org_id, test_user_id):
    from app.services import knowledge_review_service

    token = create_access_token(user_id=test_user_id, org_id=test_org_id, roles=["knowledge_reviewer"])

    version = KnowledgeItemVersionResponse(
        id=str(uuid4()),
        organization_id=test_org_id,
        item_id=str(uuid4()),
        version_number=1,
        title="T",
        content="C",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        provenance=[],
        trace_id=None,
        extra_metadata={},
        created_by_user_id=test_user_id,
    )

    monkeypatch.setattr(
        knowledge_review_service,
        "list_item_versions",
        AsyncMock(return_value=KnowledgeItemVersionsResponse(items=[version])),
    )

    res = await client.get(
        f"/api/v1/review/knowledge/items/{version.item_id}/versions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["version_number"] == 1
