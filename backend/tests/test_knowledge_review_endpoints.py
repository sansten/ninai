import pytest
from datetime import datetime, timezone
from uuid import uuid4

from unittest.mock import AsyncMock

from app.core.security import create_access_token
from app.schemas.knowledge import (
    KnowledgeReviewListResponse,
    KnowledgeReviewRequestResponse,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.mark.asyncio
async def test_knowledge_submit_requires_auth(client):
    res = await client.post("/api/v1/knowledge/review-requests", json={"title": "T", "content": "C"})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_knowledge_submit_passes_trace_id(client, auth_headers, monkeypatch, test_org_id, test_user_id):
    from app.services import knowledge_review_service

    called = {"trace_id": None}

    async def _fake_submit_for_review(db, tenant, body, trace_id=None):
        called["trace_id"] = trace_id
        return KnowledgeReviewRequestResponse(
            id=str(uuid4()),
            organization_id=test_org_id,
            item_id=str(uuid4()),
            item_version_id=str(uuid4()),
            status="pending",
            requested_by_user_id=test_user_id,
            reviewed_by_user_id=None,
            reviewed_at=None,
            decision_comment=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(knowledge_review_service, "submit_for_review", _fake_submit_for_review)

    trace_id = "trace-123"
    res = await client.post(
        "/api/v1/knowledge/review-requests",
        headers={**auth_headers, "X-Trace-ID": trace_id},
        json={"title": "How to reset password", "content": "Step 1..."},
    )
    assert res.status_code == 200
    assert called["trace_id"] == trace_id


@pytest.mark.asyncio
async def test_admin_knowledge_list_requires_admin_role(client, test_org_id, test_user_id):
    token = create_access_token(user_id=test_user_id, org_id=test_org_id, roles=["viewer"])
    res = await client.get(
        "/api/v1/admin/knowledge/review-requests",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_admin_knowledge_list_returns_items(client, auth_headers, monkeypatch, test_org_id, test_user_id):
    from app.services import knowledge_review_service

    item = KnowledgeReviewRequestResponse(
        id=str(uuid4()),
        organization_id=test_org_id,
        item_id=str(uuid4()),
        item_version_id=str(uuid4()),
        status="pending",
        requested_by_user_id=test_user_id,
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
        "/api/v1/admin/knowledge/review-requests?status=pending&limit=10",
        headers=auth_headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["status"] == "pending"


@pytest.mark.asyncio
async def test_admin_knowledge_approve_reject_wired(client, auth_headers, monkeypatch, test_org_id, test_user_id):
    from app.services import knowledge_review_service

    approved = KnowledgeReviewRequestResponse(
        id=str(uuid4()),
        organization_id=test_org_id,
        item_id=str(uuid4()),
        item_version_id=str(uuid4()),
        status="approved",
        requested_by_user_id=test_user_id,
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
        requested_by_user_id=test_user_id,
        reviewed_by_user_id=test_user_id,
        reviewed_at=datetime.now(timezone.utc),
        decision_comment="no",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    monkeypatch.setattr(knowledge_review_service, "approve_review_request", AsyncMock(return_value=approved))
    monkeypatch.setattr(knowledge_review_service, "reject_review_request", AsyncMock(return_value=rejected))

    res1 = await client.post(
        f"/api/v1/admin/knowledge/review-requests/{approved.id}/approve",
        headers=auth_headers,
        json={"comment": "ok"},
    )
    assert res1.status_code == 200
    assert res1.json()["status"] == "approved"

    res2 = await client.post(
        f"/api/v1/admin/knowledge/review-requests/{rejected.id}/reject",
        headers=auth_headers,
        json={"comment": "no"},
    )
    assert res2.status_code == 200
    assert res2.json()["status"] == "rejected"
