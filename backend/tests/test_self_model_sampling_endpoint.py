from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import select

from app.middleware.tenant_context import TenantContext
from app.models.organization import Organization
from app.models.self_model import SelfModelEvent
from app.models.user import User
from app.schemas.self_model_sampling import ToolOutcomeSampleIn


@pytest.mark.asyncio
async def test_submit_tool_outcome_sample_accepts_non_uuid_session_id(db_session, test_org_id: str, test_user_id: str, monkeypatch):
    # Seed minimal FK parents if the shared db_session fixture hasn't already.
    if await db_session.get(Organization, test_org_id) is None:
        db_session.add(
            Organization(
                id=test_org_id,
                name="Test Org",
                slug=f"test-org-{uuid4().hex[:10]}",
                is_active=True,
                settings={},
            )
        )

    if await db_session.get(User, test_user_id) is None:
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
    await db_session.commit()

    # Avoid exercising permission and celery behavior in this unit-style test.
    from app.api.v1.endpoints import self_model as self_model_endpoints

    async def _allow(*, db, tenant, permission: str) -> None:  # noqa: ANN001
        return None

    class _NoopTask:
        def delay(self, **kwargs):  # noqa: ANN003
            return None

    monkeypatch.setattr(self_model_endpoints, "_require_permission", _allow)
    monkeypatch.setattr(self_model_endpoints, "self_model_recompute_task", _NoopTask())

    tenant = TenantContext(user_id=test_user_id, org_id=test_org_id, roles=["org_admin"], clearance_level=0)

    body = ToolOutcomeSampleIn(
        tool_name="memory.search",
        success=True,
        duration_ms=125.0,
        notes="notebook sample",
        session_id="aiops-demo",
        memory_id="not-a-uuid",
        extra={},
    )

    res = await self_model_endpoints.submit_tool_outcome_sample(body=body, tenant=tenant, db=db_session)
    assert res.organization_id == test_org_id
    assert res.tool_name == "memory.search"
    assert res.event_type == "tool_success"

    saved = (await db_session.execute(select(SelfModelEvent).where(SelfModelEvent.id == res.id))).scalar_one()
    assert saved.session_id is None
    assert saved.memory_id is None

    extra = (saved.payload or {}).get("extra") or {}
    assert extra.get("session_id_raw") == "aiops-demo"
    assert extra.get("memory_id_raw") == "not-a-uuid"
