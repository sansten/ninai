from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.database import set_tenant_context
from app.models.cognitive_iteration import CognitiveIteration
from app.models.cognitive_session import CognitiveSession
from app.models.evaluation_report import EvaluationReport
from app.models.organization import Organization
from app.models.tool_call_log import ToolCallLog
from app.models.user import User
from app.services.self_model_service import SelfModelService


RUN_POSTGRES_TESTS = os.environ.get("RUN_POSTGRES_TESTS", "").lower() in {"1", "true", "yes"}
requires_postgres = pytest.mark.skipif(not RUN_POSTGRES_TESTS, reason="Set RUN_POSTGRES_TESTS=1 to run")


@requires_postgres
@pytest.mark.asyncio
async def test_self_model_ingest_and_recompute(db_session, test_org_id: str, test_user_id: str):
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

    await set_tenant_context(db_session, test_user_id, test_org_id, roles="org_admin", clearance_level=0)

    session_id = str(uuid4())
    iteration_id = str(uuid4())

    sess = CognitiveSession(
        id=session_id,
        organization_id=test_org_id,
        user_id=test_user_id,
        agent_id=None,
        status="succeeded",
        goal="Help customer with refund",
        context_snapshot={"domain": "customer_support"},
        trace_id=None,
    )
    db_session.add(sess)

    # Ensure FK parent exists before inserting child rows.
    # SQLAlchemy flush ordering isn't guaranteed without explicit relationships.
    await db_session.flush()

    it = CognitiveIteration(
        id=iteration_id,
        session_id=session_id,
        iteration_num=1,
        plan_json={},
        execution_json={},
        critique_json={},
        evaluation="pass",
        started_at=datetime.now(timezone.utc) - timedelta(seconds=2),
        finished_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        metrics={},
    )
    db_session.add(it)

    tlog_id = str(uuid4())
    db_session.add(
        ToolCallLog(
            id=tlog_id,
            session_id=session_id,
            iteration_id=iteration_id,
            tool_name="memory.search",
            tool_input={},
            tool_output_summary={},
            status="success",
            denial_reason=None,
            started_at=datetime.now(timezone.utc) - timedelta(milliseconds=900),
            finished_at=datetime.now(timezone.utc) - timedelta(milliseconds=100),
        )
    )

    rep_id = str(uuid4())
    db_session.add(
        EvaluationReport(
            id=rep_id,
            session_id=session_id,
            report={"final_decision": "pass"},
            final_decision="pass",
            created_at=datetime.now(timezone.utc),
        )
    )

    await db_session.flush()

    svc = SelfModelService(db_session)

    # Ensure RLS variables are still active for the current transaction.
    await set_tenant_context(db_session, test_user_id, test_org_id, roles="org_admin", clearance_level=0)

    # Sanity: tool logs must be visible for tool-derived self model events.
    tvis = await db_session.execute(select(ToolCallLog).where(ToolCallLog.session_id == session_id))
    assert tvis.scalar_one_or_none() is not None

    inserted = await svc.ingest_from_session(org_id=test_org_id, session_id=session_id)
    assert inserted >= 1

    await set_tenant_context(db_session, test_user_id, test_org_id, roles="org_admin", clearance_level=0)
    prof = await svc.recompute_profile(org_id=test_org_id)
    assert prof.organization_id == test_org_id

    tool_stats = (prof.tool_reliability or {}).get("memory.search")
    assert tool_stats is not None
    assert tool_stats["sample_size_30d"] >= 1
    assert tool_stats["success_rate_30d"] is not None

    dom = (prof.domain_confidence or {}).get("customer_support")
    assert dom is not None
    assert 0.0 <= float(dom) <= 1.0
