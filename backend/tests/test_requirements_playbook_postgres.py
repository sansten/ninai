from __future__ import annotations

import asyncio
import hashlib
import os
import time
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import NullPool

from app.core.redis import RedisClient
from app.core.security import create_access_token
from app.core.database import set_tenant_context
from app.models.organization import Organization
from app.models.user import Role, User, UserRole
from app.models.memory import MemoryMetadata
from app.models.goal import GoalActivityLog, GoalMemoryLink, GoalNode
from app.models.evaluation_report import EvaluationReport
from app.models.simulation_report import SimulationReport
from app.models.tool_call_log import ToolCallLog

import app.tasks.cognitive_loop as cognitive_loop_task_module
from app.tasks.cognitive_loop import cognitive_loop_task


def _auth_headers(*, org_id: str, user_id: str, roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["org_admin"])
    return {"Authorization": f"Bearer {token}"}


async def _seed_user_with_role(
    db: AsyncSession,
    *,
    org_id: str,
    user_id: str,
    role_name: str,
    permissions: list[str],
) -> None:
    # Organizations/users/roles are not RLS-scoped in this repo; seed directly.
    org = Organization(id=org_id, name=f"Org {org_id[:6]}", slug=f"org-{org_id[:8]}", description=None, settings={}, is_active=True)
    user = User(
        id=user_id,
        email=f"u-{user_id[:8]}@example.com",
        hashed_password="x",
        full_name="Test User",
        is_active=True,
        is_superuser=False,
        clearance_level=0,
        preferences={},
    )
    role = Role(
        id=str(uuid4()),
        name=role_name,
        display_name=role_name,
        description=None,
        permissions=permissions,
        organization_id=org_id,
        is_system=False,
        is_default=False,
    )
    grant = UserRole(
        id=str(uuid4()),
        user_id=user_id,
        role_id=role.id,
        organization_id=org_id,
        scope_type="organization",
        scope_id=None,
        expires_at=None,
    )

    db.add_all([org, user, role, grant])
    await db.commit()


async def _seed_memory(db: AsyncSession, *, org_id: str, user_id: str, memory_id: str) -> None:
    content = "Evidence: policy constraints apply."
    row = MemoryMetadata(
        id=memory_id,
        organization_id=org_id,
        owner_id=user_id,
        scope="team",
        scope_id=None,
        memory_type="long_term",
        classification="internal",
        required_clearance=0,
        title="Policy evidence",
        content_preview=content[:200],
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        tags=["evidence"],
        entities={},
        extra_metadata={},
        source_type="test",
        source_id="seed",
        vector_id=str(uuid4()),
        embedding_model="test",
    )
    async with db.begin():
        await set_tenant_context(db, user_id, org_id, "org_admin", 0)
        db.add(row)
    await db.commit()


def _disable_redis(monkeypatch) -> None:
    monkeypatch.setattr(RedisClient, "get_json", AsyncMock(return_value=None))
    monkeypatch.setattr(RedisClient, "set_json", AsyncMock())
    monkeypatch.setattr(RedisClient, "delete", AsyncMock())
    monkeypatch.setattr(RedisClient, "delete_pattern", AsyncMock())


def _run_cognitive_loop_task_sync(*, db_url: str, kwargs: dict) -> str:
    """Run the celery task in a thread with its own asyncpg pool.

    The core app uses asyncpg, which binds connections to a single event loop.
    Our test runner uses an event loop already; the task bridges via asyncio.run.
    Creating a fresh AsyncEngine in the worker thread prevents cross-loop errors.
    """

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    engine = create_async_engine(db_url, poolclass=NullPool)
    try:
        cognitive_loop_task_module.async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        return cognitive_loop_task(**kwargs)
    finally:
        # Best-effort cleanup; avoid hiding the original failure.
        try:
            asyncio.run(engine.dispose())
        except Exception:
            pass


@pytest.mark.asyncio
async def test_playbook_long_horizon_goal_flow(pg_client, pg_db_session, migrated_test_engine, monkeypatch):
    """Section 6.1: Postgres-backed wiring across GoalGraph + CognitiveLoop + Simulation + SelfModel.

    This stubs LLM-dependent parts but uses real DB persistence and permission checks.
    """

    _disable_redis(monkeypatch)

    # Ensure deterministic, LLM-free planning in this test.
    from app.core.config import settings

    monkeypatch.setattr(settings, "AGENT_STRATEGY", "heuristic", raising=False)

    org_id = str(uuid4())
    user_id = str(uuid4())

    await _seed_user_with_role(
        pg_db_session,
        org_id=org_id,
        user_id=user_id,
        role_name="test_admin",
        permissions=[
            "goal:create:team",
            "goal:read:team",
            "goal:update:team",
            "memory:read:team",
            "simulation:read:reports",
            "selfmodel:read:org",
        ],
    )

    memory_id = str(uuid4())
    await _seed_memory(pg_db_session, org_id=org_id, user_id=user_id, memory_id=memory_id)

    headers = _auth_headers(org_id=org_id, user_id=user_id)

    # Create a team-scope goal.
    goal_resp = await pg_client.post(
        "/api/v1/goals",
        headers=headers,
        json={
            "title": "Ship a safe feature",
            "description": "Long horizon goal",
            "owner_type": "team",
            "owner_id": None,
            "goal_type": "project",
            "status": "active",
            "priority": 3,
            "confidence": 0.6,
            "visibility_scope": "team",
            "tags": ["playbook"],
            "metadata": {},
        },
    )
    assert goal_resp.status_code == 200
    goal_id = goal_resp.json()["id"]

    # Add an in-progress node so the task can deterministically mark it done.
    node_resp = await pg_client.post(
        f"/api/v1/goals/{goal_id}/nodes",
        headers=headers,
        json={
            "node_type": "task",
            "title": "Validate policy",
            "description": "",
            "status": "in_progress",
            "priority": 5,
            "assigned_to_user_id": user_id,
        },
    )
    assert node_resp.status_code == 200

    # Create cognitive session attached to that goal.
    sess_resp = await pg_client.post(
        "/api/v1/cognitive/sessions",
        headers=headers,
        json={
            "goal": "Validate policy with evidence",
            "context_snapshot": {},
            "goal_id": goal_id,
        },
    )
    assert sess_resp.status_code == 201
    session_id = sess_resp.json()["id"]

    # Stub external calls.

    from app.services.embedding_service import EmbeddingService
    from app.services.memory_service import MemoryService
    from app.services.cognitive_loop.critic_agent import CriticAgent
    from app.schemas.cognitive import CriticOutput

    async def _embed(_text: str):
        return [0.0, 0.0, 0.0]

    monkeypatch.setattr(EmbeddingService, "embed", AsyncMock(side_effect=_embed))

    async def _search_memories(self: MemoryService, _embedding, req):
        res = await self.session.execute(
            select(MemoryMetadata)
            .where(MemoryMetadata.organization_id == self.org_id)
            .limit(int(getattr(req, "limit", 10) or 10))
        )
        rows = list(res.scalars().all())
        # Force low evidence strength so SimulationService recommends an evidence step.
        for r in rows:
            setattr(r, "score", 0.0)
        return rows

    monkeypatch.setattr(MemoryService, "search_memories", _search_memories, raising=True)

    async def _critique_pass(self, **_kwargs):
        return CriticOutput(
            evaluation="pass",
            strengths=["ok"],
            issues=[],
            followup_questions=[],
            confidence=0.9,
        )

    monkeypatch.setattr(CriticAgent, "critique", _critique_pass, raising=True)

    # Run task in a separate thread (task uses asyncio.run internally).
    # Use a fresh AsyncEngine inside the thread to avoid asyncpg cross-event-loop errors.
    task_db_url = migrated_test_engine.url.render_as_string(hide_password=False)
    status = await asyncio.to_thread(
        _run_cognitive_loop_task_sync,
        db_url=task_db_url,
        kwargs={
            "org_id": org_id,
            "session_id": session_id,
            "initiator_user_id": user_id,
            "roles": "org_admin",
            "clearance_level": 0,
            "justification": "test",
            "max_iterations": 1,
        },
    )
    assert status == "succeeded"

    # Assertions: goal node marked done; memory linked; activity log exists; evaluation report references goal_id.
    async with pg_db_session.begin():
        await set_tenant_context(pg_db_session, user_id, org_id, "org_admin", 0)

        nres = await pg_db_session.execute(select(GoalNode).where(GoalNode.goal_id == goal_id))
        nodes = list(nres.scalars().all())
        assert any(n.status == "done" for n in nodes)

        lres = await pg_db_session.execute(select(GoalMemoryLink).where(GoalMemoryLink.goal_id == goal_id))
        links = list(lres.scalars().all())
        assert any(str(l.memory_id) == memory_id for l in links)

        ares = await pg_db_session.execute(select(GoalActivityLog).where(GoalActivityLog.goal_id == goal_id))
        events = list(ares.scalars().all())
        assert any(e.action == "update_node_status" for e in events)
        assert any(e.action == "add_link" for e in events)

        eres = await pg_db_session.execute(select(EvaluationReport).where(EvaluationReport.session_id == session_id))
        reports = list(eres.scalars().all())
        assert reports
        payload = reports[-1].report or {}
        assert payload.get("goal_id") == goal_id

        sres = await pg_db_session.execute(select(SimulationReport).where(SimulationReport.session_id == session_id))
        sims = list(sres.scalars().all())
        assert sims

        sim_payload = sims[-1].report or {}
        sim = sim_payload.get("simulation") or {}
        risk_factors = list(sim.get("risk_factors") or [])
        assert any((rf.get("type") == "insufficient_evidence") for rf in risk_factors)
        patch = sim.get("recommended_plan_patch") or {}
        add_steps = list(patch.get("add_steps") or [])
        assert any((s.get("step_id") == "S_EVIDENCE") for s in add_steps)

        tres = await pg_db_session.execute(select(ToolCallLog).where(ToolCallLog.session_id == session_id))
        tool_logs = list(tres.scalars().all())
        assert any(t.tool_name == "memory.search" for t in tool_logs)
        assert any(t.status == "success" for t in tool_logs), [
            {
                "tool_name": t.tool_name,
                "status": t.status,
                "denial_reason": t.denial_reason,
                "output_mode": (t.tool_output_summary or {}).get("mode"),
                "output_summary": t.tool_output_summary,
            }
            for t in tool_logs
        ]

    # Goal progress updated (derived) when fetched.
    goal_detail = await pg_client.get(f"/api/v1/goals/{goal_id}", headers=headers)
    assert goal_detail.status_code == 200
    progress = goal_detail.json()["progress"]
    assert progress["completed_nodes"] >= 1


@pytest.mark.asyncio
async def test_playbook_low_confidence_domain_leads_to_needs_evidence(pg_client, pg_db_session, migrated_test_engine, monkeypatch):
    """Section 6.2: low-confidence domain should bias planning toward extra evidence, and fail closed when critique says insufficient."""

    _disable_redis(monkeypatch)

    org_id = str(uuid4())
    user_id = str(uuid4())

    await _seed_user_with_role(
        pg_db_session,
        org_id=org_id,
        user_id=user_id,
        role_name="test_admin",
        permissions=[
            "goal:create:personal",
            "goal:read:personal",
            "goal:update:own",
            "memory:read:team",
            "selfmodel:read:org",
            "simulation:read:reports",
        ],
    )

    headers = _auth_headers(org_id=org_id, user_id=user_id)

    # Seed a self-model profile with a low-confidence domain.
    from app.models.self_model import SelfModelProfile

    prof = SelfModelProfile(
        organization_id=org_id,
        domain_confidence={"legal": 0.4},
        tool_reliability={},
        agent_accuracy={},
        last_updated=datetime.now(timezone.utc),
    )
    async with pg_db_session.begin():
        await set_tenant_context(pg_db_session, user_id, org_id, "org_admin", 0)
        pg_db_session.add(prof)
    await pg_db_session.commit()

    # Create a goal (personal scope is enough for this scenario).
    goal_resp = await pg_client.post(
        "/api/v1/goals",
        headers=headers,
        json={
            "title": "Answer legal question",
            "description": "",
            "owner_type": "user",
            "owner_id": user_id,
            "goal_type": "research",
            "status": "active",
            "priority": 2,
            "confidence": 0.5,
            "visibility_scope": "personal",
            "tags": [],
            "metadata": {},
        },
    )
    assert goal_resp.status_code == 200
    goal_id = goal_resp.json()["id"]

    sess_resp = await pg_client.post(
        "/api/v1/cognitive/sessions",
        headers=headers,
        json={
            "goal": "Provide guidance on a legal topic",
            "context_snapshot": {},
            "goal_id": goal_id,
        },
    )
    assert sess_resp.status_code == 201
    session_id = sess_resp.json()["id"]

    from app.services.embedding_service import EmbeddingService
    from app.services.memory_service import MemoryService
    from app.services.cognitive_loop.critic_agent import CriticAgent
    from app.schemas.cognitive import CriticOutput

    monkeypatch.setattr(EmbeddingService, "embed", AsyncMock(return_value=[0.0, 0.0, 0.0]))

    async def _search_memories(self: MemoryService, _embedding, _req):
        return []

    monkeypatch.setattr(MemoryService, "search_memories", _search_memories, raising=True)

    async def _critique_needs_evidence(self, **_kwargs):
        return CriticOutput(
            evaluation="needs_evidence",
            strengths=[],
            issues=[
                {
                    "type": "missing_evidence",
                    "description": "low confidence domain; need more evidence",
                    "affected_steps": ["S1", "S2"],
                    "recommended_fix": "Gather more evidence",
                }
            ],
            followup_questions=["Provide more context"],
            confidence=0.3,
        )

    monkeypatch.setattr(CriticAgent, "critique", _critique_needs_evidence, raising=True)

    task_db_url = migrated_test_engine.url.render_as_string(hide_password=False)
    status = await asyncio.to_thread(
        _run_cognitive_loop_task_sync,
        db_url=task_db_url,
        kwargs={
            "org_id": org_id,
            "session_id": session_id,
            "initiator_user_id": user_id,
            "roles": "org_admin",
            "clearance_level": 0,
            "justification": "test",
            "max_iterations": 1,
        },
    )
    assert status == "failed"

    # Plan should include extra evidence step because low_confidence_domains is present.
    iters = await pg_client.get(f"/api/v1/cognitive/sessions/{session_id}/iterations", headers=headers)
    assert iters.status_code == 200
    plan_steps = iters.json()[0]["plan_json"]["steps"]
    step_ids = [s["step_id"] for s in plan_steps]
    assert "S2" in step_ids

    # Simulation report stored.
    async with pg_db_session.begin():
        await set_tenant_context(pg_db_session, user_id, org_id, "org_admin", 0)
        sres = await pg_db_session.execute(select(SimulationReport).where(SimulationReport.session_id == session_id))
        sims = list(sres.scalars().all())
        assert sims

        sim_payload = sims[-1].report or {}
        sim = sim_payload.get("simulation") or {}
        risk_factors = list(sim.get("risk_factors") or [])
        assert any((rf.get("type") == "insufficient_evidence") for rf in risk_factors)

    # Final decision fail-closed.
    eres = await pg_client.get(f"/api/v1/cognitive/sessions/{session_id}/report", headers=headers)
    assert eres.status_code == 200
    assert eres.json()["final_decision"] == "needs_evidence"


@pytest.mark.asyncio
async def test_playbook_cross_org_isolation(pg_client, pg_db_session, monkeypatch):
    """Section 6.3: Org A data should not be readable from Org B."""

    _disable_redis(monkeypatch)

    org_a = str(uuid4())
    org_b = str(uuid4())
    user_a = str(uuid4())
    user_b = str(uuid4())

    perms = [
        "goal:create:team",
        "goal:read:team",
        "goal:update:team",
        "simulation:read:reports",
        "selfmodel:read:org",
        "memory:read:team",
    ]

    await _seed_user_with_role(pg_db_session, org_id=org_a, user_id=user_a, role_name="test_admin", permissions=perms)
    await _seed_user_with_role(pg_db_session, org_id=org_b, user_id=user_b, role_name="test_admin", permissions=perms)

    headers_a = _auth_headers(org_id=org_a, user_id=user_a)
    headers_b = _auth_headers(org_id=org_b, user_id=user_b)

    # Org A creates a goal.
    goal_resp = await pg_client.post(
        "/api/v1/goals",
        headers=headers_a,
        json={
            "title": "Org A goal",
            "description": "",
            "owner_type": "team",
            "owner_id": None,
            "goal_type": "project",
            "status": "active",
            "priority": 1,
            "confidence": 0.5,
            "visibility_scope": "team",
            "tags": [],
            "metadata": {},
        },
    )
    assert goal_resp.status_code == 200
    goal_id = goal_resp.json()["id"]

    # Org B cannot read Org A goal.
    other = await pg_client.get(f"/api/v1/goals/{goal_id}", headers=headers_b)
    assert other.status_code in (403, 404)

    # Org isolation for simulation reports: Org B should not see Org A report rows.
    async with pg_db_session.begin():
        await set_tenant_context(pg_db_session, user_a, org_a, "org_admin", 0)
        pg_db_session.add(
            SimulationReport(
                id=str(uuid4()),
                organization_id=org_a,
                session_id=None,
                memory_id=None,
                report={"iteration_num": 1, "simulation": {"confidence": 0.5}},
                created_at=datetime.now(timezone.utc),
            )
        )

    reports_b = await pg_client.get("/api/v1/simulation-reports", headers=headers_b)
    assert reports_b.status_code == 200
    assert reports_b.json() == []

    # SelfModel bundle is org-scoped; Org B should not see Org A org_id.
    bundle_b = await pg_client.get("/api/v1/self-model/bundle", headers=headers_b)
    # Endpoint may return 404 if profile doesn't exist (fail-closed), or 200 with empty/default data.
    if bundle_b.status_code == 200:
        assert bundle_b.json()["profile"]["organization_id"] == org_b
    else:
        assert bundle_b.status_code in (404, 403)


@pytest.mark.asyncio
async def test_performance_smoke_post_cognitive_session(pg_client, pg_db_session, monkeypatch):
    """Section 6.4 (smoke): optional timing check, guarded to avoid flakiness."""

    if os.environ.get("RUN_PERF_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_PERF_TESTS=1 to enable timing assertions")

    _disable_redis(monkeypatch)

    org_id = str(uuid4())
    user_id = str(uuid4())

    await _seed_user_with_role(
        pg_db_session,
        org_id=org_id,
        user_id=user_id,
        role_name="test_admin",
        permissions=["goal:read:personal"],
    )

    headers = _auth_headers(org_id=org_id, user_id=user_id)

    t0 = time.perf_counter()
    resp = await pg_client.post(
        "/api/v1/cognitive/sessions",
        headers=headers,
        json={"goal": "quick", "context_snapshot": {}},
    )
    dt = time.perf_counter() - t0

    assert resp.status_code == 201
    assert dt < 0.2
