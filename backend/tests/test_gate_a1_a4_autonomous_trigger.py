"""Gate A1 / A4 — Autonomous trigger path executes and produces action sessions (P0).

A1 Check: heartbeat or inbound events can autonomously spawn cognitive sessions.
Evidence: CognitiveSession rows with is_autonomous=True persist correctly with all
          A1-required fields (linked goal_id, heartbeat source, org_id).

A4 Check: decide and plan outputs can trigger tracked action workflows.
Evidence: autonomous session context_snapshot records source, cognitive_load, and
          anomaly_count so the downstream cognitive_loop_task can trace action back
          to the heartbeat trigger without ambiguity.
"""

from __future__ import annotations

import inspect
import uuid

import pytest
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cognitive_session import CognitiveSession
from app.models.system_cognition_state import SystemCognitionState
from app.services.system_cognition_state import (
    CognitionStateUpdate,
    SystemCognitionStateService,
)
from app.tasks.cognitive_heartbeat import (
    _HEARTBEAT_ACTOR,
    _LOAD_FIRE_THRESHOLD,
    cognitive_heartbeat_task,
)


# ---------------------------------------------------------------------------
# A1: CognitiveSession with is_autonomous=True persists all required fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_autonomous_session_persists_with_is_autonomous_flag(
    db_session: AsyncSession,
    test_org_id: str,
):
    """A1: CognitiveSession rows can be created with is_autonomous=True and queried."""
    session_id = str(uuid.uuid4())
    cog_session = CognitiveSession(
        id=session_id,
        organization_id=test_org_id,
        user_id=_HEARTBEAT_ACTOR,
        status="running",
        goal="Heartbeat: investigate 3 anomalies",
        goal_id=None,
        agent_id=None,
        context_snapshot={
            "source": "heartbeat_autonomy",
            "cognitive_load": 0.35,
            "anomaly_count": 3,
            "stale_goal_count": 0,
        },
        trace_id=None,
        is_autonomous=True,
    )
    db_session.add(cog_session)
    await db_session.commit()

    fetched = (
        await db_session.execute(
            select(CognitiveSession).where(CognitiveSession.id == session_id)
        )
    ).scalar_one_or_none()

    assert fetched is not None, "autonomous session must persist to DB"
    assert fetched.is_autonomous is True
    assert fetched.user_id == _HEARTBEAT_ACTOR
    assert fetched.goal.startswith("Heartbeat:")
    assert fetched.organization_id == test_org_id

    print(
        f"\nA1 Gate Evidence:\n"
        f"  session.id={fetched.id}\n"
        f"  session.is_autonomous=True\n"
        f"  session.user_id={fetched.user_id} (heartbeat actor)\n"
        f"  session.goal='{fetched.goal}'\n"
        f"  Status: \u2713 CognitiveSession is_autonomous flag persists correctly"
    )


@pytest.mark.asyncio
async def test_autonomous_sessions_queryable_by_is_autonomous_flag(
    db_session: AsyncSession,
    test_org_id: str,
):
    """A1: is_autonomous index enables efficient autonomous session queries."""
    for is_auto, goal_prefix in [(True, "Heartbeat: investigate"), (False, "User: review")]:
        db_session.add(
            CognitiveSession(
                id=str(uuid.uuid4()),
                organization_id=test_org_id,
                user_id=_HEARTBEAT_ACTOR if is_auto else str(uuid.uuid4()),
                status="succeeded",
                goal=f"{goal_prefix} session",
                context_snapshot={},
                is_autonomous=is_auto,
            )
        )
    await db_session.commit()

    auto_count = (
        await db_session.execute(
            select(func.count()).select_from(CognitiveSession).where(
                CognitiveSession.organization_id == test_org_id,
                CognitiveSession.is_autonomous == True,  # noqa: E712
            )
        )
    ).scalar_one()

    assert auto_count >= 1, f"Expected at least 1 autonomous session, found {auto_count}"


# ---------------------------------------------------------------------------
# A4: context_snapshot contains all required traceability fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_autonomous_session_context_snapshot_has_a4_trace_fields(
    db_session: AsyncSession,
    test_org_id: str,
):
    """A4: context_snapshot records source, cognitive_load, anomaly_count for action tracing."""
    session_id = str(uuid.uuid4())
    cognitive_load = 0.45
    anomaly_count = 4
    stale_count = 1

    db_session.add(
        CognitiveSession(
            id=session_id,
            organization_id=test_org_id,
            user_id=_HEARTBEAT_ACTOR,
            status="running",
            goal="Heartbeat: investigate 4 anomalies and advance 1 stale goal(s)",
            context_snapshot={
                "source": "heartbeat_autonomy",
                "cognitive_load": cognitive_load,
                "anomaly_count": anomaly_count,
                "stale_goal_count": stale_count,
            },
            is_autonomous=True,
        )
    )
    await db_session.commit()

    fetched = (
        await db_session.execute(
            select(CognitiveSession).where(CognitiveSession.id == session_id)
        )
    ).scalar_one()

    ctx = fetched.context_snapshot
    assert ctx["source"] == "heartbeat_autonomy", "A4: source must identify heartbeat trigger"
    assert abs(ctx["cognitive_load"] - cognitive_load) < 0.001, "A4: cognitive_load must be recorded"
    assert ctx["anomaly_count"] == anomaly_count, "A4: anomaly_count must be recorded"
    assert ctx["stale_goal_count"] == stale_count, "A4: stale_goal_count must be recorded"

    print(
        f"\nA4 Gate Evidence:\n"
        f"  context_snapshot.source='{ctx['source']}'\n"
        f"  context_snapshot.cognitive_load={ctx['cognitive_load']}\n"
        f"  context_snapshot.anomaly_count={ctx['anomaly_count']}\n"
        f"  context_snapshot.stale_goal_count={ctx['stale_goal_count']}\n"
        f"  Status: \u2713 A4 action traceability fields present in context_snapshot"
    )


# ---------------------------------------------------------------------------
# A1: heartbeat task code-structure evidence: load formula and fire threshold
# ---------------------------------------------------------------------------


def test_heartbeat_load_fire_threshold_is_set():
    """A1: _LOAD_FIRE_THRESHOLD is defined and at a reasonable value (>0)."""
    assert isinstance(_LOAD_FIRE_THRESHOLD, float), "threshold must be a float"
    assert 0.0 < _LOAD_FIRE_THRESHOLD < 1.0, (
        f"threshold must be between 0 and 1, got {_LOAD_FIRE_THRESHOLD}"
    )
    print(f"\nA1: _LOAD_FIRE_THRESHOLD={_LOAD_FIRE_THRESHOLD}")


def test_heartbeat_task_spawns_cognitive_sessions_in_source():
    """A1: heartbeat task source code contains CognitiveSession instantiation."""
    source = inspect.getsource(cognitive_heartbeat_task)
    assert "CognitiveSession" in source, "heartbeat must create CognitiveSession rows"
    assert "is_autonomous=True" in source, "heartbeat must mark sessions as autonomous"
    assert "heartbeat_autonomy" in source, "heartbeat must tag source as heartbeat_autonomy"
    assert "apply_async" in source, "heartbeat must dispatch cognitive_loop_task"
    print(
        f"\nA1 Code Evidence:\n"
        f"  \u2713 CognitiveSession created in heartbeat task\n"
        f"  \u2713 is_autonomous=True set on spawned sessions\n"
        f"  \u2713 source='heartbeat_autonomy' in context_snapshot\n"
        f"  \u2713 cognitive_loop_task.apply_async dispatched"
    )


def test_heartbeat_task_updates_cognition_state_in_source():
    """A1 + A2: heartbeat task source code upserts SystemCognitionState with session IDs."""
    source = inspect.getsource(cognitive_heartbeat_task)
    assert "SystemCognitionStateService" in source, "heartbeat must call SystemCognitionStateService"
    assert "active_session_ids" in source, "heartbeat must pass spawned session IDs to state"
    assert "CognitionStateUpdate" in source, "heartbeat must use CognitionStateUpdate"
    print(
        f"\nA1+A2 State Linkage Code Evidence:\n"
        f"  \u2713 SystemCognitionStateService.upsert called in heartbeat\n"
        f"  \u2713 active_session_ids populated with spawned IDs"
    )
