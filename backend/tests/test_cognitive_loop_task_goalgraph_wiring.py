from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest


@dataclass
class _FakeSession:
    id: str
    status: str
    goal: str
    goal_id: str | None


@dataclass
class _FakeEvalReport:
    id: str
    session_id: str
    final_decision: str
    report: dict[str, Any]


@dataclass
class _FakeGoalNode:
    id: str
    status: str


class _ScalarOneOrNone:
    def __init__(self, item):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


class _AsyncSessionCtx:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _AsyncBeginCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeRepo:
    def __init__(self, sess: _FakeSession):
        self._sess = sess

    async def get_session(self, session_id: str):
        return self._sess if session_id == self._sess.id else None


class _FakeGoalService:
    def __init__(self, _db):
        self.updated_nodes: list[tuple[str, str]] = []
        self.links: list[tuple[str, str, str]] = []

    async def update_node_status(self, *, node_id: str, org_id: str, actor_user_id: str | None, status_value: str):
        self.updated_nodes.append((node_id, status_value))
        return None

    async def link_memory(
        self,
        *,
        org_id: str,
        actor_user_id: str | None,
        goal_id: str,
        memory_id: str,
        link_type: str,
        confidence: float,
        node_id: str | None,
        linked_by: str = "user",
    ):
        self.links.append((goal_id, memory_id, link_type))
        return None


def test_cognitive_loop_task_marks_node_done_and_links_evidence(monkeypatch):
    """Exercises the post-run wiring in cognitive_loop_task without a real DB/celery broker."""

    from app.tasks import cognitive_loop as mod

    fake_session = _FakeSession(id="s1", status="running", goal="g", goal_id="goal-1")

    # Force broker disabled so we don't enqueue Celery tasks.
    monkeypatch.setattr(mod, "_broker_enabled", lambda: False)

    # Avoid touching DB for tenant context.
    async def _noop_set_tenant_context(*args, **kwargs):
        return None

    monkeypatch.setattr(mod, "set_tenant_context", _noop_set_tenant_context)

    # Provide a fake repo that returns a session with goal_id.
    monkeypatch.setattr(mod, "CognitiveLoopRepository", lambda db: _FakeRepo(fake_session))

    # Make orchestrator instantly succeed.
    async def _orch_run(self, *, session_id: str, tool_ctx):
        return "succeeded"

    monkeypatch.setattr(mod.LoopOrchestrator, "run", _orch_run)

    # EvaluationReportService should return a report with evidence ids.
    async def _gen(*, session_id: str):
        return _FakeEvalReport(
            id="r1",
            session_id=session_id,
            final_decision="pass",
            report={"evidence_memory_ids": ["m1", "m2"], "goal_id": "goal-1"},
        )

    async def _gen_wrap(self, *, session_id: str, **kwargs):
        return await _gen(session_id=session_id)

    async def _latest_wrap(self, *, session_id: str, **kwargs):
        return await _gen(session_id=session_id)

    monkeypatch.setattr(mod.EvaluationReportService, "generate_for_session", _gen_wrap)
    monkeypatch.setattr(mod.EvaluationReportService, "get_latest_for_session", _latest_wrap)

    # SelfModel summary fetch is optional; keep it empty.
    async def _sm_summary(self, *, org_id: str):
        raise Exception("skip")

    monkeypatch.setattr(mod.SelfModelService, "get_planner_summary", _sm_summary)

    # Capture GoalService calls.
    goal_service = _FakeGoalService(None)
    monkeypatch.setattr(mod, "GoalService", lambda db: goal_service)

    # Mock DB session context.
    class _DB:
        def begin(self):
            return _AsyncBeginCtx()

        async def execute(self, stmt, *args, **kwargs):
            sql = str(stmt)
            if "FROM goal_nodes" in sql:
                calls["goal_nodes"] += 1
                if calls["goal_nodes"] == 1:
                    return _ScalarOneOrNone(None)
                return _ScalarOneOrNone(_FakeGoalNode(id="n1", status="todo"))
            return _ScalarOneOrNone(None)

    db = _DB()

    # Goal node selection: return None for in_progress, then a todo node.
    calls = {"goal_nodes": 0}

    monkeypatch.setattr(mod, "async_session_factory", lambda: _AsyncSessionCtx(db))

    # Run task (sync entrypoint).
    status = mod.cognitive_loop_task(
        org_id="o1",
        session_id="s1",
        initiator_user_id="u1",
        roles="member",
        clearance_level=0,
        justification="test",
        max_iterations=1,
    )

    assert status == "succeeded"
    assert ("n1", "done") in goal_service.updated_nodes
    assert ("goal-1", "m1", "evidence") in goal_service.links
    assert ("goal-1", "m2", "evidence") in goal_service.links
