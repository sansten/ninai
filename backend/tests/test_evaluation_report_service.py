from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.cognitive_loop.evaluation_report_service import EvaluationReportService


@dataclass
class _FakeSession:
    id: str
    status: str
    goal_id: str | None = None


@dataclass
class _FakeIteration:
    session_id: str
    iteration_num: int
    metrics: dict[str, Any]


@dataclass
class _FakeToolLog:
    id: str
    session_id: str
    status: str


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


class _ScalarOneOrNoneResult:
    def __init__(self, item: Any | None):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


@pytest.mark.asyncio
async def test_generate_report_computes_metrics_and_ids() -> None:
    fake_sess = _FakeSession(id="s1", status="succeeded", goal_id="g1")
    iterations = [
        _FakeIteration(session_id="s1", iteration_num=1, metrics={"confidence": 0.25, "evidence_memory_ids": ["m1", "m2"]}),
        _FakeIteration(session_id="s1", iteration_num=2, metrics={"confidence": 0.75, "evidence_memory_ids": ["m2"]}),
    ]
    tool_logs = [
        _FakeToolLog(id="t1", session_id="s1", status="denied"),
        _FakeToolLog(id="t2", session_id="s1", status="failed"),
        _FakeToolLog(id="t3", session_id="s1", status="success"),
    ]

    session = AsyncMock(spec=AsyncSession)
    session.flush = AsyncMock()

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "FROM evaluation_reports" in sql:
            return _ScalarOneOrNoneResult(None)
        if "FROM cognitive_sessions" in sql:
            return _ScalarOneOrNoneResult(fake_sess)
        if "FROM cognitive_iterations" in sql:
            return _ListResult(iterations)
        if "FROM tool_call_logs" in sql:
            return _ListResult(tool_logs)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    svc = EvaluationReportService(session)
    report = await svc.generate_for_session(session_id="s1")

    assert report.final_decision == "pass"
    payload = report.report
    assert payload["final_decision"] == "pass"
    assert payload["goal_id"] == "g1"
    assert payload["iteration_count"] == 2
    assert set(payload["evidence_memory_ids"]) == {"m1", "m2"}
    assert payload["quality_metrics"]["policy_denials"] == 1
    assert payload["quality_metrics"]["tool_failures"] == 1
    assert payload["quality_metrics"]["avg_confidence"] == pytest.approx(0.5)
    assert set(payload["tool_calls"]) == {"t1", "t2", "t3"}
    assert isinstance(report.created_at, datetime)
    assert report.created_at.tzinfo is not None
