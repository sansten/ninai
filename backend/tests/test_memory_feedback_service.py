from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.memory_feedback_service import MemoryFeedbackService


@dataclass
class _FakeFeedback:
    id: str
    actor_id: str
    feedback_type: str
    payload: dict
    created_at: datetime


def _execute_result_with_rows(rows: list) -> MagicMock:
    result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = rows
    result.scalars.return_value = scalars
    return result


@pytest.mark.asyncio
async def test_apply_pending_feedback_mutates_memory_and_marks_applied():
    org_id = "org"
    user_id = "user"
    memory_id = str(uuid4())
    now = datetime.now(timezone.utc)

    feedback_rows = [
        _FakeFeedback(
            id=str(uuid4()),
            actor_id="u1",
            feedback_type="tag_add",
            payload={"tag": "urgent"},
            created_at=now,
        ),
        _FakeFeedback(
            id=str(uuid4()),
            actor_id="u1",
            feedback_type="entity_add",
            payload={"key": "customer", "value": "Acme"},
            created_at=now,
        ),
        _FakeFeedback(
            id=str(uuid4()),
            actor_id="u2",
            feedback_type="classification_override",
            payload={"classification": "confidential"},
            created_at=now,
        ),
        _FakeFeedback(
            id=str(uuid4()),
            actor_id="u2",
            feedback_type="note",
            payload={"note": "Please redact PII"},
            created_at=now,
        ),
        _FakeFeedback(
            id=str(uuid4()),
            actor_id="u2",
            feedback_type="unknown_type",
            payload={},
            created_at=now,
        ),
    ]

    memory = SimpleNamespace(tags=["existing"], entities={}, extra_metadata={}, classification="internal")

    session = AsyncMock()
    # First execute() returns feedback rows for the select.
    # Second execute() is the bulk update marking feedback applied.
    session.execute = AsyncMock(
        side_effect=[_execute_result_with_rows(feedback_rows), MagicMock()]
    )
    session.get = AsyncMock(return_value=memory)

    svc = MemoryFeedbackService(session, user_id=user_id, org_id=org_id)
    summary = await svc.apply_pending_feedback(memory_id=memory_id, applied_by=user_id)

    assert summary["applied_count"] == len(feedback_rows)
    assert isinstance(summary["updates"], list)
    assert memory.classification == "confidential"
    assert "urgent" in memory.tags
    assert memory.entities.get("customer") == ["Acme"]
    assert isinstance(memory.extra_metadata.get("feedback_notes"), list)
    assert len(memory.extra_metadata["feedback_notes"]) == 1
    assert memory.extra_metadata["feedback_notes"][0]["actor_id"] == "u2"

    # Ensure we marked as applied (bulk update executed)
    assert session.execute.call_count == 2


@pytest.mark.asyncio
async def test_apply_pending_feedback_no_rows_is_noop():
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[_execute_result_with_rows([])])
    session.get = AsyncMock()

    svc = MemoryFeedbackService(session, user_id="u", org_id="org")
    summary = await svc.apply_pending_feedback(memory_id=str(uuid4()), applied_by="u")

    assert summary == {"applied_count": 0, "updates": []}
    assert session.get.call_count == 0
    assert session.execute.call_count == 1


@pytest.mark.asyncio
async def test_apply_pending_feedback_memory_missing_returns_warning_and_does_not_mark_applied():
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[_execute_result_with_rows([
        _FakeFeedback(
            id=str(uuid4()),
            actor_id="u",
            feedback_type="tag_add",
            payload={"tag": "x"},
            created_at=datetime.now(timezone.utc),
        )
    ])])
    session.get = AsyncMock(return_value=None)

    svc = MemoryFeedbackService(session, user_id="u", org_id="org")
    summary = await svc.apply_pending_feedback(memory_id=str(uuid4()), applied_by="u")

    assert summary.get("warning") == "memory_not_found"
    assert summary.get("applied_count") == 0
    # No bulk update should have been attempted
    assert session.execute.call_count == 1
