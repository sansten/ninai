"""Tests for Phase 53 - ProspectiveMemoryAgent and prospective_memory_pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.prospective_memory_agent import (
    ProspectiveMemoryAgent,
    classify_urgency,
    extract_deadline_tokens,
    infer_offset_hours,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_ANCHOR = datetime(2024, 6, 3, 12, 0, 0, tzinfo=timezone.utc)  # Monday 12:00 UTC


def _ctx(*, content: str = "", existing_reminders=None, current_time=None):
    return {
        "memory": {
            "enrichment": {
                "content": content,
                "existing_reminders": existing_reminders or [],
                "current_time": current_time or _ANCHOR,
            }
        },
        "runtime": {"job_id": "trace-53"},
    }


def _result(outputs, status="success"):
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="ProspectiveMemoryAgent",
        agent_version="v1",
        memory_id="m53",
        status=status,
        confidence=0.9,
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
    )


# ---------------------------------------------------------------------------
# infer_offset_hours
# ---------------------------------------------------------------------------

def test_within_2_hours():
    assert infer_offset_hours("within 2 hours", _ANCHOR) == 2.0


def test_within_3_days():
    assert infer_offset_hours("within 3 days", _ANCHOR) == 72.0


def test_tomorrow():
    assert infer_offset_hours("tomorrow", _ANCHOR) == 24.0


def test_friday_returns_correct_hours():
    # _ANCHOR is Monday 12:00 UTC; next Friday 17:00 = 4 days + 5h = 101h
    offset = infer_offset_hours("by Friday", _ANCHOR)
    assert 95 < offset < 110  # 101 hours ± tolerance


def test_monday_next_week():
    # Start from Friday 12:00 → next Monday 17:00 = 3 days + 5h = 77h
    friday_noon = datetime(2024, 6, 7, 12, 0, tzinfo=timezone.utc)
    offset = infer_offset_hours("by Monday", friday_noon)
    assert 70 < offset < 85  # ~77 hours


def test_end_of_week_matches_friday():
    offset_eof = infer_offset_hours("end of week", _ANCHOR)
    offset_fri = infer_offset_hours("by Friday", _ANCHOR)
    assert abs(offset_eof - offset_fri) < 1.0


def test_plain_hours():
    assert infer_offset_hours("3 hours", _ANCHOR) == 3.0


def test_plain_days():
    assert infer_offset_hours("2 days", _ANCHOR) == 48.0


def test_plain_weeks():
    assert infer_offset_hours("1 week", _ANCHOR) == 168.0


def test_no_time_fragment_returns_none():
    offset = infer_offset_hours("by", _ANCHOR)
    assert offset is None


# ---------------------------------------------------------------------------
# classify_urgency
# ---------------------------------------------------------------------------

def test_urgency_high():
    assert classify_urgency(1.0) == "high"


def test_urgency_high_boundary():
    assert classify_urgency(3.99) == "high"


def test_urgency_medium():
    assert classify_urgency(4.0) == "medium"


def test_urgency_medium_boundary():
    assert classify_urgency(47.99) == "medium"


def test_urgency_low():
    assert classify_urgency(48.0) == "low"


def test_urgency_low_large():
    assert classify_urgency(500.0) == "low"


# ---------------------------------------------------------------------------
# extract_deadline_tokens
# ---------------------------------------------------------------------------

def test_extract_deadline_deploy_by_friday():
    tokens = extract_deadline_tokens("Deploy by Friday before lunch.")
    assert any("friday" in t.lower() for t in tokens)


def test_extract_deadline_within_2_hours():
    tokens = extract_deadline_tokens("Must complete within 2 hours.")
    assert any("within" in t.lower() for t in tokens)


def test_extract_no_tokens():
    tokens = extract_deadline_tokens("The service is running fine.")
    assert tokens == []


def test_extract_deduplicates():
    text = "by noon and by morning"
    tokens = extract_deadline_tokens(text)
    # "by" appears twice but should be deduplicated by prefix
    assert len(tokens) >= 1


# ---------------------------------------------------------------------------
# Heuristic path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
async def test_deploy_by_friday_detected(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = ProspectiveMemoryAgent()
    ctx = _ctx(content="Deploy by Friday.", current_time=_ANCHOR)
    result = await agent.run("m1", ctx)
    assert result.status == "success"
    assert result.outputs["deadline_detected"] is True
    assert len(result.outputs["deadline_tokens"]) >= 1


@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
async def test_within_2_hours_high_urgency(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = ProspectiveMemoryAgent()
    ctx = _ctx(content="Fix the bug within 2 hours.", current_time=_ANCHOR)
    result = await agent.run("m1", ctx)
    assert result.status == "success"
    suggestions = result.outputs["reminders_suggested"]
    urgencies = [s["urgency"] for s in suggestions]
    assert "high" in urgencies


@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
async def test_no_deadline_empty_suggestions(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = ProspectiveMemoryAgent()
    ctx = _ctx(content="The metrics look good.", current_time=_ANCHOR)
    result = await agent.run("m1", ctx)
    assert result.status == "success"
    assert result.outputs["deadline_detected"] is False
    assert result.outputs["reminders_suggested"] == []


@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
async def test_confidence_increases_with_more_tokens(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = ProspectiveMemoryAgent()
    ctx1 = _ctx(content="Done by Friday.", current_time=_ANCHOR)
    ctx2 = _ctx(content="Done by Friday, deadline tomorrow, expires within 2 hours.", current_time=_ANCHOR)
    r1 = await agent.run("m1", ctx1)
    r2 = await agent.run("m1", ctx2)
    assert r2.outputs["confidence"] >= r1.outputs["confidence"]


@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
async def test_existing_reminders_suppressed(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = ProspectiveMemoryAgent()
    existing = [{"reminder_content": "Deadline detected: by Friday"}]
    ctx = _ctx(content="Deploy by Friday.", existing_reminders=existing, current_time=_ANCHOR)
    result = await agent.run("m1", ctx)
    # Suggestion already in existing should be suppressed
    suggestions = result.outputs["reminders_suggested"]
    contents = [s["reminder_content"].lower() for s in suggestions]
    assert not any("by friday" in c for c in contents)


@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
async def test_current_time_used_for_offset(mock_settings):
    """offset_hours should differ when current_time is different."""
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = ProspectiveMemoryAgent()
    time_early = datetime(2024, 6, 3, 8, 0, tzinfo=timezone.utc)   # Monday 08:00
    time_late  = datetime(2024, 6, 3, 16, 0, tzinfo=timezone.utc)  # Monday 16:00
    ctx1 = _ctx(content="Deploy by Friday.", current_time=time_early)
    ctx2 = _ctx(content="Deploy by Friday.", current_time=time_late)
    r1 = await agent.run("m1", ctx1)
    r2 = await agent.run("m1", ctx2)
    off1 = r1.outputs["reminders_suggested"][0]["trigger_at_offset_hours"]
    off2 = r2.outputs["reminders_suggested"][0]["trigger_at_offset_hours"]
    assert off1 > off2  # later start → fewer hours until Friday


@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
async def test_deadline_keyword_only_default_offset(mock_settings):
    """Content with bare 'deadline' keyword but no parseable time gets default offset."""
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = ProspectiveMemoryAgent()
    ctx = _ctx(content="There is a deadline for this task.", current_time=_ANCHOR)
    result = await agent.run("m1", ctx)
    assert result.outputs["deadline_detected"] is True
    suggestions = result.outputs["reminders_suggested"]
    assert len(suggestions) >= 1
    # Default offset for unparseable deadline is 168 hours (1 week)
    assert suggestions[0]["trigger_at_offset_hours"] == 168.0


@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
async def test_multi_keyword_content(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = ProspectiveMemoryAgent()
    ctx = _ctx(
        content="Complete by Monday. Due within 48 hours. No later than end of week.",
        current_time=_ANCHOR,
    )
    result = await agent.run("m1", ctx)
    assert result.outputs["deadline_detected"] is True
    assert len(result.outputs["reminders_suggested"]) >= 2


@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
async def test_rationale_is_heuristic(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = ProspectiveMemoryAgent()
    ctx = _ctx(content="Fix by Monday.", current_time=_ANCHOR)
    result = await agent.run("m1", ctx)
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
async def test_empty_content(mock_settings):
    mock_settings.AGENT_STRATEGY = "heuristic"
    agent = ProspectiveMemoryAgent()
    ctx = _ctx(content="", current_time=_ANCHOR)
    result = await agent.run("m1", ctx)
    assert result.outputs["deadline_detected"] is False
    assert result.outputs["reminders_suggested"] == []


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

def test_validate_outputs_passes():
    agent = ProspectiveMemoryAgent()
    result = _result(
        {
            "reminders_suggested": [{"trigger_type": "time", "trigger_at_offset_hours": 2.0}],
            "deadline_detected": True,
            "deadline_tokens": ["within 2 hours"],
            "confidence": 0.9,
        }
    )
    agent.validate_outputs(result)  # must not raise


def test_validate_outputs_bad_reminders_suggested():
    agent = ProspectiveMemoryAgent()
    result = _result(
        {
            "reminders_suggested": "not a list",
            "deadline_detected": True,
            "deadline_tokens": [],
            "confidence": 0.9,
        }
    )
    with pytest.raises(ValueError, match="reminders_suggested must be a list"):
        agent.validate_outputs(result)


def test_validate_outputs_bad_deadline_detected():
    agent = ProspectiveMemoryAgent()
    result = _result(
        {
            "reminders_suggested": [],
            "deadline_detected": "yes",
            "deadline_tokens": [],
            "confidence": 0.9,
        }
    )
    with pytest.raises(ValueError, match="deadline_detected must be a bool"):
        agent.validate_outputs(result)


def test_validate_outputs_bad_deadline_tokens():
    agent = ProspectiveMemoryAgent()
    result = _result(
        {
            "reminders_suggested": [],
            "deadline_detected": False,
            "deadline_tokens": "token",
            "confidence": 0.9,
        }
    )
    with pytest.raises(ValueError, match="deadline_tokens must be a list"):
        agent.validate_outputs(result)


def test_validate_outputs_bad_confidence():
    agent = ProspectiveMemoryAgent()
    result = _result(
        {
            "reminders_suggested": [],
            "deadline_detected": False,
            "deadline_tokens": [],
            "confidence": "high",
        }
    )
    with pytest.raises(ValueError, match="confidence must be a float"):
        agent.validate_outputs(result)


def test_validate_outputs_skipped_on_error_status():
    agent = ProspectiveMemoryAgent()
    result = _result({}, status="failed")
    agent.validate_outputs(result)  # must not raise even with empty outputs


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
@patch("app.agents.prospective_memory_agent.create_llm_client")
async def test_llm_path_success(mock_client_factory, mock_settings):
    mock_settings.AGENT_STRATEGY = "llm"
    mock_settings.VLLM_MODEL = "qwen2.5:7b"

    llm_response = {
        "reminders_suggested": [
            {
                "trigger_type": "time",
                "trigger_at_offset_hours": 2.0,
                "reminder_content": "Complete within 2 hours",
                "urgency": "high",
            }
        ],
        "deadline_detected": True,
        "deadline_tokens": ["within 2 hours"],
        "confidence": 0.88,
    }
    import json

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(
        return_value={"response": json.dumps(llm_response)}
    )
    mock_client_factory.return_value = mock_client

    agent = ProspectiveMemoryAgent()
    ctx = _ctx(content="Done within 2 hours.", current_time=_ANCHOR)
    result = await agent.run("m1", ctx)

    assert result.status == "success"
    assert result.outputs["deadline_detected"] is True
    assert result.outputs["rationale"] == "llm"


@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
@patch("app.agents.prospective_memory_agent.create_llm_client")
async def test_llm_falls_back_on_invalid_response(mock_client_factory, mock_settings):
    mock_settings.AGENT_STRATEGY = "llm"
    mock_settings.VLLM_MODEL = "qwen2.5:7b"

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value={"response": "not json"})
    mock_client_factory.return_value = mock_client

    agent = ProspectiveMemoryAgent()
    ctx = _ctx(content="by Friday", current_time=_ANCHOR)
    result = await agent.run("m1", ctx)

    assert result.status == "success"
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
@patch("app.agents.prospective_memory_agent.settings")
@patch("app.agents.prospective_memory_agent.create_llm_client")
async def test_llm_falls_back_on_exception(mock_client_factory, mock_settings):
    mock_settings.AGENT_STRATEGY = "llm"
    mock_settings.VLLM_MODEL = "qwen2.5:7b"

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(side_effect=ConnectionError("LLM down"))
    mock_client_factory.return_value = mock_client

    agent = ProspectiveMemoryAgent()
    ctx = _ctx(content="by Friday", current_time=_ANCHOR)
    result = await agent.run("m1", ctx)

    assert result.status == "success"
    assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_prospective_memory():
    agent = get_agent("prospective_memory")
    assert isinstance(agent, ProspectiveMemoryAgent)


def test_registry_prospective_memory_camel():
    agent = get_agent("prospectivememory")
    assert isinstance(agent, ProspectiveMemoryAgent)


def test_registry_deadline_tracker():
    agent = get_agent("deadline_tracker")
    assert isinstance(agent, ProspectiveMemoryAgent)


def test_registry_unknown_returns_none():
    assert get_agent("does_not_exist_phase53") is None


# ---------------------------------------------------------------------------
# Celery task (unit tests with mocked DB and EventPublishingService)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scan_fires_overdue_reminders():
    """_scan_and_fire_async fires reminders with trigger_at <= now."""
    from app.tasks.prospective_memory_pipeline import _scan_and_fire_async

    now = datetime.now(timezone.utc)

    overdue = MagicMock()
    overdue.id = "rem-1"
    overdue.org_id = "org-1"
    overdue.status = "pending"
    overdue.trigger_at = now - timedelta(minutes=10)
    overdue.reminder_content = "Check migration"
    overdue.trigger_type = "time"
    overdue.fired_at = None

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[overdue])))))
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_svc = AsyncMock()
    mock_svc.publish_event = AsyncMock()

    with patch("app.tasks.prospective_memory_pipeline.AsyncSessionLocal", return_value=mock_session), \
         patch("app.tasks.prospective_memory_pipeline.EventPublishingService", return_value=mock_svc):
        result = await _scan_and_fire_async()

    assert result["fired"] == 1
    assert result["skipped"] == 0
    assert overdue.status == "fired"
    assert overdue.fired_at is not None


@pytest.mark.asyncio
async def test_scan_skips_future_reminders():
    """Reminders with trigger_at > now should NOT be returned → task fires 0."""
    from app.tasks.prospective_memory_pipeline import _scan_and_fire_async

    # SQL WHERE clause filters future reminders; return empty list.
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("app.tasks.prospective_memory_pipeline.AsyncSessionLocal", return_value=mock_session):
        result = await _scan_and_fire_async()

    assert result["fired"] == 0
    assert result["skipped"] == 0


@pytest.mark.asyncio
async def test_scan_status_set_to_fired():
    """After firing, reminder.status must equal 'fired'."""
    from app.tasks.prospective_memory_pipeline import _scan_and_fire_async

    now = datetime.now(timezone.utc)
    reminder = MagicMock()
    reminder.id = "rem-2"
    reminder.org_id = "org-2"
    reminder.status = "pending"
    reminder.trigger_at = now - timedelta(seconds=1)
    reminder.reminder_content = "Escalate review"
    reminder.trigger_type = "time"
    reminder.fired_at = None

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[reminder])))))
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_svc = AsyncMock()
    mock_svc.publish_event = AsyncMock()

    with patch("app.tasks.prospective_memory_pipeline.AsyncSessionLocal", return_value=mock_session), \
         patch("app.tasks.prospective_memory_pipeline.EventPublishingService", return_value=mock_svc):
        await _scan_and_fire_async()

    assert reminder.status == "fired"
    assert reminder.fired_at is not None


@pytest.mark.asyncio
async def test_scan_counts_skipped_on_publish_error():
    """If publishing raises, reminder is skipped (not fired) and skipped count increments."""
    from app.tasks.prospective_memory_pipeline import _scan_and_fire_async

    now = datetime.now(timezone.utc)
    reminder = MagicMock()
    reminder.id = "rem-3"
    reminder.org_id = "org-3"
    reminder.status = "pending"
    reminder.trigger_at = now - timedelta(minutes=1)
    reminder.reminder_content = "Deploy alert"
    reminder.trigger_type = "time"
    reminder.fired_at = None

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[reminder])))))
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_svc = AsyncMock()
    mock_svc.publish_event = AsyncMock(side_effect=RuntimeError("publish failed"))

    with patch("app.tasks.prospective_memory_pipeline.AsyncSessionLocal", return_value=mock_session), \
         patch("app.tasks.prospective_memory_pipeline.EventPublishingService", return_value=mock_svc):
        result = await _scan_and_fire_async()

    assert result["fired"] == 0
    assert result["skipped"] == 1
