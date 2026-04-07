"""Tests for CognitiveSchedule model, service, endpoints, and Celery tasks (Phase 84)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_count_session(count: int = 0) -> AsyncMock:
    """Return a mocked async session whose execute() yields a scalar_one() count."""
    session = AsyncMock()
    # SQLAlchemy session.add is synchronous; using AsyncMock here causes
    # un-awaited coroutine warnings when service code calls add(...).
    session.add = MagicMock()
    count_result = MagicMock()
    count_result.scalar_one.return_value = count
    session.execute.return_value = count_result
    return session


def _make_get_session(obj=None) -> AsyncMock:
    """Return a mocked async session whose execute() yields a scalars().first() object."""
    session = AsyncMock()
    session.add = MagicMock()
    scalars_mock = MagicMock()
    scalars_mock.first.return_value = obj
    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock
    session.execute.return_value = result_mock
    return session


# ---------------------------------------------------------------------------
# 1. Constants and module structure (8 tests)
# ---------------------------------------------------------------------------

def test_service_module_importable():
    from app.services.cognitive_schedule_service import (  # noqa: F401
        MAX_SCHEDULES_PER_ORG,
        _VALID_COGNITIVE_VERBS,
        _CRON_PATTERN,
        validate_cron,
        compute_next_run,
        validate_verb,
        CognitiveScheduleService,
    )
    assert True


def test_max_schedules_per_org_is_20():
    from app.services.cognitive_schedule_service import MAX_SCHEDULES_PER_ORG
    assert MAX_SCHEDULES_PER_ORG == 20


def test_valid_cognitive_verbs_is_frozenset():
    from app.services.cognitive_schedule_service import _VALID_COGNITIVE_VERBS
    assert isinstance(_VALID_COGNITIVE_VERBS, frozenset)


def test_valid_cognitive_verbs_count():
    from app.services.cognitive_schedule_service import _VALID_COGNITIVE_VERBS
    assert len(_VALID_COGNITIVE_VERBS) == 7


def test_valid_cognitive_verbs_contains_all_expected():
    from app.services.cognitive_schedule_service import _VALID_COGNITIVE_VERBS
    expected = {"analyze", "summarize", "plan", "report", "monitor", "escalate", "acknowledge"}
    assert expected == _VALID_COGNITIVE_VERBS


def test_cron_pattern_is_compiled_regex():
    import re
    from app.services.cognitive_schedule_service import _CRON_PATTERN
    assert isinstance(_CRON_PATTERN, re.Pattern)


def test_model_importable_and_tablename():
    from app.models.cognitive_schedule import CognitiveSchedule
    assert CognitiveSchedule.__tablename__ == "cognitive_schedules"


def test_model_has_expected_columns():
    from app.models.cognitive_schedule import CognitiveSchedule
    for field in ("cron_expression", "cognitive_verb", "payload", "label",
                  "is_active", "next_run_at", "last_run_at"):
        assert hasattr(CognitiveSchedule, field), f"missing field: {field}"


# ---------------------------------------------------------------------------
# 2. validate_cron (10 tests)
# ---------------------------------------------------------------------------

def test_validate_cron_wildcard_all():
    from app.services.cognitive_schedule_service import validate_cron
    validate_cron("* * * * *")   # must not raise


def test_validate_cron_step_minute():
    from app.services.cognitive_schedule_service import validate_cron
    validate_cron("*/5 * * * *")


def test_validate_cron_specific_weekday():
    from app.services.cognitive_schedule_service import validate_cron
    validate_cron("0 9 * * 1")


def test_validate_cron_step_hour():
    from app.services.cognitive_schedule_service import validate_cron
    validate_cron("30 */2 * * *")


def test_validate_cron_jan_first():
    from app.services.cognitive_schedule_service import validate_cron
    validate_cron("0 0 1 1 *")


def test_validate_cron_comma_list():
    from app.services.cognitive_schedule_service import validate_cron
    validate_cron("0,30 * * * *")


def test_validate_cron_plain_text_raises():
    from app.services.cognitive_schedule_service import validate_cron
    with pytest.raises(ValueError):
        validate_cron("invalid cron")


def test_validate_cron_six_fields_raises():
    from app.services.cognitive_schedule_service import validate_cron
    with pytest.raises(ValueError):
        validate_cron("1 2 3 4 5 6")


def test_validate_cron_empty_string_raises():
    from app.services.cognitive_schedule_service import validate_cron
    with pytest.raises(ValueError):
        validate_cron("")


def test_validate_cron_double_star_raises():
    from app.services.cognitive_schedule_service import validate_cron
    with pytest.raises(ValueError):
        validate_cron("** * * * *")


# ---------------------------------------------------------------------------
# 3. compute_next_run (8 tests)
# ---------------------------------------------------------------------------

def test_compute_next_run_returns_datetime():
    from app.services.cognitive_schedule_service import compute_next_run
    result = compute_next_run("* * * * *")
    assert isinstance(result, datetime)


def test_compute_next_run_is_utc_aware():
    from app.services.cognitive_schedule_service import compute_next_run
    result = compute_next_run("* * * * *")
    assert result.tzinfo is not None


def test_compute_next_run_wildcard_is_next_minute():
    from app.services.cognitive_schedule_service import compute_next_run
    base = datetime(2024, 1, 1, 12, 10, 0, tzinfo=timezone.utc)
    result = compute_next_run("* * * * *", after=base)
    assert result == datetime(2024, 1, 1, 12, 11, 0, tzinfo=timezone.utc)


def test_compute_next_run_step5_is_multiple_of_5():
    from app.services.cognitive_schedule_service import compute_next_run
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = compute_next_run("*/5 * * * *", after=base)
    assert result.minute % 5 == 0
    assert result > base


def test_compute_next_run_specific_minute_30():
    from app.services.cognitive_schedule_service import compute_next_run
    base = datetime(2024, 1, 1, 12, 10, 0, tzinfo=timezone.utc)
    result = compute_next_run("30 * * * *", after=base)
    assert result.minute == 30


def test_compute_next_run_result_is_strictly_after_base():
    from app.services.cognitive_schedule_service import compute_next_run
    base = datetime(2024, 3, 15, 6, 30, 0, tzinfo=timezone.utc)
    result = compute_next_run("* * * * *", after=base)
    assert result > base


def test_compute_next_run_invalid_cron_raises():
    from app.services.cognitive_schedule_service import compute_next_run
    with pytest.raises(ValueError):
        compute_next_run("not a cron")


def test_compute_next_run_uses_after_not_now():
    from app.services.cognitive_schedule_service import compute_next_run
    past = datetime(2020, 6, 15, 8, 0, 0, tzinfo=timezone.utc)
    result = compute_next_run("* * * * *", after=past)
    assert result.year == 2020   # respects `after`, not system clock


# ---------------------------------------------------------------------------
# 4. validate_verb (4 tests)
# ---------------------------------------------------------------------------

def test_validate_verb_plan_does_not_raise():
    from app.services.cognitive_schedule_service import validate_verb
    validate_verb("plan")


def test_validate_verb_all_recognized_verbs():
    from app.services.cognitive_schedule_service import validate_verb
    for v in ("analyze", "summarize", "plan", "report", "monitor", "escalate", "acknowledge"):
        validate_verb(v)   # none should raise


def test_validate_verb_unknown_raises():
    from app.services.cognitive_schedule_service import validate_verb
    with pytest.raises(ValueError):
        validate_verb("launch_missiles")


def test_validate_verb_empty_raises():
    from app.services.cognitive_schedule_service import validate_verb
    with pytest.raises(ValueError):
        validate_verb("")


# ---------------------------------------------------------------------------
# 5. CognitiveScheduleService — mocked session (7 tests)
# ---------------------------------------------------------------------------

async def test_service_create_sets_next_run_at():
    from app.services.cognitive_schedule_service import CognitiveScheduleService
    svc = CognitiveScheduleService(session=_make_count_session(0), org_id="org-1")
    sched = await svc.create(cron_expression="* * * * *", cognitive_verb="plan")
    assert sched.next_run_at is not None


async def test_service_create_sets_organization_id():
    from app.services.cognitive_schedule_service import CognitiveScheduleService
    svc = CognitiveScheduleService(session=_make_count_session(0), org_id="org-42")
    sched = await svc.create(cron_expression="*/10 * * * *", cognitive_verb="analyze")
    assert sched.organization_id == "org-42"


async def test_service_create_stores_label():
    from app.services.cognitive_schedule_service import CognitiveScheduleService
    svc = CognitiveScheduleService(session=_make_count_session(0), org_id="org-1")
    sched = await svc.create(
        cron_expression="0 9 * * 1",
        cognitive_verb="report",
        label="Weekly Monday report",
    )
    assert sched.label == "Weekly Monday report"


async def test_service_create_enforces_max_limit():
    from app.services.cognitive_schedule_service import CognitiveScheduleService
    svc = CognitiveScheduleService(session=_make_count_session(20), org_id="org-1")
    with pytest.raises(ValueError, match="limit"):
        await svc.create(cron_expression="* * * * *", cognitive_verb="plan")


async def test_service_create_invalid_cron_raises():
    from app.services.cognitive_schedule_service import CognitiveScheduleService
    svc = CognitiveScheduleService(session=_make_count_session(0), org_id="org-1")
    with pytest.raises(ValueError):
        await svc.create(cron_expression="bad expression", cognitive_verb="plan")


async def test_service_create_invalid_verb_raises():
    from app.services.cognitive_schedule_service import CognitiveScheduleService
    svc = CognitiveScheduleService(session=_make_count_session(0), org_id="org-1")
    with pytest.raises(ValueError):
        await svc.create(cron_expression="* * * * *", cognitive_verb="destroy")


async def test_service_get_returns_none_when_missing():
    from app.services.cognitive_schedule_service import CognitiveScheduleService
    svc = CognitiveScheduleService(session=_make_get_session(None), org_id="org-1")
    result = await svc.get("nonexistent-id")
    assert result is None


# ---------------------------------------------------------------------------
# 6. Endpoint structure (3 tests)
# ---------------------------------------------------------------------------

def test_endpoints_router_importable():
    from app.api.v1.endpoints.cognitive_schedules import router
    assert router is not None


def test_endpoints_has_post_and_get_methods():
    from app.api.v1.endpoints.cognitive_schedules import router
    methods = {m for route in router.routes for m in (route.methods or [])}
    assert "POST" in methods
    assert "GET" in methods


def test_schedule_to_dict_returns_required_keys():
    from app.api.v1.endpoints.cognitive_schedules import _schedule_to_dict
    sched = MagicMock()
    sched.next_run_at = None
    sched.last_run_at = None
    sched.created_at = None
    result = _schedule_to_dict(sched)
    for key in ("id", "organization_id", "cron_expression", "cognitive_verb",
                "payload", "is_active", "next_run_at", "last_run_at", "created_at"):
        assert key in result, f"missing key in _schedule_to_dict output: {key}"


# ---------------------------------------------------------------------------
# 7. Celery task registration (3 tests)
# ---------------------------------------------------------------------------

def test_celery_runner_module_importable():
    from app.tasks import cognitive_schedule_runner  # noqa: F401
    assert True


def test_celery_runner_task_name():
    from app.tasks.cognitive_schedule_runner import run_cognitive_schedule_task
    assert run_cognitive_schedule_task.name == "cognitive_schedule_runner"


def test_celery_poller_task_name():
    from app.tasks.cognitive_schedule_runner import fire_due_schedules
    assert fire_due_schedules.name == "cognitive_schedule_poller"
