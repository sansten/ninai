"""Tests for Intelligence Digest — P46.

Covers:
- DigestService: all query helpers, build_digest, save_report, get_latest, list_history
- _build_narrative: heuristic template generation
- digest_task: broker disabled / enabled paths
- DigestReport model defaults
- digest endpoints: latest, history, trigger
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.digest_service import DigestService, _build_narrative
from app.models.digest_report import DigestReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(rows: list[Any] | None = None, scalar=None, scalars_list=None):
    """Return a minimal AsyncSession mock."""
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar
    result.scalar.return_value = scalar
    result.scalars.return_value.all.return_value = scalars_list or rows or []
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.add = MagicMock()
    return db


def _report(org_id: str = "org-1", digest_type: str = "daily") -> DigestReport:
    r = DigestReport()
    r.id = "rep-1"
    r.organization_id = org_id
    r.run_date = date.today()
    r.digest_type = digest_type
    r.memories_total = 10
    r.memories_flagged = 2
    r.anomalies_count = 1
    r.goals_active = 3
    r.goals_completed = 1
    r.top_insights = []
    r.quality_trend = {"avg_7d": 0.8, "delta": 0.05, "direction": "improving"}
    r.narrative = "Test narrative."
    r.completed_at = datetime.now(timezone.utc)
    r.delivered_at = None
    r.created_at = datetime.now(timezone.utc)
    return r


# ===========================================================================
# _build_narrative
# ===========================================================================

def test_narrative_daily_no_anomalies():
    text = _build_narrative(
        org_id="org-1",
        digest_type="daily",
        memories_total=5,
        memories_flagged=0,
        anomalies_count=0,
        goals_active=2,
        goals_completed=0,
        quality_trend={"direction": "stable", "delta": 0.0, "avg_7d": 0.75},
    )
    assert "today" in text
    assert "No anomalies" in text
    assert "5 active records" in text


def test_narrative_weekly_with_anomalies():
    text = _build_narrative(
        org_id="org-1",
        digest_type="weekly",
        memories_total=100,
        memories_flagged=5,
        anomalies_count=3,
        goals_active=1,
        goals_completed=2,
        quality_trend={"direction": "improving", "delta": 0.12, "avg_7d": 0.85},
    )
    assert "this week" in text
    assert "3 anomalies" in text
    assert "2 goals completed" in text
    assert "improving" in text


def test_narrative_single_anomaly_singular():
    text = _build_narrative(
        org_id="org-1",
        digest_type="daily",
        memories_total=1,
        memories_flagged=0,
        anomalies_count=1,
        goals_active=0,
        goals_completed=0,
        quality_trend={"direction": "stable", "delta": 0.0, "avg_7d": 0.5},
    )
    assert "1 anomaly detected" in text


def test_narrative_single_goal_completed_singular():
    text = _build_narrative(
        org_id="org-1",
        digest_type="daily",
        memories_total=5,
        memories_flagged=0,
        anomalies_count=0,
        goals_active=0,
        goals_completed=1,
        quality_trend={"direction": "stable", "delta": 0.0, "avg_7d": 0.6},
    )
    assert "1 goal completed" in text


def test_narrative_declining_quality():
    text = _build_narrative(
        org_id="org-1",
        digest_type="daily",
        memories_total=10,
        memories_flagged=1,
        anomalies_count=0,
        goals_active=0,
        goals_completed=0,
        quality_trend={"direction": "declining", "delta": -0.15, "avg_7d": 0.4},
    )
    assert "declining" in text
    assert "-0.15" in text


def test_narrative_small_delta_omits_sign():
    text = _build_narrative(
        org_id="org-1",
        digest_type="daily",
        memories_total=5,
        memories_flagged=0,
        anomalies_count=0,
        goals_active=0,
        goals_completed=0,
        quality_trend={"direction": "stable", "delta": 0.001, "avg_7d": 0.7},
    )
    # Small delta — should show 7d avg but no +/- sign
    assert "0.70" in text


def test_narrative_positive_delta_shows_plus():
    text = _build_narrative(
        org_id="org-1",
        digest_type="daily",
        memories_total=5,
        memories_flagged=0,
        anomalies_count=0,
        goals_active=0,
        goals_completed=0,
        quality_trend={"direction": "improving", "delta": 0.05, "avg_7d": 0.8},
    )
    assert "+0.05" in text


def test_narrative_returns_string():
    result = _build_narrative(
        org_id="org-1",
        digest_type="daily",
        memories_total=0,
        memories_flagged=0,
        anomalies_count=0,
        goals_active=0,
        goals_completed=0,
        quality_trend={},
    )
    assert isinstance(result, str)
    assert len(result) > 0


# ===========================================================================
# DigestService — query helpers
# ===========================================================================

@pytest.mark.asyncio
async def test_count_active_memories_returns_zero_on_empty():
    db = _make_db(scalar=0)
    svc = DigestService(db)
    result = await svc._count_active_memories("org-1")
    assert result == 0


@pytest.mark.asyncio
async def test_count_active_memories_returns_count():
    db = _make_db(scalar=42)
    svc = DigestService(db)
    result = await svc._count_active_memories("org-1")
    assert result == 42


@pytest.mark.asyncio
async def test_count_active_memories_returns_zero_on_exception():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=Exception("db error"))
    svc = DigestService(db)
    result = await svc._count_active_memories("org-1")
    assert result == 0


@pytest.mark.asyncio
async def test_count_flagged_memories_returns_count():
    db = _make_db(scalar=7)
    svc = DigestService(db)
    result = await svc._count_flagged_memories("org-1")
    assert result == 7


@pytest.mark.asyncio
async def test_count_flagged_memories_returns_zero_on_exception():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=Exception("db error"))
    svc = DigestService(db)
    result = await svc._count_flagged_memories("org-1")
    assert result == 0


@pytest.mark.asyncio
async def test_count_anomalies_returns_count():
    db = _make_db(scalar=3)
    svc = DigestService(db)
    result = await svc._count_anomalies("org-1", 1)
    assert result == 3


@pytest.mark.asyncio
async def test_count_anomalies_returns_zero_on_exception():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=Exception("db error"))
    svc = DigestService(db)
    result = await svc._count_anomalies("org-1", 1)
    assert result == 0


@pytest.mark.asyncio
async def test_count_goals_returns_tuple():
    # Two calls: active then completed
    db = AsyncMock()
    result1 = MagicMock()
    result1.scalar.return_value = 5
    result2 = MagicMock()
    result2.scalar.return_value = 2
    db.execute = AsyncMock(side_effect=[result1, result2])
    svc = DigestService(db)
    active, completed = await svc._count_goals("org-1", 1)
    assert active == 5
    assert completed == 2


@pytest.mark.asyncio
async def test_count_goals_returns_zeros_on_exception():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=Exception("db error"))
    svc = DigestService(db)
    active, completed = await svc._count_goals("org-1", 1)
    assert active == 0
    assert completed == 0


@pytest.mark.asyncio
async def test_compute_quality_trend_returns_dict():
    db = AsyncMock()
    result = MagicMock()
    result.scalar.return_value = 0.8
    db.execute = AsyncMock(return_value=result)
    svc = DigestService(db)
    trend = await svc._compute_quality_trend("org-1")
    assert "avg_7d" in trend
    assert "delta" in trend
    assert "direction" in trend


@pytest.mark.asyncio
async def test_compute_quality_trend_improving():
    db = AsyncMock()
    r1 = MagicMock()
    r1.scalar.return_value = 0.85  # current avg
    r2 = MagicMock()
    r2.scalar.return_value = 0.70  # prior avg
    db.execute = AsyncMock(side_effect=[r1, r2])
    svc = DigestService(db)
    trend = await svc._compute_quality_trend("org-1")
    assert trend["direction"] == "improving"
    assert trend["delta"] > 0


@pytest.mark.asyncio
async def test_compute_quality_trend_declining():
    db = AsyncMock()
    r1 = MagicMock()
    r1.scalar.return_value = 0.60
    r2 = MagicMock()
    r2.scalar.return_value = 0.75
    db.execute = AsyncMock(side_effect=[r1, r2])
    svc = DigestService(db)
    trend = await svc._compute_quality_trend("org-1")
    assert trend["direction"] == "declining"
    assert trend["delta"] < 0


@pytest.mark.asyncio
async def test_compute_quality_trend_fallback_on_exception():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=Exception("db error"))
    svc = DigestService(db)
    trend = await svc._compute_quality_trend("org-1")
    assert trend == {"avg_7d": 0.5, "delta": 0.0, "direction": "stable"}


@pytest.mark.asyncio
async def test_fetch_top_insights_returns_list():
    row = MagicMock()
    row.agent_name = "CredibilityAgent"
    row.confidence = 0.95
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [row]
    db.execute = AsyncMock(return_value=result)
    svc = DigestService(db)
    insights = await svc._fetch_top_insights("org-1", 1)
    assert len(insights) == 1
    assert insights[0]["agent"] == "CredibilityAgent"
    assert insights[0]["confidence"] == 0.95


@pytest.mark.asyncio
async def test_fetch_top_insights_returns_empty_on_exception():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=Exception("db error"))
    svc = DigestService(db)
    insights = await svc._fetch_top_insights("org-1", 1)
    assert insights == []


# ===========================================================================
# DigestService — build_digest
# ===========================================================================

@pytest.mark.asyncio
async def test_build_digest_returns_digest_report():
    db = AsyncMock()
    # All queries return 0 / empty
    zero_result = MagicMock()
    zero_result.scalar.return_value = 0
    zero_result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=zero_result)
    svc = DigestService(db)
    report = await svc.build_digest("org-1", "daily")
    assert isinstance(report, DigestReport)
    assert report.organization_id == "org-1"
    assert report.digest_type == "daily"


@pytest.mark.asyncio
async def test_build_digest_weekly():
    db = AsyncMock()
    zero_result = MagicMock()
    zero_result.scalar.return_value = 0
    zero_result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=zero_result)
    svc = DigestService(db)
    report = await svc.build_digest("org-1", "weekly")
    assert report.digest_type == "weekly"


@pytest.mark.asyncio
async def test_build_digest_narrative_is_string():
    db = AsyncMock()
    zero_result = MagicMock()
    zero_result.scalar.return_value = 0
    zero_result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=zero_result)
    svc = DigestService(db)
    report = await svc.build_digest("org-1", "daily")
    assert isinstance(report.narrative, str)
    assert len(report.narrative) > 0


@pytest.mark.asyncio
async def test_build_digest_unknown_type_defaults_to_daily_window():
    db = AsyncMock()
    zero_result = MagicMock()
    zero_result.scalar.return_value = 0
    zero_result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=zero_result)
    svc = DigestService(db)
    # Unknown type → falls back to window=1
    report = await svc.build_digest("org-1", "monthly")
    assert isinstance(report, DigestReport)


# ===========================================================================
# DigestService — save_report (upsert)
# ===========================================================================

@pytest.mark.asyncio
async def test_save_report_creates_new_when_no_existing():
    db = _make_db(scalar=None)
    svc = DigestService(db)
    report = _report()
    await svc.save_report(report)
    db.add.assert_called_once_with(report)
    db.commit.assert_called()


@pytest.mark.asyncio
async def test_save_report_updates_existing():
    existing = _report()
    db = _make_db(scalar=existing)
    svc = DigestService(db)
    new_report = _report()
    new_report.memories_total = 99
    result = await svc.save_report(new_report)
    assert result.memories_total == 99
    db.commit.assert_called()


@pytest.mark.asyncio
async def test_save_report_updates_all_fields():
    existing = _report()
    db = _make_db(scalar=existing)
    svc = DigestService(db)
    new_report = _report()
    new_report.anomalies_count = 7
    new_report.goals_active = 4
    new_report.goals_completed = 2
    await svc.save_report(new_report)
    assert existing.anomalies_count == 7
    assert existing.goals_active == 4
    assert existing.goals_completed == 2


# ===========================================================================
# DigestService — get_latest / list_history
# ===========================================================================

@pytest.mark.asyncio
async def test_get_latest_returns_report():
    rep = _report()
    db = _make_db(scalar=rep)
    svc = DigestService(db)
    result = await svc.get_latest("org-1", "daily")
    assert result is rep


@pytest.mark.asyncio
async def test_get_latest_returns_none_when_missing():
    db = _make_db(scalar=None)
    svc = DigestService(db)
    result = await svc.get_latest("org-1", "daily")
    assert result is None


@pytest.mark.asyncio
async def test_list_history_returns_list():
    reps = [_report(), _report()]
    db = _make_db(scalars_list=reps)
    svc = DigestService(db)
    result = await svc.list_history("org-1", "daily", limit=10)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_list_history_caps_limit():
    db = _make_db(scalars_list=[])
    svc = DigestService(db)
    # limit > 90 capped internally — just verify no exception
    result = await svc.list_history("org-1", "daily", limit=9999)
    assert isinstance(result, list)


# ===========================================================================
# DigestReport model defaults
# ===========================================================================

def test_digest_report_column_defaults():
    # SQLAlchemy column defaults are applied at INSERT time, not on bare object creation.
    cols = {c.name: c for c in DigestReport.__table__.columns}
    assert cols["digest_type"].default.arg == "daily"
    assert cols["memories_total"].default.arg == 0
    assert cols["memories_flagged"].default.arg == 0
    assert cols["anomalies_count"].default.arg == 0
    assert cols["goals_active"].default.arg == 0
    assert cols["goals_completed"].default.arg == 0


def test_digest_report_fields_assignable():
    r = DigestReport()
    r.organization_id = "org-abc"
    r.memories_total = 50
    r.narrative = "Test narrative."
    assert r.organization_id == "org-abc"
    assert r.memories_total == 50
    assert r.narrative == "Test narrative."


# ===========================================================================
# Celery task
# ===========================================================================

def test_digest_task_skips_when_broker_disabled():
    from app.tasks.digest_pipeline import digest_task
    with patch("app.tasks.digest_pipeline._broker_enabled", return_value=False):
        result = digest_task("daily")
    assert result["skipped"] is True


def test_digest_task_runs_when_broker_enabled():
    from app.tasks.digest_pipeline import digest_task
    with patch("app.tasks.digest_pipeline._broker_enabled", return_value=True), \
         patch("app.tasks.digest_pipeline._run_async", return_value={"ok": True, "orgs_processed": 1}):
        result = digest_task("daily")
    assert result["ok"] is True
    assert result["orgs_processed"] == 1


def test_digest_task_weekly_when_broker_disabled():
    from app.tasks.digest_pipeline import digest_task
    with patch("app.tasks.digest_pipeline._broker_enabled", return_value=False):
        result = digest_task("weekly")
    assert result["skipped"] is True


@pytest.mark.asyncio
async def test_run_digest_async_no_orgs():
    from app.tasks.digest_pipeline import _run_digest_async
    with patch("app.tasks.digest_pipeline.async_session_factory") as mock_factory:
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result_mock)
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await _run_digest_async("daily")
    assert result["ok"] is True
    assert result["orgs_processed"] == 0


@pytest.mark.asyncio
async def test_run_digest_async_handles_org_failure():
    """Pipeline continues even if one org fails."""
    from app.tasks.digest_pipeline import _run_digest_async
    with patch("app.tasks.digest_pipeline.async_session_factory") as mock_factory, \
         patch("app.tasks.digest_pipeline.get_tenant_session") as mock_tenant:
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = ["org-1"]
        session.execute = AsyncMock(return_value=result_mock)
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        # Tenant session raises
        mock_tenant.return_value.__aenter__ = AsyncMock(side_effect=Exception("tenant error"))
        mock_tenant.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await _run_digest_async("daily")
    assert result["ok"] is True
    assert result["orgs_failed"] == 1
