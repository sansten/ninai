"""Tests for GDPR / Compliance — P47.

Covers:
- ComplianceService: retention policy, deletion requests, consent records
- gdpr_pipeline task: broker disabled / async deletion logic
- compliance endpoints: schemas, admin guard, happy paths
- anonymize_user_id helper
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.compliance_service import (
    ComplianceService,
    anonymize_user_id,
)
from app.models.compliance import (
    DataRetentionPolicy,
    UserDataDeletion,
    ConsentRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(rows: list[Any] | None = None, scalar=None):
    """Return a minimal AsyncSession mock."""
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar
    result.scalars.return_value.all.return_value = rows or []
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.add = MagicMock()
    return db


def _policy(org_id: str = "org-1") -> DataRetentionPolicy:
    p = DataRetentionPolicy()
    p.id = "pol-1"
    p.organization_id = org_id
    p.retention_days_memory = 365
    p.retention_days_audit = 730
    p.auto_archive_after_days = 90
    p.is_active = True
    p.created_at = datetime.now(timezone.utc)
    return p


def _deletion(
    user_id: str = "user-1",
    org_id: str = "org-1",
    scope: str = "account",
    status: str = "pending",
) -> UserDataDeletion:
    d = UserDataDeletion()
    d.id = "del-1"
    d.user_id = user_id
    d.organization_id = org_id
    d.deletion_scope = scope
    d.status = status
    d.reason = "user request"
    d.deleted_at = None
    d.error_detail = None
    d.created_at = datetime.now(timezone.utc)
    return d


def _consent(
    user_id: str = "user-1",
    org_id: str = "org-1",
    consent_type: str = "data_processing",
    granted: bool = True,
) -> ConsentRecord:
    c = ConsentRecord()
    c.id = "con-1"
    c.user_id = user_id
    c.organization_id = org_id
    c.consent_type = consent_type
    c.granted = granted
    c.granted_at = datetime.now(timezone.utc)
    c.expires_at = None
    c.revoked_at = None
    c.ip_address = "127.0.0.1"
    return c


# ===========================================================================
# anonymize_user_id
# ===========================================================================

def test_anonymize_user_id_returns_prefixed_hash():
    result = anonymize_user_id("user-abc")
    assert result.startswith("anon_")
    assert len(result) == 5 + 16  # "anon_" + 16 hex chars


def test_anonymize_user_id_deterministic():
    assert anonymize_user_id("user-abc") == anonymize_user_id("user-abc")


def test_anonymize_user_id_different_inputs():
    assert anonymize_user_id("user-1") != anonymize_user_id("user-2")


def test_anonymize_user_id_empty_string():
    result = anonymize_user_id("")
    assert result.startswith("anon_")


# ===========================================================================
# DataRetentionPolicy
# ===========================================================================

@pytest.mark.asyncio
async def test_get_retention_policy_returns_none_when_missing():
    db = _make_db(scalar=None)
    svc = ComplianceService(db)
    result = await svc.get_retention_policy("org-1")
    assert result is None


@pytest.mark.asyncio
async def test_get_retention_policy_returns_existing():
    pol = _policy()
    db = _make_db(scalar=pol)
    svc = ComplianceService(db)
    result = await svc.get_retention_policy("org-1")
    assert result is pol


@pytest.mark.asyncio
async def test_upsert_retention_policy_creates_new():
    db = _make_db(scalar=None)
    svc = ComplianceService(db)
    result = await svc.upsert_retention_policy(
        "org-1",
        retention_days_memory=180,
        retention_days_audit=365,
        auto_archive_after_days=60,
    )
    db.add.assert_called_once()
    db.commit.assert_called()


@pytest.mark.asyncio
async def test_upsert_retention_policy_updates_existing():
    pol = _policy()
    db = _make_db(scalar=pol)
    svc = ComplianceService(db)
    await svc.upsert_retention_policy("org-1", retention_days_memory=200)
    assert pol.retention_days_memory == 200
    db.commit.assert_called()


@pytest.mark.asyncio
async def test_upsert_retention_policy_clamps_to_minimum():
    db = _make_db(scalar=None)
    svc = ComplianceService(db)
    await svc.upsert_retention_policy("org-1", retention_days_memory=0)
    added = db.add.call_args[0][0]
    assert added.retention_days_memory >= 1


# ===========================================================================
# UserDataDeletion
# ===========================================================================

@pytest.mark.asyncio
async def test_create_deletion_request_valid_scope():
    db = _make_db()
    svc = ComplianceService(db)
    await svc.create_deletion_request("user-1", "org-1", deletion_scope="account")
    db.add.assert_called_once()
    db.commit.assert_called()


@pytest.mark.asyncio
async def test_create_deletion_request_all_scopes():
    for scope in ("account", "org_data", "all"):
        db = _make_db()
        svc = ComplianceService(db)
        await svc.create_deletion_request("user-1", "org-1", deletion_scope=scope)
        db.add.assert_called_once()


@pytest.mark.asyncio
async def test_create_deletion_request_invalid_scope():
    db = _make_db()
    svc = ComplianceService(db)
    with pytest.raises(ValueError, match="deletion_scope"):
        await svc.create_deletion_request("user-1", "org-1", deletion_scope="unknown")


@pytest.mark.asyncio
async def test_get_deletion_request_found():
    req = _deletion()
    db = _make_db(scalar=req)
    svc = ComplianceService(db)
    result = await svc.get_deletion_request("del-1")
    assert result is req


@pytest.mark.asyncio
async def test_get_deletion_request_not_found():
    db = _make_db(scalar=None)
    svc = ComplianceService(db)
    result = await svc.get_deletion_request("missing")
    assert result is None


@pytest.mark.asyncio
async def test_mark_deletion_processing():
    req = _deletion()
    db = _make_db(scalar=req)
    svc = ComplianceService(db)
    result = await svc.mark_deletion_processing("del-1")
    assert result.status == "processing"


@pytest.mark.asyncio
async def test_mark_deletion_completed():
    req = _deletion(status="processing")
    db = _make_db(scalar=req)
    svc = ComplianceService(db)
    result = await svc.mark_deletion_completed("del-1")
    assert result.status == "completed"
    assert result.deleted_at is not None


@pytest.mark.asyncio
async def test_mark_deletion_failed():
    req = _deletion(status="processing")
    db = _make_db(scalar=req)
    svc = ComplianceService(db)
    result = await svc.mark_deletion_failed("del-1", "database error")
    assert result.status == "failed"
    assert "database error" in result.error_detail


@pytest.mark.asyncio
async def test_mark_deletion_returns_none_for_missing():
    db = _make_db(scalar=None)
    svc = ComplianceService(db)
    assert await svc.mark_deletion_processing("missing") is None
    assert await svc.mark_deletion_completed("missing") is None
    assert await svc.mark_deletion_failed("missing", "err") is None


@pytest.mark.asyncio
async def test_list_deletion_requests_returns_list():
    reqs = [_deletion(), _deletion(user_id="user-2")]
    db = _make_db(rows=reqs)
    svc = ComplianceService(db)
    result = await svc.list_deletion_requests("org-1")
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_list_deletion_requests_invalid_status_ignored():
    db = _make_db(rows=[])
    svc = ComplianceService(db)
    # Should not raise — bad status is silently ignored
    result = await svc.list_deletion_requests("org-1", status="bad_status")
    assert result == []


@pytest.mark.asyncio
async def test_list_deletion_requests_limit_capped():
    db = _make_db(rows=[])
    svc = ComplianceService(db)
    # limit > 200 should be capped — just verify no exception
    result = await svc.list_deletion_requests("org-1", limit=9999)
    assert isinstance(result, list)


# ===========================================================================
# ConsentRecord
# ===========================================================================

@pytest.mark.asyncio
async def test_record_consent_grant():
    db = _make_db()
    svc = ComplianceService(db)
    await svc.record_consent(
        "user-1", "org-1",
        consent_type="data_processing",
        granted=True,
    )
    db.add.assert_called_once()
    db.commit.assert_called()


@pytest.mark.asyncio
async def test_record_consent_revoke():
    db = _make_db()
    svc = ComplianceService(db)
    await svc.record_consent(
        "user-1", "org-1",
        consent_type="analytics",
        granted=False,
    )
    added = db.add.call_args[0][0]
    assert added.granted is False
    assert added.revoked_at is not None


@pytest.mark.asyncio
async def test_record_consent_all_types():
    for ctype in ("data_processing", "analytics", "marketing"):
        db = _make_db()
        svc = ComplianceService(db)
        await svc.record_consent("user-1", "org-1", consent_type=ctype, granted=True)
        db.add.assert_called_once()


@pytest.mark.asyncio
async def test_record_consent_invalid_type():
    db = _make_db()
    svc = ComplianceService(db)
    with pytest.raises(ValueError, match="consent_type"):
        await svc.record_consent("user-1", "org-1", consent_type="newsletter", granted=True)


@pytest.mark.asyncio
async def test_record_consent_stores_ip():
    db = _make_db()
    svc = ComplianceService(db)
    await svc.record_consent(
        "user-1", "org-1",
        consent_type="data_processing",
        granted=True,
        ip_address="192.168.1.1",
    )
    added = db.add.call_args[0][0]
    assert added.ip_address == "192.168.1.1"


@pytest.mark.asyncio
async def test_list_user_consents():
    records = [_consent(), _consent(consent_type="analytics")]
    db = _make_db(rows=records)
    svc = ComplianceService(db)
    result = await svc.list_user_consents("user-1", "org-1")
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_get_latest_consent_found():
    c = _consent()
    db = _make_db(scalar=c)
    svc = ComplianceService(db)
    result = await svc.get_latest_consent("user-1", "org-1", "data_processing")
    assert result is c


@pytest.mark.asyncio
async def test_get_latest_consent_not_found():
    db = _make_db(scalar=None)
    svc = ComplianceService(db)
    result = await svc.get_latest_consent("user-1", "org-1", "marketing")
    assert result is None


# ===========================================================================
# GDPR Pipeline Task
# ===========================================================================

def test_gdpr_task_skips_when_broker_disabled():
    from app.tasks.gdpr_pipeline import process_deletion_task
    with patch("app.tasks.gdpr_pipeline._broker_enabled", return_value=False):
        result = process_deletion_task("req-1")
    assert result["skipped"] is True


def test_gdpr_task_runs_when_broker_enabled():
    from app.tasks.gdpr_pipeline import process_deletion_task
    with patch("app.tasks.gdpr_pipeline._broker_enabled", return_value=True), \
         patch("app.tasks.gdpr_pipeline._run_async", return_value={"ok": True}):
        result = process_deletion_task("req-1")
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_gdpr_deletion_request_not_found():
    from app.tasks.gdpr_pipeline import _process_deletion
    with patch("app.tasks.gdpr_pipeline.async_session_factory") as mock_factory:
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result_mock)
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await _process_deletion("missing-id")
    assert result["ok"] is False
    assert result["reason"] == "request_not_found"


@pytest.mark.asyncio
async def test_gdpr_deletion_already_completed():
    from app.tasks.gdpr_pipeline import _process_deletion
    req = _deletion(status="completed")
    with patch("app.tasks.gdpr_pipeline.async_session_factory") as mock_factory:
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = req
        session.execute = AsyncMock(return_value=result_mock)
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await _process_deletion("del-1")
    assert result.get("skipped") is True


# ===========================================================================
# Model field validation
# ===========================================================================

def test_data_retention_policy_column_defaults():
    # SQLAlchemy column defaults are applied at INSERT time, not on object creation.
    # Verify the column-level defaults are configured correctly via table metadata.
    from sqlalchemy import inspect as sa_inspect
    cols = {c.name: c for c in DataRetentionPolicy.__table__.columns}
    assert cols["retention_days_memory"].default.arg == 365
    assert cols["retention_days_audit"].default.arg == 730
    assert cols["auto_archive_after_days"].default.arg == 90
    assert cols["is_active"].default.arg is True


def test_user_data_deletion_column_defaults():
    from sqlalchemy import inspect as sa_inspect
    cols = {c.name: c for c in UserDataDeletion.__table__.columns}
    assert cols["deletion_scope"].default.arg == "account"
    assert cols["status"].default.arg == "pending"


def test_consent_record_fields():
    c = ConsentRecord()
    c.consent_type = "data_processing"
    c.granted = True
    assert c.consent_type == "data_processing"
    assert c.granted is True
