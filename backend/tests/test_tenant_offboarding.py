from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.endpoints.compliance import ExportRequest, request_export_or_deletion
from app.services.tenant_offboarding_service import OffboardingReport, TenantOffboardingService


@pytest.fixture
def tenant_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        user_id="user-1",
        organization_id="org-1",
        org_id="org-1",
        roles="org_admin",
    )


def _mk_db() -> AsyncMock:
    db = AsyncMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_export_org_data_creates_json_file(tmp_path):
    db = _mk_db()
    result = MagicMock()
    row = MagicMock()
    row._mapping = {"id": "m1", "organization_id": "org-1"}
    result.all.return_value = [row]
    db.execute.return_value = result

    svc = TenantOffboardingService(db)
    output = await svc.export_org_data("org-1", export_dir=str(tmp_path))

    assert output is not None
    parsed = json.loads(Path(output).read_text(encoding="utf-8"))
    assert parsed["org_id"] == "org-1"
    assert "tables" in parsed


@pytest.mark.asyncio
async def test_export_org_data_records_failed_tables_not_silent_empty(tmp_path):
    """Regression: a table SELECT failure used to be indistinguishable from
    "no rows found" — both produced an empty list. Failures must now be
    visible in tables_failed."""
    db = _mk_db()

    async def _execute(*args, **kwargs):
        raise RuntimeError("relation does not exist")

    db.execute = AsyncMock(side_effect=_execute)

    svc = TenantOffboardingService(db)
    output = await svc.export_org_data("org-1", export_dir=str(tmp_path))

    parsed = json.loads(Path(output).read_text(encoding="utf-8"))
    assert set(parsed["tables_failed"]) == set(TenantOffboardingService.TABLES_TO_PURGE)
    for table in TenantOffboardingService.TABLES_TO_PURGE:
        assert parsed["tables"][table] == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file permission bits")
@pytest.mark.asyncio
async def test_export_org_data_file_is_owner_only(tmp_path):
    db = _mk_db()
    result = MagicMock()
    result.all.return_value = []
    db.execute.return_value = result

    svc = TenantOffboardingService(db)
    output = await svc.export_org_data("org-1", export_dir=str(tmp_path))

    mode = stat.S_IMODE(Path(output).stat().st_mode)
    assert mode == 0o600


@pytest.mark.asyncio
async def test_delete_org_data_raises_on_table_failure_instead_of_silent_zero():
    """Regression: a failed DELETE used to be recorded as counts[table] = 0,
    identical to "there was nothing to delete" — a compliance-facing silent
    failure. It must now raise so the caller knows erasure is incomplete."""
    db = _mk_db()

    async def _execute(*args, **kwargs):
        raise RuntimeError("lock timeout")

    db.execute = AsyncMock(side_effect=_execute)

    svc = TenantOffboardingService(db)
    with pytest.raises(RuntimeError):
        await svc.delete_org_data("org-1")


@pytest.mark.asyncio
async def test_delete_org_data_returns_counts_and_user_anonymized():
    db = _mk_db()

    purge_result = MagicMock()
    purge_result.rowcount = 2
    anonymize_result = MagicMock()
    anonymize_result.rowcount = 3

    # First N calls for table purge, final call for anonymize users.
    db.execute.side_effect = [purge_result] * len(TenantOffboardingService.TABLES_TO_PURGE) + [anonymize_result]

    svc = TenantOffboardingService(db)
    counts = await svc.delete_org_data("org-1")

    assert counts["memories"] == 2
    assert counts["users_anonymized"] == 3


@pytest.mark.asyncio
async def test_offboard_with_export_first_sets_export_path(tmp_path):
    db = _mk_db()
    svc = TenantOffboardingService(db)

    with patch.object(svc, "export_org_data", new=AsyncMock(return_value=str(tmp_path / "exp.json"))), patch.object(
        svc,
        "delete_org_data",
        new=AsyncMock(return_value={"memories": 5, "users_anonymized": 2}),
    ):
        sub = SimpleNamespace(status="active")
        org = SimpleNamespace(id="org-1", settings={})
        sub_result = MagicMock()
        sub_result.scalar_one_or_none.return_value = sub
        org_result = MagicMock()
        org_result.scalar_one_or_none.return_value = org
        db.execute.side_effect = [sub_result, org_result]

        report = await svc.offboard("org-1", export_first=True, export_dir=str(tmp_path))

    assert isinstance(report, OffboardingReport)
    assert report.export_path is not None
    assert report.subscription_canceled is True
    assert org.settings.get("offboarded") is True


@pytest.mark.asyncio
async def test_offboard_export_first_false_skips_export():
    db = _mk_db()
    svc = TenantOffboardingService(db)

    with patch.object(svc, "export_org_data", new=AsyncMock()) as export_mock, patch.object(
        svc,
        "delete_org_data",
        new=AsyncMock(return_value={"memories": 1, "users_anonymized": 1}),
    ):
        sub_result = MagicMock()
        sub_result.scalar_one_or_none.return_value = None
        org_result = MagicMock()
        org_result.scalar_one_or_none.return_value = None
        db.execute.side_effect = [sub_result, org_result]

        report = await svc.offboard("org-1", export_first=False)

    export_mock.assert_not_awaited()
    assert report.export_path is None


@pytest.mark.asyncio
async def test_compliance_export_request_full_deletion_calls_offboard(tenant_ctx):
    db = _mk_db()
    body = ExportRequest(request_type="full_deletion", export_first=True)
    fake_report = OffboardingReport(
        org_id="org-1",
        memories_deleted=9,
        users_anonymized=4,
        subscription_canceled=True,
        export_path="/tmp/export.json",
        completed_at="2026-04-07T00:00:00+00:00",
    )

    with patch("app.api.v1.endpoints.compliance.TenantOffboardingService") as cls:
        inst = cls.return_value
        inst.offboard = AsyncMock(return_value=fake_report)
        out = await request_export_or_deletion(body=body, db=db, ctx=tenant_ctx)

    assert out["request_type"] == "full_deletion"
    assert out["offboarding"]["org_id"] == "org-1"
    assert out["offboarding"]["subscription_canceled"] is True


@pytest.mark.asyncio
async def test_compliance_export_request_data_export_path(tenant_ctx):
    db = _mk_db()
    body = ExportRequest(request_type="data_export", user_id="user-1")

    with patch(
        "app.api.v1.endpoints.compliance._create_gdpr_export_job",
        new=AsyncMock(return_value={"job_id": "job-1", "status": "queued"}),
    ):
        out = await request_export_or_deletion(body=body, db=db, ctx=tenant_ctx)

    assert out["job_id"] == "job-1"
    assert out["status"] == "queued"
