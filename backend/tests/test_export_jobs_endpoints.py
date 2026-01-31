from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


@dataclass
class _FakeExportJob:
    id: str
    organization_id: str
    job_type: str
    status: str
    created_by_user_id: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    expires_at: datetime | None = None
    file_path: str | None = None
    file_bytes: int | None = None
    file_sha256: str | None = None
    error_message: str | None = None


class _ScalarOneOrNoneResult:
    def __init__(self, item: Any | None):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


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


class _CountResult:
    def __init__(self, value: int):
        self._value = value

    def scalar(self):
        return self._value


def _admin_headers(*, org_id: str = "o1", user_id: str = "u_admin") -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=["org_admin"])
    return {"Authorization": f"Bearer {token}"}


def _member_headers(*, org_id: str = "o1", user_id: str = "u_member") -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=["member"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_export_snapshot_job_requires_org_admin():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/v1/admin/exports/snapshots", headers=_member_headers(), json={"expires_in_seconds": 3600})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_export_jobs_requires_org_admin():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/admin/exports/jobs", headers=_member_headers())
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_export_jobs_happy_path(monkeypatch):
    now = datetime.now(timezone.utc)

    job1 = _FakeExportJob(
        id="j1",
        organization_id="o1",
        job_type="snapshot",
        status="succeeded",
        created_by_user_id="u_admin",
        created_at=now,
        updated_at=now,
    )
    job2 = _FakeExportJob(
        id="j2",
        organization_id="o1",
        job_type="snapshot",
        status="failed",
        created_by_user_id="u_admin",
        created_at=now,
        updated_at=now,
        error_message="boom",
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "count" in sql.lower() and "from export_jobs" in sql.lower():
            return _CountResult(2)
        if "FROM export_jobs" in sql:
            return _ListResult([job1, job2])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/admin/exports/jobs?page=1&page_size=50", headers=_admin_headers())
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 2
    assert payload["page"] == 1
    assert payload["page_size"] == 50
    assert len(payload["items"]) == 2


@pytest.mark.asyncio
async def test_export_snapshot_job_create_and_token_and_download(monkeypatch, tmp_path: Path):
    now = datetime.now(timezone.utc)

    # Ensure token signing works
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret", raising=False)

    job = _FakeExportJob(
        id="j1",
        organization_id="o1",
        job_type="snapshot",
        status="succeeded",
        created_by_user_id="u_admin",
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(hours=1),
    )

    # Create an on-disk zip to be returned by FileResponse
    zip_path = tmp_path / "snapshot_j1.zip"
    zip_path.write_bytes(b"PK\x03\x04")
    job.file_path = str(zip_path)
    job.file_bytes = zip_path.stat().st_size
    job.file_sha256 = "deadbeef"

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM export_jobs" in sql:
            return _ScalarOneOrNoneResult(job)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)
    session.flush = AsyncMock()
    session.commit = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    # Prevent actual celery enqueue
    calls: list[dict[str, Any]] = []

    import app.api.v1.endpoints.exports as exports_endpoints

    def _fake_send_task(name: str, kwargs: dict[str, Any]):
        calls.append({"name": name, "kwargs": kwargs})
        return None

    monkeypatch.setattr(exports_endpoints.celery_app, "send_task", _fake_send_task)

    # Make create_snapshot_job deterministic by bypassing DB-generated defaults.
    async def _fake_create_snapshot_job(self, *, organization_id: str, created_by_user_id: str | None, expires_in_seconds: int):
        assert organization_id == "o1"
        assert created_by_user_id == "u_admin"
        assert expires_in_seconds == 3600
        # Return queued job response, but keep job_id stable
        return _FakeExportJob(
            id="j1",
            organization_id="o1",
            job_type="snapshot",
            status="queued",
            created_by_user_id="u_admin",
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=3600),
        )

    monkeypatch.setattr(exports_endpoints.ExportJobService, "create_snapshot_job", _fake_create_snapshot_job)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create job
        resp = await ac.post(
            "/api/v1/admin/exports/snapshots",
            headers=_admin_headers(),
            json={"expires_in_seconds": 3600},
        )
        assert resp.status_code == 201
        payload = resp.json()
        assert payload["id"] == "j1"
        assert payload["status"] == "queued"
        assert len(calls) == 1
        assert calls[0]["name"] == "app.tasks.export_jobs.run_snapshot_export_job_task"

        # Get download token
        resp = await ac.get(
            "/api/v1/admin/exports/jobs/j1/download-token",
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        token_payload = resp.json()
        assert token_payload["job_id"] == "j1"
        assert "token" in token_payload
        token = token_payload["token"]

        # Download with token
        resp = await ac.get(f"/api/v1/exports/jobs/j1/download?token={token}")
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("application/zip")

    app.dependency_overrides.clear()
