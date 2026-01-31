from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.export_job import ExportJob


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + padding).encode("ascii"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DownloadTokenPayload:
    job_id: str
    org_id: str
    exp: int


class ExportJobService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def snapshot_base_dir() -> Path:
        base = getattr(settings, "SNAPSHOT_EXPORT_DIR", None) or "exports/snapshots"
        return Path(str(base))

    @classmethod
    def org_snapshot_dir(cls, *, organization_id: str) -> Path:
        return cls.snapshot_base_dir() / organization_id

    @classmethod
    def job_dir(cls, *, organization_id: str, job_id: str) -> Path:
        return cls.org_snapshot_dir(organization_id=organization_id) / job_id

    @classmethod
    def job_zip_path(cls, *, organization_id: str, job_id: str) -> Path:
        return cls.job_dir(organization_id=organization_id, job_id=job_id) / f"snapshot_{job_id}.zip"

    async def create_snapshot_job(
        self,
        *,
        organization_id: str,
        created_by_user_id: Optional[str],
        expires_in_seconds: int,
    ) -> ExportJob:
        now = _utcnow()
        job = ExportJob(
            organization_id=organization_id,
            created_by_user_id=created_by_user_id,
            job_type="snapshot",
            status="queued",
            expires_at=now + timedelta(seconds=int(expires_in_seconds)),
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_job(self, *, organization_id: str, job_id: str) -> ExportJob | None:
        stmt = select(ExportJob).where(ExportJob.organization_id == organization_id, ExportJob.id == job_id)
        res = await self.session.execute(stmt)
        return res.scalar_one_or_none()

    async def mark_running(self, *, job: ExportJob) -> None:
        job.status = "running"
        job.started_at = _utcnow()
        job.error_message = None
        await self.session.flush()

    async def mark_failed(self, *, job: ExportJob, error_message: str) -> None:
        job.status = "failed"
        job.finished_at = _utcnow()
        job.error_message = (error_message or "").strip()[:2000] or "failed"
        await self.session.flush()

    async def mark_succeeded(self, *, job: ExportJob, file_path: str, file_bytes: int, file_sha256: str) -> None:
        job.status = "succeeded"
        job.finished_at = _utcnow()
        job.file_path = file_path
        job.file_bytes = int(file_bytes)
        job.file_sha256 = file_sha256
        job.error_message = None
        await self.session.flush()

    @staticmethod
    def build_download_token(*, job_id: str, org_id: str, expires_in_seconds: int = 900) -> tuple[str, datetime]:
        secret = getattr(settings, "SECRET_KEY", None)
        if not isinstance(secret, str) or not secret.strip():
            raise RuntimeError("SECRET_KEY must be set to issue download tokens")

        exp_dt = _utcnow() + timedelta(seconds=int(expires_in_seconds))
        payload = {"job_id": job_id, "org_id": org_id, "exp": int(exp_dt.timestamp())}
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
        token = f"{_b64url_encode(body)}.{_b64url_encode(sig)}"
        return token, exp_dt

    @staticmethod
    def verify_download_token(token: str) -> DownloadTokenPayload | None:
        secret = getattr(settings, "SECRET_KEY", None)
        if not isinstance(secret, str) or not secret.strip():
            return None

        if not isinstance(token, str) or "." not in token:
            return None

        body_b64, sig_b64 = token.split(".", 1)
        try:
            body = _b64url_decode(body_b64)
            sig = _b64url_decode(sig_b64)
        except Exception:
            return None

        expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, sig):
            return None

        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            return None

        job_id = str(data.get("job_id") or "").strip()
        org_id = str(data.get("org_id") or "").strip()
        exp = data.get("exp")
        if not job_id or not org_id:
            return None
        try:
            exp_i = int(exp)
        except Exception:
            return None
        if _utcnow().timestamp() > exp_i:
            return None
        return DownloadTokenPayload(job_id=job_id, org_id=org_id, exp=exp_i)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
