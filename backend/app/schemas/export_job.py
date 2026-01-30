from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from app.schemas.base import BaseSchema


class SnapshotExportRequest(BaseSchema):
    """Create a snapshot export job."""

    expires_in_seconds: int = Field(86400, ge=60, le=7 * 86400)


class ExportJobResponse(BaseSchema):
    id: str
    organization_id: str
    job_type: str
    status: str

    created_by_user_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    file_bytes: Optional[int] = None
    file_sha256: Optional[str] = None

    error_message: Optional[str] = None


class ExportJobDownloadTokenResponse(BaseSchema):
    job_id: str
    expires_at: datetime
    token: str
    download_url: str
