"""Memory attachment schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from app.schemas.base import BaseSchema


class MemoryAttachmentResponse(BaseSchema):
    id: str
    organization_id: str
    memory_id: str
    uploaded_by: str

    file_name: str
    content_type: Optional[str] = None

    size_bytes: int = Field(..., ge=0)
    sha256: str

    indexed_at: Optional[datetime] = None

    created_at: datetime
    updated_at: datetime


class MemoryAttachmentListResponse(BaseSchema):
    items: list[MemoryAttachmentResponse]
    total: int
