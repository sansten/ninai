"""API key schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.base import BaseSchema


class ApiKeyCreateRequest(BaseSchema):
    name: str = Field(min_length=1, max_length=255)


class ApiKeyResponse(BaseSchema):
    id: str
    name: str
    prefix: str
    user_id: str
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


class ApiKeyCreateResponse(ApiKeyResponse):
    api_key: str
