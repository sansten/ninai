"""LLM helper API schemas (Pydantic v2)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas.base import BaseSchema


class CompleteJsonRequest(BaseSchema):
    prompt: str = Field(..., min_length=1, max_length=100_000)
    schema_hint: dict[str, Any] = Field(default_factory=dict)


class CompleteJsonResponse(BaseSchema):
    data: dict[str, Any] = Field(default_factory=dict)
