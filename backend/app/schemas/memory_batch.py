"""Batch operations schemas for memories."""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from app.schemas.base import BaseSchema
from app.schemas.memory import MemoryResponse, MemoryUpdate, MemoryShareRequest


class MemoryBatchUpdateItem(BaseSchema):
    memory_id: str
    update: MemoryUpdate


class MemoryBatchUpdateRequest(BaseSchema):
    items: List[MemoryBatchUpdateItem] = Field(..., min_length=1, max_length=200)


class MemoryBatchResult(BaseSchema):
    memory_id: str
    success: bool
    error: Optional[str] = None


class MemoryBatchUpdateResult(MemoryBatchResult):
    memory: Optional[MemoryResponse] = None


class MemoryBatchUpdateResponse(BaseSchema):
    trace_id: Optional[str] = None
    results: List[MemoryBatchUpdateResult]


class MemoryBatchDeleteRequest(BaseSchema):
    memory_ids: List[str] = Field(..., min_length=1, max_length=500)


class MemoryBatchDeleteResponse(BaseSchema):
    trace_id: Optional[str] = None
    results: List[MemoryBatchResult]


class MemoryBatchShareRequest(BaseSchema):
    memory_ids: List[str] = Field(..., min_length=1, max_length=500)
    share: MemoryShareRequest


class MemoryBatchShareResult(MemoryBatchResult):
    share_id: Optional[str] = None


class MemoryBatchShareResponse(BaseSchema):
    trace_id: Optional[str] = None
    results: List[MemoryBatchShareResult]
