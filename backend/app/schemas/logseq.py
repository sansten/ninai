"""Logseq integration schemas.

These schemas support exporting memories in Logseq-friendly Markdown and
returning a simple graph representation (nodes/edges) for visualization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import Field

from app.schemas.base import BaseSchema


class LogseqExportFileLogEntry(BaseSchema):
    id: str
    created_at: datetime
    updated_at: datetime
    relative_path: str
    bytes_written: int
    requested_by_user_id: Optional[str] = None
    trace_id: Optional[str] = None
    options: dict[str, Any] = Field(default_factory=dict)


class LogseqExportRequest(BaseSchema):
    """Request options for Logseq export."""

    memory_ids: list[str] = Field(default_factory=list, description="If provided, export only these memory IDs")
    include_short_term: bool = Field(False, description="Include STM entries (Redis) when memory_ids are provided")

    # Fallback listing behavior when memory_ids is empty.
    scope: Optional[str] = Field(None, description="Optional scope filter for LTM listing")
    limit: int = Field(100, ge=0, le=1000)

    export_mode: Optional[Literal["single_file", "vault_pages"]] = Field(
        None,
        description=(
            "Export mode for write-to-disk exports. "
            "single_file writes one .md file. vault_pages writes a Logseq vault structure with one page per memory. "
            "If omitted, server defaults apply."
        ),
    )


class LogseqExportResponse(BaseSchema):
    """Exported Markdown response."""

    markdown: str
    item_count: int


class LogseqWriteExportResponse(BaseSchema):
    """Response for admin-only write-to-disk export."""

    relative_path: str
    bytes_written: int


class LogseqExportConfig(BaseSchema):
    export_base_dir: str
    org_export_dir: str
    last_nightly_export_at: Optional[datetime] = None


class LogseqExportConfigUpdate(BaseSchema):
    """Patch-like update.

    Convention: null means "inherit env/default".
    """

    export_base_dir: Optional[str] = None


class LogseqExportConfigResponse(BaseSchema):
    effective: LogseqExportConfig
    overrides: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseSchema):
    id: str
    type: str
    label: str
    data: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseSchema):
    source: str
    target: str
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class LogseqGraphResponse(BaseSchema):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
