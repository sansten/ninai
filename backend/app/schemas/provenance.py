"""Provenance & citations schemas.

These schemas standardize how the API reports *why* an answer/result was
produced: the exact sources, snippets, versions, and trace correlation.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from app.schemas.base import BaseSchema


ProvenanceKind = Literal[
    "memory",
    "attachment",
    "document",
    "external",
]


class ProvenanceSource(BaseSchema):
    """A single source used to produce an answer/result."""

    kind: ProvenanceKind = "memory"

    # Canonical identifiers
    source_type: Optional[str] = None
    source_id: Optional[str] = None

    # Versioning / immutability hint (best-effort)
    source_version: Optional[str] = None
    content_hash: Optional[str] = None

    # Human-friendly details
    title: Optional[str] = None
    excerpt: Optional[str] = None
    uri: Optional[str] = None

    # Retrieval metadata
    score: Optional[float] = Field(default=None, ge=0.0)

    # Free-form extra metadata (e.g., created_at, updated_at, tags)
    meta: dict[str, Any] = Field(default_factory=dict)