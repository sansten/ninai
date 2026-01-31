"""Organization-level Logseq export configuration.

Supports Logseq integration admin UI requirements:
- Per-org configurable export path (base directory)
- Tracking nightly export cursor (last run timestamp)

Tenant isolation is enforced via PostgreSQL RLS on organization_id.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class OrgLogseqExportConfig(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "org_logseq_export_config"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this config belongs to (tenant isolation)",
    )

    export_base_dir: Mapped[Optional[str]] = mapped_column(
        String(length=1024),
        nullable=True,
        doc="Override base directory for Logseq exports (nullable -> env/default)",
    )

    last_nightly_export_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Cursor for nightly export: last successful run time (nullable)",
    )

    updated_by_user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="User who last updated the config (nullable)",
    )

    __table_args__ = (
        Index("ux_org_logseq_export_config_org", "organization_id", unique=True),
    )
