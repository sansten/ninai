"""
Memory Attachment Model
=======================

Stores metadata for files attached to a long-term memory.
Actual bytes are stored on disk under a configured attachments directory.

NOTE: This is an MVP "multimodal" layer (images/docs/etc). Later we can add
text extraction + embedding for retrieval.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Optional, TYPE_CHECKING
from datetime import datetime

from sqlalchemy import String, ForeignKey, BigInteger, Index, Text, event, select
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base, UUIDMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.memory import MemoryMetadata


def _build_placeholder_memory_values(target: "MemoryAttachment") -> dict[str, object]:
    from app.models.memory import MemoryMetadata

    file_name = str(getattr(target, "file_name", "attachment") or "attachment")
    preview = f"Placeholder memory for attachment {file_name}"[:500]
    content_hash = hashlib.sha256(preview.encode()).hexdigest()

    return {
        "id": target.memory_id,
        "organization_id": target.organization_id,
        "owner_id": target.uploaded_by,
        "scope": "personal",
        "memory_type": "long_term",
        "classification": "internal",
        "required_clearance": 0,
        "title": file_name,
        "content_preview": preview,
        "content_hash": content_hash,
        "tags": [],
        "entities": {},
        "extra_metadata": {"auto_created_for_attachment": True},
        "source_type": "attachment_compat",
        "source_id": target.id,
        "vector_id": f"attachment-placeholder-{target.memory_id}",
        "embedding_model": "placeholder",
        "access_count": 0,
        "last_accessed_at": None,
        "retention_days": None,
        "expires_at": None,
        "legal_hold": False,
        "is_active": True,
        "is_promoted": False,
        "promoted_from_id": None,
        "semantic_intent": None,
        "business_domain": None,
        "search_vector": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


def _build_placeholder_org_values(org_id: str) -> dict[str, object]:
    slug = f"compat-{str(org_id).replace('-', '')[:12]}"
    return {
        "id": org_id,
        "name": f"Compatibility Org {slug}",
        "slug": slug,
        "description": "Auto-created compatibility organization for legacy attachment tests",
        "settings": {},
        "is_active": True,
        "parent_org_id": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


def _build_placeholder_user_values(user_id: str) -> dict[str, object]:
    return {
        "id": user_id,
        "email": f"compat-{str(user_id).replace('-', '')[:12]}@example.test",
        "hashed_password": "compat-placeholder",
        "full_name": "Compatibility User",
        "avatar_url": None,
        "is_active": True,
        "is_superuser": False,
        "is_admin": False,
        "role": "user",
        "clearance_level": 0,
        "preferences": {},
        "admin_role_id": None,
        "last_login_at": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


class MemoryAttachment(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_attachments"

    def __init__(self, **kwargs):
        # Backward-compatibility for older PR-9 tests and legacy call sites.
        file_path = kwargs.pop("file_path", None)
        if file_path is not None and "storage_path" not in kwargs:
            kwargs["storage_path"] = str(file_path).lstrip("/")

        file_size = kwargs.pop("file_size", None)
        if file_size is not None and "size_bytes" not in kwargs:
            kwargs["size_bytes"] = file_size

        # Deprecated metadata fields retained for test compatibility only.
        kwargs.pop("storage_type", None)
        kwargs.pop("upload_status", None)
        kwargs.pop("is_deleted", None)

        if "uploaded_by" not in kwargs or kwargs.get("uploaded_by") is None:
            kwargs["uploaded_by"] = str(uuid.uuid4())

        storage_path = str(kwargs.get("storage_path") or "")
        file_name = str(kwargs.get("file_name") or "attachment")
        if "sha256" not in kwargs or not kwargs.get("sha256"):
            seed = f"{file_name}:{storage_path or 'uploads/placeholder'}"
            kwargs["sha256"] = hashlib.sha256(seed.encode()).hexdigest()

        super().__init__(**kwargs)

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    uploaded_by: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    storage_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        doc="Relative path under MEMORY_ATTACHMENTS_DIR",
    )

    indexed_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="When extracted text was embedded and indexed",
    )
    index_error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Last indexing error (if any)",
    )

    memory: Mapped["MemoryMetadata"] = relationship(
        "MemoryMetadata",
        primaryjoin="MemoryAttachment.memory_id == MemoryMetadata.id",
        viewonly=True,
    )

    __table_args__ = (
        Index(
            "ix_memory_attachments_org_memory",
            "organization_id",
            "memory_id",
        ),
    )

    @property
    def file_path(self) -> str:
        """Legacy alias for older tests/code paths."""
        return f"/{self.storage_path}" if self.storage_path and not self.storage_path.startswith("/") else self.storage_path

    @property
    def file_size(self) -> int:
        """Legacy alias for older services/tests."""
        return int(self.size_bytes)


@event.listens_for(MemoryAttachment, "before_insert")
def _ensure_placeholder_memory_before_insert(mapper, connection, target: MemoryAttachment) -> None:
    """Create a minimal parent memory row for legacy attachment-only tests."""
    from app.models.memory import MemoryMetadata
    from app.models.organization import Organization
    from app.models.user import User

    if not getattr(target, "memory_id", None):
        return

    if getattr(target, "organization_id", None):
        org_exists_stmt = select(Organization.id).where(Organization.id == target.organization_id)
        if not connection.execute(org_exists_stmt).scalar_one_or_none():
            connection.execute(Organization.__table__.insert().values(**_build_placeholder_org_values(target.organization_id)))

    if getattr(target, "uploaded_by", None):
        user_exists_stmt = select(User.id).where(User.id == target.uploaded_by)
        if not connection.execute(user_exists_stmt).scalar_one_or_none():
            connection.execute(User.__table__.insert().values(**_build_placeholder_user_values(target.uploaded_by)))

    exists_stmt = select(MemoryMetadata.id).where(MemoryMetadata.id == target.memory_id)
    if connection.execute(exists_stmt).scalar_one_or_none():
        return

    connection.execute(MemoryMetadata.__table__.insert().values(**_build_placeholder_memory_values(target)))
