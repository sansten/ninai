"""Memory attachment service.

MVP: store bytes on local disk and metadata in Postgres.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.qdrant import QdrantService
from app.models.memory import MemoryMetadata
from app.models.memory_attachment import MemoryAttachment
from app.services.audit_service import AuditService
from app.services.permission_checker import PermissionChecker
from app.services.attachment_text_extractor import extract_text_for_indexing_from_file
from app.services.embedding_service import EmbeddingService


class AttachmentTooLargeError(ValueError):
    pass


class AttachmentNotFoundError(ValueError):
    pass


class AttachmentStorageError(RuntimeError):
    pass


class MemoryAttachmentService:
    def __init__(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        clearance_level: int = 0,
    ):
        self.session = session
        self.user_id = user_id
        self.org_id = org_id
        self.clearance_level = clearance_level

        self.permission_checker = PermissionChecker(session)
        self.audit_service = AuditService(session)

    def _attachments_root(self) -> Path:
        primary = Path(settings.MEMORY_ATTACHMENTS_DIR or "data/memory_attachments")
        fallback = Path("/tmp/ninai_memory_attachments")

        for candidate in (primary, fallback):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / ".ninai_write_probe"
                with open(probe, "wb") as f:
                    f.write(b"1")
                try:
                    probe.unlink()
                except Exception:
                    pass
                return candidate
            except Exception:
                continue

        return primary

    def _attachment_relpath(self, memory_id: str, attachment_id: str) -> str:
        # Keep it deterministic and path-traversal safe.
        return str(Path(self.org_id) / memory_id / attachment_id)

    def _attachment_abspath(self, relpath: str) -> Path:
        root = self._attachments_root().resolve()
        target = (root / relpath).resolve()
        if root not in target.parents and target != root:
            raise ValueError("Invalid attachment path")
        return target

    async def _require_memory_access(self, memory_id: str, action: str) -> MemoryMetadata:
        access = await self.permission_checker.check_memory_access(
            self.user_id,
            self.org_id,
            memory_id,
            action,
            self.clearance_level,
        )

        if not access.allowed:
            await self.audit_service.log_memory_operation(
                actor_id=self.user_id,
                organization_id=self.org_id,
                memory_id=memory_id,
                operation=f"attachment_{action}_denied",
                success=False,
                error_message=access.reason,
            )
            raise PermissionError(access.reason)

        memory = await self.session.get(MemoryMetadata, memory_id)
        if memory is None:
            raise AttachmentNotFoundError("Memory not found")
        return memory

    async def list_attachments(self, memory_id: str) -> list[MemoryAttachment]:
        await self._require_memory_access(memory_id, "read")

        q = (
            select(MemoryAttachment)
            .where(MemoryAttachment.organization_id == self.org_id)
            .where(MemoryAttachment.memory_id == memory_id)
            .order_by(MemoryAttachment.created_at.desc())
        )
        res = await self.session.execute(q)
        return list(res.scalars().all())

    async def create_attachment(self, memory_id: str, file: UploadFile) -> MemoryAttachment:
        memory = await self._require_memory_access(memory_id, "write")

        if file.filename is None or file.filename.strip() == "":
            raise ValueError("File must have a filename")

        attachment_id = str(uuid4())

        relpath = self._attachment_relpath(memory_id=memory_id, attachment_id=attachment_id)
        abspath = self._attachment_abspath(relpath)
        try:
            abspath.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise AttachmentStorageError(
                f"Attachment storage not writable at {self._attachments_root()}"
            ) from e

        hasher = hashlib.sha256()
        total = 0
        max_bytes = int(settings.MAX_ATTACHMENT_SIZE_BYTES or 25 * 1024 * 1024)

        try:
            with open(abspath, "wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise AttachmentTooLargeError(
                            f"Attachment too large (max {max_bytes} bytes)"
                        )
                    hasher.update(chunk)
                    out.write(chunk)
        except PermissionError as e:
            raise AttachmentStorageError(
                f"Attachment storage not writable at {self._attachments_root()}"
            ) from e
        except Exception:
            if abspath.exists():
                try:
                    abspath.unlink()
                except Exception:
                    pass
            raise
        finally:
            await file.close()

        att = MemoryAttachment(
            id=attachment_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            uploaded_by=self.user_id,
            file_name=file.filename[:255],
            content_type=(file.content_type[:255] if file.content_type else None),
            size_bytes=total,
            sha256=hasher.hexdigest(),
            storage_path=relpath,
        )

        self.session.add(att)
        await self.session.flush()

        await self.audit_service.log_memory_operation(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            operation="attachment_add",
            success=True,
            details={
                "attachment_id": att.id,
                "file_name": att.file_name,
                "content_type": att.content_type,
                "size_bytes": att.size_bytes,
            },
        )

        # Best-effort indexing into Qdrant using extracted text.
        if bool(settings.ATTACHMENT_INDEXING_ENABLED if settings.ATTACHMENT_INDEXING_ENABLED is not None else True):
            try:
                max_chars = int(settings.ATTACHMENT_INDEX_MAX_CHARS or 20000)
                max_file_bytes = int(settings.ATTACHMENT_INDEX_MAX_FILE_BYTES or 10 * 1024 * 1024)
                text = extract_text_for_indexing_from_file(
                    content_type=att.content_type,
                    filename=att.file_name,
                    file_path=abspath,
                    max_chars=max_chars,
                    max_bytes=max_file_bytes,
                    ocr_service_url=getattr(settings, "OCR_SERVICE_URL", None),
                    ocr_timeout_seconds=float(getattr(settings, "OCR_SERVICE_TIMEOUT_SECONDS", 5.0) or 5.0),
                )

                if text and text.strip():
                    vector = await EmbeddingService.embed(text)
                    await QdrantService.upsert_memory(
                        memory_id=att.id,  # Qdrant point id
                        org_id=self.org_id,
                        vector=vector,
                        payload={
                            "kind": "attachment",
                            "memory_id": memory_id,
                            "attachment_id": att.id,
                            "file_name": att.file_name,
                            "content_type": att.content_type,
                            "scope": memory.scope,
                            "scope_id": memory.scope_id,
                            "owner_id": memory.owner_id,
                            "classification": memory.classification,
                        },
                    )
                    att.indexed_at = EmbeddingService.utcnow().replace(tzinfo=None)
                    att.index_error = None
                else:
                    att.indexed_at = None
                    att.index_error = "no_indexable_text"
            except Exception as e:
                att.indexed_at = None
                att.index_error = str(e)[:500]

        return att

    async def get_attachment(self, memory_id: str, attachment_id: str) -> tuple[MemoryAttachment, Path]:
        await self._require_memory_access(memory_id, "read")

        att = await self.session.get(MemoryAttachment, attachment_id)
        if att is None or att.organization_id != self.org_id or att.memory_id != memory_id:
            raise AttachmentNotFoundError("Attachment not found")

        abspath = self._attachment_abspath(att.storage_path)
        if not abspath.exists():
            raise AttachmentNotFoundError("Attachment file missing")

        return att, abspath

    async def delete_attachment(self, memory_id: str, attachment_id: str) -> None:
        await self._require_memory_access(memory_id, "write")

        att = await self.session.get(MemoryAttachment, attachment_id)
        if att is None or att.organization_id != self.org_id or att.memory_id != memory_id:
            raise AttachmentNotFoundError("Attachment not found")

        abspath = self._attachment_abspath(att.storage_path)

        await self.session.delete(att)
        await self.session.flush()

        # Best-effort delete from Qdrant.
        try:
            await QdrantService.delete_point(point_id=attachment_id)
        except Exception:
            pass

        if abspath.exists():
            try:
                abspath.unlink()
            except Exception:
                # Best-effort; DB is source of truth.
                pass

        await self.audit_service.log_memory_operation(
            actor_id=self.user_id,
            organization_id=self.org_id,
            memory_id=memory_id,
            operation="attachment_delete",
            success=True,
            details={"attachment_id": attachment_id},
        )
