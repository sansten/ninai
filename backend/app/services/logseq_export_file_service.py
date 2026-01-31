"""Persistence for admin write-to-disk Logseq exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.logseq_export_file import LogseqExportFile


@dataclass(frozen=True)
class LogseqExportFileRecord:
    relative_path: str
    bytes_written: int
    options: dict[str, Any]


class LogseqExportFileService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def record_export(
        self,
        *,
        organization_id: str,
        record: LogseqExportFileRecord,
        requested_by_user_id: Optional[str],
        trace_id: Optional[str],
    ) -> dict[str, Any]:
        rel = (record.relative_path or "").strip()
        if not rel:
            return {"recorded": False, "reason": "missing_path"}

        stmt = (
            insert(LogseqExportFile)
            .values(
                {
                    "id": str(uuid4()),
                    "organization_id": organization_id,
                    "relative_path": rel,
                    "bytes_written": int(record.bytes_written or 0),
                    "requested_by_user_id": requested_by_user_id,
                    "trace_id": trace_id,
                    "options": record.options or {},
                }
            )
            .on_conflict_do_nothing(index_elements=["organization_id", "relative_path"])
        )

        await self.session.execute(stmt)
        await self.session.flush()
        return {"recorded": True, "reason": "ok"}
