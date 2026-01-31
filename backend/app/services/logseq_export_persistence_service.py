"""Logseq export persistence.

Materializes LogseqExportAgent outputs into Postgres (admin-only).

Notes:
- The Logseq API already enforces admin-only for write-to-disk exports.
- The agent pipeline can run without user roles attached, so we re-check
  org admin/system admin membership via the DB before persisting.

Tenant safety:
- Table is protected by Postgres RLS.
- Callers must pass organization_id.

Idempotency:
- One export row per (organization_id, memory_id).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory_logseq_export import MemoryLogseqExport
from app.models.user import Role, User, UserRole


@dataclass(frozen=True)
class ExtractedLogseqExport:
    markdown: str
    graph: dict[str, Any]
    item_count: int


def extract_logseq_export(outputs: dict[str, Any] | None) -> Optional[ExtractedLogseqExport]:
    if not isinstance(outputs, dict):
        return None

    md = outputs.get("markdown")
    graph = outputs.get("graph")

    if not isinstance(md, str):
        return None
    if not isinstance(graph, dict):
        return None

    try:
        item_count = int(outputs.get("item_count", 0) or 0)
    except Exception:
        item_count = 0

    return ExtractedLogseqExport(markdown=md, graph=graph, item_count=max(0, item_count))


class LogseqExportPersistenceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _is_org_admin_or_system_admin(self, *, organization_id: str, user_id: str) -> bool:
        if not user_id:
            return False

        user = await self.session.get(User, user_id)
        if user is not None and bool(getattr(user, "is_superuser", False)):
            return True

        stmt = (
            select(func.count(UserRole.id))
            .select_from(UserRole)
            .join(Role, Role.id == UserRole.role_id)
            .where(
                UserRole.user_id == user_id,
                UserRole.organization_id == organization_id,
                Role.name.in_(["org_admin", "system_admin"]),
            )
        )

        res = await self.session.execute(stmt)
        count = int(res.scalar() or 0)
        return count > 0

    async def upsert_export_for_memory(
        self,
        *,
        organization_id: str,
        memory_id: str,
        outputs: dict[str, Any] | None,
        updated_by_user_id: Optional[str],
        agent_version: Optional[str],
        trace_id: Optional[str],
        created_by: str = "agent",
    ) -> dict[str, Any]:
        extracted = extract_logseq_export(outputs)
        if extracted is None:
            return {"persisted": False, "reason": "no_export"}

        if not updated_by_user_id:
            return {"persisted": False, "reason": "no_actor"}

        allowed = await self._is_org_admin_or_system_admin(organization_id=organization_id, user_id=updated_by_user_id)
        if not allowed:
            return {"persisted": False, "reason": "not_admin"}

        stmt = (
            insert(MemoryLogseqExport)
            .values(
                {
                    "id": str(uuid4()),
                    "organization_id": organization_id,
                    "memory_id": memory_id,
                    "markdown": extracted.markdown,
                    "graph": extracted.graph,
                    "item_count": int(extracted.item_count),
                    "agent_version": agent_version,
                    "trace_id": trace_id,
                    "created_by": created_by,
                    "updated_by_user_id": updated_by_user_id,
                }
            )
            .on_conflict_do_update(
                index_elements=["organization_id", "memory_id"],
                set_={
                    "markdown": extracted.markdown,
                    "graph": extracted.graph,
                    "item_count": int(extracted.item_count),
                    "agent_version": agent_version,
                    "trace_id": trace_id,
                    "created_by": created_by,
                    "updated_by_user_id": updated_by_user_id,
                    "updated_at": func.now(),
                },
            )
        )

        await self.session.execute(stmt)
        await self.session.flush()
        return {"persisted": True, "reason": "ok", "item_count": int(extracted.item_count)}
