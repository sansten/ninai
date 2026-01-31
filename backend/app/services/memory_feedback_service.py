from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import MemoryMetadata
from app.models.memory_feedback import MemoryFeedback


class MemoryFeedbackService:
    def __init__(self, session: AsyncSession, *, user_id: str, org_id: str):
        self.session = session
        self.user_id = user_id
        self.org_id = org_id

    async def create_feedback(
        self,
        *,
        memory_id: str,
        feedback_type: str,
        payload: dict[str, Any],
        target_agent: Optional[str] = None,
    ) -> MemoryFeedback:
        row = MemoryFeedback(
            organization_id=self.org_id,
            memory_id=memory_id,
            actor_id=self.user_id,
            feedback_type=feedback_type,
            target_agent=target_agent,
            payload=payload or {},
            is_applied=False,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_feedback(
        self,
        *,
        memory_id: str,
        include_applied: bool = True,
        limit: int = 50,
    ) -> tuple[list[MemoryFeedback], int]:
        stmt = select(MemoryFeedback).where(
            MemoryFeedback.organization_id == self.org_id,
            MemoryFeedback.memory_id == memory_id,
        )
        if not include_applied:
            stmt = stmt.where(MemoryFeedback.is_applied.is_(False))

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = int((await self.session.execute(count_stmt)).scalar() or 0)

        stmt = stmt.order_by(MemoryFeedback.created_at.desc()).limit(limit)
        rows = list((await self.session.execute(stmt)).scalars().all())
        return rows, total

    async def apply_pending_feedback(
        self,
        *,
        memory_id: str,
        applied_by: Optional[str] = None,
    ) -> dict[str, Any]:
        """Apply unapplied feedback signals to the memory_metadata row.

        Returns a summary dict: {applied_count:int, updates:list[dict]}
        """

        stmt = select(MemoryFeedback).where(
            MemoryFeedback.organization_id == self.org_id,
            MemoryFeedback.memory_id == memory_id,
            MemoryFeedback.is_applied.is_(False),
        ).order_by(MemoryFeedback.created_at.asc())

        feedback_rows = list((await self.session.execute(stmt)).scalars().all())
        if not feedback_rows:
            return {"applied_count": 0, "updates": []}

        memory = await self.session.get(MemoryMetadata, memory_id)
        if memory is None:
            return {"applied_count": 0, "updates": [], "warning": "memory_not_found"}

        updates: list[dict[str, Any]] = []

        def add_tag(tag: str) -> None:
            t = (tag or "").strip()
            if not t:
                return
            if t in (memory.tags or []):
                return
            memory.tags = list(memory.tags or []) + [t]

        def remove_tag(tag: str) -> None:
            t = (tag or "").strip()
            if not t:
                return
            memory.tags = [x for x in (memory.tags or []) if x != t]

        def add_entity(key: str, value: str) -> None:
            k = (key or "").strip()
            v = (value or "").strip()
            if not k or not v:
                return
            entities = dict(memory.entities or {})
            existing = entities.get(k)
            if existing is None:
                entities[k] = [v]
            elif isinstance(existing, list):
                if v not in existing:
                    entities[k] = list(existing) + [v]
            else:
                # coerce scalar to list
                if str(existing) != v:
                    entities[k] = [str(existing), v]
                else:
                    entities[k] = [v]
            memory.entities = entities

        def remove_entity(key: str, value: str) -> None:
            k = (key or "").strip()
            v = (value or "").strip()
            if not k or not v:
                return
            entities = dict(memory.entities or {})
            existing = entities.get(k)
            if isinstance(existing, list):
                entities[k] = [x for x in existing if str(x) != v]
                if not entities[k]:
                    entities.pop(k, None)
            elif existing is not None and str(existing) == v:
                entities.pop(k, None)
            memory.entities = entities

        def add_note(note: str, actor_id: str) -> None:
            n = (note or "").strip()
            if not n:
                return
            meta = dict(memory.extra_metadata or {})
            notes = meta.get("feedback_notes")
            if not isinstance(notes, list):
                notes = []
            notes.append({"note": n, "actor_id": actor_id, "at": datetime.now(timezone.utc).isoformat()})
            meta["feedback_notes"] = notes
            memory.extra_metadata = meta

        for fb in feedback_rows:
            payload = fb.payload or {}
            ftype = fb.feedback_type

            if ftype == "tag_add":
                tag = str(payload.get("tag", "")).strip()
                add_tag(tag)
                updates.append({"feedback_id": fb.id, "type": ftype, "tag": tag})
            elif ftype == "tag_remove":
                tag = str(payload.get("tag", "")).strip()
                remove_tag(tag)
                updates.append({"feedback_id": fb.id, "type": ftype, "tag": tag})
            elif ftype == "classification_override":
                classification = str(payload.get("classification", "")).strip()
                if classification:
                    memory.classification = classification
                updates.append({"feedback_id": fb.id, "type": ftype, "classification": classification})
            elif ftype == "entity_add":
                key = str(payload.get("key", "")).strip()
                value = str(payload.get("value", "")).strip()
                add_entity(key, value)
                updates.append({"feedback_id": fb.id, "type": ftype, "key": key, "value": value})
            elif ftype == "entity_remove":
                key = str(payload.get("key", "")).strip()
                value = str(payload.get("value", "")).strip()
                remove_entity(key, value)
                updates.append({"feedback_id": fb.id, "type": ftype, "key": key, "value": value})
            elif ftype == "note":
                note = str(payload.get("note", "")).strip()
                add_note(note, actor_id=fb.actor_id)
                updates.append({"feedback_id": fb.id, "type": ftype})
            else:
                updates.append({"feedback_id": fb.id, "type": ftype, "ignored": True})

        now = datetime.now(timezone.utc)
        fb_ids = [str(fb.id) for fb in feedback_rows]
        await self.session.execute(
            update(MemoryFeedback)
            .where(MemoryFeedback.id.in_(fb_ids), MemoryFeedback.organization_id == self.org_id)
            .values(is_applied=True, applied_at=now, applied_by=applied_by)
        )

        return {"applied_count": len(feedback_rows), "updates": updates}
