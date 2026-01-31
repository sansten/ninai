"""Topic persistence.

Materializes TopicModelingAgent outputs into Postgres tables:
- memory_topics
- memory_topic_memberships

The service is tenant-safe (requires org_id) and idempotent via upserts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory_topic import MemoryTopic
from app.models.memory_topic_membership import MemoryTopicMembership


_LABEL_RE = re.compile(r"[^a-z0-9_]+")


def normalize_topic_label(label: str) -> str:
    s = (label or "").strip().lower()
    s = s.replace(" ", "_")
    s = _LABEL_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass(frozen=True)
class ExtractedTopics:
    topics: list[str]
    primary_topic: str
    confidence: float


def extract_topics(outputs: dict[str, Any] | None) -> ExtractedTopics | None:
    if not isinstance(outputs, dict):
        return None

    topics = outputs.get("topics")
    primary = outputs.get("primary_topic")
    if not isinstance(topics, list) or not isinstance(primary, str):
        return None

    cleaned: list[str] = []
    seen: set[str] = set()
    for t in topics:
        if not isinstance(t, str):
            continue
        n = normalize_topic_label(t)
        if not n or n in seen:
            continue
        seen.add(n)
        cleaned.append(n)

    primary_norm = normalize_topic_label(primary)
    if primary_norm and primary_norm not in seen:
        cleaned.insert(0, primary_norm)
        seen.add(primary_norm)

    if not cleaned:
        return None

    try:
        conf = float(outputs.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    return ExtractedTopics(topics=cleaned[:8], primary_topic=primary_norm or cleaned[0], confidence=conf)


class TopicService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_topics_for_memory(
        self,
        *,
        organization_id: str,
        memory_id: str,
        scope: str,
        scope_id: str | None,
        outputs: dict[str, Any] | None,
        created_by: str = "agent",
    ) -> dict[str, Any]:
        extracted = extract_topics(outputs)
        if extracted is None:
            return {"topics_upserted": 0, "memberships_upserted": 0}

        scope_key = f"{scope}:{scope_id or ''}"

        topics_upserted = 0
        memberships_upserted = 0

        for t in extracted.topics:
            is_primary = t == extracted.primary_topic

            topic_insert = (
                insert(MemoryTopic)
                .values(
                    {
                        "id": str(uuid4()),
                        "organization_id": organization_id,
                        "scope": scope,
                        "scope_id": scope_id,
                        "scope_key": scope_key,
                        "label": t,
                        "label_normalized": t,
                        "keywords": [],
                        "created_by": created_by,
                    }
                )
                .on_conflict_do_update(
                    index_elements=["organization_id", "scope_key", "label_normalized"],
                    set_={
                        "label": t,
                        "updated_at": func.now(),
                    },
                )
                .returning(MemoryTopic.id)
            )

            topic_res = await self.session.execute(topic_insert)
            topic_id = topic_res.scalar_one()
            topics_upserted += 1

            membership_insert = (
                insert(MemoryTopicMembership)
                .values(
                    {
                        "id": str(uuid4()),
                        "organization_id": organization_id,
                        "memory_id": memory_id,
                        "topic_id": str(topic_id),
                        "is_primary": bool(is_primary),
                        "weight": float(extracted.confidence),
                        "created_by": created_by,
                    }
                )
                .on_conflict_do_update(
                    index_elements=["organization_id", "memory_id", "topic_id"],
                    set_={
                        "is_primary": bool(is_primary),
                        "weight": float(extracted.confidence),
                        "updated_at": func.now(),
                    },
                )
            )
            await self.session.execute(membership_insert)
            memberships_upserted += 1

        await self.session.flush()
        return {"topics_upserted": topics_upserted, "memberships_upserted": memberships_upserted}
