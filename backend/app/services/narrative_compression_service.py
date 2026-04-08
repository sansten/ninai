"""Narrative compression service (Phase 56)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.narrative_compression_agent import NarrativeCompressionAgent, run_heuristic
from app.models.memory import MemoryMetadata


class NarrativeCompressionService:
    """Compress episodic sequences into a narrative and archive source memories."""

    async def compress_and_archive(
        self,
        *,
        db: AsyncSession,
        org_id: str,
        user_id: str,
        episodes: list[dict[str, Any]],
        topic: str,
        max_sentences: int = 3,
    ) -> dict[str, Any]:
        agent = NarrativeCompressionAgent()
        result = await agent.run(
            memory_id=f"narrative-compression:{uuid4()}",
            context={
                "memory": {
                    "enrichment": {
                        "episodes": episodes,
                        "topic": topic,
                        "max_sentences": max_sentences,
                    }
                },
                "runtime": {},
            },
        )

        outputs = result.outputs or {}
        fallback_outputs = run_heuristic(
            episodes=episodes,
            topic=topic,
            max_sentences=max_sentences,
        )

        narrative = str(outputs.get("compressed_narrative") or "")
        archived_ids = sorted(
            {
                str(item_id)
                for item_id in [
                    *(outputs.get("archived_ids") or []),
                    *(fallback_outputs.get("archived_ids") or []),
                ]
                if item_id
            }
        )

        new_memory_id: str | None = None
        if narrative.strip():
            content_hash = hashlib.sha256(narrative.encode("utf-8")).hexdigest()
            memory = MemoryMetadata(
                id=str(uuid4()),
                organization_id=org_id,
                owner_id=user_id,
                scope="personal",
                classification="internal",
                title=f"Narrative compression: {topic}"[:500],
                content_preview=narrative[:500],
                content_hash=content_hash,
                tags=["narrative_compression"],
                entities={},
                extra_metadata={
                    "topic": topic,
                    "source_episode_count": len(episodes),
                    "compression_ratio": outputs.get("compression_ratio"),
                },
                source_type="narrative_compression",
                source_id=None,
                vector_id=f"narrative-compression:{uuid4()}",
                embedding_model="heuristic",
            )
            db.add(memory)
            new_memory_id = memory.id

        archived_count = 0
        if archived_ids:
            stmt = select(MemoryMetadata).where(
                MemoryMetadata.organization_id == org_id,
                MemoryMetadata.id.in_(archived_ids),
            )
            rows = list((await db.execute(stmt)).scalars().all())
            archived_at = datetime.now(timezone.utc)
            for row in rows:
                metadata = dict(row.extra_metadata or {})
                metadata["is_archived"] = True
                metadata["archived_at"] = archived_at.isoformat()
                row.extra_metadata = metadata
                # Keep compatibility with call sites that expect explicit archive attrs.
                setattr(row, "is_archived", True)
                setattr(row, "archived_at", archived_at)
            archived_count = len(rows)

        await db.commit()
        return {
            "new_memory_id": new_memory_id,
            "archived_count": archived_count,
            "narrative": narrative,
        }
