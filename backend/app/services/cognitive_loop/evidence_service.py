"""Evidence retrieval for CognitiveLoop planning.

This service is RLS-safe by construction: it relies on MemoryService.search_memories,
which re-validates Qdrant results through Postgres (RLS) and PermissionChecker.

Evidence is returned as summary-only cards; raw sensitive content is not included.
"""

from __future__ import annotations

from typing import Any

from app.schemas.memory import MemorySearchRequest
from app.services.embedding_service import EmbeddingService
from app.services.memory_service import MemoryService


class CognitiveEvidenceService:
    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service

    async def retrieve_evidence(
        self,
        *,
        goal: str,
        scope: str | None = None,
        team_id: str | None = None,
        limit: int = 10,
        hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        query_embedding = await EmbeddingService.embed(goal)
        req = MemorySearchRequest(
            query=goal,
            scope=scope,
            team_id=team_id,
            limit=limit,
            hybrid=hybrid,
        )

        memories = await self.memory_service.search_memories(query_embedding, req)

        cards: list[dict[str, Any]] = []
        for m in memories:
            cards.append(
                {
                    "id": str(getattr(m, "id")),
                    "summary": getattr(m, "content_preview", "") or "",
                    "title": getattr(m, "title", None),
                    "tags": list(getattr(m, "tags", []) or []),
                    "classification": getattr(m, "classification", None),
                    "scope": getattr(m, "scope", None),
                    "scope_id": getattr(m, "scope_id", None),
                    "score": float(getattr(m, "score", 0.0) or 0.0),
                }
            )

        return cards
