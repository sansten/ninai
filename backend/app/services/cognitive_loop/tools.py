"""Built-in CognitiveLoop tools.

Tools are registered into a ToolRegistry and invoked via ToolInvoker.
All implementations must be:
- async
- RLS-safe (assume set_tenant_context has been applied)
- permission-checked (MemoryService enforces this)

Outputs must be safe to summarize; raw sensitive content should not be returned.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.services.cognitive_tooling.tool_registry import ToolRegistry, ToolSpec, ToolSensitivity
from app.services.embedding_service import EmbeddingService
from app.services.memory_service import MemoryService
from app.schemas.memory import MemorySearchRequest


class MemorySearchToolInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(10, ge=1, le=50)
    scope: str | None = None
    team_id: str | None = None
    hybrid: bool = True


def register_builtin_tools(*, registry: ToolRegistry, memory_service: MemoryService) -> None:
    async def _memory_search(inp: dict[str, Any]) -> dict[str, Any]:
        validated = MemorySearchToolInput.model_validate(inp)
        emb = await EmbeddingService.embed(validated.query)
        req = MemorySearchRequest(
            query=validated.query,
            scope=validated.scope,
            team_id=validated.team_id,
            limit=validated.limit,
            hybrid=validated.hybrid,
        )
        rows = await memory_service.search_memories(emb, req)
        # Summary-only results
        results: list[dict[str, Any]] = []
        for m in rows:
            results.append(
                {
                    "id": str(getattr(m, "id")),
                    "title": getattr(m, "title", None),
                    "summary": getattr(m, "content_preview", "") or "",
                    "tags": list(getattr(m, "tags", []) or []),
                    "classification": getattr(m, "classification", None),
                    "scope": getattr(m, "scope", None),
                    "scope_id": getattr(m, "scope_id", None),
                    "score": float(getattr(m, "score", 0.0) or 0.0),
                }
            )
        return {"results": results, "count": len(results)}

    registry.register(
        ToolSpec(
            name="memory.search",
            version="v1",
            required_permissions=(),
            input_schema=MemorySearchToolInput,
            output_schema=None,
            sensitivity=ToolSensitivity(
                allow_persist_input=False,
                allow_persist_output=False,
            ),
        ),
        _memory_search,
    )
