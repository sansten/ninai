"""Memory Tier Manager Agent (Feature 24.2).

MemGPT-style tiering for cognitive sessions:
- maintain an active working set with bounded capacity
- offload overflow items to archival tier
- return a working_set_summary suitable for session payloads
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_key(item: dict[str, Any]) -> str:
    raw_id = item.get("id") or item.get("memory_id")
    if raw_id is not None:
        return f"id:{raw_id}"
    content = str(item.get("content") or item.get("content_preview") or "")
    return f"content:{content[:120]}"


class MemoryTierManagerAgent(BaseAgent):
    name = "MemoryTierManagerAgent"
    version = "v1"

    def reconcile(
        self,
        *,
        working_set: list[dict[str, Any]],
        archival: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
        max_working_set: int = 8,
    ) -> dict[str, Any]:
        """Reconcile session memory tiers and return summary.

        The newest incoming memories are favored for working-set residency.
        Existing working-set items are retained if capacity allows.
        """
        seen: set[str] = set()
        merged_working: list[dict[str, Any]] = []

        def _normalize(item: dict[str, Any], source: str) -> dict[str, Any]:
            memory_id = item.get("id") or item.get("memory_id")
            content = str(item.get("content") or item.get("content_preview") or "")
            return {
                "memory_id": str(memory_id) if memory_id is not None else None,
                "content_preview": content[:160],
                "source": source,
                "last_touched_at": _now_iso(),
            }

        for item in list(incoming or []) + list(working_set or []):
            if not isinstance(item, dict):
                continue
            key = _memory_key(item)
            if key in seen:
                continue
            seen.add(key)
            source = "incoming" if item in incoming else "retained"
            merged_working.append(_normalize(item, source))

        max_size = max(1, int(max_working_set or 8))
        kept = merged_working[:max_size]
        offloaded = merged_working[max_size:]

        next_archival = list(archival or [])
        for item in offloaded:
            next_archival.append(
                {
                    **item,
                    "source": "offloaded",
                    "offloaded_at": _now_iso(),
                }
            )

        return {
            "working_set": kept,
            "archival": next_archival,
            "working_set_size": len(kept),
            "archival_size": len(next_archival),
            "loaded_ids": [i.get("memory_id") for i in kept if i.get("memory_id")],
            "offloaded_ids": [i.get("memory_id") for i in offloaded if i.get("memory_id")],
            "updated_at": _now_iso(),
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started = datetime.now(timezone.utc)

        tier_ctx = context.get("memory_tiers") if isinstance(context, dict) else None
        tier_ctx = tier_ctx if isinstance(tier_ctx, dict) else {}

        summary = self.reconcile(
            working_set=list(tier_ctx.get("working_set") or []),
            archival=list(tier_ctx.get("archival") or []),
            incoming=list(tier_ctx.get("incoming") or []),
            max_working_set=int(tier_ctx.get("max_working_set") or 8),
        )

        finished = datetime.now(timezone.utc)
        return AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=0.83,
            outputs={"working_set_summary": summary},
            warnings=[],
            errors=[],
            started_at=started,
            finished_at=finished,
            trace_id=None,
        )
