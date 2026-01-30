from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.services.logseq_service import ExportableMemory, build_logseq_graph, render_logseq_markdown


class LogseqExportAgent(BaseAgent):
    name = "LogseqExportAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        md = outputs.get("markdown")
        graph = outputs.get("graph")
        if not isinstance(md, str):
            raise ValueError("logseq export outputs.markdown must be a string")
        if not isinstance(graph, dict):
            raise ValueError("logseq export outputs.graph must be an object")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        memory_ctx = context.get("memory") or {}
        content = memory_ctx.get("content", "")
        enrichment = memory_ctx.get("enrichment") if isinstance(memory_ctx, dict) else {}

        md_enrichment = (enrichment or {}).get("metadata") if isinstance(enrichment, dict) else {}
        tags = (md_enrichment or {}).get("tags") if isinstance(md_enrichment, dict) else []
        entities = (md_enrichment or {}).get("entities") if isinstance(md_enrichment, dict) else {}

        export_mem = ExportableMemory(
            id=memory_id,
            title=None,
            content_preview=content,
            created_at=datetime.now(timezone.utc),
            scope=None,
            classification=memory_ctx.get("classification") if isinstance(memory_ctx, dict) else None,
            tags=[str(t) for t in (tags or []) if str(t).strip()],
            entities=entities if isinstance(entities, dict) else {},
        )

        markdown, item_count = render_logseq_markdown([export_mem])
        graph = build_logseq_graph([export_mem])

        outputs: dict[str, Any] = {
            "markdown": markdown,
            "graph": graph,
            "item_count": int(item_count),
            "confidence": 0.6,
            "rationale": "rendered_from_enrichment",
        }

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.6)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )

        self.validate_outputs(result)
        return result
