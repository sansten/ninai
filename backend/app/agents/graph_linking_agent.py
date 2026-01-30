from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


class GraphLinkingAgent(BaseAgent):
    name = "GraphLinkingAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        graph = outputs.get("graph")
        if not isinstance(graph, dict):
            raise ValueError("graph linking outputs.graph must be an object")
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) or not all(isinstance(n, dict) for n in nodes):
            raise ValueError("graph linking graph.nodes must be a list[object]")
        if not isinstance(edges, list) or not all(isinstance(e, dict) for e in edges):
            raise ValueError("graph linking graph.edges must be a list[object]")

    def _heuristic(self, *, memory_id: str, enrichment: dict | None) -> dict[str, Any]:
        md = (enrichment or {}).get("metadata") if isinstance(enrichment, dict) else None
        topics = (enrichment or {}).get("topics") if isinstance(enrichment, dict) else None

        tags = (md or {}).get("tags") if isinstance(md, dict) else []
        entities = (md or {}).get("entities") if isinstance(md, dict) else {}
        topic_list = (topics or {}).get("topics") if isinstance(topics, dict) else []

        nodes: list[dict[str, Any]] = [{"id": f"memory:{memory_id}", "type": "memory", "label": memory_id}]
        edges: list[dict[str, Any]] = []

        def add_node(node_id: str, *, ntype: str, label: str):
            if any(n.get("id") == node_id for n in nodes):
                return
            nodes.append({"id": node_id, "type": ntype, "label": label})

        def add_edge(src: str, dst: str, *, rel: str):
            edges.append({"source": src, "target": dst, "relation": rel})

        # Topics as nodes
        for t in topic_list or []:
            tt = str(t).strip()
            if not tt:
                continue
            tid = f"topic:{tt.lower()}"
            add_node(tid, ntype="topic", label=tt)
            add_edge(f"memory:{memory_id}", tid, rel="about")

        # Tags as nodes
        for tag in tags or []:
            tg = str(tag).strip()
            if not tg:
                continue
            tid = f"tag:{tg.lower()}"
            add_node(tid, ntype="tag", label=tg)
            add_edge(f"memory:{memory_id}", tid, rel="tagged")

        # Entities
        if isinstance(entities, dict):
            for key, values in entities.items():
                if not isinstance(values, list):
                    continue
                for v in values:
                    vv = str(v).strip()
                    if not vv:
                        continue
                    eid = f"entity:{str(key).lower()}:{vv.lower()}"
                    add_node(eid, ntype="entity", label=vv)
                    add_edge(f"memory:{memory_id}", eid, rel=f"mentions:{str(key).lower()}")

        confidence = 0.4
        signal_count = (len(topic_list or []) + len(tags or []) + sum(len(v) for v in entities.values() if isinstance(v, list)))
        if signal_count:
            confidence = min(0.85, 0.45 + 0.08 * min(signal_count, 5))

        return {
            "graph": {"nodes": nodes, "edges": edges},
            "confidence": float(confidence),
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        memory_ctx = context.get("memory") or {}
        enrichment = memory_ctx.get("enrichment") if isinstance(memory_ctx, dict) else None

        outputs: dict[str, Any]

        strategy = str(getattr(settings, "AGENT_STRATEGY", "llm") or "llm").strip().lower()
        if strategy == "heuristic" or not isinstance(enrichment, dict):
            outputs = self._heuristic(memory_id=memory_id, enrichment=enrichment if isinstance(enrichment, dict) else None)
        else:
            prompt = (
                "You are a graph linking engine for an enterprise memory system. Output JSON only.\n\n"
                "Create nodes and edges from the enrichment (topics/tags/entities). Do not hallucinate.\n\n"
                f"MEMORY_ID: {memory_id}\n\n"
                f"ENRICHMENT: {enrichment}\n\n"
                "Return JSON with keys: graph:{nodes:[{id,type,label}], edges:[{source,target,relation}]}, confidence (0..1), rationale"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(getattr(settings, "OLLAMA_MODEL", "llama3.1:8b")),
                timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
                use_circuit_breaker=True,
            )
            resp = await client.complete_json(prompt=prompt, schema_hint={}, tool_event_sink=context.get("tool_event_sink"))
            if isinstance(resp, dict) and isinstance(resp.get("graph"), dict):
                outputs = resp
            else:
                outputs = self._heuristic(memory_id=memory_id, enrichment=enrichment)

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.5)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )

        self.validate_outputs(result)
        return result
