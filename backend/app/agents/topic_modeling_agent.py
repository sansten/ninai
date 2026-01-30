from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


class TopicModelingAgent(BaseAgent):
    name = "TopicModelingAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        topics = outputs.get("topics")
        if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
            raise ValueError("topic modeling outputs.topics must be a list[str]")
        if "primary_topic" not in outputs or not isinstance(outputs.get("primary_topic"), str):
            raise ValueError("topic modeling outputs.primary_topic must be a string")
        conf = outputs.get("confidence", result.confidence)
        if conf is None or not (0.0 <= float(conf) <= 1.0):
            raise ValueError("topic modeling confidence must be between 0 and 1")

    def _heuristic(self, *, content: str, tags: list[str] | None = None) -> dict[str, Any]:
        text = (content or "")
        lower = text.lower()

        # Prefer metadata tags when present.
        normalized_tags = [t.strip().lower() for t in (tags or []) if isinstance(t, str) and t.strip()]

        topic_scores: dict[str, int] = {}

        def bump(topic: str, n: int = 1):
            topic_scores[topic] = topic_scores.get(topic, 0) + n

        # Tag-based hints
        for t in normalized_tags:
            if t in {"billing", "payments", "refund"}:
                bump("billing", 3)
            if t in {"security", "auth"}:
                bump("security", 3)
            if t in {"engineering", "bug", "incident"}:
                bump("engineering", 3)
            if t in {"logistics", "shipping", "delivery"}:
                bump("logistics", 3)
            if t in {"meeting"}:
                bump("meeting", 3)

        # Content-based hints
        billing_kw = ["refund", "charge", "invoice", "payment", "chargeback", "billing"]
        security_kw = ["password", "login", "token", "2fa", "mfa", "ssn", "credit card", "api key"]
        eng_kw = ["bug", "error", "stack trace", "exception", "crash", "deploy"]
        logistics_kw = ["shipping", "delivery", "tracking", "warehouse", "carrier"]
        meeting_kw = ["meeting", "agenda", "minutes", "call", "zoom", "teams meeting"]

        for k in billing_kw:
            if k in lower:
                bump("billing")
        for k in security_kw:
            if k in lower:
                bump("security")
        for k in eng_kw:
            if k in lower:
                bump("engineering")
        for k in logistics_kw:
            if k in lower:
                bump("logistics")
        for k in meeting_kw:
            if k in lower:
                bump("meeting")

        if not topic_scores:
            topics = ["general"]
            primary = "general"
            confidence = 0.35
        else:
            # Sort by score desc, then name.
            sorted_topics = sorted(topic_scores.items(), key=lambda kv: (-kv[1], kv[0]))
            primary = sorted_topics[0][0]
            topics = [t for t, _ in sorted_topics[:5]]
            confidence = min(0.85, 0.45 + 0.1 * min(sorted_topics[0][1], 4))

        return {
            "topics": topics,
            "primary_topic": primary,
            "confidence": float(confidence),
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        memory_ctx = context.get("memory") or {}
        content = memory_ctx.get("content", "")

        enrichment = memory_ctx.get("enrichment") or {}
        tags = None
        if isinstance(enrichment, dict):
            md = enrichment.get("metadata") or {}
            if isinstance(md, dict) and isinstance(md.get("tags"), list):
                tags = [str(t) for t in md.get("tags")]

        outputs: dict[str, Any]

        strategy = str(getattr(settings, "AGENT_STRATEGY", "llm") or "llm").strip().lower()
        if strategy == "heuristic" or not content:
            outputs = self._heuristic(content=content, tags=tags)
        else:
            prompt = (
                "You are a topic modeling engine for an enterprise memory system. Output JSON only.\n\n"
                "Given the content (and optional tags), produce a few coarse topics suitable for routing and discovery.\n"
                "Be conservative: do not hallucinate.\n\n"
                f"TAGS: {tags or []}\n\n"
                f"CONTENT:\n{content}\n\n"
                "Return JSON with keys: topics (string[]), primary_topic (string), confidence (0..1), rationale"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(getattr(settings, "OLLAMA_MODEL", "llama3.1:8b")),
                timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
                use_circuit_breaker=True,
            )
            resp = await client.complete_json(prompt=prompt, schema_hint={}, tool_event_sink=context.get("tool_event_sink"))
            if isinstance(resp, dict) and isinstance(resp.get("topics"), list) and isinstance(resp.get("primary_topic"), str):
                resp["topics"] = [str(t) for t in resp.get("topics") if str(t).strip()]
                outputs = resp
            else:
                outputs = self._heuristic(content=content, tags=tags)

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
