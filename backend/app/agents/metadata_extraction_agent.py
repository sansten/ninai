from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?1[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b")
_ORDER_RE = re.compile(r"\b(?:order|ord|ticket|case)[\s:#-]*([A-Z0-9-]{3,})\b", re.IGNORECASE)


def _uniq_sorted(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        v = (x or "").strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return sorted(out, key=lambda s: s.lower())


class MetadataExtractionAgent(BaseAgent):
    name = "MetadataExtractionAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("tags", []), list):
            raise ValueError("metadata outputs.tags must be a list")
        if not isinstance(outputs.get("entities", {}), dict):
            raise ValueError("metadata outputs.entities must be a dict")
        conf = outputs.get("confidence", result.confidence)
        if conf is None or not (0.0 <= float(conf) <= 1.0):
            raise ValueError("metadata confidence must be between 0 and 1")

    def _heuristic(self, *, content: str) -> dict[str, Any]:
        text = (content or "")
        lower = text.lower()

        emails = _EMAIL_RE.findall(text)
        urls = _URL_RE.findall(text)
        phones = _PHONE_RE.findall(text)
        order_ids = [m.group(1) for m in _ORDER_RE.finditer(text)]

        tags: list[str] = []
        keyword_tags = {
            "refund": "billing",
            "chargeback": "billing",
            "invoice": "billing",
            "payment": "billing",
            "password": "security",
            "login": "security",
            "auth": "security",
            "ssn": "security",
            "credit card": "security",
            "bug": "engineering",
            "error": "engineering",
            "stack trace": "engineering",
            "meeting": "meeting",
            "agenda": "meeting",
            "call": "meeting",
            "delivery": "logistics",
            "shipping": "logistics",
        }
        for k, tag in keyword_tags.items():
            if k in lower:
                tags.append(tag)

        entities: dict[str, Any] = {
            "email": _uniq_sorted([e.lower() for e in emails]),
            "url": _uniq_sorted(urls),
            "phone": _uniq_sorted(phones),
            "order_id": _uniq_sorted(order_ids),
        }
        # Remove empty entity lists
        entities = {k: v for k, v in entities.items() if v}

        # Lightweight summary: first sentence or preview
        summary = text.strip().split("\n", 1)[0]
        summary = summary.split(".", 1)[0].strip()
        if len(summary) > 200:
            summary = summary[:200].rstrip() + "…"

        signal_count = sum(len(v) for v in entities.values()) + len(set(tags))
        confidence = min(0.85, 0.35 + 0.1 * min(signal_count, 5))

        return {
            "tags": _uniq_sorted(tags),
            "entities": entities,
            "summary": summary,
            "confidence": float(confidence),
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        content = (context.get("memory") or {}).get("content", "")

        outputs: dict[str, Any]

        # Per-agent override takes precedence, otherwise follow global strategy.
        strategy = getattr(settings, "METADATA_EXTRACTION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()
        if strategy == "heuristic" or not content:
            outputs = self._heuristic(content=content)
        else:
            prompt = (
                "You are an enterprise metadata extraction engine for a memory system. Output JSON only.\n\n"
                "Extract tags and entities from the content. Do not hallucinate; if unknown, omit.\n\n"
                f"CONTENT:\n{content}\n\n"
                "Return JSON with keys: tags (string[]), entities (object), summary (string), confidence (0..1), rationale"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(getattr(settings, "OLLAMA_MODEL", "llama3.1:8b")),
                timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
            )
            resp = await client.complete_json(prompt=prompt, schema_hint={}, tool_event_sink=context.get("tool_event_sink"))
            if isinstance(resp, dict) and isinstance(resp.get("tags"), list) and isinstance(resp.get("entities"), dict):
                resp["tags"] = _uniq_sorted([str(t) for t in resp.get("tags") or []])
                outputs = resp
            else:
                outputs = self._heuristic(content=content)

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
