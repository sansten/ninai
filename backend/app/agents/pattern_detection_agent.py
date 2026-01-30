from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


_STEP_RE = re.compile(r"\b(step\s+\d+|steps?:)\b", re.IGNORECASE)
_RESOLUTION_RE = re.compile(r"\b(resolved|fixed|workaround|mitigation)\b", re.IGNORECASE)


class PatternDetectionAgent(BaseAgent):
    name = "PatternDetectionAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        patterns = outputs.get("patterns")
        if not isinstance(patterns, list):
            raise ValueError("pattern detection outputs.patterns must be a list")
        for p in patterns:
            if not isinstance(p, dict):
                raise ValueError("pattern detection pattern entries must be objects")
            if not isinstance(p.get("pattern"), str) or not p.get("pattern"):
                raise ValueError("pattern detection pattern.pattern must be a non-empty string")
            conf = p.get("confidence")
            if conf is None or not (0.0 <= float(conf) <= 1.0):
                raise ValueError("pattern detection pattern.confidence must be between 0 and 1")

        conf = outputs.get("confidence", result.confidence)
        if conf is None or not (0.0 <= float(conf) <= 1.0):
            raise ValueError("pattern detection confidence must be between 0 and 1")

    def _heuristic(self, *, content: str, enrichment: dict | None) -> dict[str, Any]:
        text = (content or "")
        lower = text.lower()

        md = (enrichment or {}).get("metadata") if isinstance(enrichment, dict) else None
        tags = (md or {}).get("tags") if isinstance(md, dict) else None
        topics = ((enrichment or {}).get("topics") or {}).get("topics") if isinstance(enrichment, dict) else None

        normalized_tags = [str(t).strip().lower() for t in (tags or []) if str(t).strip()]
        normalized_topics = [str(t).strip().lower() for t in (topics or []) if str(t).strip()]

        patterns: list[dict[str, Any]] = []

        def add(pattern: str, *, kind: str, confidence: float, evidence: list[str] | None = None):
            patterns.append(
                {
                    "pattern": pattern,
                    "type": kind,
                    "confidence": float(max(0.0, min(1.0, confidence))),
                    "evidence": evidence or [],
                }
            )

        # Procedural shape
        if _STEP_RE.search(text) or "how to" in lower or "runbook" in lower:
            add("procedural_template", kind="structure", confidence=0.75, evidence=["steps/how-to markers"])

        # Incident/support resolution shape
        if _RESOLUTION_RE.search(text) and any(k in lower for k in ["issue", "incident", "bug", "error", "escalation"]):
            add("issue_resolution", kind="support", confidence=0.72, evidence=["resolved/fixed + issue markers"])

        # Billing pattern
        if any(t in normalized_tags for t in ["billing"]) or any(t in normalized_topics for t in ["billing"]):
            if any(k in lower for k in ["refund", "chargeback", "invoice", "payment", "charged"]):
                add("billing_dispute_or_refund", kind="domain", confidence=0.7, evidence=["billing topic + billing keywords"])

        # Security pattern
        if any(t in normalized_tags for t in ["security"]) or any(t in normalized_topics for t in ["security"]):
            if any(k in lower for k in ["password", "login", "token", "mfa", "2fa"]):
                add("auth_or_credential_issue", kind="domain", confidence=0.68, evidence=["security topic + auth keywords"])

        # Default if nothing detected
        if not patterns:
            patterns = []
            confidence = 0.35
        else:
            confidence = min(0.85, 0.5 + 0.05 * min(len(patterns), 4))

        return {
            "patterns": patterns,
            "confidence": float(confidence),
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        memory_ctx = context.get("memory") or {}
        content = memory_ctx.get("content", "")
        enrichment = memory_ctx.get("enrichment") if isinstance(memory_ctx, dict) else None

        outputs: dict[str, Any]

        strategy = str(getattr(settings, "AGENT_STRATEGY", "llm") or "llm").strip().lower()
        if strategy == "heuristic" or not content:
            outputs = self._heuristic(content=content, enrichment=enrichment if isinstance(enrichment, dict) else None)
        else:
            prompt = (
                "You are a pattern detection engine for an enterprise memory system. Output JSON only.\n\n"
                "Given a single memory (and optional enrichment), identify any reusable patterns or templates. "
                "Be conservative; do not hallucinate.\n\n"
                f"ENRICHMENT: {enrichment or {}}\n\n"
                f"CONTENT:\n{content}\n\n"
                "Return JSON with keys: patterns (array of {pattern,type,confidence,evidence}), confidence (0..1), rationale"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(getattr(settings, "OLLAMA_MODEL", "llama3.1:8b")),
                timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
                use_circuit_breaker=True,
            )
            resp = await client.complete_json(prompt=prompt, schema_hint={}, tool_event_sink=context.get("tool_event_sink"))
            if isinstance(resp, dict) and isinstance(resp.get("patterns"), list):
                outputs = resp
            else:
                outputs = self._heuristic(content=content, enrichment=enrichment if isinstance(enrichment, dict) else None)

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
