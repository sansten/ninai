from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.utils import max_classification
from app.agents.llm.llm_breaker import create_llm_client
from app.core.config import settings


class ClassificationAgent(BaseAgent):
    name = "ClassificationAgent"
    version = "v1"

    def _heuristic(self, *, content: str, existing_classification: str | None) -> dict[str, Any]:
        text = (content or "").lower()

        sensitive_markers = [
            "ssn",
            "social security",
            "credit card",
            "password",
            "api key",
            "secret",
            "confidential",
        ]
        is_sensitive = any(m in text for m in sensitive_markers)

        classification = "confidential" if is_sensitive else "internal"
        classification = max_classification(existing_classification, classification)

        if "step" in text or "how to" in text or "runbook" in text:
            memory_type = "procedural"
        elif len(text) < 400:
            memory_type = "short_term"
        else:
            memory_type = "long_term"

        importance_keywords = ["order", "purchase", "payment", "contract", "issue", "escalation"]
        importance_hits = sum(1 for k in importance_keywords if k in text)
        importance_score = min(1.0, 0.2 + 0.15 * importance_hits)

        return {
            "memory_type_suggestion": memory_type,
            "importance_score": float(importance_score),
            "is_sensitive": bool(is_sensitive),
            "classification": classification,
            "domain_signals": [],
            "confidence": 0.6 if is_sensitive else 0.4,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        content = (context.get("memory") or {}).get("content", "")  # optional
        existing_classification = (context.get("memory") or {}).get("classification")

        outputs: dict[str, Any]

        strategy = str(getattr(settings, "AGENT_STRATEGY", "llm") or "llm").strip().lower()
        if strategy == "heuristic" or not content:
            outputs = self._heuristic(content=content, existing_classification=existing_classification)
        else:
            prompt = (
                "You are a classification engine for an enterprise memory system. Output JSON only.\n\n"
                "Classify the memory. Consider sensitivity and classification. Do not hallucinate.\n\n"
                "Allowed classification values: public, internal, confidential, restricted.\n"
                "Sensitivity policy (fail safe):\n"
                "- If payment card data (card number/PAN/CVV), passwords, API keys, secrets, government IDs (SSN/passport), or authentication credentials are present, set is_sensitive=true and classification to confidential or restricted (never public/internal).\n"
                "- If credentials can grant account/system access, prefer restricted.\n"
                "- If content contains direct personal identifiers with financial, health, legal, or security context, classification must be at least confidential.\n"
                "- If unsure between two levels, choose the more restrictive level.\n\n"
                "Examples:\n"
                "- 'Customer credit card 4111 ... and password reset': is_sensitive=true, classification=confidential or restricted.\n"
                "- 'Internal sprint update without sensitive data': is_sensitive=false, classification=internal.\n\n"
                f"CONTENT:\n{content}\n\n"
                "Return JSON with keys: memory_type_suggestion, importance_score, is_sensitive, classification, domain_signals, confidence, rationale"
            )
            client = create_llm_client(
                base_url=str(getattr(settings, "VLLM_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_llm_model("agents")),
                timeout_seconds=float(getattr(settings, "VLLM_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "VLLM_MAX_CONCURRENCY", 2)),
                use_circuit_breaker=True,
            )
            resp = await client.complete_json(prompt=prompt, schema_hint={}, tool_event_sink=context.get("tool_event_sink"))
            # fail-closed: if llm didn't return expected fields, fall back
            if isinstance(resp, dict) and resp.get("classification"):
                resp["classification"] = max_classification(existing_classification, resp.get("classification"))
                outputs = resp
            else:
                outputs = self._heuristic(content=content, existing_classification=existing_classification)

        finished_at = datetime.now(timezone.utc)

        # Normalize confidence to 0-1 range (handle LLMs that return percentages like 95.0)
        raw_confidence = float(outputs.get("confidence", 0.5))
        if raw_confidence > 1.0:
            raw_confidence = raw_confidence / 100.0
        confidence = max(0.0, min(1.0, raw_confidence))

        return AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=confidence,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
