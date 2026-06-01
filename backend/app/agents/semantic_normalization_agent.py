"""Semantic Normalization Agent — Phase 6.

Extracts intent, business_domain, and semantic_relationships from memory content.
These fields power downstream phases:
  - Phase 7 (Ontology): domain links entities across departments
  - Phase 8 (Context Amplifier): relationships fan-out on expert read
  - Phase 9 (Silo Propagation): domain routes signals across teams
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Intent classification keywords
# ---------------------------------------------------------------------------
_INTENT_MAP: list[tuple[str, list[str]]] = [
    ("document_decision", ["decided", "decision", "resolved", "agreed", "approved", "we will", "we have chosen"]),
    ("report_issue", ["bug", "error", "issue", "problem", "broken", "fail", "outage", "incident", "crash"]),
    ("share_learning", ["learned", "finding", "discovery", "insight", "retrospective", "takeaway", "lesson"]),
    ("request_action", ["please", "action item", "todo", "must", "need to", "should", "follow up", "assign"]),
    ("record_status", ["status", "update", "progress", "completed", "done", "deployed", "released", "shipped"]),
    ("capture_context", ["background", "context", "overview", "summary", "note that", "fyi", "for reference"]),
]

# ---------------------------------------------------------------------------
# Business domain keywords
# ---------------------------------------------------------------------------
_DOMAIN_MAP: list[tuple[str, list[str]]] = [
    ("engineering", ["bug", "deploy", "code", "api", "repo", "pull request", "pr ", "merge", "stack trace",
                     "incident", "sre", "ci/cd", "pipeline", "build", "unit test", "microservice"]),
    ("sales", ["deal", "prospect", "lead", "revenue", "quota", "pipeline", "crm", "close", "opportunity",
               "account executive", "demo", "proposal", "contract value", "arr"]),
    ("finance", ["invoice", "payment", "budget", "expense", "billing", "cost", "p&l", "profit", "loss",
                 "accounts payable", "accounts receivable", "tax", "audit", "forecast"]),
    ("support", ["ticket", "sla", "escalation", "complaint", "resolution", "customer issue", "helpdesk",
                 "support case", "refund", "chargeback"]),
    ("hr", ["hiring", "employee", "onboarding", "offboarding", "payroll", "performance review", "headcount",
            "recruiter", "job offer", "resignation"]),
    ("legal", ["contract", "nda", "compliance", "regulation", "gdpr", "policy", "legal", "attorney",
               "lawsuit", "ip", "trademark", "liability"]),
    ("product", ["roadmap", "feature", "design", "requirements", "backlog", "sprint", "user story",
                 "acceptance criteria", "product spec", "wireframe"]),
    ("marketing", ["campaign", "content", "launch", "brand", "social media", "seo", "ads", "email blast",
                   "analytics", "conversion", "landing page"]),
    ("operations", ["process", "logistics", "vendor", "infrastructure", "sla", "runbook", "capacity",
                    "on-call", "alert", "monitoring", "datacenter"]),
]

# Mention patterns — @name or "John Smith said", system/tool names
_MENTION_RE = re.compile(r"@(\w[\w.-]{1,50})")
_SYSTEM_RE = re.compile(
    r"\b(salesforce|jira|github|slack|zendesk|hubspot|confluence|notion|asana|pagerduty|datadog|"
    r"aws|gcp|azure|postgres|redis|qdrant|celery|kafka|kubernetes|k8s)\b",
    re.IGNORECASE,
)


def _classify_intent(text: str) -> str:
    lower = text.lower()
    for intent, keywords in _INTENT_MAP:
        if any(kw in lower for kw in keywords):
            return intent
    return "capture_context"


def _classify_domain(text: str) -> str:
    lower = text.lower()
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_MAP:
        hits = sum(1 for kw in keywords if kw in lower)
        if hits:
            scores[domain] = hits
    if not scores:
        return "general"
    return max(scores, key=lambda d: scores[d])


def _extract_relationships(text: str) -> list[dict[str, str]]:
    rels: list[dict[str, str]] = []
    seen: set[str] = set()

    for mention in _MENTION_RE.findall(text):
        key = f"person:{mention.lower()}"
        if key not in seen:
            rels.append({"type": "mentions_person", "concept": mention})
            seen.add(key)

    for match in _SYSTEM_RE.finditer(text):
        system = match.group(1).lower()
        key = f"system:{system}"
        if key not in seen:
            rels.append({"type": "references_system", "concept": system})
            seen.add(key)

    return rels[:10]  # cap at 10 to avoid noise


class SemanticNormalizationAgent(BaseAgent):
    """Phase 6: Extract semantic intent, business domain, and relationships."""

    name = "SemanticNormalizationAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["MetadataExtractionAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        valid_intents = {
            "document_decision", "report_issue", "share_learning",
            "request_action", "record_status", "capture_context", "general",
        }
        if outputs.get("intent") not in valid_intents:
            raise ValueError(f"invalid intent: {outputs.get('intent')!r}")
        valid_domains = {
            "engineering", "sales", "finance", "support", "hr",
            "legal", "product", "marketing", "operations", "general",
        }
        if outputs.get("business_domain") not in valid_domains:
            raise ValueError(f"invalid business_domain: {outputs.get('business_domain')!r}")
        if not isinstance(outputs.get("semantic_relationships", []), list):
            raise ValueError("semantic_relationships must be a list")

    def _heuristic(self, *, content: str) -> dict[str, Any]:
        text = content or ""
        intent = _classify_intent(text)
        domain = _classify_domain(text)
        rels = _extract_relationships(text)

        # Confidence: higher when we have clear domain + intent signals
        domain_hits = sum(
            1 for _, kws in _DOMAIN_MAP
            if any(kw in text.lower() for kw in kws)
        )
        confidence = min(0.85, 0.35 + 0.08 * min(domain_hits, 6))

        return {
            "intent": intent,
            "business_domain": domain,
            "semantic_relationships": rels,
            "confidence": round(confidence, 3),
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        content = (context.get("memory") or {}).get("content", "")

        strategy = getattr(settings, "SEMANTIC_NORMALIZATION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(content=content)
        else:
            prompt = (
                "You are a semantic normalization engine for an enterprise Cognitive OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Analyze the content and return:\n"
                "- intent: one of document_decision|report_issue|share_learning|"
                "request_action|record_status|capture_context|general\n"
                "- business_domain: one of engineering|sales|finance|support|hr|"
                "legal|product|marketing|operations|general\n"
                "- semantic_relationships: list of {type, concept} objects (max 10)\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation\n\n"
                f"CONTENT:\n{content}\n\n"
                "Return JSON with keys: intent, business_domain, semantic_relationships, confidence, rationale"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_ollama_model("agents")),
                timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
            )
            try:
                resp = await client.complete_json(
                    prompt=prompt,
                    schema_hint={},
                    tool_event_sink=context.get("tool_event_sink"),
                )
            except Exception:
                resp = None
            valid_intents = {
                "document_decision", "report_issue", "share_learning",
                "request_action", "record_status", "capture_context", "general",
            }
            valid_domains = {
                "engineering", "sales", "finance", "support", "hr",
                "legal", "product", "marketing", "operations", "general",
            }
            if (
                isinstance(resp, dict)
                and resp.get("intent") in valid_intents
                and resp.get("business_domain") in valid_domains
                and isinstance(resp.get("semantic_relationships"), list)
            ):
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
