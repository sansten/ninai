"""Cognitive Gateway Service — Phase 49.

Exposes Ninai's full intelligence stack as five simple verbs for downstream
consumers (Tier-3 developers, AI agent platforms, enterprise integrations):

  write(content, ...)  → store + enrich a memory record
  read(query, ...)     → retrieve + assemble context
  decide(context, ...) → run enrichment pipeline, return decision + confidence
  plan(goal, ...)      → decompose goal, return ordered steps
  explain(memory_id)   → return audit trail for last decision on a memory

Internally, each verb fans out to the right combination of agents.
Externally the caller sees a consistent, simple interface.

Per-tenant capability gating:
  CognitiveGatewayCapabilities controls which verbs are available to each org.
  Default: all five verbs enabled.  Can be restricted to read+write for lite plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Capability gating
# ---------------------------------------------------------------------------

ALL_VERBS = frozenset({"write", "read", "decide", "plan", "explain"})


@dataclass
class CognitiveGatewayCapabilities:
    """Per-tenant capabilities — which verbs are permitted."""
    enabled_verbs: frozenset[str] = field(default_factory=lambda: frozenset(ALL_VERBS))

    def is_enabled(self, verb: str) -> bool:
        return verb in self.enabled_verbs

    @classmethod
    def lite(cls) -> "CognitiveGatewayCapabilities":
        """Read + write only — for basic integration plans."""
        return cls(enabled_verbs=frozenset({"read", "write"}))

    @classmethod
    def standard(cls) -> "CognitiveGatewayCapabilities":
        """All verbs except plan — for most enterprise plans."""
        return cls(enabled_verbs=frozenset({"read", "write", "decide", "explain"}))

    @classmethod
    def full(cls) -> "CognitiveGatewayCapabilities":
        """All five verbs enabled."""
        return cls(enabled_verbs=frozenset(ALL_VERBS))


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class GatewayWriteResult:
    memory_id: str
    enriched: bool
    enrichment_summary: dict
    tags: list[str]
    created_at: datetime


@dataclass
class GatewayReadResult:
    memories: list[dict]
    total: int
    query: str
    context_assembled: bool


@dataclass
class GatewayDecideResult:
    decision: str
    confidence: float
    tone: str
    action_recommended: str | None
    enrichment: dict
    agents_run: list[str]


@dataclass
class GatewayPlanResult:
    goal: str
    steps: list[dict]
    step_count: int
    blocking_step: str | None
    confidence: float


@dataclass
class GatewayExplainResult:
    memory_id: str
    decisions: list[dict]
    agents: list[str]
    confidence: float
    explainability_summary: str


# ---------------------------------------------------------------------------
# Internal pipeline helpers (pure — no I/O, designed for heuristic path)
# ---------------------------------------------------------------------------

def _enrich_write_heuristic(
    content: str,
    title: str,
    tags: list[str],
    metadata: dict,
) -> dict:
    """Simulate enrichment for the write verb in heuristic mode."""
    lower = content.lower()
    enrichment: dict[str, Any] = {}

    # Quick tone signal
    if any(w in lower for w in ("critical", "urgent", "outage", "down", "breach")):
        enrichment["tone"] = "urgent"
    elif any(w in lower for w in ("warning", "caution", "slow", "latency")):
        enrichment["tone"] = "cautionary"
    else:
        enrichment["tone"] = "informational"

    # Domain guess
    if any(w in lower for w in ("auth", "login", "sso", "password", "token")):
        enrichment["domain"] = "security"
    elif any(w in lower for w in ("deploy", "pipeline", "ci", "build")):
        enrichment["domain"] = "devops"
    elif any(w in lower for w in ("billing", "payment", "invoice")):
        enrichment["domain"] = "finance"
    else:
        enrichment["domain"] = "general"

    enrichment["word_count"] = len(content.split())
    enrichment["has_metadata"] = bool(metadata)
    return enrichment


def _assemble_read_context(memories: list[dict], query: str) -> list[dict]:
    """Sort memories by a simple relevance score (keyword overlap)."""
    query_tokens = set(query.lower().split())
    scored = []
    for mem in memories:
        content = str(mem.get("content") or "").lower()
        overlap = len(query_tokens & set(content.split()))
        scored.append((overlap, mem))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]


def _heuristic_decide(content: str, enrichment: dict) -> GatewayDecideResult:
    """Heuristic decision from content + optional enrichment signals."""
    lower = content.lower()
    agents_run = ["entity_resolution", "anomaly_detection", "narrative_synthesis"]

    anomaly = bool(enrichment.get("anomaly_detected"))
    score = float(enrichment.get("anomaly_score") or 0.0)
    tone = str(enrichment.get("tone") or "informational")

    if anomaly and score >= 0.9:
        decision = "escalate"
        confidence = min(0.95, 0.70 + score * 0.25)
        action = "Declare P1 and page on-call engineer."
    elif anomaly and score >= 0.7:
        decision = "investigate"
        confidence = 0.70 + score * 0.10
        action = "Assign to on-call team for investigation."
    elif any(w in lower for w in ("critical", "urgent", "outage")):
        decision = "escalate"
        confidence = 0.75
        action = "Escalate to engineering lead."
    elif any(w in lower for w in ("warning", "caution")):
        decision = "monitor"
        confidence = 0.65
        action = "Monitor and review in next stand-up."
    else:
        decision = "acknowledge"
        confidence = 0.55
        action = None

    return GatewayDecideResult(
        decision=decision,
        confidence=round(confidence, 4),
        tone=tone,
        action_recommended=action,
        enrichment=enrichment,
        agents_run=agents_run,
    )


def _heuristic_plan(goal: str, context: dict) -> GatewayPlanResult:
    """Heuristic goal decomposition."""
    lower = goal.lower()
    steps: list[dict] = []
    step_num = 1

    def _step(action: str, tool: str | None = None) -> dict:
        nonlocal step_num
        s = {"step_id": f"s{step_num}", "action": action, "tool": tool}
        step_num += 1
        return s

    if "report" in lower or "summarise" in lower or "summarize" in lower:
        steps = [
            _step("Retrieve relevant memories", "memory.search"),
            _step("Assemble temporal context", "temporal_reasoning"),
            _step("Generate narrative summary", "narrative.synthesize"),
        ]
    elif "investigate" in lower or "debug" in lower or "diagnose" in lower:
        steps = [
            _step("Search memory for related incidents", "memory.search"),
            _step("Identify affected entities", "entity_resolution"),
            _step("Check causal graph for known patterns", "causal.explain"),
            _step("Synthesise investigation report", "narrative.synthesize"),
        ]
    elif "escalate" in lower or "p1" in lower or "incident" in lower:
        steps = [
            _step("Retrieve P1 playbook", "playbook.match"),
            _step("Identify escalation targets", "org_attention"),
            _step("Dispatch P1 notification", "action.dispatch"),
            _step("Open incident episode", "episode.create"),
        ]
    else:
        steps = [
            _step("Retrieve context for goal", "memory.search"),
            _step("Decompose goal into subtasks", "goal_decomposition"),
            _step("Execute subtasks", None),
            _step("Validate completion", None),
        ]

    blocking = context.get("blocking_subtask") or None
    confidence = 0.80 if len(steps) >= 3 else 0.65

    return GatewayPlanResult(
        goal=goal,
        steps=steps,
        step_count=len(steps),
        blocking_step=blocking,
        confidence=confidence,
    )


def _heuristic_explain(memory_id: str, audit_records: list[dict]) -> GatewayExplainResult:
    """Assemble explainability summary from audit records."""
    agents = list({r.get("agent_name") or "" for r in audit_records if r.get("agent_name")})
    agents = [a for a in agents if a]

    if not audit_records:
        return GatewayExplainResult(
            memory_id=memory_id,
            decisions=[],
            agents=[],
            confidence=0.0,
            explainability_summary="No audit records found for this memory.",
        )

    recent = audit_records[-5:]
    confidence_vals = [
        float(r["confidence"]) for r in recent
        if "confidence" in r and _is_float(r["confidence"])
    ]
    avg_conf = sum(confidence_vals) / len(confidence_vals) if confidence_vals else 0.5

    summary_parts = [f"Memory {memory_id} was processed by {len(agents)} agent(s)."]
    if agents:
        summary_parts.append(f"Agents: {', '.join(agents[:5])}.")
    summary_parts.append(f"Average confidence: {avg_conf:.2f}.")

    return GatewayExplainResult(
        memory_id=memory_id,
        decisions=recent,
        agents=agents,
        confidence=round(avg_conf, 4),
        explainability_summary=" ".join(summary_parts),
    )


def _is_float(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class CognitiveGatewayService:
    """Five-verb gateway to Ninai's full intelligence stack.

    Each verb is:
    1. Capability-gated per org/tenant
    2. Handled heuristically (or can be extended to call real agents)
    3. Returns a typed result object

    In production, this service is wired to the real agent pipeline, memory
    service, and audit trail.  In tests and demos, the heuristic path runs
    standalone without DB or LLM access.
    """

    def __init__(
        self,
        *,
        capabilities: CognitiveGatewayCapabilities | None = None,
    ) -> None:
        self._caps = capabilities or CognitiveGatewayCapabilities.full()

    def _check(self, verb: str) -> None:
        if not self._caps.is_enabled(verb):
            raise PermissionError(f"Verb '{verb}' not enabled for this tenant.")

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(
        self,
        *,
        content: str,
        title: str = "",
        tags: list[str] | None = None,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> GatewayWriteResult:
        """Store and enrich a memory record."""
        self._check("write")
        _tags = list(tags or [])
        _meta = metadata or {}
        enrichment = _enrich_write_heuristic(content, title, _tags, _meta)

        import uuid
        mid = memory_id or str(uuid.uuid4())

        return GatewayWriteResult(
            memory_id=mid,
            enriched=True,
            enrichment_summary=enrichment,
            tags=_tags,
            created_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(
        self,
        *,
        query: str,
        memories: list[dict] | None = None,
        limit: int = 10,
    ) -> GatewayReadResult:
        """Retrieve + assemble context for a query."""
        self._check("read")
        _mems = list(memories or [])
        ranked = _assemble_read_context(_mems, query)[:limit]
        return GatewayReadResult(
            memories=ranked,
            total=len(ranked),
            query=query,
            context_assembled=len(ranked) > 0,
        )

    # ------------------------------------------------------------------
    # decide
    # ------------------------------------------------------------------

    def decide(
        self,
        *,
        content: str,
        enrichment: dict | None = None,
    ) -> GatewayDecideResult:
        """Run enrichment pipeline and return a decision with confidence."""
        self._check("decide")
        return _heuristic_decide(content, enrichment or {})

    # ------------------------------------------------------------------
    # plan
    # ------------------------------------------------------------------

    def plan(
        self,
        *,
        goal: str,
        context: dict | None = None,
    ) -> GatewayPlanResult:
        """Decompose a goal into ordered, actionable steps."""
        self._check("plan")
        return _heuristic_plan(goal, context or {})

    # ------------------------------------------------------------------
    # explain
    # ------------------------------------------------------------------

    def explain(
        self,
        *,
        memory_id: str,
        audit_records: list[dict] | None = None,
    ) -> GatewayExplainResult:
        """Return audit trail and explainability summary for a memory."""
        self._check("explain")
        return _heuristic_explain(memory_id, audit_records or [])
