"""Cognitive Gateway Service — Phase 49 (revised).

Exposes Ninai's full intelligence stack as five simple verbs for downstream
consumers (Tier-3 developers, AI agent platforms, enterprise integrations):

  write(content, ...)      → store + enrich a memory record
  read(query, ...)         → retrieve + rank context
  decide(context, ...)     → run anomaly detection, return decision + confidence
  plan(goal, ...)          → decompose goal into real subtasks, return ordered steps
  explain(memory_id)       → return audit trail for last decision on a memory

decide() delegates to AnomalyDetectionAgent's heuristic for real signal
detection rather than plain keyword scanning.

plan() uses GoalDecompositionAgent's extraction helpers (extract_subtasks,
detect_blocking_subtask) to pull real subtasks from goal text before falling
back to keyword-matched templates when the text has no extractable structure.

read() accepts an optional vector_fn callback for semantic ranking (production
path via Qdrant); falls back to token-overlap sort when absent.

Per-tenant capability gating:
  CognitiveGatewayCapabilities controls which verbs are available to each org.
  Default: all five verbs enabled.  Can be restricted to read+write for lite plans.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from app.agents.anomaly_detection_agent import run_heuristic as _anomaly_detect
from app.agents.debate_ensemble_agent import DebateEnsembleAgent
from app.agents.memory_tier_manager_agent import MemoryTierManagerAgent
from app.core.requester_context import RequesterContext
from app.services.cognitive_fingerprint_service import CognitiveFingerprintService
from app.services.context_compression_service import ContextCompressionService
from app.services.corrective_rag_service import CorrectiveRagService
from app.services.enterprise_fallbacks import (
    detect_blocking_subtask,
    detect_goal,
    extract_subtasks,
)
from app.services.attention_retrieval_service import AttentionRetrievalService
from app.services.self_rag_service import SelfRagService


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
    retrieval_confidence: float = 0.0
    corrected_by: str | None = None
    reasoning_steps: list[dict] = field(default_factory=list)
    compression_ratio: float = 1.0
    information_density: float = 0.0


@dataclass
class GatewayDecideResult:
    decision: str
    confidence: float
    tone: str
    action_recommended: str | None
    enrichment: dict
    agents_run: list[str]
    debate_transcript: list[dict] = field(default_factory=list)
    fingerprint_alerts: list[dict] = field(default_factory=list)


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

    if any(w in lower for w in ("critical", "urgent", "outage", "down", "breach")):
        enrichment["tone"] = "urgent"
    elif any(w in lower for w in ("warning", "caution", "slow", "latency")):
        enrichment["tone"] = "cautionary"
    else:
        enrichment["tone"] = "informational"

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


def _credibility_weight(mem: dict) -> float:
    """Extract credibility weight from a memory dict (0.1–1.0).

    Falls back to 1.0 if no credibility signal is present so that memories
    without an explicit score are not penalised.
    """
    raw = mem.get("credibility_score")
    if raw is None:
        raw = mem.get("enrichment", {}).get("credibility_score")
    try:
        val = float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
        val = 1.0
    return max(0.1, min(1.0, val))


_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalise_tokens(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    return {t for t in _PUNCT_RE.sub(" ", text.lower()).split() if t}


def _assemble_read_context(
    memories: list[dict],
    query: str,
    requester: "RequesterContext | None" = None,
) -> list[dict]:
    """Sort memories by credibility-weighted keyword-overlap relevance score.

    Score = (token_overlap + domain_boost) * credibility_weight

    domain_boost: memories whose content overlaps with the requester's
    dominant_domains get a 0.5-per-token additive boost.  This surfaces
    domain-relevant signals for e.g. a CFO without suppressing unrelated
    memories entirely.

    In production this is replaced by a Qdrant vector similarity query via
    the vector_fn callback accepted by CognitiveGatewayService.read().
    """
    query_tokens = _normalise_tokens(query)

    # Build domain token set from requester profile (empty if not available)
    domain_tokens: set[str] = set()
    if requester and requester.dominant_domains:
        for domain in requester.dominant_domains:
            domain_tokens.update(_normalise_tokens(domain.replace("_", " ")))

    scored = []
    for mem in memories:
        content_tokens = _normalise_tokens(str(mem.get("content") or ""))
        overlap = len(query_tokens & content_tokens)
        domain_boost = len(domain_tokens & content_tokens) * 0.5 if domain_tokens else 0.0
        weighted = (overlap + domain_boost) * _credibility_weight(mem)
        scored.append((weighted, mem))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]


def _escalate_action(requester: "RequesterContext | None") -> str:
    """Return a role-appropriate escalation action string."""
    if requester and requester.job_role:
        role = requester.job_role.lower()
        if any(r in role for r in ("ceo", "cto", "cfo", "vp", "director", "exec")):
            return "Brief executive team and prepare board communication."
        if any(r in role for r in ("engineer", "sre", "devops", "oncall", "on-call")):
            return "Declare P1, page on-call team, and open incident channel."
        if any(r in role for r in ("csm", "customer", "success", "support")):
            return "Notify affected enterprise accounts and escalate to account managers."
    return "Declare P1 and page on-call engineer."


def _heuristic_decide(
    content: str,
    enrichment: dict,
    requester: "RequesterContext | None" = None,
) -> GatewayDecideResult:
    """Decision from content and enrichment signals.

    Delegates to AnomalyDetectionAgent.run_heuristic() for real anomaly
    signal extraction.  Caller-supplied anomaly_detected / anomaly_score
    values take precedence and override agent outputs (allows pre-computed
    pipeline enrichment to flow through unchanged).
    """
    agents_run = ["anomaly_detection", "entity_resolution", "narrative_synthesis"]

    # Run real anomaly detection; caller overrides take precedence
    agent_signals = _anomaly_detect(enrichment)
    merged: dict[str, Any] = {**agent_signals, **enrichment}

    anomaly = bool(merged.get("anomaly_detected"))
    score = float(merged.get("anomaly_score") or 0.0)
    tone = str(merged.get("tone") or "informational")

    lower = content.lower()

    # Crisis urgency (requester paging outside business hours) → lower the
    # escalation threshold so marginal signals still get routed correctly.
    crisis_offset = 0.1 if (requester and requester.urgency_signal == "crisis") else 0.0

    if anomaly and score >= (0.9 - crisis_offset):
        decision = "escalate"
        confidence = min(0.95, 0.70 + score * 0.25)
        action: str | None = _escalate_action(requester)
    elif anomaly and score >= (0.7 - crisis_offset):
        decision = "investigate"
        confidence = 0.70 + score * 0.10
        action = "Assign to on-call team for investigation."
    elif any(w in lower for w in ("critical", "urgent", "outage")):
        decision = "escalate"
        confidence = 0.75
        action = _escalate_action(requester)
    elif any(w in lower for w in ("warning", "caution")):
        decision = "monitor"
        confidence = 0.65
        action = "Monitor and review in next stand-up."
    else:
        decision = "acknowledge"
        confidence = 0.55
        action = None

    debate_transcript: list[dict] = []
    if confidence >= 0.7:
        debate = DebateEnsembleAgent().generate_transcript(
            content=content,
            decision=decision,
            confidence=confidence,
            enrichment=merged,
        )
        debate_transcript = list(debate.get("debate_transcript") or [])
        moderator_decision = str(debate.get("moderator_decision") or decision)
        if moderator_decision:
            decision_rank = {
                "acknowledge": 0,
                "monitor": 1,
                "investigate": 2,
                "escalate": 3,
            }
            base_rank = decision_rank.get(decision, 1)
            mod_rank = decision_rank.get(moderator_decision, base_rank)
            # Moderator synthesis can strengthen or keep severity, but should
            # not downgrade a stronger safety posture.
            if mod_rank >= base_rank:
                decision = moderator_decision
        agents_run.append("debate_ensemble")

    return GatewayDecideResult(
        decision=decision,
        confidence=round(confidence, 4),
        tone=tone,
        action_recommended=action,
        enrichment=merged,
        agents_run=agents_run,
        debate_transcript=debate_transcript,
    )


_TECHNICAL_TOOLS = {"causal.explain", "entity_resolution", "causal.predict", "goal_decomposition"}
_NARRATIVE_TOOLS = {"narrative.synthesize", "narrative.generate"}


def _filter_steps_for_requester(
    steps: list[dict],
    requester: "RequesterContext",
) -> list[dict]:
    """Trim and reorder plan steps based on who is reading the plan.

    Rules:
    - pre_meeting urgency → cap at 3 steps (decision-maker needs concise list)
    - crisis urgency → push action steps first (act now, explain later)
    - executive roles (CEO/CFO/VP/Director) → drop purely technical tool steps
    - engineering/sre roles → drop narrative-synthesis-only steps
    """
    role = (requester.job_role or "").lower()
    urgency = requester.urgency_signal

    is_executive = any(r in role for r in ("ceo", "cfo", "cto", "vp", "director", "exec", "chief"))
    is_engineer = any(r in role for r in ("engineer", "sre", "devops", "oncall", "developer"))

    filtered = list(steps)

    if is_executive:
        filtered = [
            s for s in filtered
            if s.get("tool") not in _TECHNICAL_TOOLS
        ]

    if is_engineer:
        filtered = [
            s for s in filtered
            if s.get("tool") not in _NARRATIVE_TOOLS
        ]

    if urgency == "crisis":
        # Bring action/dispatch steps to the front
        action_steps = [s for s in filtered if s.get("tool") and "action" in s.get("tool", "")]
        other_steps = [s for s in filtered if s not in action_steps]
        filtered = action_steps + other_steps

    if urgency == "pre_meeting":
        filtered = filtered[:3]

    # Always keep at least one step
    return filtered or steps[:1]


def _heuristic_plan(
    goal: str,
    context: dict,
    requester: "RequesterContext | None" = None,
) -> GatewayPlanResult:
    """Goal decomposition using real subtask extraction.

    First attempts structural extraction via GoalDecompositionAgent helpers
    (numbered lists, bullet points, action-verb sentences).  Falls back to
    keyword-matched templates when extraction yields no subtasks — which is
    the common case for short, unstructured goal strings.
    """
    step_num = 0

    def _step(action: str, tool: str | None = None) -> dict:
        nonlocal step_num
        step_num += 1
        return {"step_id": f"s{step_num}", "action": action, "tool": tool}

    # --- Try real extraction first (require ≥ 2 subtasks; single-sentence
    #     extraction is just the goal itself and doesn't add value over templates) ---
    raw_subtasks = extract_subtasks(goal)
    if len(raw_subtasks) >= 2:
        steps = [_step(t) for t in raw_subtasks]
        blocking = (
            detect_blocking_subtask(raw_subtasks, goal)
            or context.get("blocking_subtask")
            or None
        )
        confidence = 0.85 if len(steps) >= 3 else 0.70
        if requester:
            steps = _filter_steps_for_requester(steps, requester)
        return GatewayPlanResult(
            goal=goal,
            steps=steps,
            step_count=len(steps),
            blocking_step=blocking,
            confidence=confidence,
        )

    # --- Keyword-template fallback (unstructured goal strings) ---
    lower = goal.lower()

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

    # Apply requester-aware step filtering.
    if requester:
        steps = _filter_steps_for_requester(steps, requester)

    return GatewayPlanResult(
        goal=goal,
        steps=[s for s in steps if s],
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
# Gateway context chain (stateful multi-call sessions via Redis)
# ---------------------------------------------------------------------------

@dataclass
class GatewayContextSession:
    """Accumulated state across multiple gateway verb calls sharing a context_id."""

    context_id: str
    org_id: str
    prior_decision: str | None = None
    prior_steps: list = field(default_factory=list)
    prior_memories: list = field(default_factory=list)
    accumulated_enrichment: dict = field(default_factory=dict)
    working_set_summary: dict = field(default_factory=lambda: {
        "working_set": [],
        "archival": [],
        "working_set_size": 0,
        "archival_size": 0,
        "loaded_ids": [],
        "offloaded_ids": [],
        "updated_at": None,
    })
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    call_count: int = 0


def _ctx_key(context_id: str, org_id: str) -> str:
    return f"gateway:ctx:{context_id}:{org_id}"


async def load_gateway_context(context_id: str, org_id: str) -> GatewayContextSession | None:
    """Load a gateway context session from Redis. Returns None if missing or expired."""
    try:
        from app.core.redis import get_redis_client  # type: ignore[import]
        r = get_redis_client()
        raw = r.get(_ctx_key(context_id, org_id))
        if raw is None:
            return None
        data = json.loads(raw)
        return GatewayContextSession(**data)
    except Exception:
        return None


async def save_gateway_context(ctx: GatewayContextSession, ttl: int = 3600) -> None:
    """Persist updated context to Redis with TTL (default 1 hour)."""
    try:
        from app.core.redis import get_redis_client  # type: ignore[import]
        r = get_redis_client()
        ctx.updated_at = datetime.now(timezone.utc).isoformat()
        r.set(_ctx_key(ctx.context_id, ctx.org_id), json.dumps(dataclasses.asdict(ctx)), ex=ttl)
    except Exception:
        pass  # Context chaining degrades gracefully when Redis is unavailable


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class CognitiveGatewayService:
    """Five-verb gateway to Ninai's full intelligence stack.

    Each verb is:
    1. Capability-gated per org/tenant
    2. Backed by real agent logic (anomaly detection, goal decomposition)
       with token-overlap / template fallbacks for offline/test paths
    3. Returns a typed result object

    read() accepts an optional vector_fn(query, memories) → list[dict]
    callback for Qdrant-backed semantic ranking in production.
    """

    def __init__(
        self,
        *,
        capabilities: CognitiveGatewayCapabilities | None = None,
    ) -> None:
        self._caps = capabilities or CognitiveGatewayCapabilities.full()
        self._tier_manager = MemoryTierManagerAgent()
        self._fingerprint = CognitiveFingerprintService()

    def _check(self, verb: str) -> None:
        if not self._caps.is_enabled(verb):
            raise PermissionError(f"Verb '{verb}' not enabled for this tenant.")

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    async def write(
        self,
        *,
        content: str,
        title: str = "",
        tags: list[str] | None = None,
        metadata: dict | None = None,
        memory_id: str | None = None,
        context_id: str | None = None,
        org_id: str | None = None,
        requester: RequesterContext | None = None,
    ) -> GatewayWriteResult:
        """Store and enrich a memory record."""
        self._check("write")
        _tags = list(tags or [])
        _meta = metadata or {}
        enrichment = _enrich_write_heuristic(content, title, _tags, _meta)

        import uuid
        mid = memory_id or str(uuid.uuid4())

        # Merge requester context into enrichment so downstream agents
        # know the provenance of this write (who wrote it, their role, when).
        if requester:
            _meta.setdefault("_requester_job_role", requester.job_role)
            _meta.setdefault("_requester_timezone", requester.timezone)
            _meta.setdefault("_requester_urgency", requester.urgency_signal)
            # Seed domain from requester's top domain if enrichment didn't resolve one
            if enrichment.get("domain") == "general" and requester.dominant_domains:
                enrichment["domain"] = requester.dominant_domains[0]

        result = GatewayWriteResult(
            memory_id=mid,
            enriched=True,
            enrichment_summary=enrichment,
            tags=_tags,
            created_at=datetime.now(timezone.utc),
        )

        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id) or GatewayContextSession(
                context_id=context_id, org_id=org_id
            )
            ctx.accumulated_enrichment.update(enrichment)
            ctx.working_set_summary = self._tier_manager.reconcile(
                working_set=list(ctx.working_set_summary.get("working_set") or []),
                archival=list(ctx.working_set_summary.get("archival") or []),
                incoming=[{"id": mid, "content": content}],
            )
            ctx.call_count += 1
            await save_gateway_context(ctx)

        return result

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    async def read(
        self,
        *,
        query: str,
        memories: list[dict] | None = None,
        limit: int = 10,
        context_token_budget: int = 4000,
        filter_tags: list[str] | None = None,
        vector_fn: Callable[[str, list[dict]], list[dict]] | None = None,
        external_connector_fn: Callable[[str, int], list[dict]] | None = None,
        cross_encoder_fn: Callable[[str, list[dict]], list[dict]] | None = None,
        context_id: str | None = None,
        org_id: str | None = None,
        requester: RequesterContext | None = None,
    ) -> GatewayReadResult:
        """Retrieve and rank context for a query.

        vector_fn(query, memories) → sorted list[dict]
          When provided, used instead of token-overlap sort.
          Production: pass a closure over the org's Qdrant collection.
          Tests / offline: omit to use the token-overlap fallback.
        """
        self._check("read")

        # Merge prior memories from context chain as additional candidates
        _mems = list(memories or [])
        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id)
            if ctx and ctx.prior_memories:
                _mems = ctx.prior_memories + _mems

        # Optional tag-based scoping: keep only memories that carry ALL filter_tags.
        # Useful for conversation-scoped retrieval without passing the full candidate list.
        if filter_tags:
            _tag_set = set(filter_tags)
            _mems = [m for m in _mems if _tag_set.issubset(set(m.get("tags") or []))]

        if vector_fn is not None:
            # Caller-supplied semantic ranking takes precedence; skip attention rerank.
            ranked = vector_fn(query, _mems)[:limit]
        else:
            ranked = _assemble_read_context(_mems, query, requester)[:limit]
            # Attention-weighted reranking (Phase 58): re-scores candidates using
            # recency, query-token relevance, and goal/incident alignment before
            # passing to SelfRag. Works on plain dicts — no DB access required.
            _query_tokens = frozenset(
                re.findall(r"\b[a-z0-9_]+\b", query.lower())
            )
            ranked = AttentionRetrievalService().rank(
                memories=ranked,
                active_goals=[],
                active_incidents=[],
                query_tokens=_query_tokens,
                now=datetime.now(timezone.utc),
                limit=limit,
            )

        # Self-RAG verification before answer-time consumption.
        # strict_support requires keyword overlap between query and memory — valid for
        # open-corpus vector search but over-filters pre-scoped memory pools where the
        # caller has already narrowed candidates.  Always use non-strict mode so that
        # memories with sufficient content and credibility are not discarded purely
        # because the query phrasing differs from the stored text.
        self_rag = SelfRagService()
        verification = self_rag.verify_and_filter(
            query=query,
            memories=ranked,
            strict_support=False,
        )
        verified = verification.verified_memories

        # CRAG corrective pass if verified context quality is low.
        corrective = CorrectiveRagService().apply(
            query=query,
            memories=verified,
            limit=limit,
            external_connector_fn=external_connector_fn,
            cross_encoder_fn=cross_encoder_fn,
        )

        compression = ContextCompressionService().compress(
            memories=corrective.memories,
            token_budget=context_token_budget,
        )

        result = GatewayReadResult(
            memories=compression.memories,
            total=len(compression.memories),
            query=query,
            context_assembled=len(compression.memories) > 0,
            retrieval_confidence=corrective.retrieval_confidence,
            corrected_by=corrective.corrected_by,
            reasoning_steps=verification.reasoning_steps,
            compression_ratio=compression.compression_ratio,
            information_density=compression.information_density,
        )

        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id) or GatewayContextSession(
                context_id=context_id, org_id=org_id
            )
            ctx.prior_memories = compression.memories
            ctx.working_set_summary = self._tier_manager.reconcile(
                working_set=list(ctx.working_set_summary.get("working_set") or []),
                archival=list(ctx.working_set_summary.get("archival") or []),
                incoming=list(compression.memories or []),
            )
            ctx.call_count += 1
            await save_gateway_context(ctx)

        return result

    # ------------------------------------------------------------------
    # decide
    # ------------------------------------------------------------------

    async def decide(
        self,
        *,
        content: str,
        enrichment: dict | None = None,
        context_id: str | None = None,
        org_id: str | None = None,
        requester: RequesterContext | None = None,
    ) -> GatewayDecideResult:
        """Run anomaly detection pipeline and return a decision with confidence."""
        self._check("decide")

        _enrichment = dict(enrichment or {})
        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id)
            if ctx:
                # Merge accumulated enrichment from prior calls
                merged = {**ctx.accumulated_enrichment, **_enrichment}
                _enrichment = merged

        result = _heuristic_decide(content, _enrichment, requester)

        # Fingerprint the anomaly detection agent's output distribution.
        fp_result = self._fingerprint.detect_anomaly("anomaly_detection", result.enrichment)
        self._fingerprint.update_fingerprint("anomaly_detection", result.enrichment)
        result.fingerprint_alerts = [
            {
                "agent": a.agent_name,
                "field": a.field,
                "current_value": a.current_value,
                "expected_mean": a.expected_mean,
                "z_score": a.z_score,
            }
            for a in fp_result.alerts
        ]
        if fp_result.anomalous and "cognitive_fingerprint" not in result.agents_run:
            result.agents_run.append("cognitive_fingerprint")

        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id) or GatewayContextSession(
                context_id=context_id, org_id=org_id
            )
            ctx.prior_decision = result.decision
            ctx.accumulated_enrichment.update(result.enrichment)
            ctx.working_set_summary = self._tier_manager.reconcile(
                working_set=list(ctx.working_set_summary.get("working_set") or []),
                archival=list(ctx.working_set_summary.get("archival") or []),
                incoming=[],
            )
            ctx.call_count += 1
            await save_gateway_context(ctx)

        return result

    # ------------------------------------------------------------------
    # plan
    # ------------------------------------------------------------------

    async def plan(
        self,
        *,
        goal: str,
        context: dict | None = None,
        context_id: str | None = None,
        org_id: str | None = None,
        requester: RequesterContext | None = None,
    ) -> GatewayPlanResult:
        """Decompose a goal into ordered, actionable steps."""
        self._check("plan")

        _ctx = dict(context or {})
        if context_id and org_id:
            prior = await load_gateway_context(context_id, org_id)
            if prior and prior.prior_decision:
                _ctx.setdefault("prior_decision", prior.prior_decision)

        result = _heuristic_plan(goal, _ctx, requester)

        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id) or GatewayContextSession(
                context_id=context_id, org_id=org_id
            )
            ctx.prior_steps = result.steps
            ctx.working_set_summary = self._tier_manager.reconcile(
                working_set=list(ctx.working_set_summary.get("working_set") or []),
                archival=list(ctx.working_set_summary.get("archival") or []),
                incoming=[],
            )
            ctx.call_count += 1
            await save_gateway_context(ctx)

        return result

    # ------------------------------------------------------------------
    # explain
    # ------------------------------------------------------------------

    async def explain(
        self,
        *,
        memory_id: str,
        audit_records: list[dict] | None = None,
        context_id: str | None = None,
        org_id: str | None = None,
    ) -> GatewayExplainResult:
        """Return audit trail and explainability summary for a memory."""
        self._check("explain")
        result = _heuristic_explain(memory_id, audit_records or [])

        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id) or GatewayContextSession(
                context_id=context_id, org_id=org_id
            )
            ctx.working_set_summary = self._tier_manager.reconcile(
                working_set=list(ctx.working_set_summary.get("working_set") or []),
                archival=list(ctx.working_set_summary.get("archival") or []),
                incoming=[],
            )
            ctx.call_count += 1
            await save_gateway_context(ctx)

        return result
