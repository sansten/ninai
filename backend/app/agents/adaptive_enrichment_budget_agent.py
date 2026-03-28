"""Adaptive Enrichment Budget — Phase 35.

Uses feedback signal (Phase 24) and feature readiness/importance (Phase 28) to
dynamically decide which agents run per memory.  Learns which enrichment steps
actually improved retrieval quality and skips low-value agents for
low-importance writes.

Depends on:
  - FeedbackIntegrationAgent (Phase 24) — feedback_applied, correction_delta,
                                          feedback_confidence
  - FeatureReadinessService  (Phase 28) — per-agent {is_ready, active, params}
                                          via context["feature_readiness"]
  - OrchestrationBusAgent    (Phase 29) — execution_plan, agent_order
                                          via context["memory"]["enrichment"]

Inputs (via context["memory"]["enrichment"]):
  - feedback_applied:     bool
  - correction_delta:     dict[str, float]   — agent/field → signed delta
  - feedback_confidence:  float
  - uncertainty_level:    str                — "low" | "medium" | "high"
  - anomaly_detected:     bool
  - credibility_score:    float
  - agent_order:          list[str]          — from OrchestrationBusAgent

Inputs (via context["feature_readiness"]):
  - dict[agent_name, {is_ready: bool, active: bool, params: dict}]

Outputs:
  - agent_budget:      list[str]         — agents approved to run
  - skipped_agents:    list[str]         — agents skipped to save budget
  - budget_reason:     dict[str, str]    — agent → reason string
  - budget_confidence: float
  - rationale:         "heuristic" | "llm"
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
# Constants
# ---------------------------------------------------------------------------

# Agents that always run regardless of budget — removing them risks data loss.
_CORE_AGENTS: frozenset[str] = frozenset({
    "MetadataExtractionAgent",
    "SemanticNormalizationAgent",
    "EntityResolutionAgent",
    "CredibilityAgent",
    "UncertaintyReportingAgent",
    "AuditTrailAgent",
})

_IMPORTANCE_HIGH = 0.65   # run all ready agents above this threshold
_IMPORTANCE_LOW  = 0.30   # aggressively prune below this threshold

_MIN_FEEDBACK_CONF = 0.40  # ignore correction_delta below this confidence
_MIN_VALUE_TO_SKIP = 0.05  # agents with |delta| < this are treated as zero-value


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def compute_importance_score(enrichment: dict[str, Any]) -> float:
    """Compute a [0, 1] importance score for this memory write.

    Higher importance → fewer agents get skipped.
    """
    score = 0.30  # baseline

    credibility = enrichment.get("credibility_score")
    if isinstance(credibility, (int, float)):
        score += float(credibility) * 0.25

    uncertainty = str(enrichment.get("uncertainty_level") or "").lower()
    if uncertainty == "high":
        score += 0.25
    elif uncertainty == "medium":
        score += 0.12

    if bool(enrichment.get("anomaly_detected")):
        score += 0.15

    return round(min(1.0, score), 4)


def score_agent_value(
    agent_name: str,
    correction_delta: dict[str, float],
    feedback_confidence: float,
) -> float:
    """Return the absolute value contribution this agent made to feedback corrections.

    Matches correction_delta keys that contain any CamelCase word from the agent
    name (e.g. "AnomalyDetectionAgent" matches keys containing "anomaly" or
    "detection").  Returns 0.0 when feedback_confidence is below the minimum.
    """
    if feedback_confidence < _MIN_FEEDBACK_CONF:
        return 0.0

    # Split CamelCase into words, drop generic "agent" suffix
    raw_parts = re.findall(r"[A-Z][a-z]+", agent_name)
    name_parts = [p.lower() for p in raw_parts if p.lower() not in {"agent"} and len(p) > 3]
    if not name_parts:
        return 0.0

    total = 0.0
    for field, delta in (correction_delta or {}).items():
        field_lower = field.lower()
        if any(part in field_lower for part in name_parts):
            total += abs(float(delta) if isinstance(delta, (int, float)) else 0.0)

    return round(total, 4)


def should_skip_agent(
    agent_name: str,
    importance: float,
    value: float,
    readiness: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Decide whether to skip an agent.

    Returns (skip: bool, reason: str).
    """
    if agent_name in _CORE_AGENTS:
        return False, "core"

    # Readiness gate — if readiness data exists and agent is disabled
    if readiness is not None:
        if not readiness.get("active", True):
            return True, "inactive"
        if not readiness.get("is_ready", True):
            return True, "not_ready"

    # High importance — run everything that is ready
    if importance >= _IMPORTANCE_HIGH:
        return False, "high_importance"

    # Low importance + low value → skip
    if importance < _IMPORTANCE_LOW and value < _MIN_VALUE_TO_SKIP:
        return True, "low_importance_low_value"

    # Medium importance — skip if proven zero-value
    if value < _MIN_VALUE_TO_SKIP and importance < _IMPORTANCE_HIGH:
        return True, "zero_value"

    return False, "included"


def compute_budget_confidence(
    *,
    has_feedback: bool,
    feedback_confidence: float,
    has_readiness: bool,
    importance: float,
    skip_count: int,
    total: int,
) -> float:
    """Confidence in the budget decision, in [0, 1]."""
    base = 0.45
    if has_feedback and feedback_confidence >= _MIN_FEEDBACK_CONF:
        base += 0.15
    if has_readiness:
        base += 0.10
    if importance >= _IMPORTANCE_HIGH:
        base += 0.08
    if total > 0:
        skip_ratio = skip_count / total
        # confidence is lower when we skip a large fraction (more uncertainty)
        base -= round(skip_ratio * 0.10, 4)
    return round(min(0.90, max(0.0, base)), 4)


def run_heuristic(
    enrichment: dict[str, Any],
    feature_readiness: dict[str, Any],
    candidate_agents: list[str],
) -> dict[str, Any]:
    """Heuristic adaptive enrichment budget."""
    correction_delta: dict[str, float] = dict(enrichment.get("correction_delta") or {})
    feedback_confidence = float(enrichment.get("feedback_confidence") or 0.0)
    importance = compute_importance_score(enrichment)

    agent_budget: list[str] = []
    skipped_agents: list[str] = []
    budget_reason: dict[str, str] = {}

    for agent_name in candidate_agents:
        readiness = (feature_readiness or {}).get(agent_name)
        value = score_agent_value(agent_name, correction_delta, feedback_confidence)
        skip, reason = should_skip_agent(agent_name, importance, value, readiness)

        budget_reason[agent_name] = reason
        if skip:
            skipped_agents.append(agent_name)
        else:
            agent_budget.append(agent_name)

    budget_confidence = compute_budget_confidence(
        has_feedback=bool(enrichment.get("feedback_applied")),
        feedback_confidence=feedback_confidence,
        has_readiness=bool(feature_readiness),
        importance=importance,
        skip_count=len(skipped_agents),
        total=len(candidate_agents),
    )

    return {
        "agent_budget": agent_budget,
        "skipped_agents": skipped_agents,
        "budget_reason": budget_reason,
        "budget_confidence": budget_confidence,
        "rationale": "heuristic",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AdaptiveEnrichmentBudgetAgent(BaseAgent):
    """Phase 35: Decide which enrichment agents run per memory write.

    Reads feedback signal and feature readiness to prune low-value agents for
    low-importance memories while ensuring core agents always execute.
    """

    name = "AdaptiveEnrichmentBudgetAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["FeedbackIntegrationAgent", "OrchestrationBusAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("agent_budget"), list):
            raise ValueError("agent_budget must be a list")

        if not isinstance(outputs.get("skipped_agents"), list):
            raise ValueError("skipped_agents must be a list")

        if not isinstance(outputs.get("budget_reason"), dict):
            raise ValueError("budget_reason must be a dict")

        confidence = outputs.get("budget_confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            raise ValueError("budget_confidence must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment: dict[str, Any] = memory.get("enrichment") or {}
        feature_readiness: dict[str, Any] = dict(context.get("feature_readiness") or {})

        # Candidate agents come from OrchestrationBusAgent's agent_order if
        # available, otherwise fall back to an empty list (caller supplies them).
        candidate_agents: list[str] = list(
            enrichment.get("agent_order")
            or context.get("candidate_agents")
            or []
        )

        strategy = getattr(settings, "ENRICHMENT_BUDGET_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = run_heuristic(enrichment, feature_readiness, candidate_agents)
        else:
            prompt = (
                "You are an adaptive enrichment budget planner for an enterprise "
                "memory OS.  Output JSON only.  Do not hallucinate.\n\n"
                "Given the memory enrichment signals and feature readiness, decide "
                "which agents from the candidate list should run (agent_budget) and "
                "which should be skipped (skipped_agents) for this memory write.\n\n"
                f"IMPORTANCE_SIGNALS:\n"
                f"  credibility_score: {enrichment.get('credibility_score')}\n"
                f"  uncertainty_level: {enrichment.get('uncertainty_level')}\n"
                f"  anomaly_detected:  {enrichment.get('anomaly_detected')}\n"
                f"FEEDBACK:\n"
                f"  feedback_applied:    {enrichment.get('feedback_applied')}\n"
                f"  feedback_confidence: {enrichment.get('feedback_confidence')}\n"
                f"  correction_delta:    {enrichment.get('correction_delta')}\n"
                f"CANDIDATE_AGENTS: {candidate_agents}\n"
                f"FEATURE_READINESS_KEYS: {list(feature_readiness.keys())}\n\n"
                "Core agents that MUST always be included: "
                f"{sorted(_CORE_AGENTS)}\n\n"
                "Return JSON with keys:\n"
                "- agent_budget: list[str]\n"
                "- skipped_agents: list[str]\n"
                "- budget_reason: dict[str, str]  (agent → reason)\n"
                "- budget_confidence: float 0..1\n"
                "- rationale: brief explanation"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_ollama_model("agents")),
                timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
            )
            resp = await client.complete_json(
                prompt=prompt,
                schema_hint={},
                tool_event_sink=context.get("tool_event_sink"),
            )
            if (
                isinstance(resp, dict)
                and isinstance(resp.get("agent_budget"), list)
                and isinstance(resp.get("skipped_agents"), list)
                and isinstance(resp.get("budget_reason"), dict)
            ):
                outputs = resp
            else:
                outputs = run_heuristic(enrichment, feature_readiness, candidate_agents)

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("budget_confidence", 0.45)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
