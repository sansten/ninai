"""Silo Propagation Agent — Phase 9.

On memory write, scores cross-silo signals from Phase 8 context_bundle for
relevance to recipient domains and emits propagation-ready signals for
injection into other teams' cognitive context on next query.

Depends on Phase 8 (ContextAmplifierAgent) via memory enrichment
context_bundle field.
Also uses Phase 6 (SemanticNormalizationAgent) business_domain.

Outputs power downstream phases:
  - Phase 10 (Organizational Attention Model): signals update team attention
  - Phase 11 (Proactive Memory Push): high-relevance signals trigger proactive push
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.llm_breaker import create_llm_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Entity type → extra relevance weight
# ---------------------------------------------------------------------------

_ENTITY_TYPE_WEIGHT: dict[str, float] = {
    "person": 0.20,
    "org": 0.20,
    "organization": 0.20,
    "customer": 0.15,
    "project": 0.10,
    "system": 0.10,
    "concept": 0.05,
    "metric": 0.08,
    "role": 0.08,
}

_DEFAULT_ENTITY_WEIGHT = 0.05

_DEFAULT_THRESHOLD = 0.5


def score_signal(item: dict[str, Any]) -> float:
    """Compute propagation relevance score for a context_bundle item.

    Score components:
      - Base: 0.40
      - source == "cross_silo": +0.10
      - silo_count >= 2: +0.10
      - silo_count >= 3: +0.10 (additional)
      - entity_type weight: 0.05..0.20

    Capped at 1.0.
    """
    score = 0.40

    if item.get("source") == "cross_silo":
        score += 0.10

    silo_count = int(item.get("silo_count", 1))
    if silo_count >= 2:
        score += 0.10
    if silo_count >= 3:
        score += 0.10

    entity_type = (item.get("entity_type") or "concept").lower()
    score += _ENTITY_TYPE_WEIGHT.get(entity_type, _DEFAULT_ENTITY_WEIGHT)

    return min(round(score, 3), 1.0)


def propagate_signals(
    *,
    context_bundle: list[dict[str, Any]],
    primary_domain: str,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Filter and score context_bundle items for cross-silo propagation.

    Only cross_silo source items with relevance_score >= threshold are
    propagated. Resolved (primary-domain) items are not propagated — they
    stay local.

    Returns (propagation_signals[:50], sorted(target_domains)).
    """
    signals: list[dict[str, Any]] = []
    target_domains: set[str] = set()

    for item in context_bundle:
        if item.get("source") != "cross_silo":
            continue  # primary-domain resolved signals are not propagated

        relevance = score_signal(item)
        if relevance < threshold:
            continue

        domain = item.get("domain", "")
        signals.append(
            {
                "entity": item.get("entity", ""),
                "original": item.get("original", item.get("entity", "")),
                "entity_type": item.get("entity_type", "concept"),
                "signal": item.get("signal", ""),
                "source_domain": primary_domain,
                "target_domain": domain,
                "relevance_score": relevance,
                "silo_count": item.get("silo_count", 1),
            }
        )
        target_domains.add(domain)

    return signals[:50], sorted(target_domains)


class SiloPropagationAgent(BaseAgent):
    """Phase 9: Score and emit cross-silo propagation signals.

    Inputs (via context["memory"]["enrichment"]):
      - context_bundle:   list[{entity, original, entity_type, domain,
                               signal, source, silo_count}]  — from Phase 8
      - business_domain:  str — primary domain from Phase 6

    Outputs:
      - propagation_signals:  list[{entity, original, entity_type, signal,
                                    source_domain, target_domain,
                                    relevance_score, silo_count}]
      - target_domains:       list[str]
      - signal_count:         int
      - threshold_used:       float
      - confidence:           float
      - rationale:            "heuristic" | "llm"
    """

    name = "SiloPropagationAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["ContextAmplifierAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("propagation_signals", []), list):
            raise ValueError("propagation_signals must be a list")
        if not isinstance(outputs.get("target_domains", []), list):
            raise ValueError("target_domains must be a list")
        for item in outputs.get("propagation_signals", []):
            if not isinstance(item, dict):
                raise ValueError("each propagation_signals item must be a dict")
            if "entity" not in item or "target_domain" not in item:
                raise ValueError(
                    "each propagation_signals item must have entity and target_domain"
                )

    def _get_threshold(self) -> float:
        raw = getattr(settings, "SILO_PROPAGATION_THRESHOLD", None)
        try:
            return float(raw) if raw is not None else _DEFAULT_THRESHOLD
        except (TypeError, ValueError):
            return _DEFAULT_THRESHOLD

    def _heuristic(
        self,
        *,
        context_bundle: list[dict[str, Any]],
        primary_domain: str,
        threshold: float,
    ) -> dict[str, Any]:
        signals, target_domains = propagate_signals(
            context_bundle=context_bundle,
            primary_domain=primary_domain,
            threshold=threshold,
        )

        cross_items = [b for b in context_bundle if b.get("source") == "cross_silo"]
        total = len(cross_items)
        if total == 0:
            confidence = 0.4
        else:
            confidence = min(0.9, round(0.4 + 0.4 * (len(signals) / max(total, 1)), 3))

        return {
            "propagation_signals": signals,
            "target_domains": target_domains,
            "signal_count": len(signals),
            "threshold_used": threshold,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}
        content = (context.get("memory") or {}).get("content", "")

        # Phase 8 output
        context_bundle: list[dict[str, Any]] = enrichment.get("context_bundle", [])

        # Phase 6 primary domain
        primary_domain: str = enrichment.get("business_domain", "general")

        threshold = self._get_threshold()

        strategy = getattr(settings, "SILO_PROPAGATION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                context_bundle=context_bundle,
                primary_domain=primary_domain,
                threshold=threshold,
            )
        else:
            prompt = (
                "You are a cross-silo signal propagation engine for an enterprise Cognitive OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Given a context_bundle of cross-silo signals, score each for propagation relevance "
                "and emit only those above the threshold as propagation_signals.\n\n"
                f"PRIMARY DOMAIN: {primary_domain}\n"
                f"THRESHOLD: {threshold}\n"
                f"CONTEXT BUNDLE: {context_bundle}\n\n"
                "Return JSON with keys:\n"
                "- propagation_signals: list of {entity, original, entity_type, signal, "
                "source_domain, target_domain, relevance_score, silo_count}\n"
                "- target_domains: list of domain strings that will receive signals\n"
                "- signal_count: int\n"
                "- threshold_used: float\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation"
            )
            client = create_llm_client(
                base_url=str(getattr(settings, "VLLM_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_llm_model("agents")),
                timeout_seconds=float(getattr(settings, "VLLM_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "VLLM_MAX_CONCURRENCY", 2)),
            )
            resp = await client.complete_json(
                prompt=prompt,
                schema_hint={},
                tool_event_sink=context.get("tool_event_sink"),
            )
            if (
                isinstance(resp, dict)
                and isinstance(resp.get("propagation_signals"), list)
                and isinstance(resp.get("target_domains"), list)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    context_bundle=context_bundle,
                    primary_domain=primary_domain,
                    threshold=threshold,
                )

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
