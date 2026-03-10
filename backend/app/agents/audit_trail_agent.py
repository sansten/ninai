"""Audit & Explainability Trail — Phase 31.

Reads enrichment produced by all prior agents and assembles an append-only
audit_log explaining every agent decision made on this memory.  Answers
questions like "why was this flagged as anomaly?" or "why was this conflict
resolved this way?" for enterprise compliance audits.

Depends on (reads enrichment from):
  - SemanticNormalizationAgent (Phase 6)         — canonical_form, normalization_confidence
  - EntityResolutionAgent (Phase 7)              — resolved_entities, entity_count
  - ContextAmplifierAgent (Phase 8)              — amplification_confidence
  - SiloPropagationAgent (Phase 9)               — propagated_signals, propagation_confidence
  - OrgAttentionAgent (Phase 10)                 — attention_score, attention_tier
  - ProactiveMemoryPushAgent (Phase 11)          — push_recommended, push_confidence
  - CausalReasoningAgent (Phase 12)              — causal_links, causal_confidence
  - ConflictDetectionAgent (Phase 13)            — conflict_detected, conflict_count
  - AdaptiveConflictResolutionAgent (Phase 14)   — resolution_strategy, resolution_confidence
  - MemoryDecayAgent (Phase 15)                  — decay_factor, is_stale
  - MemoryConsolidationAgent (Phase 16)          — consolidation_recommended, consolidation_score
  - TemporalReasoningAgent (Phase 17)            — temporal_confidence, sequence_position
  - EpisodicGroupingAgent (Phase 18)             — episode_id, episode_confidence
  - CredibilityAgent (Phase 19)                  — credibility_score, source_tier
  - PlaybookAgent (Phase 20)                     — matched_playbook_id, playbook_confidence
  - GoalDecompositionAgent (Phase 21)            — goal_detected, completion_fraction
  - UncertaintyReportingAgent (Phase 22)         — uncertainty_level, review_recommended
  - NarrativeSynthesisAgent (Phase 23)           — narrative_confidence
  - FeedbackIntegrationAgent (Phase 24)          — feedback_applied, correction_made
  - AnomalyDetectionAgent (Phase 25)             — anomaly_detected, anomaly_type, severity

Outputs:
  - audit_log:       list[dict]  — per-agent entries: agent_name, decision,
                                    confidence, reasoning_snapshot, timestamp
  - agents_audited:  int         — number of agents with entries
  - total_entries:   int         — total audit_log length
  - coverage_score:  float       — fraction of expected agents that produced output
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Expected agents (used for coverage score)
# ---------------------------------------------------------------------------

_EXPECTED_AGENTS: list[str] = [
    "SemanticNormalizationAgent",
    "EntityResolutionAgent",
    "ContextAmplifierAgent",
    "SiloPropagationAgent",
    "OrgAttentionAgent",
    "ProactiveMemoryPushAgent",
    "CausalReasoningAgent",
    "ConflictDetectionAgent",
    "AdaptiveConflictResolutionAgent",
    "MemoryDecayAgent",
    "MemoryConsolidationAgent",
    "TemporalReasoningAgent",
    "EpisodicGroupingAgent",
    "CredibilityAgent",
    "PlaybookAgent",
    "GoalDecompositionAgent",
    "UncertaintyReportingAgent",
    "NarrativeSynthesisAgent",
    "FeedbackIntegrationAgent",
    "AnomalyDetectionAgent",
]


# ---------------------------------------------------------------------------
# Per-agent decision extractors
# ---------------------------------------------------------------------------

def _extract_semantic_normalization(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "canonical_form" not in enrichment and "normalization_confidence" not in enrichment:
        return None
    canonical = enrichment.get("canonical_form") or "unknown"
    conf = float(enrichment.get("normalization_confidence") or 0.65)
    decision = f"normalized to canonical form: {str(canonical)[:80]}"
    return {
        "agent_name": "SemanticNormalizationAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["canonical_form", "normalization_confidence", "normalization_strategy"]
            if enrichment.get(k) is not None
        },
    }


def _extract_entity_resolution(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "resolved_entities" not in enrichment and "entity_count" not in enrichment:
        return None
    count = int(enrichment.get("entity_count") or 0)
    conf = float(enrichment.get("resolution_confidence") or 0.65)
    decision = f"resolved {count} entities from raw content"
    return {
        "agent_name": "EntityResolutionAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["entity_count", "resolved_entities", "resolution_confidence"]
            if enrichment.get(k) is not None
        },
    }


def _extract_context_amplifier(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "amplification_confidence" not in enrichment:
        return None
    conf = float(enrichment.get("amplification_confidence") or 0.65)
    strategy = enrichment.get("amplification_strategy") or "heuristic"
    decision = f"context amplified via {strategy}"
    return {
        "agent_name": "ContextAmplifierAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["amplification_confidence", "amplification_strategy", "amplified_context"]
            if enrichment.get(k) is not None
        },
    }


def _extract_silo_propagation(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "propagated_signals" not in enrichment and "propagation_confidence" not in enrichment:
        return None
    signals = enrichment.get("propagated_signals") or []
    count = len(signals) if isinstance(signals, list) else 0
    conf = float(enrichment.get("propagation_confidence") or 0.65)
    decision = f"propagated {count} cross-silo signals"
    return {
        "agent_name": "SiloPropagationAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["propagated_signals", "propagation_confidence", "source_silos"]
            if enrichment.get(k) is not None
        },
    }


def _extract_org_attention(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "attention_score" not in enrichment and "attention_tier" not in enrichment:
        return None
    score = float(enrichment.get("attention_score") or 0.0)
    tier = enrichment.get("attention_tier") or "low"
    decision = f"attention tier={tier} (score={score:.2f})"
    return {
        "agent_name": "OrgAttentionAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, score)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["attention_score", "attention_tier", "attention_signals"]
            if enrichment.get(k) is not None
        },
    }


def _extract_proactive_push(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "push_recommended" not in enrichment:
        return None
    recommended = bool(enrichment.get("push_recommended"))
    conf = float(enrichment.get("push_confidence") or 0.65)
    decision = "push recommended to relevant users" if recommended else "no proactive push needed"
    return {
        "agent_name": "ProactiveMemoryPushAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["push_recommended", "push_confidence", "push_targets"]
            if enrichment.get(k) is not None
        },
    }


def _extract_causal_reasoning(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "causal_links" not in enrichment and "causal_confidence" not in enrichment:
        return None
    links = enrichment.get("causal_links") or []
    count = len(links) if isinstance(links, list) else 0
    conf = float(enrichment.get("causal_confidence") or 0.65)
    decision = f"identified {count} causal link(s)"
    return {
        "agent_name": "CausalReasoningAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["causal_links", "causal_confidence", "causal_depth"]
            if enrichment.get(k) is not None
        },
    }


def _extract_conflict_detection(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "conflict_detected" not in enrichment and "conflict_count" not in enrichment:
        return None
    detected = bool(enrichment.get("conflict_detected"))
    count = int(enrichment.get("conflict_count") or 0)
    conf = float(enrichment.get("conflict_confidence") or 0.70)
    if detected:
        decision = f"conflict detected: {count} conflict(s) found"
    else:
        decision = "no conflicts detected"
    return {
        "agent_name": "ConflictDetectionAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["conflict_detected", "conflict_count", "conflicts", "conflict_confidence"]
            if enrichment.get(k) is not None
        },
    }


def _extract_conflict_resolution(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "resolution_strategy" not in enrichment:
        return None
    strategy = enrichment.get("resolution_strategy") or "unknown"
    conf = float(enrichment.get("resolution_confidence") or 0.65)
    decision = f"conflict resolved via strategy: {strategy}"
    return {
        "agent_name": "AdaptiveConflictResolutionAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["resolution_strategy", "resolution_confidence", "resolution_rationale"]
            if enrichment.get(k) is not None
        },
    }


def _extract_memory_decay(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "decay_factor" not in enrichment:
        return None
    decay = float(enrichment.get("decay_factor") or 1.0)
    is_stale = bool(enrichment.get("is_stale"))
    stale_str = "stale" if is_stale else "fresh"
    decision = f"memory is {stale_str}; decay_factor={decay:.3f}"
    return {
        "agent_name": "MemoryDecayAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, 1.0 - abs(decay - 1.0))), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["decay_factor", "is_stale", "decay_rate", "age_days"]
            if enrichment.get(k) is not None
        },
    }


def _extract_memory_consolidation(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "consolidation_recommended" not in enrichment and "consolidation_score" not in enrichment:
        return None
    recommended = bool(enrichment.get("consolidation_recommended"))
    score = float(enrichment.get("consolidation_score") or 0.0)
    decision = "consolidation recommended" if recommended else f"no consolidation needed (score={score:.2f})"
    return {
        "agent_name": "MemoryConsolidationAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, score if score > 0 else 0.65)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["consolidation_recommended", "consolidation_score", "consolidation_target"]
            if enrichment.get(k) is not None
        },
    }


def _extract_temporal_reasoning(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "temporal_confidence" not in enrichment and "sequence_position" not in enrichment:
        return None
    conf = float(enrichment.get("temporal_confidence") or 0.65)
    position = enrichment.get("sequence_position")
    pos_str = f"; sequence_position={position}" if position is not None else ""
    decision = f"temporal context assessed (confidence={conf:.2f}){pos_str}"
    return {
        "agent_name": "TemporalReasoningAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["temporal_confidence", "sequence_position", "time_references", "temporal_order"]
            if enrichment.get(k) is not None
        },
    }


def _extract_episodic_grouping(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "episode_id" not in enrichment and "episode_confidence" not in enrichment:
        return None
    episode_id = enrichment.get("episode_id")
    conf = float(enrichment.get("episode_confidence") or 0.65)
    if episode_id:
        decision = f"assigned to episode: {str(episode_id)[:40]}"
    else:
        decision = "no episode assignment"
    return {
        "agent_name": "EpisodicGroupingAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["episode_id", "episode_confidence", "episode_theme"]
            if enrichment.get(k) is not None
        },
    }


def _extract_credibility(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "credibility_score" not in enrichment:
        return None
    score = float(enrichment.get("credibility_score") or 0.5)
    tier = enrichment.get("source_tier") or "unknown"
    decision = f"credibility_score={score:.2f}; source_tier={tier}"
    return {
        "agent_name": "CredibilityAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, score)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["credibility_score", "source_tier", "credibility_signals"]
            if enrichment.get(k) is not None
        },
    }


def _extract_playbook(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "matched_playbook_id" not in enrichment and "is_new_playbook_candidate" not in enrichment:
        return None
    playbook_id = enrichment.get("matched_playbook_id")
    conf = float(enrichment.get("playbook_confidence") or 0.65)
    is_new = bool(enrichment.get("is_new_playbook_candidate"))
    if playbook_id:
        decision = f"matched playbook: {str(playbook_id)[:40]}"
    elif is_new:
        decision = "identified as new playbook candidate"
    else:
        decision = "no playbook matched"
    return {
        "agent_name": "PlaybookAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["matched_playbook_id", "playbook_confidence", "is_new_playbook_candidate", "step_position"]
            if enrichment.get(k) is not None
        },
    }


def _extract_goal_decomposition(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "goal_detected" not in enrichment:
        return None
    detected = bool(enrichment.get("goal_detected"))
    fraction = enrichment.get("completion_fraction")
    blocking = enrichment.get("blocking_subtask")
    if detected:
        frac_str = f"; completion={fraction:.0%}" if isinstance(fraction, float) else ""
        block_str = f"; blocked on: {blocking}" if blocking else ""
        decision = f"goal detected{frac_str}{block_str}"
    else:
        decision = "no goal detected in memory"
    conf = float(fraction) if detected and isinstance(fraction, float) else 0.65
    return {
        "agent_name": "GoalDecompositionAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["goal_detected", "subtasks", "completion_fraction", "blocking_subtask"]
            if enrichment.get(k) is not None
        },
    }


def _extract_uncertainty_reporting(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "uncertainty_level" not in enrichment and "review_recommended" not in enrichment:
        return None
    level = enrichment.get("uncertainty_level") or "unknown"
    review = bool(enrichment.get("review_recommended"))
    review_str = "; human review recommended" if review else ""
    decision = f"uncertainty_level={level}{review_str}"
    level_conf = {"low": 0.85, "medium": 0.65, "high": 0.45, "critical": 0.30}
    conf = level_conf.get(str(level).lower(), 0.65)
    return {
        "agent_name": "UncertaintyReportingAgent",
        "decision": decision,
        "confidence": conf,
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["uncertainty_level", "uncertainty_score", "review_recommended",
                      "unresolved_conflicts", "low_confidence_fields"]
            if enrichment.get(k) is not None
        },
    }


def _extract_narrative_synthesis(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "narrative_confidence" not in enrichment and "narrative" not in enrichment:
        return None
    conf = float(enrichment.get("narrative_confidence") or 0.65)
    narrative = enrichment.get("narrative") or ""
    preview = str(narrative)[:60] + "…" if len(str(narrative)) > 60 else str(narrative)
    decision = f"narrative synthesized: {preview}" if narrative else "narrative synthesis attempted"
    return {
        "agent_name": "NarrativeSynthesisAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["narrative_confidence", "narrative_length", "narrative_strategy"]
            if enrichment.get(k) is not None
        },
    }


def _extract_feedback_integration(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "feedback_applied" not in enrichment and "correction_made" not in enrichment:
        return None
    applied = bool(enrichment.get("feedback_applied"))
    corrected = bool(enrichment.get("correction_made"))
    if corrected:
        decision = "correction applied based on feedback"
    elif applied:
        decision = "feedback integrated (no correction needed)"
    else:
        decision = "no feedback to integrate"
    conf = float(enrichment.get("feedback_confidence") or 0.70)
    return {
        "agent_name": "FeedbackIntegrationAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["feedback_applied", "correction_made", "feedback_source", "feedback_confidence"]
            if enrichment.get(k) is not None
        },
    }


def _extract_anomaly_detection(enrichment: dict[str, Any]) -> dict[str, Any] | None:
    if "anomaly_detected" not in enrichment:
        return None
    detected = bool(enrichment.get("anomaly_detected"))
    atype = enrichment.get("anomaly_type")
    severity = enrichment.get("severity")
    if detected:
        decision = f"anomaly flagged: {atype or 'unknown'} ({severity or 'unclassified'})"
    else:
        decision = "no anomaly detected"
    conf = float(enrichment.get("confidence") or enrichment.get("anomaly_score") or 0.65)
    return {
        "agent_name": "AnomalyDetectionAgent",
        "decision": decision,
        "confidence": round(min(1.0, max(0.0, conf)), 4),
        "reasoning_snapshot": {
            k: enrichment.get(k)
            for k in ["anomaly_detected", "anomaly_type", "severity", "anomaly_score", "affected_fields"]
            if enrichment.get(k) is not None
        },
    }


# Ordered list of extractor functions
_EXTRACTORS = [
    _extract_semantic_normalization,
    _extract_entity_resolution,
    _extract_context_amplifier,
    _extract_silo_propagation,
    _extract_org_attention,
    _extract_proactive_push,
    _extract_causal_reasoning,
    _extract_conflict_detection,
    _extract_conflict_resolution,
    _extract_memory_decay,
    _extract_memory_consolidation,
    _extract_temporal_reasoning,
    _extract_episodic_grouping,
    _extract_credibility,
    _extract_playbook,
    _extract_goal_decomposition,
    _extract_uncertainty_reporting,
    _extract_narrative_synthesis,
    _extract_feedback_integration,
    _extract_anomaly_detection,
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_audit_log(enrichment: dict[str, Any], timestamp: str) -> list[dict[str, Any]]:
    """Run all extractors and assemble the audit_log list."""
    log: list[dict[str, Any]] = []
    for extractor in _EXTRACTORS:
        entry = extractor(enrichment)
        if entry is not None:
            entry["timestamp"] = timestamp
            log.append(entry)
    return log


def compute_coverage(audit_log: list[dict[str, Any]]) -> float:
    """Fraction of expected agents that produced an audit entry."""
    if not _EXPECTED_AGENTS:
        return 0.0
    covered = {e["agent_name"] for e in audit_log}
    return round(len(covered) / len(_EXPECTED_AGENTS), 4)


def run_heuristic(enrichment: dict[str, Any]) -> dict[str, Any]:
    """Build the audit trail from enrichment without LLM involvement."""
    timestamp = datetime.now(timezone.utc).isoformat()
    audit_log = build_audit_log(enrichment, timestamp)
    coverage = compute_coverage(audit_log)
    agents_audited = len({e["agent_name"] for e in audit_log})

    return {
        "audit_log": audit_log,
        "agents_audited": agents_audited,
        "total_entries": len(audit_log),
        "coverage_score": coverage,
        "rationale": "heuristic",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AuditTrailAgent(BaseAgent):
    """Phase 31: Assemble an append-only audit trail of all agent decisions.

    Reads enrichment from every prior agent and produces a structured
    audit_log that answers "why did the system decide X?" for any memory.
    Required for enterprise compliance and explainability.
    """

    name = "AuditTrailAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return _EXPECTED_AGENTS[:]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        audit_log = outputs.get("audit_log")
        if not isinstance(audit_log, list):
            raise ValueError("audit_log must be a list")

        for entry in audit_log:
            if not isinstance(entry, dict):
                raise ValueError("each audit_log entry must be a dict")
            if not isinstance(entry.get("agent_name"), str):
                raise ValueError("audit_log entry missing string agent_name")
            if not isinstance(entry.get("decision"), str):
                raise ValueError("audit_log entry missing string decision")
            conf = entry.get("confidence")
            if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
                raise ValueError(f"audit_log entry confidence must be float in [0,1], got {conf!r}")
            if not isinstance(entry.get("reasoning_snapshot"), dict):
                raise ValueError("audit_log entry missing dict reasoning_snapshot")

        agents_audited = outputs.get("agents_audited")
        if not isinstance(agents_audited, int) or agents_audited < 0:
            raise ValueError("agents_audited must be a non-negative int")

        total = outputs.get("total_entries")
        if not isinstance(total, int) or total < 0:
            raise ValueError("total_entries must be a non-negative int")

        coverage = outputs.get("coverage_score")
        if not isinstance(coverage, (int, float)) or not (0.0 <= float(coverage) <= 1.0):
            raise ValueError("coverage_score must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment: dict[str, Any] = memory.get("enrichment") or {}

        strategy = getattr(settings, "AUDIT_TRAIL_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "heuristic")
        strategy = str(strategy or "heuristic").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = run_heuristic(enrichment)
        else:
            base_outputs = run_heuristic(enrichment)
            prompt = (
                "You are an enterprise audit trail agent. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Below is a summary of agent decisions on one memory. "
                "Return a JSON object with the same structure plus a "
                "'summary' key: a 1-2 sentence plain-English explanation "
                "of the most significant decisions.\n\n"
                f"AGENTS_AUDITED: {base_outputs['agents_audited']}\n"
                f"COVERAGE_SCORE: {base_outputs['coverage_score']}\n"
                f"AUDIT_LOG_SAMPLE: {str(base_outputs['audit_log'][:3])}\n\n"
                "Return JSON with keys:\n"
                "- audit_log: list (copy from input)\n"
                "- agents_audited: int\n"
                "- total_entries: int\n"
                "- coverage_score: float 0..1\n"
                "- summary: string\n"
                "- rationale: 'llm'"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")),
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
                and isinstance(resp.get("audit_log"), list)
                and isinstance(resp.get("agents_audited"), int)
                and isinstance(resp.get("coverage_score"), (int, float))
            ):
                outputs = resp
                if "rationale" not in outputs:
                    outputs["rationale"] = "llm"
            else:
                outputs = base_outputs

        finished_at = datetime.now(timezone.utc)
        avg_confidence = (
            round(
                sum(e.get("confidence", 0.65) for e in outputs.get("audit_log", [])) /
                max(1, len(outputs.get("audit_log", []))),
                4,
            )
            if outputs.get("audit_log")
            else 0.65
        )

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=avg_confidence,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
