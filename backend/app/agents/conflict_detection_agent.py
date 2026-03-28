"""Conflict Detection Agent — Phase 13.

Detects contradictory states reported by different enterprise silos for the
same entity. Surfaces data inconsistencies, coverage gaps, and circular causal
dependencies so that humans know which entities need reconciliation.

Depends on Phase 12 (CausalReasoningAgent).

Inputs (via context["memory"]["enrichment"]):
  - world_nodes:        list[{entity, entity_type, domains, silo_count,
                              state_version, node_confidence}]       — Phase 4
  - propagated_signals: list[{content, source_silo, target_domains,
                              relevance_score, entities, ...}]       — Phase 9
  - causal_chains:      list[{effect_entity, causal_path, hop_count,
                              chain_strength, silo_span, narrative}] — Phase 12

Outputs:
  - conflicts:               list[{entity, entity_type, conflict_type, silos,
                                   descriptions, severity, resolution_hint}]
  - conflict_count:          int
  - high_severity_conflicts: list[{entity, entity_type, conflict_type, silos, severity}]
  - resolution_hints:        list[str]
  - confidence:              float
  - rationale:               "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Entity-type weights (same hierarchy as Phase 12 cause weights).
# Higher-weight entities are more likely to produce high-severity conflicts.
# ---------------------------------------------------------------------------

_ENTITY_TYPE_WEIGHT: dict[str, float] = {
    "person": 1.0,
    "org": 0.95,
    "organization": 0.95,
    "customer": 0.90,
    "project": 0.80,
    "system": 0.75,
    "deployment": 0.70,
    "sprint": 0.65,
    "event": 0.60,
    "incident": 0.55,
    "document": 0.50,
    "concept": 0.40,
    "metric": 0.30,
}

_DEFAULT_TYPE_WEIGHT = 0.45

_POSITIVE_KEYWORDS: frozenset[str] = frozenset([
    "resolved", "completed", "closed", "success", "done",
    "fixed", "approved", "healthy", "on track", "stable",
    "green", "confirmed", "delivered", "deployed",
])

_NEGATIVE_KEYWORDS: frozenset[str] = frozenset([
    "failed", "blocked", "critical", "at risk", "error",
    "issue", "escalated", "broken", "overdue", "red",
    "delayed", "incident", "outage", "degraded", "rejected",
    "missed", "open", "unresolved",
])

_SEVERITY_HIGH = "high"
_SEVERITY_MEDIUM = "medium"
_SEVERITY_LOW = "low"

_SEVERITY_ORDER = {_SEVERITY_HIGH: 0, _SEVERITY_MEDIUM: 1, _SEVERITY_LOW: 2}

_MIN_ENTITY_NAME_LEN = 4  # skip very short names in content matching


def _entity_type_weight(entity_type: str) -> float:
    return _ENTITY_TYPE_WEIGHT.get((entity_type or "").lower(), _DEFAULT_TYPE_WEIGHT)


def _conflict_severity(entity_type: str, conflicting_silo_count: int) -> str:
    """Compute conflict severity from entity type weight and silo spread."""
    tw = _entity_type_weight(entity_type)
    if tw >= 0.75 and conflicting_silo_count >= 2:
        return _SEVERITY_HIGH
    if tw >= 0.45 or conflicting_silo_count >= 2:
        return _SEVERITY_MEDIUM
    return _SEVERITY_LOW


def classify_signal_sentiment(text: str) -> str:
    """Classify text as 'positive', 'negative', or 'neutral' by keyword count."""
    lower = (text or "").lower()
    pos = sum(1 for kw in _POSITIVE_KEYWORDS if kw in lower)
    neg = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in lower)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def detect_state_mismatches(
    *,
    propagated_signals: list[dict[str, Any]],
    node_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect entities with contradictory sentiments across silos.

    An entity is conflicted when at least one signal describes it positively
    and another describes it negatively, and those signals come from different
    source silos.

    Returns list of conflict dicts sorted by entity name.
    """
    # entity → {source_silo: (sentiment, snippet)}
    entity_silo_map: dict[str, dict[str, tuple[str, str]]] = {}

    for sig in propagated_signals:
        source_silo = sig.get("source_silo", "")
        content = sig.get("content", "")
        sentiment = classify_signal_sentiment(content)
        if sentiment == "neutral":
            continue
        snippet = content[:200]

        # Entities explicitly listed in the signal
        for entity in sig.get("entities", []):
            entry = entity_silo_map.setdefault(entity, {})
            if source_silo not in entry:
                entry[source_silo] = (sentiment, snippet)

        # Best-effort: match world node names found in signal content
        content_lower = content.lower()
        for entity, node in node_lookup.items():
            if len(entity) < _MIN_ENTITY_NAME_LEN:
                continue
            if entity.lower() in content_lower:
                entry = entity_silo_map.setdefault(entity, {})
                if source_silo not in entry:
                    entry[source_silo] = (sentiment, snippet)

    conflicts: list[dict[str, Any]] = []
    for entity, silo_data in entity_silo_map.items():
        if len(silo_data) < 2:
            continue
        sentiments = {v[0] for v in silo_data.values()}
        if "positive" not in sentiments or "negative" not in sentiments:
            continue

        silos = list(silo_data.keys())
        descriptions = [silo_data[s][1] for s in silos]
        node = node_lookup.get(entity, {})
        entity_type = node.get("entity_type", "concept")
        severity = _conflict_severity(entity_type, len(silos))

        conflicts.append(
            {
                "entity": entity,
                "entity_type": entity_type,
                "conflict_type": "state_mismatch",
                "silos": silos,
                "descriptions": descriptions,
                "severity": severity,
            }
        )

    conflicts.sort(key=lambda x: x["entity"])
    return conflicts


def detect_silo_gaps(
    *,
    world_nodes: list[dict[str, Any]],
    causal_chains: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect entities referenced in causal chains but absent from the world model.

    A silo gap means some silo is referencing an entity that was never
    reconciled into the unified world model, which indicates a data coverage
    problem.
    """
    node_entities = {n["entity"] for n in world_nodes}

    chain_entities: set[str] = set()
    for chain in causal_chains:
        for entity in chain.get("causal_path", []):
            chain_entities.add(entity)

    gap_entities = chain_entities - node_entities

    conflicts: list[dict[str, Any]] = []
    for entity in sorted(gap_entities):
        conflicts.append(
            {
                "entity": entity,
                "entity_type": "unknown",
                "conflict_type": "silo_gap",
                "silos": [],
                "descriptions": [
                    f"{entity} appears in causal chains but has no world model entry"
                ],
                "severity": _SEVERITY_LOW,
            }
        )
    return conflicts


def detect_causal_loops(
    *,
    causal_chains: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect circular causality: entity A causes entity B and B causes A.

    Checks pairwise: for each effect→root pair in causal_chains, see if the
    root is also an effect whose root is the original effect.
    """
    # effect_entity → set of root cause entities (last element of causal_path)
    effect_to_roots: dict[str, set[str]] = {}
    for chain in causal_chains:
        effect = chain.get("effect_entity", "")
        path = chain.get("causal_path", [])
        if len(path) >= 2:
            root = path[-1]
            effect_to_roots.setdefault(effect, set()).add(root)

    conflicts: list[dict[str, Any]] = []
    seen_pairs: set[frozenset[str]] = set()

    for effect, roots in effect_to_roots.items():
        for root in roots:
            if root in effect_to_roots and effect in effect_to_roots[root]:
                pair: frozenset[str] = frozenset([effect, root])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                conflicts.append(
                    {
                        "entity": effect,
                        "entity_type": "concept",
                        "conflict_type": "causal_loop",
                        "silos": [],
                        "descriptions": [f"Circular causality detected: {effect} ↔ {root}"],
                        "severity": _SEVERITY_MEDIUM,
                    }
                )

    return conflicts


def build_resolution_hint(*, conflict: dict[str, Any]) -> str:
    """Generate an actionable resolution suggestion for a conflict."""
    entity = conflict.get("entity", "")
    ctype = conflict.get("conflict_type", "")
    silos = conflict.get("silos", [])
    silo_str = f" (silos: {', '.join(silos)})" if silos else ""

    if ctype == "state_mismatch":
        return (
            f"Reconcile {entity}{silo_str}: verify current status in each silo "
            f"and establish a single source of truth"
        )
    if ctype == "silo_gap":
        return (
            f"Register {entity} in the world model to enable unified cross-silo tracking"
        )
    if ctype == "causal_loop":
        return (
            f"Review causal model for {entity}: circular dependency may indicate "
            f"a missing intermediate entity or incorrect causal direction"
        )
    return f"Investigate {entity}{silo_str} for data consistency"


def merge_and_rank_conflicts(
    conflicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add resolution_hint and sort by severity (high → medium → low), then entity name."""
    for c in conflicts:
        c["resolution_hint"] = build_resolution_hint(conflict=c)

    conflicts.sort(
        key=lambda x: (_SEVERITY_ORDER.get(x["severity"], 3), x["entity"])
    )
    return conflicts


class ConflictDetectionAgent(BaseAgent):
    """Phase 13: Detect cross-silo state conflicts, data gaps, and causal loops.

    Inputs (via context["memory"]["enrichment"]):
      - world_nodes:        list[{entity, entity_type, domains, silo_count,
                                  state_version, node_confidence}]       — Phase 4
      - propagated_signals: list[{content, source_silo, target_domains,
                                  relevance_score, entities, ...}]       — Phase 9
      - causal_chains:      list[{effect_entity, causal_path, hop_count,
                                  chain_strength, silo_span, narrative}] — Phase 12

    Outputs:
      - conflicts:               list[{entity, entity_type, conflict_type, silos,
                                       descriptions, severity, resolution_hint}]
      - conflict_count:          int
      - high_severity_conflicts: list[{entity, entity_type, conflict_type, silos, severity}]
      - resolution_hints:        list[str]
      - confidence:              float
      - rationale:               "heuristic" | "llm"
    """

    name = "ConflictDetectionAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["CausalReasoningAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("conflicts", []), list):
            raise ValueError("conflicts must be a list")
        if not isinstance(outputs.get("resolution_hints", []), list):
            raise ValueError("resolution_hints must be a list")
        for conflict in outputs.get("conflicts", []):
            if not isinstance(conflict, dict):
                raise ValueError("each conflicts item must be a dict")
            if "entity" not in conflict:
                raise ValueError("each conflicts item must have entity")
            if "conflict_type" not in conflict:
                raise ValueError("each conflicts item must have conflict_type")
            if conflict.get("severity") not in {
                _SEVERITY_HIGH, _SEVERITY_MEDIUM, _SEVERITY_LOW
            }:
                raise ValueError(
                    f"conflict severity must be high/medium/low, got: {conflict.get('severity')}"
                )

    def _heuristic(
        self,
        *,
        world_nodes: list[dict[str, Any]],
        propagated_signals: list[dict[str, Any]],
        causal_chains: list[dict[str, Any]],
    ) -> dict[str, Any]:
        node_lookup: dict[str, dict[str, Any]] = {n["entity"]: n for n in world_nodes}

        state_conflicts = detect_state_mismatches(
            propagated_signals=propagated_signals,
            node_lookup=node_lookup,
        )
        gap_conflicts = detect_silo_gaps(
            world_nodes=world_nodes,
            causal_chains=causal_chains,
        )
        loop_conflicts = detect_causal_loops(causal_chains=causal_chains)

        all_conflicts = merge_and_rank_conflicts(
            state_conflicts + gap_conflicts + loop_conflicts
        )

        high_severity = [
            {
                "entity": c["entity"],
                "entity_type": c["entity_type"],
                "conflict_type": c["conflict_type"],
                "silos": c["silos"],
                "severity": c["severity"],
            }
            for c in all_conflicts
            if c["severity"] == _SEVERITY_HIGH
        ]

        resolution_hints = [c["resolution_hint"] for c in all_conflicts]

        # Confidence scales with signal richness — more signals → more reliable detection
        n_signals = len(propagated_signals)
        confidence = min(0.85, round(0.35 + 0.05 * n_signals, 3))

        return {
            "conflicts": all_conflicts,
            "conflict_count": len(all_conflicts),
            "high_severity_conflicts": high_severity,
            "resolution_hints": resolution_hints,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        content = (context.get("memory") or {}).get("content", "")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        world_nodes: list[dict[str, Any]] = enrichment.get("world_nodes", [])
        propagated_signals: list[dict[str, Any]] = enrichment.get("propagated_signals", [])
        causal_chains: list[dict[str, Any]] = enrichment.get("causal_chains", [])

        strategy = getattr(settings, "CONFLICT_DETECTION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                world_nodes=world_nodes,
                propagated_signals=propagated_signals,
                causal_chains=causal_chains,
            )
        else:
            prompt = (
                "You are an enterprise conflict detection engine. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Given a world model, cross-silo propagated signals, and causal chains, "
                "identify conflicts where different silos report contradictory states "
                "for the same entity, entities missing from the world model, "
                "and circular causal dependencies.\n\n"
                f"WORLD NODES: {world_nodes[:20]}\n"
                f"PROPAGATED SIGNALS: {propagated_signals[:15]}\n"
                f"CAUSAL CHAINS: {causal_chains[:10]}\n\n"
                "Return JSON with keys:\n"
                "- conflicts: list of {entity, entity_type, conflict_type "
                "(state_mismatch|silo_gap|causal_loop), silos (list), "
                "descriptions (list), severity (high|medium|low), resolution_hint (str)}\n"
                "- conflict_count: int\n"
                "- high_severity_conflicts: list of {entity, entity_type, "
                "conflict_type, silos, severity}\n"
                "- resolution_hints: list of str\n"
                "- confidence: float 0..1\n"
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
                and isinstance(resp.get("conflicts"), list)
                and isinstance(resp.get("resolution_hints"), list)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    world_nodes=world_nodes,
                    propagated_signals=propagated_signals,
                    causal_chains=causal_chains,
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
