"""Hypothesis Service — Cognitive OS gap 3.

When AnomalyDetectionAgent fires, the previous behaviour was:
  anomaly detected → escalate → page someone

This service adds a reasoning step between detection and escalation:
  anomaly detected → form hypothesis (root cause + confidence) →
  query data to confirm/deny → update confidence →
  return evidence-backed decision instead of blind escalation

Internally delegates to:
- AnomalyDetectionAgent.run_heuristic() for anomaly signals
- CausalReasoningAgent helpers (extract_root_causes, trace_causal_chain,
  generate_counterfactual_hint) for causal structure
- CognitiveGatewayService.read() for evidence retrieval

Design rules:
- Pure service: no I/O, no DB session, operable in tests without mocks.
- Enrichment dict is the only required input; world_nodes and entity_edges
  are optional — without them only content-signal hypotheses are generated.
- Returns a typed HypothesisReport with ranked hypotheses, evidence, and
  a confirmed_hypothesis (the one with sufficient confirming evidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents.anomaly_detection_agent import run_heuristic as _anomaly_heuristic
from app.services.cognitive_gateway_service import CognitiveGatewayService
from app.services.enterprise_fallbacks import (
    build_adjacency,
    extract_root_causes,
    generate_counterfactual_hint,
    identify_effect_candidates,
    trace_causal_chain,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum anomaly score to warrant hypothesis formation
_MIN_ANOMALY_SCORE = 0.30

# Confidence boost per corroborating evidence item
_EVIDENCE_BOOST = 0.10

# Confidence penalty per contradicting evidence item
_CONTRADICTION_PENALTY = 0.08

# Threshold above which a hypothesis is considered confirmed
_CONFIRMATION_THRESHOLD = 0.70

# Max hypotheses to surface
_MAX_HYPOTHESES = 5

# Tokens that indicate a memory contradicts (negates) the hypothesis
_NEGATION_TOKENS = frozenset({
    "not", "no", "never", "resolved", "fixed", "closed", "false",
    "incorrect", "wrong", "disproved", "denied", "absent", "unrelated",
})


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """One candidate root cause with associated confidence."""
    entity: str
    entity_type: str
    domains: list[str]
    initial_confidence: float    # from causal chain strength
    confirmed_confidence: float  # after evidence + contradiction pass
    supporting_evidence: list[str]   # memory IDs that corroborate
    contradicting_evidence: list[str]  # memory IDs that negate
    causal_path: list[str]           # [effect, ..., root_cause]
    counterfactual_hint: str | None


@dataclass
class HypothesisReport:
    """Full output of a hypothesis generation and confirmation pass."""
    memory_id: str
    anomaly_detected: bool
    anomaly_score: float
    anomaly_type: str | None
    hypotheses: list[Hypothesis]          # ranked by confirmed_confidence
    confirmed_hypothesis: Hypothesis | None  # top hypothesis above threshold
    recommended_action: str               # "escalate_with_cause" | "investigate" | "monitor" | "no_action"
    counterfactual_hint: str | None
    evidence_count: int


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _content_to_effect_candidates(content: str, anomaly_type: str | None) -> list[dict[str, Any]]:
    """Derive effect candidates from plain content when world_nodes are absent.

    Splits content into noun-like tokens and treats each as a candidate effect
    entity so causal tracing has something to work with even without a graph.
    """
    if not content:
        return []
    # Weight anomaly_type token highest
    candidates: list[dict[str, Any]] = []
    if anomaly_type:
        candidates.append({
            "entity": anomaly_type,
            "entity_type": "concept",
            "domains": [],
            "effect_score": 0.9,
        })
    # Simple heuristic: capitalised words or known system terms as effect entities
    words = [w.strip(".,;:!?") for w in content.split() if len(w) > 4]
    for w in words[:10]:
        candidates.append({
            "entity": w.lower(),
            "entity_type": "concept",
            "domains": [],
            "effect_score": 0.3,
        })
    return candidates[:5]


def _build_hypotheses_from_graph(
    *,
    world_nodes: list[dict[str, Any]],
    entity_edges: list[dict[str, Any]],
    push_candidates: list[dict[str, Any]],
    content: str,
    anomaly_type: str | None,
) -> list[tuple[dict[str, Any], list[str]]]:
    """Return list of (root_cause_dict, causal_path) using graph structure."""
    if world_nodes and entity_edges:
        effect_candidates = identify_effect_candidates(
            world_nodes=world_nodes,
            push_candidates=push_candidates,
        )
        adj = build_adjacency(entity_edges)
        node_lookup = {n["entity"]: n for n in world_nodes}
    else:
        effect_candidates = _content_to_effect_candidates(content, anomaly_type)
        adj = {}
        node_lookup = {}

    if not effect_candidates:
        return []

    top_effect = effect_candidates[0]["entity"]
    chains = trace_causal_chain(
        effect_entity=top_effect,
        adjacency=adj,
        node_lookup=node_lookup,
        propagated_signal_entities=set(),
        max_hops=3,
    )

    root_causes = extract_root_causes(chains, node_lookup)

    results: list[tuple[dict[str, Any], list[str]]] = []
    chain_by_root: dict[str, list[str]] = {}
    for chain in chains:
        path = chain.get("causal_path", [])
        if path:
            root = path[-1]
            chain_by_root[root] = path

    for rc in root_causes[:_MAX_HYPOTHESES]:
        path = chain_by_root.get(rc["entity"], [rc["entity"]])
        results.append((rc, path))

    return results


def _credibility_weight(mem: dict[str, Any]) -> float:
    """Extract credibility weight from memory dict (0.1–1.0), default 1.0."""
    raw = mem.get("credibility_score")
    if raw is None:
        raw = mem.get("enrichment", {}).get("credibility_score")
    try:
        val = float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
        val = 1.0
    return max(0.1, min(1.0, val))


def _find_corroborating_evidence(
    *,
    hypothesis_entity: str,
    domains: list[str],
    candidate_memories: list[dict[str, Any]],
    gateway: CognitiveGatewayService,
) -> list[str]:
    """Return memory IDs that mention the hypothesis entity or its domains.

    Only memories with credibility_weight * token_overlap >= 0.1 qualify,
    so low-credibility sources contribute less to hypothesis confirmation.
    """
    query_terms = [hypothesis_entity] + (domains[:2] if domains else [])
    query = " ".join(query_terms)
    if not query.strip():
        return []
    read_result = gateway.read(
        query=query,
        memories=candidate_memories,
        limit=5,
    )
    evidence: list[str] = []
    query_tokens = set(query.lower().split())
    for mem in read_result.memories:
        content = str(mem.get("content") or "").lower()
        overlap = len(query_tokens & set(content.split()))
        weighted = overlap * _credibility_weight(mem)
        if weighted >= 0.1:
            mem_id = str(mem.get("id") or mem.get("memory_id") or "")
            if mem_id:
                evidence.append(mem_id)
    return evidence[:3]


def _find_contradicting_evidence(
    *,
    hypothesis_entity: str,
    candidate_memories: list[dict[str, Any]],
) -> list[str]:
    """Return memory IDs that mention the entity but negate it.

    A memory contradicts a hypothesis when its content contains the entity
    and at least one strong negation token in close proximity (within 5 words).
    Only memories with credibility_weight >= 0.3 are trusted as contradictions
    — an unreliable source claiming something is resolved should not suppress
    a well-supported hypothesis.
    """
    entity_lower = hypothesis_entity.lower()
    contradicting: list[str] = []
    for mem in candidate_memories:
        if _credibility_weight(mem) < 0.3:
            continue
        content = str(mem.get("content") or "").lower()
        if entity_lower not in content:
            continue
        words = content.split()
        entity_positions = [i for i, w in enumerate(words) if entity_lower in w]
        for pos in entity_positions:
            window = set(words[max(0, pos - 5): pos + 6])
            if window & _NEGATION_TOKENS:
                mem_id = str(mem.get("id") or mem.get("memory_id") or "")
                if mem_id:
                    contradicting.append(mem_id)
                break
    return contradicting[:3]


def _determine_action(
    confirmed: Hypothesis | None,
    anomaly_score: float,
) -> str:
    if confirmed is not None and confirmed.confirmed_confidence >= _CONFIRMATION_THRESHOLD:
        return "escalate_with_cause"
    if anomaly_score >= 0.7:
        return "investigate"
    if anomaly_score >= _MIN_ANOMALY_SCORE:
        return "monitor"
    return "no_action"


# EMA alpha for edge weight reinforcement / suppression
_EDGE_REINFORCE_ALPHA = 0.25   # confirmed path: blend toward 1.0
_EDGE_SUPPRESS_ALPHA = 0.25    # contradicted path: blend toward 0.0


def reinforce_causal_edges(
    *,
    entity_edges: list[dict[str, Any]],
    causal_path: list[str],
    confirmed: bool,
) -> list[dict[str, Any]]:
    """Update edge weights along a causal path using EMA reinforcement.

    For each consecutive pair in causal_path, find the matching edge in
    entity_edges and adjust its weight:
      - confirmed=True  → weight = (1-α)*weight + α*1.0  (strengthen)
      - confirmed=False → weight = (1-α)*weight + α*0.0  (weaken)

    Edges are matched by (from_entity, to_entity) or (to_entity, from_entity)
    since build_adjacency treats the graph as undirected.

    Returns the updated entity_edges list (mutated in place; also returned for
    convenience so callers can chain the call).  Edges not found in the list
    are ignored — the caller is responsible for persisting the updated edges.
    """
    if len(causal_path) < 2:
        return entity_edges

    target = 1.0 if confirmed else 0.0
    alpha = _EDGE_REINFORCE_ALPHA if confirmed else _EDGE_SUPPRESS_ALPHA

    # Build an index for fast lookup: (a, b) → edge dict
    edge_index: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in entity_edges:
        a = str(edge.get("from_entity") or "")
        b = str(edge.get("to_entity") or "")
        if a and b:
            edge_index[(a, b)] = edge
            edge_index[(b, a)] = edge  # undirected

    for i in range(len(causal_path) - 1):
        src = causal_path[i]
        dst = causal_path[i + 1]
        edge = edge_index.get((src, dst))
        if edge is not None:
            old_w = float(edge.get("weight", 0.5))
            new_w = round((1.0 - alpha) * old_w + alpha * target, 4)
            edge["weight"] = max(0.05, min(1.0, new_w))

    return entity_edges


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class HypothesisService:
    """Causal hypothesis engine for anomalous memories.

    Typical usage:
        svc = HypothesisService()
        report = svc.analyse(
            memory_id="mem-abc",
            content="DB response time exceeded 5s for the third time this hour",
            enrichment=memory.enrichment,
            candidate_memories=recent_memories,
        )
        if report.confirmed_hypothesis:
            # escalate with root cause instead of blind page
            notify(root_cause=report.confirmed_hypothesis.entity,
                   hint=report.counterfactual_hint)
    """

    def __init__(self, *, gateway: CognitiveGatewayService | None = None) -> None:
        self._gateway = gateway or CognitiveGatewayService()

    def analyse(
        self,
        *,
        memory_id: str,
        content: str,
        enrichment: dict[str, Any],
        candidate_memories: list[dict[str, Any]] | None = None,
        world_nodes: list[dict[str, Any]] | None = None,
        entity_edges: list[dict[str, Any]] | None = None,
        push_candidates: list[dict[str, Any]] | None = None,
    ) -> HypothesisReport:
        """Form and confirm hypotheses for an anomalous memory.

        Parameters
        ----------
        memory_id:           ID of the memory under analysis
        content:             Raw text content of the memory
        enrichment:          Full enrichment dict from the agent pipeline
        candidate_memories:  Pool of recent memories to search for corroborating evidence
        world_nodes:         World model graph nodes (Phase 4) — optional
        entity_edges:        World model graph edges (Phase 4) — optional
        push_candidates:     Push candidates from ProactiveMemoryPushAgent — optional
        """
        # Run anomaly detection
        anomaly_signals = _anomaly_heuristic(enrichment)
        anomaly_detected = bool(anomaly_signals.get("anomaly_detected"))
        anomaly_score = float(anomaly_signals.get("anomaly_score") or 0.0)
        anomaly_type = anomaly_signals.get("anomaly_type")

        # Short-circuit if score below threshold
        if not anomaly_detected or anomaly_score < _MIN_ANOMALY_SCORE:
            return HypothesisReport(
                memory_id=memory_id,
                anomaly_detected=anomaly_detected,
                anomaly_score=anomaly_score,
                anomaly_type=anomaly_type,
                hypotheses=[],
                confirmed_hypothesis=None,
                recommended_action="no_action",
                counterfactual_hint=None,
                evidence_count=0,
            )

        # Build candidate hypotheses from causal graph (or content fallback)
        raw_hypotheses = _build_hypotheses_from_graph(
            world_nodes=world_nodes or [],
            entity_edges=entity_edges or [],
            push_candidates=push_candidates or [],
            content=content,
            anomaly_type=anomaly_type,
        )

        _mems = list(candidate_memories or [])
        hypotheses: list[Hypothesis] = []

        for rc, path in raw_hypotheses:
            entity = str(rc.get("entity") or "")
            if not entity:
                continue
            initial_conf = float(rc.get("cause_score") or 0.4)
            domains = list(rc.get("domains") or [])

            # Evidence pass — boost for corroboration, penalise for contradictions
            evidence_ids = _find_corroborating_evidence(
                hypothesis_entity=entity,
                domains=domains,
                candidate_memories=_mems,
                gateway=self._gateway,
            )
            contra_ids = _find_contradicting_evidence(
                hypothesis_entity=entity,
                candidate_memories=_mems,
            )

            confirmed_conf = round(
                max(0.0, min(0.95,
                    initial_conf
                    + _EVIDENCE_BOOST * len(evidence_ids)
                    - _CONTRADICTION_PENALTY * len(contra_ids)
                )), 3
            )

            # Counterfactual for top effect
            top_effect = path[0] if path else None
            c_hint = generate_counterfactual_hint([rc], top_effect)

            hypotheses.append(Hypothesis(
                entity=entity,
                entity_type=str(rc.get("entity_type") or "concept"),
                domains=domains,
                initial_confidence=round(initial_conf, 3),
                confirmed_confidence=confirmed_conf,
                supporting_evidence=evidence_ids,
                contradicting_evidence=contra_ids,
                causal_path=path,
                counterfactual_hint=c_hint,
            ))

        # Sort by confirmed confidence
        hypotheses.sort(key=lambda h: -h.confirmed_confidence)

        confirmed = next(
            (h for h in hypotheses if h.confirmed_confidence >= _CONFIRMATION_THRESHOLD),
            None,
        )

        top_hint = hypotheses[0].counterfactual_hint if hypotheses else None
        action = _determine_action(confirmed=confirmed, anomaly_score=anomaly_score)

        # Auto-reinforce causal edges when entity_edges were supplied
        _edges = list(entity_edges or [])
        if _edges:
            for h in hypotheses:
                is_confirmed = h.confirmed_confidence >= _CONFIRMATION_THRESHOLD
                has_contradictions = bool(h.contradicting_evidence)
                # Reinforce confirmed paths; suppress paths with net contradictions
                if is_confirmed:
                    reinforce_causal_edges(
                        entity_edges=_edges,
                        causal_path=h.causal_path,
                        confirmed=True,
                    )
                elif has_contradictions and not h.supporting_evidence:
                    reinforce_causal_edges(
                        entity_edges=_edges,
                        causal_path=h.causal_path,
                        confirmed=False,
                    )

        return HypothesisReport(
            memory_id=memory_id,
            anomaly_detected=anomaly_detected,
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            hypotheses=hypotheses,
            confirmed_hypothesis=confirmed,
            recommended_action=action,
            counterfactual_hint=top_hint,
            evidence_count=sum(len(h.supporting_evidence) for h in hypotheses),
        )

    def reinforce_from_report(
        self,
        *,
        report: HypothesisReport,
        entity_edges: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply causal reinforcement from a completed report to entity_edges.

        Useful when the caller persists entity_edges separately and wants to
        update them after analyse() has run.  Returns the mutated edge list.
        """
        for h in report.hypotheses:
            is_confirmed = h.confirmed_confidence >= _CONFIRMATION_THRESHOLD
            has_contradictions = bool(h.contradicting_evidence)
            if is_confirmed:
                reinforce_causal_edges(
                    entity_edges=entity_edges,
                    causal_path=h.causal_path,
                    confirmed=True,
                )
            elif has_contradictions and not h.supporting_evidence:
                reinforce_causal_edges(
                    entity_edges=entity_edges,
                    causal_path=h.causal_path,
                    confirmed=False,
                )
        return entity_edges

    def is_anomalous(self, enrichment: dict[str, Any]) -> bool:
        """Quick check: does this enrichment have an actionable anomaly?"""
        signals = _anomaly_heuristic(enrichment)
        return (
            bool(signals.get("anomaly_detected"))
            and float(signals.get("anomaly_score") or 0.0) >= _MIN_ANOMALY_SCORE
        )
