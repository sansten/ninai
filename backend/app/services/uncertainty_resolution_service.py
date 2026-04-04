"""Uncertainty Resolution Service — Cognitive OS gap 2.

When UncertaintyReportingAgent flags a memory with uncertainty_level='high'
or 'critical', nothing used to act on that signal.  This service closes
the loop:

  1. Read the uncertainty signals from the enrichment dict.
  2. Build a set of structured resolution queries targeted at the
     specific uncertainty sources (unresolved conflicts, low-confidence
     fields, stale data, credibility issues).
  3. Search existing memories for corroborating or contradicting evidence.
  4. Return a resolution report: which fields remain uncertain, what
     evidence was found, and a recommended action.

Design rules:
- Pure service: no DB session required (operates on dicts, callable from
  any layer including tests).
- Delegates to CognitiveGatewayService.read() for memory retrieval so it
  honours the vector_fn hook in production.
- Does NOT modify enrichment in place; callers decide what to persist.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any

from app.agents.uncertainty_reporting_agent import (
    run_heuristic as _uncertainty_heuristic,
    collect_low_confidence_fields,
    collect_unresolved_conflicts,
)
from app.services.cognitive_gateway_service import CognitiveGatewayService


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# uncertainty_level values that trigger active resolution
_ACTIVE_RESOLUTION_LEVELS = frozenset({"high", "critical"})

# Maximum memories to scan per resolution query
_MAX_MEMORIES_PER_QUERY = 10

# Minimum token-overlap score to count a memory as corroborating evidence
_MIN_RELEVANCE_OVERLAP = 1


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ResolutionQuery:
    """One structured question generated to resolve uncertainty."""
    field: str              # which field / conflict is being queried
    query_text: str         # the actual search string
    uncertainty_source: str  # "low_confidence" | "unresolved_conflict" | "stale" | "credibility"


@dataclass
class ResolutionEvidence:
    """A memory record that provides evidence for resolving a query."""
    memory_id: str
    content_preview: str
    relevance: float        # 0..1 proxy (token overlap / max_overlap)


@dataclass
class UncertaintyResolutionReport:
    """Full output of a resolution pass."""
    memory_id: str
    uncertainty_level: str
    uncertainty_score: float
    queries_generated: list[ResolutionQuery]
    evidence_found: list[ResolutionEvidence]
    still_unresolved: list[str]     # fields that no evidence addressed
    recommended_action: str          # "resolved" | "escalate" | "human_review" | "monitor"
    resolution_confidence: float


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _build_queries(
    enrichment: dict[str, Any],
    uncertainty_signals: dict[str, Any],
) -> list[ResolutionQuery]:
    """Build resolution queries from uncertainty signals.

    Each low-confidence field → one query.
    Each unresolved conflict → one query.
    Stale data and low credibility → generic corroboration queries.
    """
    queries: list[ResolutionQuery] = []

    # Low-confidence fields → ask the gateway to find recent information
    for fld in uncertainty_signals.get("low_confidence_fields") or []:
        label = fld.replace("_", " ")
        queries.append(ResolutionQuery(
            field=fld,
            query_text=f"current value status {label} update",
            uncertainty_source="low_confidence",
        ))

    # Unresolved conflicts → search for resolution or authoritative source
    for conflict in uncertainty_signals.get("unresolved_conflicts") or []:
        topic = str(conflict.get("topic") or conflict.get("field") or "conflict")
        queries.append(ResolutionQuery(
            field=topic,
            query_text=f"resolution decision authoritative {topic}",
            uncertainty_source="unresolved_conflict",
        ))

    # Stale data → look for fresher records
    if bool(enrichment.get("is_stale")):
        queries.append(ResolutionQuery(
            field="freshness",
            query_text="recent update current state latest",
            uncertainty_source="stale",
        ))

    # Low credibility → look for corroborating sources
    cred = enrichment.get("credibility_score")
    try:
        cred_val = float(cred) if cred is not None else 1.0
    except (TypeError, ValueError):
        cred_val = 1.0
    if cred_val < 0.40:
        queries.append(ResolutionQuery(
            field="credibility",
            query_text="verified confirmed authoritative source",
            uncertainty_source="credibility",
        ))

    return queries[:8]  # cap at 8 queries per memory


def _credibility_weight(mem: dict) -> float:
    """Extract credibility weight from memory dict (0.1–1.0), default 1.0."""
    raw = mem.get("credibility_score")
    if raw is None:
        raw = mem.get("enrichment", {}).get("credibility_score")
    try:
        val = float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
        val = 1.0
    return max(0.1, min(1.0, val))


def _score_evidence(memories: list[dict], query_text: str) -> list[ResolutionEvidence]:
    """Score candidate memories by credibility-weighted token overlap.

    raw_score = token_overlap * credibility_weight
    Normalised relevance is relative to the highest raw score seen.
    """
    query_tokens = set(query_text.lower().split())
    scored: list[tuple[float, dict]] = []
    for mem in memories:
        content = str(mem.get("content") or "").lower()
        overlap = len(query_tokens & set(content.split()))
        if overlap >= _MIN_RELEVANCE_OVERLAP:
            weighted = overlap * _credibility_weight(mem)
            scored.append((weighted, mem))

    if not scored:
        return []

    max_score = max(s for s, _ in scored)
    results: list[ResolutionEvidence] = []
    for score, mem in sorted(scored, key=lambda x: -x[0]):
        mem_id = str(mem.get("id") or mem.get("memory_id") or "")
        content = str(mem.get("content") or "")
        results.append(ResolutionEvidence(
            memory_id=mem_id,
            content_preview=content[:200],
            relevance=round(score / max_score, 3) if max_score else 0.0,
        ))
    return results[:3]  # top 3 per query


def _determine_action(
    *,
    uncertainty_level: str,
    n_queries: int,
    n_evidence: int,
    still_unresolved_count: int,
) -> tuple[str, float]:
    """Return (recommended_action, resolution_confidence)."""
    if still_unresolved_count == 0 and n_evidence > 0:
        return "resolved", round(min(0.90, 0.60 + 0.10 * n_evidence), 3)
    if uncertainty_level == "critical" and still_unresolved_count > 0:
        return "escalate", 0.30
    if still_unresolved_count > 0 and uncertainty_level == "high":
        return "human_review", 0.45
    return "monitor", 0.60


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class UncertaintyResolutionService:
    """Proactive uncertainty resolution for flagged memories.

    Typical usage:
        svc = UncertaintyResolutionService(gateway=CognitiveGatewayService())
        report = svc.resolve(
            memory_id="mem-123",
            enrichment=memory.enrichment,
            candidate_memories=recent_memories,
        )
        if report.recommended_action == "escalate":
            # hand off to human review queue
    """

    def __init__(
        self,
        *,
        gateway: CognitiveGatewayService | None = None,
    ) -> None:
        self._gateway = gateway or CognitiveGatewayService()

    @staticmethod
    def _await_if_needed(value: Any) -> Any:
        """Resolve an awaitable result when gateway methods are async.

        This service remains sync-callable for task and test code paths,
        while still supporting async CognitiveGatewayService methods.
        """
        if not inspect.isawaitable(value):
            return value

        try:
            return asyncio.run(value)
        except RuntimeError:
            # If already inside a running loop, run the coroutine in a short-lived
            # loop on a helper thread so this sync service still returns a value.
            import threading

            container: dict[str, Any] = {}

            def _runner() -> None:
                container["value"] = asyncio.run(value)

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join()
            return container.get("value")

    def resolve(
        self,
        *,
        memory_id: str,
        enrichment: dict[str, Any],
        candidate_memories: list[dict] | None = None,
    ) -> UncertaintyResolutionReport:
        """Run a full uncertainty resolution pass for one memory.

        Parameters
        ----------
        memory_id:           ID of the memory being resolved
        enrichment:          The memory's enrichment dict (output of the agent pipeline)
        candidate_memories:  Pre-fetched pool of memories to search for evidence.
                             If None, only structural queries are generated (no evidence).
        """
        # Run uncertainty agent to get structured signals
        signals = _uncertainty_heuristic(enrichment)
        level = str(signals.get("uncertainty_level") or "low")
        score = float(signals.get("uncertainty_score") or 0.0)

        queries = _build_queries(enrichment, signals)
        all_evidence: list[ResolutionEvidence] = []
        addressed_fields: set[str] = set()

        if candidate_memories and queries:
            for q in queries:
                # Use gateway read (honours vector_fn in production)
                read_result = self._gateway.read(
                    query=q.query_text,
                    memories=candidate_memories,
                    limit=_MAX_MEMORIES_PER_QUERY,
                )
                read_result = self._await_if_needed(read_result)
                evidence = _score_evidence(read_result.memories, q.query_text)
                if evidence:
                    all_evidence.extend(evidence)
                    addressed_fields.add(q.field)

        # Deduplicate evidence by memory_id
        seen: set[str] = set()
        unique_evidence: list[ResolutionEvidence] = []
        for ev in all_evidence:
            if ev.memory_id and ev.memory_id not in seen:
                seen.add(ev.memory_id)
                unique_evidence.append(ev)

        query_fields = {q.field for q in queries}
        still_unresolved = sorted(query_fields - addressed_fields)

        action, confidence = _determine_action(
            uncertainty_level=level,
            n_queries=len(queries),
            n_evidence=len(unique_evidence),
            still_unresolved_count=len(still_unresolved),
        )

        return UncertaintyResolutionReport(
            memory_id=memory_id,
            uncertainty_level=level,
            uncertainty_score=round(score, 4),
            queries_generated=queries,
            evidence_found=unique_evidence,
            still_unresolved=still_unresolved,
            recommended_action=action,
            resolution_confidence=confidence,
        )

    def needs_resolution(self, enrichment: dict[str, Any]) -> bool:
        """Quick check: does this enrichment warrant active resolution?"""
        signals = _uncertainty_heuristic(enrichment)
        return str(signals.get("uncertainty_level") or "low") in _ACTIVE_RESOLUTION_LEVELS
