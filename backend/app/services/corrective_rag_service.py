"""Corrective RAG service (Feature 24.1).

Implements a lightweight CRAG pipeline for the gateway read path:
1) Estimate retrieval confidence from current candidates.
2) If confidence is low, fetch corrective candidates from an external source.
3) Re-rank combined candidates with a cross-encoder-like scoring heuristic.
4) Surface correction metadata, including a corrected_by provenance signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CorrectiveRagResult:
    memories: list[dict]
    retrieval_confidence: float
    corrected: bool
    corrected_by: str | None
    provenance_edges: list[dict]


def _token_overlap_ratio(query: str, content: str) -> float:
    q = set(str(query or "").lower().split())
    if not q:
        return 0.0
    c = set(str(content or "").lower().split())
    return len(q & c) / max(1, len(q))


def _score_memory(query: str, memory: dict) -> float:
    overlap = _token_overlap_ratio(query, str(memory.get("content") or ""))
    cred = memory.get("credibility_score")
    if cred is None:
        cred = memory.get("enrichment", {}).get("credibility_score")
    try:
        cred_f = float(cred) if cred is not None else 1.0
    except (TypeError, ValueError):
        cred_f = 1.0
    cred_f = max(0.1, min(1.0, cred_f))
    return round(0.7 * overlap + 0.3 * cred_f, 4)


def compute_retrieval_confidence(query: str, memories: list[dict]) -> float:
    """Estimate retrieval quality in [0, 1] from top candidates."""
    if not memories:
        return 0.0

    scored = sorted((_score_memory(query, m) for m in memories), reverse=True)
    top = scored[:3]
    return round(sum(top) / len(top), 4)


class CorrectiveRagService:
    """Apply corrective retrieval when initial context confidence is low."""

    def apply(
        self,
        *,
        query: str,
        memories: list[dict],
        limit: int,
        confidence_threshold: float = 0.45,
        external_connector_fn: Callable[[str, int], list[dict]] | None = None,
        cross_encoder_fn: Callable[[str, list[dict]], list[dict]] | None = None,
    ) -> CorrectiveRagResult:
        baseline_conf = compute_retrieval_confidence(query, memories)

        if baseline_conf >= confidence_threshold or external_connector_fn is None:
            return CorrectiveRagResult(
                memories=memories[:limit],
                retrieval_confidence=baseline_conf,
                corrected=False,
                corrected_by=None,
                provenance_edges=[],
            )

        external = list(external_connector_fn(query, limit) or [])
        if not external:
            return CorrectiveRagResult(
                memories=memories[:limit],
                retrieval_confidence=baseline_conf,
                corrected=False,
                corrected_by=None,
                provenance_edges=[],
            )

        merged = list(memories) + external
        if cross_encoder_fn is not None:
            ranked = list(cross_encoder_fn(query, merged) or [])
        else:
            ranked = sorted(merged, key=lambda m: _score_memory(query, m), reverse=True)

        final = ranked[:limit]
        for mem in final:
            if mem in external:
                mem.setdefault("provenance", {})
                mem["provenance"]["corrected_by"] = "external_connector"

        return CorrectiveRagResult(
            memories=final,
            retrieval_confidence=baseline_conf,
            corrected=True,
            corrected_by="external_connector",
            provenance_edges=[
                {
                    "edge": "corrected_by",
                    "source": "external_connector",
                    "reason": "low_retrieval_confidence",
                }
            ],
        )
