"""CurriculumSequencerService — Phase 89.

Answers the question: "What should Ninai learn next to maximally reduce its
overall epistemic uncertainty?"

Rather than asking questions randomly (ActiveKnowledgeSeekerAgent) or
waiting for users to surface gaps, the curriculum sequencer:

1. Maps current knowledge gaps from UncertaintyPropagation + CredibilityAgent
2. Builds a concept dependency graph (what blocks what)
3. Prioritizes gaps whose resolution would unblock the most downstream reasoning
4. Emits an ordered KnowledgeAcquisitionPlan

This is inspired by curriculum learning in ML: learn prerequisites before
advanced topics; learn what is most load-bearing first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeGap:
    concept: str
    gap_type: str          # "missing" | "low_confidence" | "conflicted" | "stale"
    confidence: float      # current confidence in this concept, [0, 1]
    importance: float      # how often this concept appears in reasoning chains
    blocking_count: int    # number of downstream concepts blocked by this gap
    suggested_query: str   # query to issue to fill this gap


@dataclass
class KnowledgeAcquisitionPlan:
    gaps: list[KnowledgeGap]   # ordered by priority (highest first)
    total_gaps: int
    estimated_uncertainty_reduction: float  # rough aggregate if top-N gaps filled


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class CurriculumSequencerService:
    """Stateless service — reads uncertainty signals and produces a learning plan."""

    # ------------------------------------------------------------------
    # Gap identification
    # ------------------------------------------------------------------

    def identify_gaps_from_chunks(
        self,
        question: str,
        chunks: list[dict],
        *,
        low_confidence_threshold: float = 0.45,
    ) -> list[KnowledgeGap]:
        """
        Identify knowledge gaps by inspecting retrieval chunks for a given question.

        Uses three gap signals:
          1. Missing: question concepts absent from all retrieved chunks
          2. Low-confidence: chunks present but with low score / credibility
          3. Conflicted: chunks that directly contradict each other
        """
        import re

        q_concepts = self._extract_concepts(question)
        ctx_words: set[str] = set()
        chunk_texts: list[str] = []
        chunk_scores: list[float] = []

        for c in chunks:
            text = (c.get("payload") or {}).get("text") or c.get("text") or ""
            chunk_texts.append(text.lower())
            ctx_words.update(text.lower().split())
            score = float(c.get("score") or c.get("payload", {}).get("credibility") or 0.5)
            chunk_scores.append(score)

        gaps: list[KnowledgeGap] = []

        # Gap 1: Missing concepts
        for concept in q_concepts:
            if concept.lower() not in ctx_words:
                gaps.append(KnowledgeGap(
                    concept=concept,
                    gap_type="missing",
                    confidence=0.0,
                    importance=self._estimate_importance(concept, question),
                    blocking_count=0,
                    suggested_query=f"{concept} context background",
                ))

        # Gap 2: Low-confidence chunks present
        avg_score = sum(chunk_scores) / max(len(chunk_scores), 1)
        if avg_score < low_confidence_threshold and chunks:
            top_concepts = q_concepts[:2]
            for concept in top_concepts:
                if concept.lower() in ctx_words:
                    gaps.append(KnowledgeGap(
                        concept=concept,
                        gap_type="low_confidence",
                        confidence=avg_score,
                        importance=self._estimate_importance(concept, question),
                        blocking_count=0,
                        suggested_query=f"{concept} verified source details",
                    ))

        # Gap 3: Conflicted — chunks that partially negate each other
        conflicts = self._detect_conflicts(chunk_texts)
        for conflict_concept in conflicts:
            gaps.append(KnowledgeGap(
                concept=conflict_concept,
                gap_type="conflicted",
                confidence=0.20,
                importance=self._estimate_importance(conflict_concept, question),
                blocking_count=1,
                suggested_query=f"{conflict_concept} clarification authoritative",
            ))

        return gaps

    def identify_gaps_from_state(
        self,
        uncertain_concepts: list[dict],
    ) -> list[KnowledgeGap]:
        """
        Build gaps from an external uncertainty signal.

        uncertain_concepts: list of dicts with keys:
          concept, confidence, importance, gap_type (optional)
        """
        gaps: list[KnowledgeGap] = []
        for uc in uncertain_concepts:
            concept = str(uc.get("concept") or "")
            if not concept:
                continue
            confidence = float(uc.get("confidence") or 0.0)
            importance = float(uc.get("importance") or 0.5)
            gap_type = str(uc.get("gap_type") or "missing")
            gaps.append(KnowledgeGap(
                concept=concept,
                gap_type=gap_type,
                confidence=confidence,
                importance=importance,
                blocking_count=int(uc.get("blocking_count") or 0),
                suggested_query=f"{concept} {gap_type} explanation",
            ))
        return gaps

    # ------------------------------------------------------------------
    # Dependency scoring + plan generation
    # ------------------------------------------------------------------

    def build_acquisition_plan(
        self,
        gaps: list[KnowledgeGap],
        *,
        dependency_graph: dict[str, list[str]] | None = None,
    ) -> KnowledgeAcquisitionPlan:
        """
        Prioritize gaps and return an ordered KnowledgeAcquisitionPlan.

        Priority score per gap:
          = importance × (1 - confidence) × (1 + log(1 + blocking_count))

        dependency_graph: {concept → [concepts it blocks]}.
        If provided, blocking_count is updated to reflect transitive dependencies.
        """
        import math

        enriched = list(gaps)
        if dependency_graph:
            for gap in enriched:
                blocked = self._count_transitive_blocked(
                    gap.concept, dependency_graph
                )
                gap.blocking_count = max(gap.blocking_count, blocked)

        def priority(g: KnowledgeGap) -> float:
            return (
                g.importance
                * (1.0 - g.confidence)
                * (1.0 + math.log1p(g.blocking_count))
            )

        ordered = sorted(enriched, key=priority, reverse=True)

        # Rough uncertainty-reduction estimate if top-5 gaps filled
        top5 = ordered[:5]
        est_reduction = sum(
            g.importance * (1.0 - g.confidence) for g in top5
        ) / max(len(top5), 1)

        return KnowledgeAcquisitionPlan(
            gaps=ordered,
            total_gaps=len(ordered),
            estimated_uncertainty_reduction=round(min(est_reduction, 1.0), 3),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_concepts(text: str) -> list[str]:
        """Extract proper nouns and key noun phrases from text."""
        import re
        proper = re.findall(r"\b([A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})?)\b", text)
        stopwords = frozenset({"When", "What", "Who", "Where", "How", "Did", "Does",
                                "Was", "Were", "The", "This", "That", "There"})
        return [p for p in proper if p not in stopwords][:8]

    @staticmethod
    def _estimate_importance(concept: str, question: str) -> float:
        """Rough importance: higher if concept appears multiple times or is specific."""
        count = question.lower().count(concept.lower())
        length_bonus = min(len(concept.split()) * 0.1, 0.3)
        return min(0.40 + count * 0.15 + length_bonus, 1.0)

    @staticmethod
    def _detect_conflicts(chunk_texts: list[str]) -> list[str]:
        """Very lightweight conflict detection: find negation pairs in chunks."""
        import re
        neg_re = re.compile(r"\b(not|never|no|didn\'t|wasn\'t|hasn\'t|can\'t)\b", re.IGNORECASE)
        conflict_concepts: list[str] = []
        for i, text_a in enumerate(chunk_texts):
            for text_b in chunk_texts[i + 1:]:
                has_neg_a = bool(neg_re.search(text_a))
                has_neg_b = bool(neg_re.search(text_b))
                # If one negates and the other doesn't, flag their common nouns
                if has_neg_a != has_neg_b:
                    words_a = set(text_a.split())
                    words_b = set(text_b.split())
                    common = words_a & words_b - frozenset({
                        "the", "a", "an", "is", "was", "in", "at", "of", "to", "and",
                    })
                    for w in list(common)[:2]:
                        if w not in conflict_concepts:
                            conflict_concepts.append(w)
        return conflict_concepts[:3]

    @staticmethod
    def _count_transitive_blocked(
        concept: str,
        graph: dict[str, list[str]],
        visited: frozenset[str] | None = None,
    ) -> int:
        """Count concepts transitively blocked by this one in the dependency graph."""
        if visited is None:
            visited = frozenset()
        direct = graph.get(concept, [])
        total = 0
        for dep in direct:
            if dep not in visited:
                total += 1 + CurriculumSequencerService._count_transitive_blocked(
                    dep, graph, visited | {concept}
                )
        return total
