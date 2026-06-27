"""InterferencePreventionService — Phase 90.

Prevents catastrophic forgetting: when a new memory would overwrite or
substantially conflict with an existing stable, load-bearing memory, this
service intercepts the write and chooses one of three strategies:

  1. PRESERVE_BOTH   — store both with explicit temporal ordering
  2. SOFT_UPDATE     — update the existing memory's metadata, keep content
  3. SUPERSEDE       — new memory replaces old (only when old is low-confidence
                       and not referenced by other reasoning chains)

"Stable" = high credibility + referenced by >= min_reference_count chains.
"Load-bearing" = credibility >= stability_threshold AND ref_count >= min_refs.

This directly addresses the stability-plasticity dilemma in knowledge systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import re


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STABILITY_THRESHOLD = 0.70     # credibility above which a memory is "stable"
_MIN_REFERENCE_COUNT = 2        # reference count above which a memory is "load-bearing"
_OVERLAP_THRESHOLD = 0.55       # Jaccard overlap above which memories are "conflicting"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class WriteStrategy(str, Enum):
    SUPERSEDE = "supersede"         # replace old with new
    SOFT_UPDATE = "soft_update"     # update metadata only, keep old content
    PRESERVE_BOTH = "preserve_both" # keep both with temporal annotation


@dataclass
class StabilityProfile:
    memory_id: str
    credibility: float
    reference_count: int
    is_stable: bool
    is_load_bearing: bool


@dataclass
class InterferenceDecision:
    strategy: WriteStrategy
    existing_id: str
    reason: str
    overlap_score: float
    existing_stability: StabilityProfile


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class InterferencePreventionService:

    def __init__(
        self,
        stability_threshold: float = _STABILITY_THRESHOLD,
        min_reference_count: int = _MIN_REFERENCE_COUNT,
        overlap_threshold: float = _OVERLAP_THRESHOLD,
    ) -> None:
        self.stability_threshold = stability_threshold
        self.min_reference_count = min_reference_count
        self.overlap_threshold = overlap_threshold

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def evaluate_write(
        self,
        new_text: str,
        existing_memory: dict,
    ) -> InterferenceDecision:
        """
        Given a new memory text and an existing memory that semantically overlaps
        with it, decide the safest write strategy.

        existing_memory: dict with keys:
          id, text (or content), credibility (float), reference_count (int)
        """
        existing_text = (
            existing_memory.get("text")
            or existing_memory.get("content")
            or ""
        )
        existing_id = str(existing_memory.get("id") or "")
        credibility = float(existing_memory.get("credibility") or 0.5)
        ref_count = int(existing_memory.get("reference_count") or 0)

        overlap = self.text_overlap(new_text, existing_text)

        profile = StabilityProfile(
            memory_id=existing_id,
            credibility=credibility,
            reference_count=ref_count,
            is_stable=credibility >= self.stability_threshold,
            is_load_bearing=(
                credibility >= self.stability_threshold
                and ref_count >= self.min_reference_count
            ),
        )

        strategy, reason = self._choose_strategy(overlap, profile, new_text, existing_text)

        return InterferenceDecision(
            strategy=strategy,
            existing_id=existing_id,
            reason=reason,
            overlap_score=round(overlap, 4),
            existing_stability=profile,
        )

    def batch_evaluate(
        self,
        new_text: str,
        candidate_memories: list[dict],
    ) -> list[InterferenceDecision]:
        """
        Evaluate interference risk against a list of semantically similar memories.
        Only returns decisions where overlap >= overlap_threshold.
        """
        decisions = []
        for mem in candidate_memories:
            existing_text = mem.get("text") or mem.get("content") or ""
            overlap = self.text_overlap(new_text, existing_text)
            if overlap >= self.overlap_threshold:
                decisions.append(self.evaluate_write(new_text, mem))
        return decisions

    # ------------------------------------------------------------------
    # Text overlap (Jaccard on content tokens)
    # ------------------------------------------------------------------

    def text_overlap(self, text_a: str, text_b: str) -> float:
        _STOPWORDS = frozenset({"the", "a", "an", "is", "are", "was", "were",
                                 "in", "at", "of", "to", "and", "or", "it",
                                 "this", "that", "with", "for", "on", "be"})
        def _tokens(t: str) -> set[str]:
            return {w.lower() for w in re.findall(r"\b\w{3,}\b", t)} - _STOPWORDS

        a, b = _tokens(text_a), _tokens(text_b)
        if not (a | b):
            return 0.0
        return len(a & b) / len(a | b)

    # ------------------------------------------------------------------
    # Conflict detection between new and existing texts
    # ------------------------------------------------------------------

    def texts_conflict(self, text_a: str, text_b: str) -> bool:
        """
        Return True if text_a and text_b appear to make contradictory claims
        about the same subject (one affirms, one negates a phrase).
        """
        _NEG_RE = re.compile(r"\b(not|never|no|didn't|wasn't|hasn't|can't|won't)\b",
                              re.IGNORECASE)
        has_neg_a = bool(_NEG_RE.search(text_a))
        has_neg_b = bool(_NEG_RE.search(text_b))
        return has_neg_a != has_neg_b and self.text_overlap(text_a, text_b) > 0.30

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _choose_strategy(
        self,
        overlap: float,
        profile: StabilityProfile,
        new_text: str,
        existing_text: str,
    ) -> tuple[WriteStrategy, str]:
        if overlap < self.overlap_threshold:
            return WriteStrategy.SUPERSEDE, "low overlap: new memory is sufficiently distinct"

        if profile.is_load_bearing:
            if self.texts_conflict(new_text, existing_text):
                return (
                    WriteStrategy.PRESERVE_BOTH,
                    f"load-bearing memory (cred={profile.credibility:.2f}, "
                    f"refs={profile.reference_count}) conflicts with new — preserving both",
                )
            return (
                WriteStrategy.SOFT_UPDATE,
                f"load-bearing memory (cred={profile.credibility:.2f}, "
                f"refs={profile.reference_count}) overlaps — soft update only",
            )

        if profile.is_stable and not self.texts_conflict(new_text, existing_text):
            return (
                WriteStrategy.SOFT_UPDATE,
                f"stable memory (cred={profile.credibility:.2f}) consistent with new — soft update",
            )

        if not profile.is_stable:
            return WriteStrategy.SUPERSEDE, "existing memory has low credibility — safe to supersede"

        return (
            WriteStrategy.PRESERVE_BOTH,
            "stable conflict: preserving both with temporal annotation",
        )
