from __future__ import annotations

from dataclasses import dataclass


_CLASSIFICATION_ORDER = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}


def resolve_classification(candidates: list[str]) -> str:
    """Resolve to the most restrictive (max) classification.

    Fail-closed: unknown values raise.
    """

    if not candidates:
        return "internal"

    normalized: list[str] = [str(c).strip().lower() for c in candidates if str(c).strip()]
    if not normalized:
        return "internal"

    for c in normalized:
        if c not in _CLASSIFICATION_ORDER:
            raise ValueError("Unknown classification")

    return max(normalized, key=lambda c: _CLASSIFICATION_ORDER[c])


@dataclass(frozen=True)
class ClassificationCandidate:
    classification: str
    confidence: float


def resolve_classification_candidates(
    candidates: list[ClassificationCandidate], *, confidence_gap_threshold: float = 0.60
) -> str:
    """Arbitrate classification.

    Rule:
    - default to most restrictive classification across candidates
    - unless a top-confidence candidate exceeds runner-up by > threshold
    """

    if not candidates:
        return "internal"

    cleaned: list[ClassificationCandidate] = []
    for c in candidates:
        val = str(c.classification).strip().lower()
        if not val:
            continue
        if val not in _CLASSIFICATION_ORDER:
            raise ValueError("Unknown classification")
        conf = float(c.confidence)
        if conf < 0 or conf > 1:
            raise ValueError("Candidate confidence must be 0..1")
        cleaned.append(ClassificationCandidate(classification=val, confidence=conf))

    if not cleaned:
        return "internal"

    cleaned_sorted = sorted(cleaned, key=lambda x: x.confidence, reverse=True)
    top = cleaned_sorted[0]
    runner_up_conf = cleaned_sorted[1].confidence if len(cleaned_sorted) > 1 else 0.0
    if (top.confidence - runner_up_conf) > float(confidence_gap_threshold or 0.0):
        return top.classification

    # No clear winner: fail safe (most restrictive)
    return max((c.classification for c in cleaned_sorted), key=lambda v: _CLASSIFICATION_ORDER[v])


@dataclass(frozen=True)
class ConflictDetection:
    has_conflict: bool
    conflict_type: str | None = None
    details: dict | None = None


def detect_classification_conflict(candidates: list[str]) -> ConflictDetection:
    normalized = [str(c).strip().lower() for c in candidates if str(c).strip()]
    unique = sorted(set(normalized))
    if len(unique) <= 1:
        return ConflictDetection(has_conflict=False)

    return ConflictDetection(
        has_conflict=True,
        conflict_type="classification",
        details={"candidates": unique},
    )
