"""EpistemicProvenanceService — Phase 88.

Evaluates the quality of evidence for a claim by distinguishing between:
  - Genuinely independent sources (convergent evidence → high warrant)
  - Citation copies of the same original source (single-source illusion)

Without this, a fact that appears in 10 documents but all citing the same
blog post is treated as 10x more credible than it actually is.

Algorithm:
  1. source_fingerprint(chunk)     — extract the original source identity
  2. cluster_by_origin(chunks)     — group by fingerprint
  3. score_independence(clusters)  — score = unique_origins / total_chunks
  4. adjust_credibility(score, raw_credibility) → calibrated_credibility

Also exposes:
  trace_citation_chain(chunks)  — returns the likely original source + chain
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s\"\'\]]+", re.IGNORECASE)
_DOC_REF_RE = re.compile(
    r"\b(doc(?:ument)?[:\s#]\s*\w+|report[:\s]\w+|paper[:\s]\w+|article[:\s]\w+)",
    re.IGNORECASE,
)
_AUTHOR_RE = re.compile(r"\b([A-Z][a-z]+ (?:et al\.?|and [A-Z][a-z]+))\b")

# Phrases that mark a document as a re-citation, not a primary source
_CITATION_MARKERS = frozenset({
    "as reported by", "according to", "citing", "references",
    "sourced from", "based on", "adapted from", "via", "quoting",
})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SourceFingerprint:
    chunk_id: str
    origin_hash: str       # hash of the inferred original source identity
    origin_label: str      # human-readable description (URL, doc ref, or text hash)
    is_derivative: bool    # True if this chunk appears to be a re-citation


@dataclass
class ProvenanceAssessment:
    total_chunks: int
    unique_origins: int
    independence_score: float   # unique_origins / total_chunks, [0, 1]
    is_single_source: bool      # True if all chunks trace to one origin
    adjusted_credibility: float
    origin_labels: list[str]    # list of distinct source identifiers


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class EpistemicProvenanceService:

    def source_fingerprint(self, chunk: dict) -> SourceFingerprint:
        """Extract a stable origin identifier from a chunk's payload."""
        payload = chunk.get("payload") or {}
        text = (payload.get("text") or chunk.get("text") or "").strip()
        chunk_id = str(chunk.get("id") or "")

        # Priority: explicit source field → URL in text → doc reference → author → text hash
        source = (
            payload.get("source")
            or payload.get("url")
            or payload.get("origin")
            or payload.get("document_id")
        )

        is_derivative = False
        origin_label: str

        if source:
            origin_label = str(source)[:120]
        else:
            url_match = _URL_RE.search(text)
            if url_match:
                origin_label = url_match.group(0)[:120]
            elif _DOC_REF_RE.search(text):
                origin_label = _DOC_REF_RE.search(text).group(0)[:120]  # type: ignore[union-attr]
            elif _AUTHOR_RE.search(text):
                origin_label = _AUTHOR_RE.search(text).group(0)[:60]  # type: ignore[union-attr]
            else:
                # Fall back to hash of first 200 chars (same text ≈ same source)
                origin_label = "text:" + hashlib.sha256(text[:200].encode()).hexdigest()[:12]

        # Detect re-citation
        low = text.lower()
        if any(m in low for m in _CITATION_MARKERS):
            is_derivative = True

        origin_hash = hashlib.sha256(origin_label.encode()).hexdigest()[:16]
        return SourceFingerprint(
            chunk_id=chunk_id,
            origin_hash=origin_hash,
            origin_label=origin_label,
            is_derivative=is_derivative,
        )

    def assess_provenance(
        self,
        chunks: list[dict],
        raw_credibility: float = 0.5,
    ) -> ProvenanceAssessment:
        """
        Score the independence of a set of chunks that support the same claim.

        raw_credibility: the agent's prior credibility estimate (0–1).
        Returns an adjusted credibility that accounts for source redundancy.
        """
        if not chunks:
            return ProvenanceAssessment(
                total_chunks=0,
                unique_origins=0,
                independence_score=0.0,
                is_single_source=True,
                adjusted_credibility=raw_credibility,
                origin_labels=[],
            )

        fingerprints = [self.source_fingerprint(c) for c in chunks]
        origins: dict[str, str] = {}  # origin_hash → label
        for fp in fingerprints:
            if fp.origin_hash not in origins:
                origins[fp.origin_hash] = fp.origin_label

        n = len(chunks)
        unique = len(origins)
        independence = unique / n

        # Adjust credibility:
        # - All from one source: penalise (divide by ln(n+1) to shrink overcount)
        # - Fully independent: slight boost (up to +10%)
        import math
        if unique == 1:
            # Single-source illusion: raw × (1 / log2(n + 1))
            deflation = 1.0 / max(math.log2(n + 1), 1.0)
            adjusted = raw_credibility * deflation
        else:
            # Independence bonus: scales with sqrt(unique / n)
            bonus = 0.10 * math.sqrt(independence)
            adjusted = min(raw_credibility + bonus, 1.0)

        return ProvenanceAssessment(
            total_chunks=n,
            unique_origins=unique,
            independence_score=round(independence, 4),
            is_single_source=(unique == 1),
            adjusted_credibility=round(adjusted, 4),
            origin_labels=list(origins.values()),
        )

    def trace_citation_chain(self, chunks: list[dict]) -> dict[str, Any]:
        """
        Identify the likely primary source and list derivative re-citations.
        Returns {primary_source, derivatives: list, chain_depth: int}.
        """
        fingerprints = [self.source_fingerprint(c) for c in chunks]

        primaries = [fp for fp in fingerprints if not fp.is_derivative]
        derivatives = [fp for fp in fingerprints if fp.is_derivative]

        # Most-common origin among non-derivatives is likely the root source
        if primaries:
            from collections import Counter
            origin_counts = Counter(fp.origin_hash for fp in primaries)
            most_common_hash = origin_counts.most_common(1)[0][0]
            primary_label = next(
                fp.origin_label for fp in primaries if fp.origin_hash == most_common_hash
            )
        elif fingerprints:
            primary_label = fingerprints[0].origin_label
        else:
            primary_label = "unknown"

        return {
            "primary_source": primary_label,
            "derivatives": [fp.origin_label for fp in derivatives],
            "chain_depth": len(derivatives),
            "total_sources": len({fp.origin_hash for fp in fingerprints}),
        }
