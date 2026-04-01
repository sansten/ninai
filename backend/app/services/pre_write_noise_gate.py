"""Pre-Write Noise Gate — Gap E.

Every memory write enters the full enrichment pipeline regardless of whether
the content is near-duplicate of something already stored.  Inbound connector
events are especially prone to this — the same alert fires three times.

This service provides a cheap similarity check before enrichment starts:

  1. Fingerprint the incoming content (normalised token set hash).
  2. Compare against a small in-memory recency window of recent fingerprints
     per org (configurable size, default 200 entries).
  3. If the Jaccard similarity of the token sets exceeds the noise threshold,
     the write is classified as noise and should be skipped (or de-prioritised).

Design rules:
- Pure in-memory: no DB session, no Qdrant.  Fast enough to run synchronously
  on every write path.
- Thread-safe reads via a plain dict (writes are single-threaded in async context).
- Returns a NoiseGateResult with is_noise=True/False and the closest match's
  similarity score so callers can log or escalate partial duplicates.
- Does NOT delete or modify any existing memory.
- The recency window uses an LRU-like eviction: when capacity is exceeded, the
  oldest entry per org is dropped.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Jaccard threshold above which content is considered near-duplicate noise
_DEFAULT_NOISE_THRESHOLD = 0.85

# Max fingerprints to keep per org in the recency window
_DEFAULT_WINDOW_SIZE = 200

# Minimum token count to run the gate (very short strings are not deduplicated)
_MIN_TOKENS = 5


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class NoiseGateResult:
    is_noise: bool
    similarity: float        # highest similarity found (0.0 if no match)
    closest_fingerprint: str | None  # fingerprint of the nearest duplicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(content: str) -> frozenset[str]:
    """Lowercase token set from content string."""
    return frozenset(
        w.strip(".,;:!?\"'()[]{}") for w in content.lower().split()
        if len(w) > 2
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _fingerprint(tokens: frozenset[str]) -> str:
    """Stable string key for a token set (sorted join, truncated)."""
    return " ".join(sorted(tokens))[:256]


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class PreWriteNoiseGate:
    """Stateful deduplication gate for memory writes.

    One instance should be shared per process (or injected as a singleton).

    Usage:
        gate = PreWriteNoiseGate()

        # Before enrichment:
        result = gate.check(org_id="org-1", content=raw_content)
        if result.is_noise:
            return  # skip enrichment + write

        # After successful write, register the content:
        gate.register(org_id="org-1", content=raw_content)
    """

    def __init__(
        self,
        *,
        noise_threshold: float = _DEFAULT_NOISE_THRESHOLD,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ) -> None:
        self._threshold = max(0.0, min(1.0, noise_threshold))
        self._window_size = max(10, window_size)
        # org_id -> OrderedDict[fingerprint -> frozenset[str]]
        self._windows: dict[str, OrderedDict[str, frozenset[str]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, *, org_id: str, content: str) -> NoiseGateResult:
        """Check whether content is near-duplicate of a recently-seen write.

        Returns NoiseGateResult(is_noise=False) if the recency window is empty
        or the content is sufficiently novel.
        """
        tokens = _tokenize(content)
        if len(tokens) < _MIN_TOKENS:
            return NoiseGateResult(is_noise=False, similarity=0.0, closest_fingerprint=None)

        window = self._windows.get(org_id)
        if not window:
            return NoiseGateResult(is_noise=False, similarity=0.0, closest_fingerprint=None)

        best_sim = 0.0
        best_fp: str | None = None
        for fp, existing_tokens in window.items():
            sim = _jaccard(tokens, existing_tokens)
            if sim > best_sim:
                best_sim = sim
                best_fp = fp

        is_noise = best_sim >= self._threshold
        return NoiseGateResult(
            is_noise=is_noise,
            similarity=round(best_sim, 4),
            closest_fingerprint=best_fp,
        )

    def register(self, *, org_id: str, content: str) -> None:
        """Add content to the recency window for this org.

        Call this after a successful write to track the content for future
        duplicate detection.  Evicts the oldest entry when the window is full.
        """
        tokens = _tokenize(content)
        if len(tokens) < _MIN_TOKENS:
            return

        fp = _fingerprint(tokens)
        window = self._windows.setdefault(org_id, OrderedDict())

        if fp in window:
            # Move to end (most-recently-seen)
            window.move_to_end(fp)
            return

        window[fp] = tokens
        if len(window) > self._window_size:
            window.popitem(last=False)  # evict oldest

    def check_and_register(self, *, org_id: str, content: str) -> NoiseGateResult:
        """Convenience: check, then register if not noise.

        Returns the NoiseGateResult.  If is_noise=False, the content has been
        registered so subsequent identical writes are detected.
        """
        result = self.check(org_id=org_id, content=content)
        if not result.is_noise:
            self.register(org_id=org_id, content=content)
        return result

    def clear(self, *, org_id: str | None = None) -> None:
        """Clear the recency window.  Pass org_id to clear one org, or None for all."""
        if org_id is None:
            self._windows.clear()
        else:
            self._windows.pop(org_id, None)

    def window_size_for_org(self, org_id: str) -> int:
        """Return the number of fingerprints currently tracked for an org."""
        return len(self._windows.get(org_id, {}))


# Process-level singleton — import and use directly:
#   from app.services.pre_write_noise_gate import default_noise_gate
default_noise_gate = PreWriteNoiseGate()
