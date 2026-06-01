"""
Query tier classifier — decides whether a query needs deep reasoning (LLM)
or is a simple factual lookup that a fast SLM can handle.

Returns one of two tiers:
  'fast'      — direct retrieval + extraction; SLM sufficient
  'reasoning' — multi-step inference, counterfactual, or synthesis; needs LLM

The classifier is purely heuristic (regex + word-count). It runs in
microseconds with zero LLM overhead and is called before model selection
in the cognitive loop whenever no explicit model_hint is provided.
"""

from __future__ import annotations

import re

# ── Reasoning signals ──────────────────────────────────────────────────────
# Past counterfactual / conditional structure.
# "would she have", "could he have", "might they have",
# "if she had done", "if he were to", "what if", "suppose"
_COUNTERFACTUAL_RE = re.compile(
    r'\bwould\s+(?:\w+\s+)?have\b'
    r'|\bcould\s+(?:\w+\s+)?have\b'
    r'|\bmight\s+(?:\w+\s+)?have\b'
    r'|\bhad\s+\w+\s+(?:not\s+)?(?:done|been|gone|said|met|decided|chosen|started|attended|gotten|taken|left|stayed)\b'
    r'|\bif\s+\w+\s+(?:had|were|was|did)\b'
    r'|\bwhat\s+if\b'
    r'|\bsuppose(?:d)?\b'
    r'|\bwere\s+to\b',
    re.IGNORECASE,
)

# Likelihood / probability questions — require inference beyond simple recall.
# "is it likely that", "is it possible that", "likely to have"
_LIKELIHOOD_RE = re.compile(
    r'\bis\s+it\s+(?:likely|possible|probable)\b'
    r'|\blikely\s+(?:that|to)\b'
    r'|\bprobably\s+(?:did|would|could|have)\b',
    re.IGNORECASE,
)

# Synthesis / explanation / evolution questions.
# "why", "describe", "explain", "how did X evolve/change/happen",
# "what caused", "what led", "how has"
_SYNTHESIS_RE = re.compile(
    r'\bwhy\b'
    r'|\bdescribe\b'
    r'|\bexplain\b'
    r'|\bsummar(?:ize|ise)\b'
    r'|\bwhat\s+caused\b'
    r'|\bwhat\s+led\b'
    r'|\bwhat\s+resulted\b'
    r'|\bhow\s+(?:did|has|have|was|were)\b'
    r'|\bhow\s+(?:do|does)\s+.{0,30}\s+(?:affect|influence|relate|connect)\b',
    re.IGNORECASE,
)


def classify_query_tier(text: str) -> str:
    """Return 'fast' or 'reasoning' for the given query text.

    'fast'      → SLM path: single-fact lookup, temporal counting, presence checks.
    'reasoning' → LLM path: counterfactual, multi-hop chain, synthesis, explanation.
    """
    if not text or not text.strip():
        return 'fast'

    n_words = len(text.split())

    # Counterfactual / conditional structure always needs reasoning
    if _COUNTERFACTUAL_RE.search(text):
        return 'reasoning'

    # Likelihood inference needs reasoning
    if _LIKELIHOOD_RE.search(text):
        return 'reasoning'

    # Synthesis / explanation needs reasoning
    if _SYNTHESIS_RE.search(text):
        return 'reasoning'

    # Very long questions almost always require multi-hop reasoning
    if n_words > 20:
        return 'reasoning'

    # Default: simple factual lookup — SLM is sufficient
    return 'fast'
