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


# Inferential / opinion / open-ended verbs — need the stronger model.
# "what did X realize/decide/learn", "what are X's plans/thoughts/views/reaction"
_INFERENTIAL_RE = re.compile(
    r'\b(realiz|decide[ds]?|learn(?:ed|t)?|think[s]?|thought|feel[s]?|felt|'
    r'believe[ds]?|prefer(?:s|red)?|want[s]?|wanted|plan(?:s|ned|ning)?|'
    r'consider(?:s|ed)?|expect[s]?|hope[ds]?|wish(?:es|ed)?|intend[s]?|'
    r'reaction|opinion|view[s]?|reason[s]?|motivat|inspire[ds]?|enjoy)\b',
    re.IGNORECASE,
)

# List / enumeration / comparison — broad retrieval + synthesis, not one lookup.
_LIST_COMPARE_RE = re.compile(
    r'\bwhat\s+(?:\w+\s+){0,2}(activities|events|things|ways|hobbies|interests|'
    r'plans|goals|books|movies|places|topics|kinds?|types?|reasons?)\b'
    r'|\bin\s+what\s+ways\b'
    r'|\b(compare|comparison|difference|differ|more\s+than|less\s+than|better|worse|'
    r'rather\s+than|versus|vs)\b',
    re.IGNORECASE,
)

# Short, DIRECT single-fact lookups the 7b handles reliably — the only "fast" cases.
_SIMPLE_FACT_RE = re.compile(
    r'^\s*(what\s+(is|was|are|were)\s|who\s+(is|was|are|were)\s|'
    r'when\s+(did|was|is|were)\s|where\s+(did|was|is|were)\s|'
    r'how\s+(many|much|old|long\s+ago)\b|which\b)',
    re.IGNORECASE,
)


def classify_query_tier(text: str) -> str:
    """Return 'fast' or 'reasoning' for the given query text.

    Conservative bias: only SHORT, DIRECT single-fact lookups go to the fast SLM
    (qwen2.5:7b). Anything inferential, open-ended, comparative, list-style,
    counterfactual, or long goes to the stronger reasoning model (deepseek-r1:14b).
    This corrects the earlier 35.0 regression where too many open_domain /
    inferential questions hit the weak 7b and underperformed / refused.
    """
    if not text or not text.strip():
        return 'fast'

    n_words = len(text.split())

    # Anything requiring inference / synthesis / comparison / counterfactual → reasoning
    if (_COUNTERFACTUAL_RE.search(text)
            or _LIKELIHOOD_RE.search(text)
            or _SYNTHESIS_RE.search(text)
            or _INFERENTIAL_RE.search(text)
            or _LIST_COMPARE_RE.search(text)):
        return 'reasoning'

    # Longer questions tend to be multi-hop / compositional → reasoning
    if n_words > 14:
        return 'reasoning'

    # Fast tier ONLY for short, direct single-fact lookups.
    if _SIMPLE_FACT_RE.search(text):
        return 'fast'

    # Default to the stronger model when unsure (accuracy > speed).
    return 'reasoning'
