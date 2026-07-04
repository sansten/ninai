"""Deterministic temporal-arithmetic helper.

Small local models are unreliable at date/duration arithmetic — asking "how
many days between the barbecue and the retirement party" makes the model try
to subtract dates itself, which it frequently gets wrong even when both
dates are present in the retrieved context.

This module moves that arithmetic into Python: it looks at the question,
finds two anchor dates in the already-retrieved evidence pool, computes the
delta deterministically, and returns a short computed-fact string that can
be appended to the prompt as grounding — the LLM still writes the final
answer sentence, it just no longer has to do the subtraction itself.

Pure function, no I/O, no LLM calls, no gold-answer access — only the
question text and whatever evidence retrieval already surfaced, so it
generalizes to any unseen duration question of this shape.
"""
from __future__ import annotations

import re
from datetime import date

_DURATION_Q_RE = re.compile(
    r"\bhow\s+(?:many|long)\b.{0,40}\b(days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "how", "many", "long", "did", "does",
    "was", "were", "is", "are", "between", "from", "to", "after", "before",
    "since", "until", "of", "in", "on", "at", "for", "with", "by",
    "days", "day", "weeks", "week", "months", "month", "years", "year",
})


def _payload_date(chunk: dict) -> date | None:
    p = chunk.get("payload") or {}
    for key in ("anchor_date", "date_prefix"):
        raw = p.get(key)
        if raw:
            m = _ISO_DATE_RE.search(str(raw))
            if m:
                try:
                    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    pass
    text = str(p.get("text") or p.get("content") or "")
    m = _ISO_DATE_RE.search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


_BETWEEN_RE = re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)\s*\??$", re.IGNORECASE)
_FROM_TO_RE = re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)\s*\??$", re.IGNORECASE)


def _terms(phrase: str) -> str:
    return " ".join(
        t for t in re.findall(r"[a-z']+", phrase.lower()) if t not in _STOPWORDS and len(t) > 2
    )


def _reference_phrases(question: str) -> list[str]:
    """Pull the two candidate event-reference phrases out of a duration question.

    Only fires on an explicit "between X and Y" / "from X to Y" anchor —
    earlier versions split on the first bare " and "/" to " anywhere in the
    question, which misfires when an event name itself contains "and" (e.g.
    "the trip Bob and Alice took"). Requiring the anchor keyword means we
    return no phrases (safe no-op) rather than a wrong split in that case.
    """
    q = question.strip()
    m = _BETWEEN_RE.search(q) or _FROM_TO_RE.search(q)
    if not m:
        return []
    left, right = _terms(m.group(1)), _terms(m.group(2))
    if not left or not right:
        return []
    return [left, right]


def _best_matching_dated_chunk(phrase: str, chunks: list[dict]) -> date | None:
    phrase_tokens = set(phrase.lower().split())
    if not phrase_tokens:
        return None
    best_score = 0
    best_date: date | None = None
    for ch in chunks:
        d = _payload_date(ch)
        if not d:
            continue
        p = ch.get("payload") or {}
        text = str(p.get("text") or p.get("content") or p.get("summary") or "").lower()
        score = sum(1 for t in phrase_tokens if t in text)
        if score > best_score:
            best_score = score
            best_date = d
    return best_date if best_score > 0 else None


def compute_temporal_arithmetic(question: str, chunks: list[dict]) -> str | None:
    """Return a computed-fact hint string, or None if not confidently computable.

    Only fires for explicit "how many days/weeks/months/years" style
    questions, and only when two distinct reference dates can be matched
    against the already-retrieved evidence pool. Returning None is always
    safe — the caller's existing prompt/answer path is untouched.
    """
    if not chunks or not _DURATION_Q_RE.search(question or ""):
        return None

    phrases = _reference_phrases(question)
    if len(phrases) != 2:
        return None

    d1 = _best_matching_dated_chunk(phrases[0], chunks)
    d2 = _best_matching_dated_chunk(phrases[1], chunks)
    if not d1 or not d2 or d1 == d2:
        return None

    earlier, later = (d1, d2) if d1 <= d2 else (d2, d1)
    delta_days = (later - earlier).days
    weeks = delta_days // 7
    months = round(delta_days / 30.44, 1)

    return (
        f"[Computed fact - do not recompute: the earlier referenced date is "
        f"{earlier.isoformat()} and the later referenced date is {later.isoformat()}; "
        f"the gap is {delta_days} days (~{weeks} weeks, ~{months} months). "
        f"Use this number directly in your answer.]"
    )
