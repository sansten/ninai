"""Pure helpers for structured LoCoMo evidence states.

These functions keep benchmark diagnostics and prompt shaping out of the main
runner so they can be tested in isolation.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "how", "i", "if", "in", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "she", "that", "the", "their", "them",
    "they", "this", "to", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "with", "would", "you", "your",
}

_RELATIVE_TIME_MARKERS = (
    "today", "tomorrow", "yesterday", "last week", "last month", "last year",
    "next week", "next month", "next year", "this week", "this month",
    "this year", "a few weeks", "few weeks", "mid-summer", "during the pandemic",
)


def _compact(text: str, max_len: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _question_terms(question: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9']+", str(question or "").lower())
    seen: set[str] = set()
    terms: list[str] = []
    for tok in tokens:
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
    return terms[:10]


def _extract_hit_entities(hit: dict[str, Any]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def _add(value: Any) -> None:
        text = _compact(str(value or ""), max_len=80)
        if not text:
            return
        lowered = text.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        found.append(text)

    raw = hit.get("entities")
    if isinstance(raw, dict):
        for _, value in raw.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _add(item.get("name") or item.get("text") or item.get("value"))
                    else:
                        _add(item)
            else:
                _add(value)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                _add(item.get("name") or item.get("text") or item.get("value"))
            else:
                _add(item)

    if found:
        return found[:8]

    content = str(hit.get("content") or "")
    for match in re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", content):
        _add(match)
    return found[:8]


def build_evidence_state(
    question: str,
    category: str,
    hits: list[dict[str, Any]] | None,
    *,
    max_facts: int = 4,
) -> dict[str, Any]:
    """Summarize retrieved hits into a compact, prompt-safe evidence state."""
    hits = list(hits or [])
    q_terms = _question_terms(question)
    q_term_set = set(q_terms)

    facts: list[dict[str, str]] = []
    session_dates: list[str] = []
    relative_markers: list[str] = []
    entity_counts: Counter[str] = Counter()
    bridge_counts: Counter[str] = Counter()
    first_person_turns = 0

    for hit in hits[: max(max_facts * 2, 8)]:
        content = _compact(hit.get("content", ""))
        if not content:
            continue

        occurred_at = str(hit.get("occurred_at") or "")[:10]
        if occurred_at:
            session_dates.append(occurred_at)

        facts.append({
            "date": occurred_at,
            "text": content,
        })

        lower_content = content.lower()
        if re.search(r"\b(i|me|my|we|our|us)\b", lower_content):
            first_person_turns += 1

        for marker in _RELATIVE_TIME_MARKERS:
            if marker in lower_content and marker not in relative_markers:
                relative_markers.append(marker)

        for ent in _extract_hit_entities(hit):
            entity_counts[ent] += 1
            ent_tokens = {tok for tok in re.findall(r"[A-Za-z0-9']+", ent.lower()) if len(tok) >= 3}
            if ent_tokens and ent_tokens.isdisjoint(q_term_set):
                bridge_counts[ent] += 1

    top_entities = [name for name, _ in entity_counts.most_common(6)]
    bridge_entities = [name for name, count in bridge_counts.most_common(4) if count >= 1]

    unique_dates: list[str] = []
    for dt in session_dates:
        if dt and dt not in unique_dates:
            unique_dates.append(dt)

    selected_facts: list[dict[str, str]] = []
    seen_facts: set[str] = set()
    for fact in facts:
        key = fact["text"].lower()
        if key in seen_facts:
            continue
        seen_facts.add(key)
        selected_facts.append(fact)
        if len(selected_facts) >= max_facts:
            break

    return {
        "question_terms": q_terms,
        "top_entities": top_entities,
        "bridge_entities": bridge_entities if category == "multi_hop" else [],
        "session_dates": unique_dates[:5],
        "relative_time_markers": relative_markers[:6] if category == "temporal" else [],
        "first_person_turn_count": first_person_turns if category == "adversarial" else 0,
        "fact_candidates": selected_facts,
        "has_cross_session_evidence": len(unique_dates) > 1,
    }


def build_evidence_block(state: dict[str, Any]) -> str:
    """Render a compact text block that can be fed to prompts/debug logs."""
    if not state:
        return ""

    lines: list[str] = []
    q_terms = state.get("question_terms") or []
    if q_terms:
        lines.append("Question terms: " + ", ".join(q_terms))

    top_entities = state.get("top_entities") or []
    if top_entities:
        lines.append("Top entities: " + ", ".join(top_entities[:6]))

    bridge_entities = state.get("bridge_entities") or []
    if bridge_entities:
        lines.append("Bridge entities: " + ", ".join(bridge_entities[:4]))

    session_dates = state.get("session_dates") or []
    if session_dates:
        lines.append("Session dates: " + ", ".join(session_dates[:5]))

    relative_markers = state.get("relative_time_markers") or []
    if relative_markers:
        lines.append("Temporal markers: " + ", ".join(relative_markers[:6]))

    first_person_turn_count = int(state.get("first_person_turn_count") or 0)
    if first_person_turn_count:
        lines.append(f"Perspective cues: {first_person_turn_count} first-person turn(s)")

    fact_candidates = state.get("fact_candidates") or []
    if fact_candidates:
        lines.append("Candidate facts:")
        for fact in fact_candidates[:4]:
            date_prefix = f"[{fact.get('date')}] " if fact.get("date") else ""
            lines.append("- " + date_prefix + _compact(fact.get("text", ""), max_len=160))

    return "\n".join(lines)


def classify_failure_layer(row: dict[str, Any]) -> str:
    """Higher-level failure attribution aligned to the AGI plan."""
    semantic = row.get("semantic_correct", -1)
    rouge = float(row.get("rouge1_f1", 0) or 0)
    category = str(row.get("category", "") or "")
    evidence_state = row.get("evidence_state") or {}
    generated = str(row.get("generated_answer", "") or "").lower()

    if semantic == 1 and rouge < 40:
        return "format_only_miss"
    if semantic == 1:
        return "ok"
    if not row.get("retrieval_has_gold", False):
        return "retrieval_miss"
    if category == "multi_hop" and not evidence_state.get("bridge_entities"):
        return "bridge_miss"
    if category == "temporal" and not evidence_state.get("relative_time_markers") and not evidence_state.get("session_dates"):
        return "temporal_anchor_miss"
    if category == "adversarial":
        if evidence_state.get("first_person_turn_count", 0) > 0:
            return "perspective_miss"
        if "no mention" in generated or "not mention" in generated:
            return "answer_extraction_miss"
    if rouge < 25:
        return "answer_extraction_miss"
    return "reasoning_miss"
