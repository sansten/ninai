"""Sequential sub-query decomposition for nested multi-hop questions.

Questions like "Who reported to the person who approved the Q3 budget?" pack
two lookups into one call: (1) resolve the embedded clause ("who approved
the Q3 budget?") to an entity, then (2) answer the outer question with that
entity substituted in. A single small-model call has to hold both hops in
its head at once and frequently drops one; this module resolves them as two
independent calls and lets Python do the substitution between them.

The split point is found purely structurally — the last "who" relative-
clause connector in the question text — never by reading the gold
answer or branching on a question id, so it generalizes to any unseen
question of this shape. Falls back to None (caller uses its existing
single-shot path) whenever the split isn't confident or either hop refuses,
so this can never make an answer worse than the baseline.
"""
from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

InferFn = Callable[[str], Awaitable[Any]]
PromptBuilderFn = Callable[[str, list[dict]], str]

# Only "who" maps cleanly onto a standalone wh-question ("Who approved X?").
# "that"/"which" clauses don't have a natural interrogative form ("That
# approved the Q3 budget?" isn't a real question), which used to produce
# a garbage inner question and an unreliable substituted answer — so those
# two connectors are intentionally not supported.
_CONNECTORS = (" who ",)

_REFUSAL_FRAGMENTS = (
    "not mentioned", "not provided", "not available", "no information",
    "cannot determine", "does not contain", "not in the context",
    "not specified", "not stated", "not found", "i don't know", "i do not know",
)


def _is_refusal(text: str) -> bool:
    low = (text or "").lower().strip()
    if len(low) < 2:
        return True
    return any(p in low for p in _REFUSAL_FRAGMENTS)


def split_nested_question(question: str) -> tuple[str, str] | None:
    """Split "<outer> <connector> <inner>?" into (inner_question, outer_template).

    outer_template contains "{ANSWER}" where the inner answer should be
    substituted. Returns None when no confident nested-clause split exists.
    """
    q = question.strip()
    low = q.lower()

    best_idx = -1
    best_connector = ""
    for connector in _CONNECTORS:
        idx = low.rfind(connector)
        # Leave enough room before AND after the connector for two real clauses.
        if idx > 8 and idx > best_idx:
            best_idx = idx
            best_connector = connector

    if best_idx < 0:
        return None

    before = q[:best_idx].strip()
    after = q[best_idx + len(best_connector):].strip().rstrip("?").strip()

    if len(before.split()) < 3 or len(after.split()) < 3:
        return None

    wh_word = best_connector.strip().capitalize()
    inner_question = f"{wh_word} {after}?"
    outer_template = f"{before} {{ANSWER}}?"
    return inner_question, outer_template


def chunks_for_query(query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    """Lexical-overlap filter over the already-retrieved pool — no new retrieval I/O."""
    terms = {t.lower() for t in re.findall(r"[A-Za-z0-9']+", query) if len(t) > 3}
    scored = []
    for ch in chunks:
        p = ch.get("payload") or {}
        text = str(p.get("text") or p.get("content") or p.get("summary") or "").lower()
        score = sum(1 for t in terms if t in text)
        if score:
            scored.append((score, ch))
    scored.sort(key=lambda x: -x[0])
    return [ch for _, ch in scored[:top_k]] if scored else chunks[:top_k]


async def decompose_and_compose(
    question: str,
    chunks: list[dict],
    infer_fn: InferFn,
    build_prompt_fn: PromptBuilderFn,
) -> str | None:
    """Resolve a nested multi-hop question as two sequential sub-answers.

    Returns the composed final answer, or None if the split failed or either
    hop came back empty/refused — callers should fall back to their existing
    single-shot answer in that case.
    """
    split = split_nested_question(question)
    if not split:
        return None
    inner_question, outer_template = split

    inner_chunks = chunks_for_query(inner_question, chunks)
    if not inner_chunks:
        return None
    inner_prompt = build_prompt_fn(inner_question, inner_chunks)
    inner_result = await infer_fn(inner_prompt)
    inner_answer = str(getattr(inner_result, "response", inner_result) or "").strip()
    # A long/refused inner answer isn't a substitutable entity.
    if _is_refusal(inner_answer) or len(inner_answer) > 80:
        return None

    # .replace, not .format — the question text can legitimately contain
    # literal "{"/"}" (quoted JSON, code snippets), which would otherwise
    # raise inside .format() and silently disable decomposition for those
    # questions via the caller's blanket except.
    outer_question = outer_template.replace("{ANSWER}", inner_answer)
    outer_chunks = chunks_for_query(outer_question + " " + inner_answer, chunks)
    if not outer_chunks:
        return None
    outer_prompt = build_prompt_fn(outer_question, outer_chunks)
    outer_result = await infer_fn(outer_prompt)
    outer_answer = str(getattr(outer_result, "response", outer_result) or "").strip()
    if _is_refusal(outer_answer):
        return None

    return outer_answer
