"""Unit tests for sequential multi-hop decomposition (NINAI_MULTIHOP_DECOMPOSE)."""
from __future__ import annotations

import pytest

from app.v2.llm.multihop_decompose import (
    chunks_for_query,
    decompose_and_compose,
    split_nested_question,
)


def _chunk(text: str) -> dict:
    return {"id": text[:12], "payload": {"text": text}}


class _Result:
    def __init__(self, response: str) -> None:
        self.response = response


class TestSplitNestedQuestion:
    def test_splits_who_clause(self):
        split = split_nested_question(
            "Who reported to the person who approved the Q3 budget?"
        )
        assert split is not None
        inner, outer_template = split
        assert inner == "Who approved the Q3 budget?"
        assert "{ANSWER}" in outer_template

    def test_returns_none_for_simple_question(self):
        assert split_nested_question("What is the capital of France?") is None

    def test_returns_none_when_clause_too_short(self):
        assert split_nested_question("Who is who?") is None

    def test_that_and_which_connectors_not_supported(self):
        """Regression: "that"/"which" clauses don't map to a real standalone
        question ("That approved the Q3 budget?" isn't valid English), so
        they must not attempt a split."""
        assert split_nested_question(
            "What was the project that Alice approved?"
        ) is None
        assert split_nested_question(
            "What is the budget which Alice approved?"
        ) is None


class TestChunksForQuery:
    def test_filters_by_lexical_overlap(self):
        chunks = [
            _chunk("Alice approved the Q3 budget last week."),
            _chunk("Bob likes coffee in the morning."),
        ]
        result = chunks_for_query("who approved the budget", chunks)
        assert result[0]["payload"]["text"].startswith("Alice")

    def test_falls_back_to_all_chunks_when_no_overlap(self):
        chunks = [_chunk("Something unrelated.")]
        result = chunks_for_query("zzzzz nomatch", chunks)
        assert result == chunks


class TestDecomposeAndCompose:
    @pytest.mark.asyncio
    async def test_composes_two_hop_answer(self):
        chunks = [
            _chunk("Alice approved the Q3 budget."),
            _chunk("Bob reported to Alice on the engineering team."),
        ]

        async def infer_fn(prompt: str):
            if prompt.lower().startswith("who approved"):
                return _Result("Alice")
            return _Result("Bob")

        def build_prompt(sub_q: str, sub_chunks: list[dict]) -> str:
            return sub_q + "\n" + "\n".join(c["payload"]["text"] for c in sub_chunks)

        answer = await decompose_and_compose(
            "Who reported to the person who approved the Q3 budget?",
            chunks,
            infer_fn,
            build_prompt,
        )
        assert answer == "Bob"

    @pytest.mark.asyncio
    async def test_returns_none_when_split_fails(self):
        async def infer_fn(prompt: str):
            return _Result("anything")

        def build_prompt(sub_q: str, sub_chunks: list[dict]) -> str:
            return sub_q

        answer = await decompose_and_compose(
            "What is the capital of France?", [], infer_fn, build_prompt
        )
        assert answer is None

    @pytest.mark.asyncio
    async def test_returns_none_when_inner_hop_refuses(self):
        chunks = [_chunk("Nothing relevant here.")]

        async def infer_fn(prompt: str):
            return _Result("Not mentioned in the context.")

        def build_prompt(sub_q: str, sub_chunks: list[dict]) -> str:
            return sub_q

        answer = await decompose_and_compose(
            "Who reported to the person who approved the Q3 budget?",
            chunks,
            infer_fn,
            build_prompt,
        )
        assert answer is None

    @pytest.mark.asyncio
    async def test_literal_braces_in_question_do_not_raise(self):
        """Regression: outer_template used str.format(), which raised on a
        question containing literal "{"/"}" (e.g. quoted JSON), silently
        disabling decomposition via the caller's broad except. .replace()
        handles it without a crash."""
        chunks = [
            _chunk("Alice approved the {Q3} budget."),
            _chunk("Bob reported to Alice on the engineering team."),
        ]

        async def infer_fn(prompt: str):
            if prompt.lower().startswith("who approved"):
                return _Result("Alice")
            return _Result("Bob")

        def build_prompt(sub_q: str, sub_chunks: list[dict]) -> str:
            return sub_q + "\n" + "\n".join(c["payload"]["text"] for c in sub_chunks)

        answer = await decompose_and_compose(
            "Who reported to the person who approved the {Q3} budget?",
            chunks,
            infer_fn,
            build_prompt,
        )
        assert answer == "Bob"

    @pytest.mark.asyncio
    async def test_returns_none_when_outer_hop_refuses(self):
        chunks = [
            _chunk("Alice approved the Q3 budget."),
            _chunk("No one else is mentioned."),
        ]

        async def infer_fn(prompt: str):
            if prompt.lower().startswith("who approved"):
                return _Result("Alice")
            return _Result("I don't know")

        def build_prompt(sub_q: str, sub_chunks: list[dict]) -> str:
            return sub_q + "\n" + "\n".join(c["payload"]["text"] for c in sub_chunks)

        answer = await decompose_and_compose(
            "Who reported to the person who approved the Q3 budget?",
            chunks,
            infer_fn,
            build_prompt,
        )
        assert answer is None
