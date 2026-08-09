"""Query-focused distillation (map step) — verbatim extraction parsing."""

from unittest.mock import patch, MagicMock
import pytest

from app.v2.llm.inference_engine import InferenceEngine


class _FakeResp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _client_returning(*contents):
    """Build a fake httpx.AsyncClient whose .post returns the given contents in order."""
    calls = {"i": 0}

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            c = contents[min(calls["i"], len(contents) - 1)]
            calls["i"] += 1
            return _FakeResp(c)

    return MagicMock(return_value=_FakeClient())


@pytest.mark.asyncio
async def test_distill_extracts_verbatim_lines():
    eng = InferenceEngine(base_url="http://x", model="m")
    with patch("app.v2.llm.inference_engine.httpx.AsyncClient",
               _client_returning("- John started surfing in 2018\n- John surfs at dawn")):
        facts = await eng.distill_evidence("When did John start surfing?",
                                           ["John: I started surfing 5 years ago"])
    assert facts == ["John started surfing in 2018", "John surfs at dawn"]


@pytest.mark.asyncio
async def test_distill_none_yields_empty():
    eng = InferenceEngine(base_url="http://x", model="m")
    with patch("app.v2.llm.inference_engine.httpx.AsyncClient", _client_returning("NONE")):
        facts = await eng.distill_evidence("irrelevant?", ["chatter", "more chatter"])
    assert facts == []


@pytest.mark.asyncio
async def test_distill_empty_input_no_call():
    eng = InferenceEngine(base_url="http://x", model="m")
    # No HTTP patching: must not make a call for empty input.
    assert await eng.distill_evidence("q", []) == []


@pytest.mark.asyncio
async def test_distill_batches_large_input():
    eng = InferenceEngine(base_url="http://x", model="m")
    items = [f"line {i}" for i in range(20)]
    # batch_size=12 → 2 batches; each returns one fact.
    with patch("app.v2.llm.inference_engine.httpx.AsyncClient",
               _client_returning("- fact A", "- fact B")):
        facts = await eng.distill_evidence("q", items, batch_size=12, max_items=24)
    assert facts == ["fact A", "fact B"]
