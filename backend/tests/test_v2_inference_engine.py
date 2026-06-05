from __future__ import annotations

import pytest

from app.v2.llm.inference_engine import InferenceEngine, _parse_final_answer


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, responses: list[dict], calls: list[dict], timeout: float | None = None) -> None:
        self._responses = responses
        self._calls = calls
        self._timeout = timeout

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def post(self, url: str, json: dict) -> _FakeResponse:
        self._calls.append(json)
        if not self._responses:
            raise AssertionError("No fake responses remaining")
        return _FakeResponse(self._responses.pop(0))


def test_parse_final_answer_strips_marker() -> None:
    assert _parse_final_answer("Some reasoning\nFINAL ANSWER: Sweden") == "Sweden"


@pytest.mark.asyncio
async def test_infer_plain_retries_when_first_parse_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "<think>searching context</think>",
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "FINAL ANSWER: Sweden",
                    }
                }
            ]
        },
    ]
    calls: list[dict] = []

    def _client_factory(*args, **kwargs):
        return _FakeAsyncClient(responses, calls, timeout=kwargs.get("timeout"))

    monkeypatch.setattr("app.v2.llm.inference_engine.httpx.AsyncClient", _client_factory)

    engine = InferenceEngine(base_url="http://example.test", model="qwen2.5:14b")
    result = await engine.infer_plain("QUESTION: Where did Caroline move from 4 years ago?")

    assert result.response == "Sweden"
    assert result.error == ""
    assert len(calls) == 2
    assert "stop" in calls[0]
    assert "stop" not in calls[1]
