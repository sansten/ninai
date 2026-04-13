"""Tests for Google ADK adapter integration points that do not require google-adk."""

from integrations import google_adk_adapter as adk


class _DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _DummyClient:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None, json=None):
        self.calls.append(("POST", url, json))
        if url.endswith("/cognitive/gateway/read"):
            return _DummyResponse(
                {
                    "memories": [{"id": "m1", "content": "cached context"}],
                    "context_assembled": True,
                }
            )
        return _DummyResponse({"ok": True, "url": url, "payload": json})

    def get(self, url, headers=None):
        self.calls.append(("GET", url, None))
        return _DummyResponse({"ok": True, "url": url})


def test_adk_toolset_gateway_calls(monkeypatch):
    """Toolset should call Ninai gateway endpoints and return parsed JSON."""
    monkeypatch.setattr(adk.httpx, "Client", _DummyClient)

    toolset = adk.NinaiADKToolset(base_url="http://localhost:8002", api_key="test-token")

    write_result = toolset.ninai_write(content="hello", title="t", tags=["x"])
    read_result = toolset.ninai_read(query="hello", limit=2)
    explain_result = toolset.ninai_explain(memory_id="m1")

    assert write_result["ok"] is True
    assert read_result["context_assembled"] is True
    assert read_result["memories"][0]["id"] == "m1"
    assert explain_result["ok"] is True


def test_adk_session_hook_round_trip(monkeypatch):
    """Session hook should inject context before a turn and persist output after a turn."""
    monkeypatch.setattr(adk.httpx, "Client", _DummyClient)

    hook = adk.NinaiADKSessionHook(
        base_url="http://localhost:8002",
        api_key="test-token",
        context_id="ctx-1",
        recall_limit=3,
    )

    state = {"last_user_message": "what happened yesterday?"}
    updated = hook.before_turn(state)

    assert "ninai_context" in updated
    assert updated["ninai_context_assembled"] is True

    # Should not raise.
    hook.after_turn(updated, "Summary: issue was mitigated.")
