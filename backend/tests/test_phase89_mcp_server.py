"""Phase 89 — MCP server wiring tests.

Tests verify:
  1. Protocol handshake (initialize, tools/list).
  2. All 6 tools are exposed and have valid schemas.
  3. tools/call routes to the correct HTTP handler.
  4. HTTP failures are returned as error content, not crashes.
  5. Unknown tool raises ValueError.
  6. Unsupported method raises ValueError.
  7. Environment variables configure backend URL and API key.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import mcp_server as mcp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dispatch(method: str, params: dict | None = None) -> dict:
    return mcp._dispatch(method, params or {})


def _mock_post(return_value: dict):
    return patch("mcp_server._post", return_value=return_value)


def _mock_get(return_value: dict):
    return patch("mcp_server._get", return_value=return_value)


# ---------------------------------------------------------------------------
# Protocol handshake
# ---------------------------------------------------------------------------

class TestProtocolHandshake:
    def test_initialize_returns_protocol_version(self):
        result = _dispatch("initialize")
        assert result["protocolVersion"] == "2025-03-26"

    def test_initialize_returns_server_info(self):
        result = _dispatch("initialize")
        assert result["serverInfo"]["name"] == "ninai-mcp"

    def test_initialize_has_tools_capability(self):
        result = _dispatch("initialize")
        assert "tools" in result["capabilities"]

    def test_tools_list_returns_tools_key(self):
        result = _dispatch("tools/list")
        assert "tools" in result

    def test_tools_list_returns_6_tools(self):
        result = _dispatch("tools/list")
        assert len(result["tools"]) == 6

    def test_all_tools_have_name_and_schema(self):
        result = _dispatch("tools/list")
        for tool in result["tools"]:
            assert "name" in tool
            assert "inputSchema" in tool
            assert "type" in tool["inputSchema"]

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported method"):
            _dispatch("notifications/message")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_cognitive_decide_present(self):
        assert "cognitive.decide" in mcp._TOOL_NAMES

    def test_cognitive_plan_present(self):
        assert "cognitive.plan" in mcp._TOOL_NAMES

    def test_cognitive_read_present(self):
        assert "cognitive.read" in mcp._TOOL_NAMES

    def test_memory_ingest_present(self):
        assert "memory.ingest" in mcp._TOOL_NAMES

    def test_memory_search_present(self):
        assert "memory.search" in mcp._TOOL_NAMES

    def test_memory_consensus_submit_present(self):
        assert "memory.consensus.submit" in mcp._TOOL_NAMES

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            _dispatch("tools/call", {"name": "nonexistent.tool", "arguments": {}})


# ---------------------------------------------------------------------------
# cognitive.decide
# ---------------------------------------------------------------------------

class TestCognitiveDecide:
    def test_posts_to_gateway_decide(self):
        with _mock_post({"decision": "proceed"}) as mock_post:
            result = _dispatch("tools/call", {"name": "cognitive.decide", "arguments": {"query": "should we ship?"}})
        mock_post.assert_called_once()
        path = mock_post.call_args[0][0]
        assert "gateway" in path and "decide" in path

    def test_query_in_request_body(self):
        with _mock_post({}) as mock_post:
            _dispatch("tools/call", {"name": "cognitive.decide", "arguments": {"query": "test?"}})
        body = mock_post.call_args[0][1]
        assert body["query"] == "test?"

    def test_context_forwarded_when_present(self):
        ctx = {"user": "ts", "role": "admin"}
        with _mock_post({}) as mock_post:
            _dispatch("tools/call", {"name": "cognitive.decide", "arguments": {"query": "q", "context": ctx}})
        body = mock_post.call_args[0][1]
        assert body.get("context") == ctx

    def test_result_returned_as_text_content(self):
        with _mock_post({"decision": "go"}):
            result = _dispatch("tools/call", {"name": "cognitive.decide", "arguments": {"query": "go?"}})
        content = result["content"]
        assert content[0]["type"] == "text"
        parsed = json.loads(content[0]["text"])
        assert parsed["decision"] == "go"

    def test_backend_error_returned_not_raised(self):
        with _mock_post({"error": "500 Internal Server Error"}):
            result = _dispatch("tools/call", {"name": "cognitive.decide", "arguments": {"query": "q"}})
        parsed = json.loads(result["content"][0]["text"])
        assert "error" in parsed


# ---------------------------------------------------------------------------
# cognitive.plan
# ---------------------------------------------------------------------------

class TestCognitivePlan:
    def test_posts_to_gateway_plan(self):
        with _mock_post({"plan": []}) as mock_post:
            _dispatch("tools/call", {"name": "cognitive.plan", "arguments": {"goal": "ship v2"}})
        path = mock_post.call_args[0][0]
        assert "plan" in path

    def test_goal_in_body(self):
        with _mock_post({}) as mock_post:
            _dispatch("tools/call", {"name": "cognitive.plan", "arguments": {"goal": "launch"}})
        assert mock_post.call_args[0][1]["goal"] == "launch"


# ---------------------------------------------------------------------------
# cognitive.read
# ---------------------------------------------------------------------------

class TestCognitiveRead:
    def test_posts_to_gateway_read(self):
        with _mock_post({"memories": []}) as mock_post:
            _dispatch("tools/call", {"name": "cognitive.read", "arguments": {"query": "last meeting"}})
        path = mock_post.call_args[0][0]
        assert "read" in path

    def test_default_limit_is_10(self):
        with _mock_post({}) as mock_post:
            _dispatch("tools/call", {"name": "cognitive.read", "arguments": {"query": "q"}})
        assert mock_post.call_args[0][1]["limit"] == 10

    def test_custom_limit_forwarded(self):
        with _mock_post({}) as mock_post:
            _dispatch("tools/call", {"name": "cognitive.read", "arguments": {"query": "q", "limit": 25}})
        assert mock_post.call_args[0][1]["limit"] == 25


# ---------------------------------------------------------------------------
# memory.ingest
# ---------------------------------------------------------------------------

class TestMemoryIngest:
    def test_posts_to_memory_endpoint(self):
        with _mock_post({"id": "mem-1"}) as mock_post:
            _dispatch("tools/call", {"name": "memory.ingest", "arguments": {"content": "Alice joined the team"}})
        path = mock_post.call_args[0][0]
        assert "memory" in path.lower() or "mem" in path.lower()

    def test_content_in_body(self):
        with _mock_post({}) as mock_post:
            _dispatch("tools/call", {"name": "memory.ingest", "arguments": {"content": "fact"}})
        assert mock_post.call_args[0][1]["content"] == "fact"

    def test_source_forwarded(self):
        with _mock_post({}) as mock_post:
            _dispatch("tools/call", {"name": "memory.ingest", "arguments": {"content": "c", "source": "slack"}})
        assert mock_post.call_args[0][1].get("source") == "slack"


# ---------------------------------------------------------------------------
# memory.search
# ---------------------------------------------------------------------------

class TestMemorySearch:
    def test_posts_to_search_endpoint(self):
        with _mock_post({"results": []}) as mock_post:
            _dispatch("tools/call", {"name": "memory.search", "arguments": {"query": "project"}})
        path = mock_post.call_args[0][0]
        assert "search" in path

    def test_session_id_forwarded(self):
        with _mock_post({}) as mock_post:
            _dispatch("tools/call", {"name": "memory.search", "arguments": {"query": "q", "session_id": "sess-1"}})
        body = mock_post.call_args[0][1]
        assert body.get("session_id") == "sess-1"


# ---------------------------------------------------------------------------
# memory.consensus.submit
# ---------------------------------------------------------------------------

class TestMemoryConsensusSubmit:
    def test_posts_to_consensus_endpoint(self):
        with _mock_post({"claim_id": "c1"}) as mock_post:
            _dispatch("tools/call", {"name": "memory.consensus.submit", "arguments": {
                "content": "Alice is the CTO", "submitter": "agent-a"
            }})
        path = mock_post.call_args[0][0]
        assert "consensus" in path

    def test_default_confidence_is_07(self):
        with _mock_post({}) as mock_post:
            _dispatch("tools/call", {"name": "memory.consensus.submit", "arguments": {
                "content": "f", "submitter": "a"
            }})
        body = mock_post.call_args[0][1]
        assert body["confidence"] == 0.7

    def test_custom_confidence_forwarded(self):
        with _mock_post({}) as mock_post:
            _dispatch("tools/call", {"name": "memory.consensus.submit", "arguments": {
                "content": "f", "submitter": "a", "confidence": 0.95
            }})
        assert mock_post.call_args[0][1]["confidence"] == 0.95


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

class TestHttpHelper:
    def test_api_key_added_to_headers_when_set(self, monkeypatch):
        monkeypatch.setattr(mcp, "_API_KEY", "test-key-123")
        headers = mcp._headers()
        assert headers.get("Authorization") == "Bearer test-key-123"

    def test_no_authorization_header_when_key_empty(self, monkeypatch):
        monkeypatch.setattr(mcp, "_API_KEY", "")
        headers = mcp._headers()
        assert "Authorization" not in headers

    def test_backend_url_configurable(self, monkeypatch):
        monkeypatch.setattr(mcp, "_BACKEND_URL", "http://custom:9000")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"ok": true}'
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            result = mcp._post("/test", {})
        req = mock_urlopen.call_args[0][0]
        assert "custom:9000" in req.full_url
