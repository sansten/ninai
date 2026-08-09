"""Ninai MCP server (stdio JSON-RPC).

Implements the Model Context Protocol 2025-03-26 for tool discovery and
invocation. Wires each tool to the Ninai backend HTTP API.

Backend URL is read from NINAI_BACKEND_URL (default http://localhost:8000).
API key / Bearer token from NINAI_API_KEY (passed as Authorization header).

Transport: stdio JSON-RPC (works with Claude Desktop, Cursor, Continue, etc.).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from typing import Any

_BACKEND_URL = os.environ.get("NINAI_BACKEND_URL", "http://localhost:8000").rstrip("/")
_API_KEY = os.environ.get("NINAI_API_KEY", "")
_TIMEOUT = float(os.environ.get("NINAI_MCP_TIMEOUT", "15"))

TOOLS = [
    {
        "name": "cognitive.decide",
        "description": "Run a decision cycle against the tenant's cognitive memory context",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The decision question or situation"},
                "context": {"type": "object", "description": "Optional additional context"},
                "context_id": {"type": "string", "description": "Chain decisions with the same context_id"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "cognitive.plan",
        "description": "Generate a bounded cognitive plan for a goal using memory context",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The goal to plan for"},
                "context_id": {"type": "string", "description": "Optional context chain ID"},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "cognitive.read",
        "description": "Read relevant memory context for a query",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in memory"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory.ingest",
        "description": "Write a new memory into the cognitive store",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory content to store"},
                "session_id": {"type": "string", "description": "Session scope for the memory"},
                "source": {"type": "string", "description": "Origin label (e.g. 'slack', 'email', 'mcp')"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory.search",
        "description": "Search the memory store semantically",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "session_id": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory.consensus.submit",
        "description": "Submit a fact to the consensus-gated shared memory pool (Phase 85)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The fact to submit for consensus"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "submitter": {"type": "string", "description": "Agent or user identifier"},
            },
            "required": ["content", "submitter"],
        },
    },
]

_TOOL_NAMES = {t["name"] for t in TOOLS}


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if _API_KEY:
        h["Authorization"] = f"Bearer {_API_KEY}"
    return h


def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    url = f"{_BACKEND_URL}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            return {"error": json.loads(raw)}
        except Exception:
            return {"error": raw, "http_status": exc.code}
    except Exception as exc:
        return {"error": str(exc)}


def _get(path: str) -> dict[str, Any]:
    url = f"{_BACKEND_URL}{path}"
    req = urllib.request.Request(url, headers=_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _call_cognitive_decide(args: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {"query": str(args.get("query", ""))}
    if args.get("context"):
        body["context"] = args["context"]
    if args.get("context_id"):
        body["context_id"] = str(args["context_id"])
    return _post("/api/v1/cognitive/gateway/decide", body)


def _call_cognitive_plan(args: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {"goal": str(args.get("goal", ""))}
    if args.get("context_id"):
        body["context_id"] = str(args["context_id"])
    return _post("/api/v1/cognitive/gateway/plan", body)


def _call_cognitive_read(args: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": str(args.get("query", "")),
        "limit": int(args.get("limit", 10)),
    }
    return _post("/api/v1/cognitive/gateway/read", body)


def _call_memory_ingest(args: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {"content": str(args.get("content", ""))}
    if args.get("session_id"):
        body["session_id"] = str(args["session_id"])
    if args.get("source"):
        body["source"] = str(args["source"])
    return _post("/api/v1/memory/", body)


def _call_memory_search(args: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": str(args.get("query", "")),
        "limit": int(args.get("limit", 10)),
    }
    if args.get("session_id"):
        body["session_id"] = str(args["session_id"])
    return _post("/api/v1/memory/search", body)


def _call_memory_consensus_submit(args: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "content": str(args.get("content", "")),
        "submitter": str(args.get("submitter", "")),
        "confidence": float(args.get("confidence", 0.7)),
    }
    return _post("/api/v1/consensus/claims", body)


_TOOL_HANDLERS = {
    "cognitive.decide": _call_cognitive_decide,
    "cognitive.plan": _call_cognitive_plan,
    "cognitive.read": _call_cognitive_read,
    "memory.ingest": _call_memory_ingest,
    "memory.search": _call_memory_search,
    "memory.consensus.submit": _call_memory_consensus_submit,
}


# ---------------------------------------------------------------------------
# JSON-RPC dispatch
# ---------------------------------------------------------------------------

def _ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _err(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "initialize":
        return {
            "protocolVersion": "2025-03-26",
            "serverInfo": {"name": "ninai-mcp", "version": "0.2.0"},
            "capabilities": {
                "tools": {"listChanged": False},
            },
        }

    if method == "tools/list":
        return {"tools": TOOLS}

    if method == "tools/call":
        tool_name = str(params.get("name") or "")
        arguments = dict(params.get("arguments") or {})
        if tool_name not in _TOOL_NAMES:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = _TOOL_HANDLERS[tool_name]
        backend_result = handler(arguments)

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(backend_result, default=str),
                }
            ]
        }

    raise ValueError(f"Unsupported method: {method}")


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            request_id = request.get("id")
            method = str(request.get("method") or "")
            params = request.get("params") or {}
            result = _dispatch(method, params)
            response = _ok(request_id, result)
        except Exception as exc:
            request_id = None
            try:
                request_id = request.get("id") if "request" in locals() else None
            except Exception:
                pass
            response = _err(request_id, -32601, str(exc))

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
