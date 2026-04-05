"""Ninai MCP server (stdio JSON-RPC).

Implements a minimal Model Context Protocol compatible surface for tool discovery
and invocation. This is intentionally dependency-light and can be expanded to
full MCP SDK integration later.
"""

from __future__ import annotations

import json
import sys
from typing import Any


TOOLS = [
    {
        "name": "cognitive.decide",
        "description": "Run a decision cycle against current cognitive context",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "context": {"type": "object"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "cognitive.plan",
        "description": "Generate a bounded cognitive plan for a goal",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "cognitive.read",
        "description": "Read relevant cognitive memory context",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["query"],
        },
    },
]


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
            "serverInfo": {"name": "ninai-mcp", "version": "0.1.0"},
            "capabilities": {
                "tools": {"listChanged": False},
            },
        }

    if method == "tools/list":
        return {"tools": TOOLS}

    if method == "tools/call":
        tool_name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if tool_name not in {t["name"] for t in TOOLS}:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Placeholder response for now; can be wired to backend HTTP services.
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "tool": tool_name,
                            "status": "ok",
                            "arguments": arguments,
                        }
                    ),
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
            response = _err(request.get("id") if "request" in locals() else None, -32601, str(exc))

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
