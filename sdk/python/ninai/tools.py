"""SDK tool invocation helpers.

The backend runs the authoritative PolicyGuard + ToolInvoker flow.
This module provides an ergonomic async wrapper for notebooks and the book.

Example:
    from ninai.tools import ToolInvoker

    invoker = ToolInvoker()
    res = await invoker.invoke("memory.search", params={"query": "refund"})
    if res.warnings:
        print(res.warnings)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from ninai.client import NinaiClient
from ninai.models import ToolInvocationResult
from ninai.observability import ToolEventSink, emit_event


async def _to_thread(func, /, *args, **kwargs):
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except AttributeError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


class ToolInvoker:
    def __init__(self, client: NinaiClient | None = None, *, tool_event_sink: ToolEventSink | None = None):
        self.client = client or NinaiClient()
        self.tool_event_sink = tool_event_sink

    async def invoke(
        self,
        tool_name: str,
        *,
        params: Dict[str, Any] | None = None,
        session_id: str | None = None,
        iteration_id: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
        classification: str | None = None,
        justification: str | None = None,
        trace_id: str | None = None,
        tool_event_sink: ToolEventSink | None = None,
    ) -> ToolInvocationResult:
        sink = tool_event_sink or self.tool_event_sink
        t0 = time.perf_counter()
        await emit_event(
            sink,
            {
                "type": "tool.invoke",
                "name": tool_name,
                "trace_id": trace_id,
                "phase": "start",
            },
        )

        try:
            res = await _to_thread(
                self.client.tools.invoke,
                tool_name=tool_name,
                tool_input=params,
                session_id=session_id,
                iteration_id=iteration_id,
                scope=scope,
                scope_id=scope_id,
                classification=classification,
                justification=justification,
                trace_id=trace_id,
            )
            await emit_event(
                sink,
                {
                    "type": "tool.invoke",
                    "name": tool_name,
                    "trace_id": trace_id,
                    "phase": "end",
                    "ok": True,
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    "status": getattr(res, "status", None),
                    "success": getattr(res, "success", None),
                    "warnings": getattr(res, "warnings", None),
                },
            )
            return res
        except Exception as e:
            await emit_event(
                sink,
                {
                    "type": "tool.invoke",
                    "name": tool_name,
                    "trace_id": trace_id,
                    "phase": "end",
                    "ok": False,
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            raise
