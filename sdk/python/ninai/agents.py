"""High-level agent helpers for the Ninai SDK.

The Practical Ninai book treats agents as deterministic wrappers around:
- an LLM adapter (e.g. ``client.llm``)
- optional observability sinks

These agents are async-first so they compose naturally in notebooks/services.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ninai.models import GoalLinkingAgentOutput, GoalPlannerAgentOutput
from ninai.observability import ToolEventSink, emit_event


async def _to_thread(func, /, *args, **kwargs):
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except AttributeError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def _call_maybe_sync(func, /, *args, **kwargs):
    """Call an API that might be sync (runs in thread) or async (awaits)."""
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await _to_thread(func, *args, **kwargs)


def _extract_llm_data(resp: Any) -> Dict[str, Any]:
    if resp is None:
        return {}
    # LLMResource.complete_json returns a model with `data`.
    data = getattr(resp, "data", None)
    if isinstance(data, dict):
        return data
    if isinstance(resp, dict):
        # tolerate raw dict or {"data": {...}}
        inner = resp.get("data")
        return inner if isinstance(inner, dict) else resp
    return {}


class GoalPlannerAgent:
    """Propose a goal graph from a user request."""

    def __init__(self, *, llm_client: Any, tool_event_sink: ToolEventSink | None = None):
        self.llm_client = llm_client
        self.tool_event_sink = tool_event_sink

    def _fail_closed(self) -> GoalPlannerAgentOutput:
        return GoalPlannerAgentOutput(create_goal=False, confidence=0.0)

    async def propose_goal(
        self,
        *,
        user_request: str,
        session_context: Optional[Dict[str, Any]] = None,
        existing_goals: Optional[List[Dict[str, Any]]] = None,
        trace_id: str | None = None,
        tool_event_sink: ToolEventSink | None = None,
    ) -> GoalPlannerAgentOutput:
        cleaned = (user_request or "").strip()
        if not cleaned:
            return self._fail_closed()

        schema_hint: Dict[str, Any] = {
            "create_goal": "bool",
            "goal": {
                "title": "string",
                "description": "string|null",
                "goal_type": "task|project|objective|policy|research",
                "visibility_scope": "personal|team|department|division|organization",
                "priority": "int 0-5",
            },
            "nodes": [
                {
                    "temp_id": "string",
                    "node_type": "subgoal|task|milestone",
                    "title": "string",
                }
            ],
            "edges": [
                {
                    "from_temp_id": "string",
                    "to_temp_id": "string",
                    "edge_type": "depends_on|blocks|related_to",
                }
            ],
            "confidence": "float 0-1",
        }

        prompt = (
            "You are a goal planner. Produce strict JSON only.\n\n"
            f"User request:\n{cleaned}\n\n"
            f"Session context (json):\n{session_context or {}}\n\n"
            f"Existing goals (json):\n{existing_goals or []}\n"
        )

        sink = tool_event_sink or self.tool_event_sink
        t0 = time.perf_counter()
        try:
            resp = await _call_maybe_sync(self.llm_client.complete_json, prompt=prompt, schema_hint=schema_hint)
            data = _extract_llm_data(resp)
            out = GoalPlannerAgentOutput(**(data or {}))
            await emit_event(
                sink,
                {
                    "type": "llm.complete_json",
                    "name": "GoalPlannerAgent.propose_goal",
                    "trace_id": trace_id,
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    "ok": True,
                },
            )
            return out
        except Exception as e:
            await emit_event(
                sink,
                {
                    "type": "llm.complete_json",
                    "name": "GoalPlannerAgent.propose_goal",
                    "trace_id": trace_id,
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            return self._fail_closed()


class GoalLinkingAgent:
    """Suggest links between a memory and active goals."""

    def __init__(self, *, llm_client: Any, tool_event_sink: ToolEventSink | None = None):
        self.llm_client = llm_client
        self.tool_event_sink = tool_event_sink

    def _fail_closed(self) -> GoalLinkingAgentOutput:
        return GoalLinkingAgentOutput(links=[], confidence=0.0)

    async def suggest_links(
        self,
        *,
        memory: Dict[str, Any],
        active_goals: List[Dict[str, Any]],
        trace_id: str | None = None,
        tool_event_sink: ToolEventSink | None = None,
    ) -> GoalLinkingAgentOutput:
        if not memory or not active_goals:
            return self._fail_closed()

        schema_hint: Dict[str, Any] = {
            "links": [
                {
                    "goal_id": "string",
                    "node_id": "string|null",
                    "memory_id": "string",
                    "link_type": "evidence|progress|blocker|reference",
                    "confidence": "float 0-1",
                    "reason": "string|null",
                }
            ],
            "confidence": "float 0-1",
        }

        prompt = (
            "You are a goal linking assistant. Produce strict JSON only.\n\n"
            f"Memory (json):\n{memory}\n\n"
            f"Active goals (json):\n{active_goals}\n"
        )

        sink = tool_event_sink or self.tool_event_sink
        t0 = time.perf_counter()
        try:
            resp = await _call_maybe_sync(self.llm_client.complete_json, prompt=prompt, schema_hint=schema_hint)
            data = _extract_llm_data(resp)
            out = GoalLinkingAgentOutput(**(data or {}))
            await emit_event(
                sink,
                {
                    "type": "llm.complete_json",
                    "name": "GoalLinkingAgent.suggest_links",
                    "trace_id": trace_id,
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    "ok": True,
                },
            )
            return out
        except Exception as e:
            await emit_event(
                sink,
                {
                    "type": "llm.complete_json",
                    "name": "GoalLinkingAgent.suggest_links",
                    "trace_id": trace_id,
                    "duration_ms": (time.perf_counter() - t0) * 1000.0,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            return self._fail_closed()


class MetaAgent:
    """Meta review primitives (server-side)."""

    def __init__(self, client: "NinaiClient"):
        self._client = client

    async def review_memory(self, memory_id: str, *, trace_id: str | None = None) -> Dict[str, Any]:
        def _call() -> Dict[str, Any]:
            headers = self._client._get_headers()
            if trace_id:
                headers = dict(headers)
                headers["X-Trace-ID"] = trace_id

            resp = self._client._client.post(
                f"/meta/review/memories/{memory_id}",
                headers=headers,
            )
            self._client._handle_response_errors(resp)
            return resp.json()

        return await _to_thread(_call)

    async def review_cognitive_session(self, session_id: str, *, trace_id: str | None = None) -> Dict[str, Any]:
        def _call() -> Dict[str, Any]:
            headers = self._client._get_headers()
            if trace_id:
                headers = dict(headers)
                headers["X-Trace-ID"] = trace_id

            resp = self._client._client.post(
                f"/meta/review/cognitive-sessions/{session_id}",
                headers=headers,
            )
            self._client._handle_response_errors(resp)
            return resp.json()

        return await _to_thread(_call)


if TYPE_CHECKING:
    from ninai.client import NinaiClient
