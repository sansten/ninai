"""Ninai streaming integration layer.

Two responsibilities:
  1. Server-side SSE generators — used by FastAPI streaming endpoints.
     Each generator wraps an existing gateway service method and yields
     SSE-formatted strings: progress frames, per-result frames, done/error.

  2. Client-side NinaiStreamingClient — used by external callers (LangChain,
     Google ADK, etc.) to consume the streaming gateway endpoints.
     Provides both async generators and sync wrappers.

SSE format:
  data: {"event": "<type>", "payload": {...}}\n\n

Event types:
  progress  — intermediate status update
  result    — a data item (step, memory, decision)
  error     — terminal error
  done      — stream complete
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING, Any, AsyncGenerator

if TYPE_CHECKING:
    from app.services.cognitive_gateway_service import CognitiveGatewayService


# ---------------------------------------------------------------------------
# SSE frame helpers
# ---------------------------------------------------------------------------

def _frame(event: str, payload: Any) -> str:
    return f"data: {json.dumps({'event': event, 'payload': payload})}\n\n"


def sse_progress(message: str, step: int | None = None) -> str:
    p: dict[str, Any] = {"message": message}
    if step is not None:
        p["step"] = step
    return _frame("progress", p)


def sse_result(data: dict) -> str:
    return _frame("result", data)


def sse_error(detail: str) -> str:
    return _frame("error", {"detail": detail})


def sse_done(summary: dict | None = None) -> str:
    return _frame("done", summary or {})


# ---------------------------------------------------------------------------
# Server-side async generators
# ---------------------------------------------------------------------------

async def stream_decide(
    gateway: "CognitiveGatewayService",
    content: str,
    enrichment: dict,
    context_id: str | None,
    org_id: str | None,
) -> AsyncGenerator[str, None]:
    """Yield SSE frames for the decide verb: progress → result → done."""
    yield sse_progress("running anomaly detection", step=1)
    try:
        result = await gateway.decide(
            content=content,
            enrichment=enrichment,
            context_id=context_id,
            org_id=org_id,
        )
        yield sse_progress("decision computed", step=2)
        yield sse_result({
            "decision": result.decision,
            "confidence": result.confidence,
            "tone": result.tone,
            "action_recommended": result.action_recommended,
            "enrichment": result.enrichment,
            "agents_run": result.agents_run,
            "context_id": context_id,
        })
        yield sse_done({"decision": result.decision, "confidence": result.confidence})
    except Exception as exc:
        yield sse_error(str(exc))
        yield sse_done()


async def stream_plan(
    gateway: "CognitiveGatewayService",
    goal: str,
    context: dict,
    context_id: str | None,
    org_id: str | None,
) -> AsyncGenerator[str, None]:
    """Yield SSE frames for the plan verb: one frame per step, then done."""
    yield sse_progress("decomposing goal", step=1)
    try:
        result = await gateway.plan(
            goal=goal,
            context=context,
            context_id=context_id,
            org_id=org_id,
        )
        yield sse_progress(f"{result.step_count} steps extracted", step=2)
        for i, step in enumerate(result.steps, 1):
            yield sse_result({"step_number": i, "step": step})
        yield sse_done({
            "goal": result.goal,
            "step_count": result.step_count,
            "blocking_step": result.blocking_step,
            "confidence": result.confidence,
            "context_id": context_id,
        })
    except Exception as exc:
        yield sse_error(str(exc))
        yield sse_done()


async def stream_read(
    gateway: "CognitiveGatewayService",
    query: str,
    memories: list[dict],
    limit: int,
    context_id: str | None,
    org_id: str | None,
) -> AsyncGenerator[str, None]:
    """Yield SSE frames for the read verb: one frame per memory, then done."""
    yield sse_progress("retrieving context", step=1)
    try:
        result = await gateway.read(
            query=query,
            memories=memories,
            limit=limit,
            context_id=context_id,
            org_id=org_id,
        )
        yield sse_progress(f"{result.total} memories retrieved", step=2)
        for i, mem in enumerate(result.memories, 1):
            yield sse_result({"rank": i, "memory": mem})
        yield sse_done({
            "total": result.total,
            "context_assembled": result.context_assembled,
            "context_id": context_id,
        })
    except Exception as exc:
        yield sse_error(str(exc))
        yield sse_done()


# ---------------------------------------------------------------------------
# Client-side: NinaiStreamingClient
# ---------------------------------------------------------------------------

class NinaiStreamingClient:
    """Async HTTP client for Ninai's streaming gateway endpoints.

    Provides async generators that yield parsed SSE event dicts, plus
    sync wrappers for non-async callers (e.g., LangChain sync chains).

    Usage (async):
        client = NinaiStreamingClient(
            base_url="http://ninai-api.internal",
            api_key="cap_ninai_...",
        )
        async for event in client.stream_gateway_decide("Database is down"):
            if event["event"] == "result":
                print(event["payload"]["decision"])

    Usage (sync):
        events = client.stream_gateway_decide_sync("Database is down")
        decision = next(e for e in events if e["event"] == "result")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }

    async def _stream(
        self, path: str, body: dict
    ) -> AsyncGenerator[dict, None]:
        import httpx
        url = f"{self._base_url}/api/v1{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", url, headers=self._headers(), json=body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            yield json.loads(line[6:])
                        except json.JSONDecodeError:
                            pass

    async def stream_gateway_decide(
        self,
        content: str,
        enrichment: dict | None = None,
        context_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream decide results from POST /cognitive/gateway/stream/decide."""
        async for event in self._stream(
            "/cognitive/gateway/stream/decide",
            {"content": content, "enrichment": enrichment or {}, "context_id": context_id},
        ):
            yield event

    async def stream_gateway_plan(
        self,
        goal: str,
        context: dict | None = None,
        context_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream plan results from POST /cognitive/gateway/stream/plan."""
        async for event in self._stream(
            "/cognitive/gateway/stream/plan",
            {"goal": goal, "context": context or {}, "context_id": context_id},
        ):
            yield event

    async def stream_gateway_read(
        self,
        query: str,
        limit: int = 10,
        context_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream read results from POST /cognitive/gateway/stream/read."""
        async for event in self._stream(
            "/cognitive/gateway/stream/read",
            {"query": query, "limit": limit, "context_id": context_id},
        ):
            yield event

    # ------------------------------------------------------------------
    # Sync wrappers (collect all events, block until stream is done)
    # ------------------------------------------------------------------

    def _collect_sync(self, coro_gen_factory) -> list[dict]:
        """Run an async generator synchronously and return all events."""
        results: list[dict] = []
        exc_holder: list[Exception] = []

        def _run():
            import asyncio

            async def _collect():
                async for item in coro_gen_factory():
                    results.append(item)

            try:
                asyncio.run(_collect())
            except Exception as exc:
                exc_holder.append(exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join()
        if exc_holder:
            raise exc_holder[0]
        return results

    def stream_gateway_decide_sync(
        self,
        content: str,
        enrichment: dict | None = None,
        context_id: str | None = None,
    ) -> list[dict]:
        return self._collect_sync(
            lambda: self.stream_gateway_decide(content, enrichment, context_id)
        )

    def stream_gateway_plan_sync(
        self,
        goal: str,
        context: dict | None = None,
        context_id: str | None = None,
    ) -> list[dict]:
        return self._collect_sync(
            lambda: self.stream_gateway_plan(goal, context, context_id)
        )

    def stream_gateway_read_sync(
        self,
        query: str,
        limit: int = 10,
        context_id: str | None = None,
    ) -> list[dict]:
        return self._collect_sync(
            lambda: self.stream_gateway_read(query, limit, context_id)
        )
