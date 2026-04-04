"""Google ADK adapter for Ninai Cognitive OS.

Provides four integration points:

1. NinaiADKToolset
   Returns FunctionTool objects for all 5 gateway verbs.
   Drop them into any ADK Agent's tools list.

2. NinaiADKSessionHook
   before_turn(): restores Ninai memory context into ADK session state.
   after_turn():  persists the turn output to Ninai as a memory.

3. NinaiADKEventBridge
   Background async task that subscribes to Ninai's /ws/stream WebSocket
   and routes high-severity events to an ADK agent runner.

4. NinaiADKAgent
   Convenience wrapper: ADK Agent pre-configured with toolset + session hook.

Usage:
    from integrations.google_adk_adapter import NinaiADKAgent

    agent = NinaiADKAgent(
        base_url="http://ninai-api.internal",
        api_key="cap_ninai_...",
        model="gemini-2.0-flash",
    )
    # agent.agent is a fully configured google.adk.agents.Agent
    # agent.get_hook() returns the NinaiADKSessionHook for runner integration
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

try:
    from google.adk.agents import Agent as _ADKAgent
    from google.adk.tools import FunctionTool as _FunctionTool
    ADK_AVAILABLE = True
    _ADK_IMPORT_ERROR: Exception | None = None
except Exception as _e:
    ADK_AVAILABLE = False
    _ADKAgent = object  # type: ignore[assignment,misc]
    _FunctionTool = object  # type: ignore[assignment,misc]
    _ADK_IMPORT_ERROR = _e


# ---------------------------------------------------------------------------
# 1. NinaiADKToolset
# ---------------------------------------------------------------------------

class NinaiADKToolset:
    """Returns FunctionTool objects wrapping all 5 Ninai gateway verbs.

    Each tool calls the Ninai REST API synchronously (ADK tools are
    invoked synchronously by the ADK runner).
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(
                f"{self._base_url}/api/v1{path}",
                headers=self._headers,
                json=body,
            )
            r.raise_for_status()
            return r.json()

    def ninai_write(self, content: str, title: str = "", tags: list | None = None) -> dict:
        """Store a memory in the Ninai enterprise knowledge base with enrichment."""
        return self._post("/cognitive/gateway/write", {
            "content": content,
            "title": title,
            "tags": tags or [],
        })

    def ninai_read(self, query: str, limit: int = 5) -> dict:
        """Retrieve the most relevant enterprise memories for a query."""
        return self._post("/cognitive/gateway/read", {
            "query": query,
            "limit": limit,
        })

    def ninai_decide(self, content: str) -> dict:
        """Run Ninai anomaly detection and get a decision: escalate / investigate / monitor / acknowledge."""
        return self._post("/cognitive/gateway/decide", {"content": content})

    def ninai_plan(self, goal: str) -> dict:
        """Decompose a goal into ordered, actionable steps using Ninai's GoalDecompositionAgent."""
        return self._post("/cognitive/gateway/plan", {"goal": goal})

    def ninai_explain(self, memory_id: str) -> dict:
        """Return the audit trail and explainability summary for a Ninai memory."""
        with httpx.Client(timeout=self._timeout) as client:
            r = client.get(
                f"{self._base_url}/api/v1/cognitive/gateway/explain/{memory_id}",
                headers=self._headers,
            )
            r.raise_for_status()
            return r.json()

    def get_tools(self) -> list:
        """Return a list of FunctionTool objects for all 5 gateway verbs."""
        if not ADK_AVAILABLE:
            raise ImportError(
                "google-adk is not installed. Install it with: pip install google-adk"
            ) from _ADK_IMPORT_ERROR
        return [
            _FunctionTool(func=self.ninai_write),
            _FunctionTool(func=self.ninai_read),
            _FunctionTool(func=self.ninai_decide),
            _FunctionTool(func=self.ninai_plan),
            _FunctionTool(func=self.ninai_explain),
        ]


# ---------------------------------------------------------------------------
# 2. NinaiADKSessionHook
# ---------------------------------------------------------------------------

class NinaiADKSessionHook:
    """ADK session lifecycle hook that bridges Ninai memory to ADK session state.

    Register with your ADK runner:
        hook = agent_wrapper.get_hook()
        session_state = hook.before_turn(session_state)
        # ... run turn ...
        await hook.after_turn(session_state, turn_output)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        context_id: str | None = None,
        recall_limit: int = 5,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._context_id = context_id
        self._recall_limit = recall_limit
        self._timeout = timeout

    def before_turn(self, session_state: dict, query: str | None = None) -> dict:
        """Load relevant Ninai memories into the ADK session state.

        Adds a 'ninai_context' key to session_state containing the top-K
        memories most relevant to the current query (or recent generic context).
        """
        search_query = query or session_state.get("last_user_message", "context")
        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.post(
                    f"{self._base_url}/api/v1/cognitive/gateway/read",
                    headers=self._headers,
                    json={
                        "query": search_query,
                        "limit": self._recall_limit,
                        "context_id": self._context_id,
                    },
                )
                r.raise_for_status()
                data = r.json()
                session_state["ninai_context"] = data.get("memories", [])
                session_state["ninai_context_assembled"] = data.get("context_assembled", False)
        except Exception as exc:
            logger.warning("NinaiADKSessionHook.before_turn failed: %s", exc)
            session_state.setdefault("ninai_context", [])
        return session_state

    def after_turn(self, session_state: dict, output: str) -> None:
        """Persist the turn output to Ninai as a memory.

        The memory is tagged with 'adk' and 'session' for later retrieval.
        """
        if not output or not output.strip():
            return
        try:
            with httpx.Client(timeout=self._timeout) as client:
                client.post(
                    f"{self._base_url}/api/v1/cognitive/gateway/write",
                    headers=self._headers,
                    json={
                        "content": output,
                        "title": "ADK agent turn output",
                        "tags": ["adk", "session", "agent-output"],
                        "context_id": self._context_id,
                    },
                )
        except Exception as exc:
            logger.warning("NinaiADKSessionHook.after_turn failed: %s", exc)


# ---------------------------------------------------------------------------
# 3. NinaiADKEventBridge
# ---------------------------------------------------------------------------

class NinaiADKEventBridge:
    """Background async task: subscribes to Ninai's /ws/stream and routes
    high-severity events to an ADK agent runner.

    Usage:
        bridge = NinaiADKEventBridge(
            ws_base_url="ws://ninai-api.internal",
            token="<jwt>",
            agent_runner=runner,   # ADK Runner instance
            user_id="system",
            session_id_prefix="event-bridge",
        )
        asyncio.create_task(bridge.run())
    """

    def __init__(
        self,
        ws_base_url: str,
        token: str,
        agent_runner: Any,
        user_id: str = "system",
        session_id_prefix: str = "ninai-event",
        high_severity_types: tuple[str, ...] = ("memory.anomaly", "memory.conflict"),
    ) -> None:
        self._ws_url = f"{ws_base_url.rstrip('/')}/api/v1/ws/stream?token={token}"
        self._runner = agent_runner
        self._user_id = user_id
        self._prefix = session_id_prefix
        self._high_severity_types = high_severity_types

    async def run(self) -> None:
        """Connect to Ninai WebSocket and forward events indefinitely.

        Reconnects on disconnect with exponential backoff up to 60s.
        """
        import asyncio

        try:
            import websockets  # type: ignore[import]
        except ImportError:
            logger.error(
                "NinaiADKEventBridge requires 'websockets'. "
                "Install with: pip install websockets"
            )
            return

        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self._ws_url) as ws:
                    backoff = 1.0
                    async for message in ws:
                        try:
                            event = json.loads(message)
                            if event.get("type") == "ping":
                                continue
                            await self._route_event(event)
                        except Exception as exc:
                            logger.warning("Event routing error: %s", exc)
            except Exception as exc:
                logger.warning("WebSocket disconnected (%s), reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _route_event(self, event: dict) -> None:
        event_type = event.get("event_type", "")
        if event_type not in self._high_severity_types:
            return

        payload = event.get("payload") or {}
        message = (
            f"Ninai event [{event_type}]: "
            f"{payload.get('summary') or payload.get('content') or json.dumps(payload)[:200]}"
        )

        import uuid as _uuid
        session_id = f"{self._prefix}-{_uuid.uuid4()}"

        try:
            if hasattr(self._runner, "run_async"):
                await self._runner.run_async(
                    user_id=self._user_id,
                    session_id=session_id,
                    new_message=message,
                )
            else:
                # Fallback: call synchronous run in thread
                import asyncio
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._runner.run(
                        user_id=self._user_id,
                        session_id=session_id,
                        new_message=message,
                    ),
                )
        except Exception as exc:
            logger.warning("ADK runner invocation failed for event %s: %s", event_type, exc)


# ---------------------------------------------------------------------------
# 4. NinaiADKAgent
# ---------------------------------------------------------------------------

class NinaiADKAgent:
    """Convenience wrapper: ADK Agent pre-configured with Ninai toolset + session hook.

    Usage:
        wrapper = NinaiADKAgent(
            base_url="http://ninai-api.internal",
            api_key="cap_ninai_...",
            model="gemini-2.0-flash",
            instruction="You are an enterprise incident response agent.",
        )
        # Use wrapper.agent as a standard ADK Agent
        # Use wrapper.get_hook() to register before/after_turn callbacks
        hook = wrapper.get_hook()
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gemini-2.0-flash",
        name: str = "ninai-cognitive-agent",
        instruction: str = (
            "You are a Ninai cognitive agent with access to an enterprise memory OS. "
            "Use ninai_read to recall context, ninai_decide to evaluate anomalies, "
            "ninai_plan to decompose goals, and ninai_write to store important findings."
        ),
        context_id: str | None = None,
    ) -> None:
        if not ADK_AVAILABLE:
            raise ImportError(
                "google-adk is not installed. Install it with: pip install google-adk"
            ) from _ADK_IMPORT_ERROR

        self._toolset = NinaiADKToolset(base_url, api_key)
        self._hook = NinaiADKSessionHook(base_url, api_key, context_id=context_id)

        self.agent = _ADKAgent(
            name=name,
            model=model,
            instruction=instruction,
            tools=self._toolset.get_tools(),
        )

    def get_hook(self) -> NinaiADKSessionHook:
        """Return the session hook for before/after_turn integration."""
        return self._hook

    def get_toolset(self) -> NinaiADKToolset:
        """Return the toolset for direct tool access."""
        return self._toolset
