"""NinaiSSEStream — consume Ninai's Server-Sent Events stream.

Connects to ``/sse/events`` using ``httpx`` (already a dependency — no extras
needed).  Suitable for environments where WebSockets are blocked or for
simpler read-only event consumption.

Example::

    from ninai import NinaiClient
    from ninai.streaming.sse import NinaiSSEStream

    client = NinaiClient(api_key="nai_...")

    async for event in NinaiSSEStream(client):
        print(event)

    # Filter by event type
    async for event in NinaiSSEStream(client, event_types=["memory.created"]):
        handle(event)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, AsyncIterator, List, Optional

import httpx

if TYPE_CHECKING:
    from ninai.client import NinaiClient

logger = logging.getLogger(__name__)


class NinaiSSEStream:
    """Async iterator that yields events from Ninai's ``/sse/events`` endpoint.

    Uses ``httpx`` streaming — no additional dependencies.

    Args:
        client: An authenticated :class:`~ninai.NinaiClient` instance.
        event_types: Optional list of event type filters passed as query params.
        goal_id: Stream events for a specific goal (uses ``/sse/goals/{goal_id}``).
        session_id: Stream events for a specific session
            (uses ``/sse/session/{session_id}``).
        timeout: HTTP read timeout in seconds (default 300 — long-poll friendly).
    """

    def __init__(
        self,
        client: "NinaiClient",
        *,
        event_types: Optional[List[str]] = None,
        goal_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        self._client = client
        self._event_types = event_types
        self._goal_id = goal_id
        self._session_id = session_id
        self._timeout = timeout

    def _build_url_and_params(self) -> tuple[str, dict]:
        base = self._client.base_url.rstrip("/")
        if self._goal_id:
            url = f"{base}/sse/goals/{self._goal_id}"
            params: dict = {}
        elif self._session_id:
            url = f"{base}/sse/session/{self._session_id}"
            params = {}
        else:
            url = f"{base}/sse/events"
            params = {}
        if self._event_types:
            params["event_types"] = ",".join(self._event_types)
        return url, params

    def __aiter__(self) -> AsyncIterator[dict]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[dict]:
        url, params = self._build_url_and_params()
        headers = self._client._get_headers()
        headers["Accept"] = "text/event-stream"

        async with httpx.AsyncClient(timeout=self._timeout) as http:
            async with http.stream("GET", url, headers=headers, params=params) as resp:
                resp.raise_for_status()
                event_data: list[str] = []
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        event_data.append(line[5:].strip())
                    elif line == "" and event_data:
                        # Blank line = end of one SSE event
                        raw = "\n".join(event_data)
                        event_data.clear()
                        if raw in ("", "ping"):
                            continue
                        try:
                            yield json.loads(raw)
                        except json.JSONDecodeError:
                            logger.warning("NinaiSSEStream: non-JSON data: %r", raw)
                    elif line.startswith(":"):
                        pass  # SSE comment / keepalive
