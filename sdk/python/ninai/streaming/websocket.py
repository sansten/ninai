"""NinaiEventStream — consume the Ninai real-time WebSocket event stream.

Connects to the backend ``/ws/stream`` endpoint and yields events as they
arrive.  Handles reconnection, heartbeat pings, and graceful shutdown.

Requires::

    pip install "ninai[streaming]"

Example::

    from ninai import NinaiClient
    from ninai.streaming.websocket import NinaiEventStream

    client = NinaiClient(api_key="nai_...")

    async for event in NinaiEventStream(client):
        if event.get("type") == "ping":
            continue  # heartbeat — ignore
        print(event["event_type"], event["payload"])

    # With type filtering
    async for event in NinaiEventStream(client, event_types=["memory.created", "goal.updated"]):
        handle(event)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, AsyncIterator, List, Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "WebSocket streaming requires the 'streaming' extra.\n"
        "Install with: pip install \"ninai[streaming]\""
    ) from exc

if TYPE_CHECKING:
    from ninai.client import NinaiClient

logger = logging.getLogger(__name__)

_RECONNECT_DELAY = 2.0   # seconds between reconnect attempts
_MAX_RECONNECTS = 10


class NinaiEventStream:
    """Async iterator that yields real-time events from ``/ws/stream``.

    Args:
        client: An authenticated :class:`~ninai.NinaiClient` instance.
        event_types: If given, only yield events whose ``event_type`` is in
            this list.  All events pass through when omitted.
        reconnect: Whether to reconnect automatically on disconnect
            (default ``True``).
        max_reconnects: Max consecutive reconnect attempts before giving up
            (default 10).  ``-1`` means unlimited.

    Example::

        client = NinaiClient(api_key="nai_...")

        async for event in NinaiEventStream(client, event_types=["memory.created"]):
            print(event)
    """

    def __init__(
        self,
        client: "NinaiClient",
        *,
        event_types: Optional[List[str]] = None,
        reconnect: bool = True,
        max_reconnects: int = _MAX_RECONNECTS,
    ) -> None:
        self._client = client
        self._event_types = set(event_types) if event_types else None
        self._reconnect = reconnect
        self._max_reconnects = max_reconnects

    def _ws_url(self) -> str:
        """Build the WebSocket URL from the client's base URL + access token."""
        base = self._client.base_url.rstrip("/")
        # Convert http(s) → ws(s)
        ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
        # Strip /api/v1 suffix if present — ws endpoint is at root
        ws_base = ws_base.rstrip("/api/v1").rstrip("/")
        token = self._client._access_token or self._client.api_key or ""
        return f"{ws_base}/ws/stream?token={token}"

    def _should_yield(self, event: dict) -> bool:
        if self._event_types is None:
            return True
        return event.get("event_type") in self._event_types

    async def _connect_and_stream(self, url: str) -> AsyncIterator[dict]:
        async with websockets.connect(url) as ws:
            logger.info("NinaiEventStream connected to %s", url.split("?")[0])
            async for raw in ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON frame: %r", raw)
                    continue
                if self._should_yield(event):
                    yield event

    def __aiter__(self) -> AsyncIterator[dict]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[dict]:
        url = self._ws_url()
        attempts = 0
        while True:
            try:
                async for event in self._connect_and_stream(url):
                    attempts = 0  # reset on successful receive
                    yield event
                # Generator exhausted cleanly — done
                return
            except ConnectionClosed as exc:
                if not self._reconnect:
                    raise
                attempts += 1
                if self._max_reconnects >= 0 and attempts > self._max_reconnects:
                    raise RuntimeError(
                        f"NinaiEventStream: exceeded {self._max_reconnects} reconnect attempts"
                    ) from exc
                logger.warning(
                    "NinaiEventStream: disconnected (%s), reconnecting in %.1fs (attempt %d)…",
                    exc, _RECONNECT_DELAY, attempts,
                )
                await asyncio.sleep(_RECONNECT_DELAY)
            except Exception:
                raise
