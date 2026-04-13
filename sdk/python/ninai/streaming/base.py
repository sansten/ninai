"""Base protocol and pipeline for Ninai streaming adapters.

Any event source that implements ``BaseStreamAdapter`` can be plugged into a
``StreamPipeline`` to automatically ingest events into Ninai memory.

Example — custom adapter::

    from ninai.streaming.base import BaseStreamAdapter, StreamPipeline
    from ninai.streaming.ingestor import EventIngestor

    class MyRedisStreamAdapter(BaseStreamAdapter):
        async def __aenter__(self):
            self._conn = await connect_redis()
            return self
        async def __aexit__(self, *args):
            await self._conn.close()
        async def events(self):
            async for msg in self._conn.xread("my-stream"):
                yield {"title": msg["title"], "content": msg["body"]}

    async with MyRedisStreamAdapter() as adapter:
        ingestor = EventIngestor(client, source_type="redis")
        pipeline = StreamPipeline(adapter=adapter, ingestor=ingestor)
        await pipeline.run()
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseStreamAdapter(Protocol):
    """Protocol every streaming adapter must satisfy.

    Adapters are async context managers that expose an ``events()`` async
    generator.  Each yielded item is a plain ``dict`` with at least a
    ``content`` key (other keys are passed through to
    :class:`~ninai.streaming.ingestor.EventIngestor` as metadata).
    """

    async def __aenter__(self) -> "BaseStreamAdapter": ...

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...

    async def events(self) -> AsyncIterator[dict]:
        """Yield events from the underlying source.

        Each event is a plain dict.  Recognised keys:

        - ``content`` *(required)* — main text body stored as Ninai memory content.
        - ``title`` — optional human-readable title.
        - ``tags`` — list of strings.
        - ``severity`` — ``"critical" | "high" | "medium" | "low"``.
        - ``source_id`` — ID in the originating system.
        - ``metadata`` — extra dict merged into ``extra_metadata``.
        """
        # yield is required to make this a generator method in the Protocol.
        # Implementations override this.
        return  # type: ignore[misc]
        yield  # noqa: unreachable


class StreamPipeline:
    """Connects a :class:`BaseStreamAdapter` to an
    :class:`~ninai.streaming.ingestor.EventIngestor`.

    Runs until cancelled or the adapter's ``events()`` generator is exhausted.

    Args:
        adapter: Any object satisfying :class:`BaseStreamAdapter`.
        ingestor: An :class:`~ninai.streaming.ingestor.EventIngestor` instance.
        concurrency: Max number of events processed in parallel (default 8).
        on_error: ``"log"`` (default) or ``"raise"`` — what to do when a
            single event fails ingestion.

    Example::

        pipeline = StreamPipeline(adapter=kafka_adapter, ingestor=ingestor)
        await pipeline.run()          # blocks until done / cancelled
    """

    def __init__(
        self,
        *,
        adapter: BaseStreamAdapter,
        ingestor,
        concurrency: int = 8,
        on_error: str = "log",
    ) -> None:
        self._adapter = adapter
        self._ingestor = ingestor
        self._concurrency = concurrency
        self._on_error = on_error
        self._sem = asyncio.Semaphore(concurrency)
        self._processed = 0
        self._errors = 0

    @property
    def stats(self) -> dict:
        """Return a snapshot of pipeline stats."""
        return {"processed": self._processed, "errors": self._errors}

    async def _handle(self, event: dict) -> None:
        async with self._sem:
            try:
                await self._ingestor.ingest(event)
                self._processed += 1
            except Exception as exc:
                self._errors += 1
                if self._on_error == "raise":
                    raise
                logger.warning("StreamPipeline: ingestion error — %s", exc)

    async def run(self) -> None:
        """Start the pipeline.  Blocks until the adapter is exhausted or cancelled."""
        tasks: list[asyncio.Task] = []
        async with self._adapter:
            async for event in self._adapter.events():
                task = asyncio.create_task(self._handle(event))
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(
            "StreamPipeline finished — processed=%d errors=%d",
            self._processed,
            self._errors,
        )
