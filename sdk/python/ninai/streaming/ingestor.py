"""EventIngestor — write any streaming event into Ninai memory.

This is the core of the streaming layer.  No extra dependencies required —
just the base ``ninai`` package.

Example::

    from ninai import NinaiClient
    from ninai.streaming.ingestor import EventIngestor

    client = NinaiClient(api_key="nai_...")
    ingestor = EventIngestor(client, source_type="pagerduty", tags=["prod", "on-call"])

    # Single event
    await ingestor.ingest({
        "title": "Disk usage above 90%",
        "content": "Host db-prod-1 disk /dev/sda1 at 92%.",
        "severity": "high",
        "source_id": "INC-4421",
        "tags": ["infra", "disk"],
    })

    # Batch
    await ingestor.ingest_batch([event1, event2, event3])
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from ninai.client import NinaiClient

logger = logging.getLogger(__name__)

# Severity → Ninai classification mapping
_SEVERITY_TO_CLASSIFICATION = {
    "critical": "restricted",
    "high": "confidential",
    "medium": "internal",
    "low": "internal",
}

_DEFAULT_CLASSIFICATION = "internal"


class EventIngestor:
    """Write streaming events into Ninai as memory records.

    Field mapping (event dict → Ninai memory):

    +-----------------+-------------------------------------------+
    | Event key       | Ninai field                               |
    +=================+===========================================+
    | ``content``     | ``content`` *(required)*                  |
    | ``title``       | ``title``                                 |
    | ``tags``        | merged with ``base_tags``                 |
    | ``severity``    | maps to ``classification``                |
    | ``source_id``   | ``source_id``                             |
    | ``metadata``    | ``extra_metadata``                        |
    | ``memory_type`` | ``memory_type`` (default ``"long_term"``) |
    | ``scope``       | ``scope`` (default ``"organization"``)    |
    +-----------------+-------------------------------------------+

    Args:
        client: An authenticated :class:`~ninai.NinaiClient` instance.
        source_type: Label stored as ``source_type`` on every memory
            (e.g. ``"kafka"``, ``"pagerduty"``, ``"webhook"``).
        tags: Base tags applied to every ingested event.
        scope: Ninai memory scope (default ``"organization"``).
        memory_type: Ninai memory type (default ``"long_term"``).
        classification: Override severity-based classification with a fixed
            value (``"public" | "internal" | "confidential" | "restricted"``).
        concurrency: Max parallel memory writes for batch ingestion.
    """

    def __init__(
        self,
        client: "NinaiClient",
        *,
        source_type: str = "stream",
        tags: Optional[List[str]] = None,
        scope: str = "organization",
        memory_type: str = "long_term",
        classification: Optional[str] = None,
        concurrency: int = 16,
    ) -> None:
        self._client = client
        self._source_type = source_type
        self._base_tags = list(tags or [])
        self._scope = scope
        self._memory_type = memory_type
        self._classification = classification
        self._sem = asyncio.Semaphore(concurrency)

    def _build_payload(self, event: dict) -> dict:
        """Map an event dict to a Ninai memory create payload."""
        content = event.get("content") or event.get("body") or event.get("message", "")
        if not content:
            raise ValueError("Event must have a 'content', 'body', or 'message' key.")

        # Tags: base + event-level
        event_tags = event.get("tags") or []
        if isinstance(event_tags, str):
            event_tags = [t.strip() for t in event_tags.split(",") if t.strip()]
        merged_tags = list({*self._base_tags, *event_tags})

        # Classification from severity
        severity = (event.get("severity") or "").lower()
        classification = (
            self._classification
            or _SEVERITY_TO_CLASSIFICATION.get(severity, _DEFAULT_CLASSIFICATION)
        )

        payload: dict[str, Any] = {
            "content": content,
            "scope": event.get("scope", self._scope),
            "memory_type": event.get("memory_type", self._memory_type),
            "classification": classification,
            "source_type": self._source_type,
            "tags": merged_tags,
        }

        if title := event.get("title"):
            payload["title"] = title
        if source_id := event.get("source_id"):
            payload["source_id"] = str(source_id)

        # Merge extra metadata
        meta: dict = dict(event.get("metadata") or {})
        if severity:
            meta.setdefault("severity", severity)
        if event.get("url"):
            meta.setdefault("url", event["url"])
        if event.get("actor"):
            meta.setdefault("actor", event["actor"])
        if meta:
            payload["metadata"] = meta

        return payload

    async def ingest(self, event: dict) -> Any:
        """Ingest a single event into Ninai memory.

        Args:
            event: Dict with at least a ``content`` / ``body`` / ``message`` key.

        Returns:
            The created :class:`~ninai.models.Memory` object.

        Raises:
            ValueError: If the event has no usable content.
            :class:`~ninai.exceptions.NinaiError`: On API errors.
        """
        payload = self._build_payload(event)
        loop = asyncio.get_event_loop()
        async with self._sem:
            memory = await loop.run_in_executor(
                None,
                lambda: self._client.memories.create(**payload),
            )
        logger.debug("Ingested event → memory %s", memory.id)
        return memory

    async def ingest_batch(self, events: List[dict]) -> List[Any]:
        """Ingest multiple events concurrently.

        Args:
            events: List of event dicts.

        Returns:
            List of created :class:`~ninai.models.Memory` objects (same order).
            Failed items are ``None`` (errors are logged).
        """
        results: List[Any] = [None] * len(events)

        async def _one(i: int, ev: dict) -> None:
            try:
                results[i] = await self.ingest(ev)
            except Exception as exc:
                logger.warning("Batch ingest[%d] failed: %s", i, exc)

        await asyncio.gather(*[_one(i, ev) for i, ev in enumerate(events)])
        return results
