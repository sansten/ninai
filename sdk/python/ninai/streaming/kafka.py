"""NinaiKafkaIngestor — consume a Kafka topic and write events into Ninai memory.

Requires::

    pip install "ninai[kafka]"

Example — simple consumer loop::

    from ninai import NinaiClient
    from ninai.streaming.kafka import NinaiKafkaIngestor
    from ninai.streaming.ingestor import EventIngestor

    client = NinaiClient(api_key="nai_...")
    ingestor = EventIngestor(client, source_type="kafka", tags=["prod", "alerts"])

    async with NinaiKafkaIngestor(
        ingestor=ingestor,
        bootstrap_servers="kafka:9092",
        topic="alerts",
        group_id="ninai-consumer",
    ) as consumer:
        await consumer.run()

Example — custom event transformer::

    def transform(raw: dict) -> dict:
        return {
            "title": raw.get("summary"),
            "content": raw.get("description", ""),
            "severity": raw.get("priority", "medium"),
            "tags": ["kafka", raw.get("service", "unknown")],
            "source_id": raw.get("id"),
        }

    async with NinaiKafkaIngestor(
        ingestor=ingestor,
        bootstrap_servers="kafka:9092",
        topic="incidents",
        transform=transform,
    ) as consumer:
        await consumer.run()
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Union

try:
    from aiokafka import AIOKafkaConsumer
    from aiokafka.errors import KafkaError
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Kafka streaming requires the 'kafka' extra.\n"
        "Install with: pip install \"ninai[kafka]\""
    ) from exc

if TYPE_CHECKING:
    from ninai.streaming.ingestor import EventIngestor

logger = logging.getLogger(__name__)


def _default_transform(raw: dict) -> dict:
    """Pass-through: the raw Kafka message is used as-is."""
    return raw


class NinaiKafkaIngestor:
    """Consume one or more Kafka topics and ingest messages into Ninai memory.

    Each Kafka message value is JSON-decoded and passed through an optional
    ``transform`` callable before being handed to
    :class:`~ninai.streaming.ingestor.EventIngestor`.

    Args:
        ingestor: A configured :class:`~ninai.streaming.ingestor.EventIngestor`.
        bootstrap_servers: Kafka broker address(es) — string or list.
        topic: Topic name, or list of topic names to subscribe to.
        group_id: Consumer group ID (default ``"ninai-sdk-consumer"``).
        transform: Optional callable ``(raw_dict) → event_dict`` to reshape
            the Kafka message before ingestion.
        auto_offset_reset: ``"earliest"`` or ``"latest"`` (default ``"latest"``).
        max_poll_records: Max records fetched per poll (default 50).
        stop_on_empty: Stop the run loop when the topic has no new messages
            for ``idle_timeout`` seconds.  Useful for batch-style jobs
            (default ``False`` — run forever).
        idle_timeout: Seconds to wait with no messages before stopping when
            ``stop_on_empty=True`` (default 30).
        extra_consumer_kwargs: Additional kwargs forwarded to
            ``AIOKafkaConsumer``.

    Example::

        async with NinaiKafkaIngestor(ingestor, "kafka:9092", "alerts") as c:
            await c.run()
    """

    def __init__(
        self,
        ingestor: "EventIngestor",
        bootstrap_servers: Union[str, List[str]],
        topic: Union[str, List[str]],
        *,
        group_id: str = "ninai-sdk-consumer",
        transform: Optional[Callable[[dict], dict]] = None,
        auto_offset_reset: str = "latest",
        max_poll_records: int = 50,
        stop_on_empty: bool = False,
        idle_timeout: float = 30.0,
        **extra_consumer_kwargs,
    ) -> None:
        self._ingestor = ingestor
        self._servers = (
            bootstrap_servers
            if isinstance(bootstrap_servers, str)
            else ",".join(bootstrap_servers)
        )
        self._topics = [topic] if isinstance(topic, str) else list(topic)
        self._group_id = group_id
        self._transform = transform or _default_transform
        self._auto_offset_reset = auto_offset_reset
        self._max_poll_records = max_poll_records
        self._stop_on_empty = stop_on_empty
        self._idle_timeout = idle_timeout
        self._extra = extra_consumer_kwargs
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._running = False
        self._ingested = 0
        self._errors = 0

    @property
    def stats(self) -> dict:
        return {"ingested": self._ingested, "errors": self._errors}

    async def __aenter__(self) -> "NinaiKafkaIngestor":
        self._consumer = AIOKafkaConsumer(
            *self._topics,
            bootstrap_servers=self._servers,
            group_id=self._group_id,
            auto_offset_reset=self._auto_offset_reset,
            max_poll_records=self._max_poll_records,
            value_deserializer=lambda b: json.loads(b.decode("utf-8", errors="replace")),
            **self._extra,
        )
        await self._consumer.start()
        self._running = True
        logger.info(
            "NinaiKafkaIngestor started — brokers=%s topics=%s group=%s",
            self._servers, self._topics, self._group_id,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            logger.info(
                "NinaiKafkaIngestor stopped — ingested=%d errors=%d",
                self._ingested, self._errors,
            )

    async def run(self) -> None:
        """Consume messages and ingest them until stopped or idle.

        Call this after entering the context manager::

            async with NinaiKafkaIngestor(...) as c:
                await c.run()

        Or use :class:`~ninai.streaming.base.StreamPipeline` which calls
        ``events()`` directly.
        """
        if not self._consumer:
            raise RuntimeError("Must be used as an async context manager.")

        idle_since: Optional[float] = None

        async for msg in self._consumer:
            idle_since = None
            raw = msg.value if isinstance(msg.value, dict) else {}
            try:
                event = self._transform(raw)
                await self._ingestor.ingest(event)
                self._ingested += 1
            except Exception as exc:
                self._errors += 1
                logger.warning(
                    "KafkaIngestor: failed to ingest message offset=%s — %s",
                    msg.offset, exc,
                )

        # If stop_on_empty, handle via getmany with a timeout
        if self._stop_on_empty:
            logger.info("NinaiKafkaIngestor: stop_on_empty mode, idle_timeout=%.0fs", self._idle_timeout)
            while self._running:
                batch = await self._consumer.getmany(
                    timeout_ms=int(self._idle_timeout * 1000), max_records=self._max_poll_records
                )
                if not batch:
                    logger.info("NinaiKafkaIngestor: no messages in %.0fs, stopping.", self._idle_timeout)
                    break
                for _tp, msgs in batch.items():
                    for msg in msgs:
                        raw = msg.value if isinstance(msg.value, dict) else {}
                        try:
                            event = self._transform(raw)
                            await self._ingestor.ingest(event)
                            self._ingested += 1
                        except Exception as exc:
                            self._errors += 1
                            logger.warning("KafkaIngestor batch error: %s", exc)

    async def events(self):
        """Async generator yielding raw transformed event dicts.

        Used by :class:`~ninai.streaming.base.StreamPipeline`.
        """
        if not self._consumer:
            raise RuntimeError("Must be used as an async context manager.")
        async for msg in self._consumer:
            raw = msg.value if isinstance(msg.value, dict) else {}
            try:
                yield self._transform(raw)
            except Exception as exc:
                logger.warning("KafkaIngestor transform error: %s", exc)
