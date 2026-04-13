"""Ninai streaming — real-time event ingestion and consumption.

Write events from any source (Kafka, WebSocket, SSE, webhooks, custom)
into Ninai memory, or consume the live Ninai event stream.

Quick start::

    # Ingest from any source (no extra deps)
    from ninai.streaming import EventIngestor
    ingestor = EventIngestor(client, source_type="webhook", tags=["prod"])
    await ingestor.ingest({"title": "Alert", "content": "Disk at 95%", "severity": "high"})

    # Read real-time events FROM Ninai (WebSocket)
    # pip install "ninai[streaming]"
    from ninai.streaming import NinaiEventStream
    async for event in NinaiEventStream(client):
        print(event["event_type"])

    # Read events as SSE (no extra deps)
    from ninai.streaming import NinaiSSEStream
    async for event in NinaiSSEStream(client):
        print(event)

    # Consume Kafka → Ninai pipeline
    # pip install "ninai[kafka]"
    from ninai.streaming import EventIngestor, StreamPipeline
    from ninai.streaming.kafka import NinaiKafkaIngestor

    ingestor = EventIngestor(client, source_type="kafka", tags=["prod"])
    async with NinaiKafkaIngestor(ingestor, "kafka:9092", "alerts") as adapter:
        pipeline = StreamPipeline(adapter=adapter, ingestor=ingestor)
        await pipeline.run()

Optional extras
---------------
- ``ninai[kafka]``      — Kafka support via ``aiokafka``
- ``ninai[streaming]``  — WebSocket stream via ``websockets``
- ``ninai[all]``        — all of the above
"""

from ninai.streaming.base import BaseStreamAdapter, StreamPipeline
from ninai.streaming.ingestor import EventIngestor
from ninai.streaming.sse import NinaiSSEStream  # uses httpx — always available


def NinaiEventStream(*args, **kwargs):
    """WebSocket event stream from Ninai.  Requires ``pip install "ninai[streaming]"``."""
    from ninai.streaming.websocket import NinaiEventStream as _NinaiEventStream
    return _NinaiEventStream(*args, **kwargs)


def NinaiKafkaIngestor(*args, **kwargs):
    """Kafka consumer → Ninai ingestor.  Requires ``pip install "ninai[kafka]"``."""
    from ninai.streaming.kafka import NinaiKafkaIngestor as _NinaiKafkaIngestor
    return _NinaiKafkaIngestor(*args, **kwargs)


__all__ = [
    "BaseStreamAdapter",
    "StreamPipeline",
    "EventIngestor",
    "NinaiSSEStream",
    "NinaiEventStream",
    "NinaiKafkaIngestor",
]
