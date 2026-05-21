"""
V2 Pipeline Factory

Constructs the V2CognitiveLoop from app settings.
Called once at startup and cached for the process lifetime.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.core.config import settings
from app.v2.graph.client import V2GraphClient
from app.v2.llm.ollama_engine import OllamaReasoningEngine
from app.v2.memory.dnc_router import DNCMemoryRouter
from app.v2.pipeline.cognitive_loop import V2CognitiveLoop

logger = logging.getLogger(__name__)

# Module-level singleton (reset by tests via _reset_v2_loop())
_v2_loop: V2CognitiveLoop | None = None


def get_v2_loop() -> V2CognitiveLoop:
    global _v2_loop
    if _v2_loop is None:
        _v2_loop = _build_v2_loop()
    return _v2_loop


def _build_v2_loop() -> V2CognitiveLoop:
    graph_redis_url = (
        getattr(settings, "GRAPH_REDIS_URL", None)
        or getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
    )
    graph_client = V2GraphClient(redis_url=graph_redis_url)

    engine = OllamaReasoningEngine(
        base_url=getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"),
        model=getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b"),
        embed_model=getattr(settings, "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        timeout=getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 45.0),
    )

    # Qdrant client — optional (graceful degrade when not configured)
    qdrant_client = _try_get_qdrant()

    router = DNCMemoryRouter(
        graph_client=graph_client,
        qdrant_service=qdrant_client,
        embedding_fn=engine.embed,
        entity_extractor=engine.extract_entities,
        top_k_qdrant=10,
        top_m_graph=25,
        graph_hops=2,
    )

    loop = V2CognitiveLoop(dnc_router=router, reasoning_engine=engine)
    logger.info(
        "V2 cognitive loop initialised (graph_available=%s, qdrant=%s)",
        graph_client.is_available(),
        qdrant_client is not None,
    )
    return loop


def _try_get_qdrant() -> object | None:
    try:
        from qdrant_client import AsyncQdrantClient  # type: ignore

        host = getattr(settings, "QDRANT_HOST", None)
        port = getattr(settings, "QDRANT_PORT", None)
        url = getattr(settings, "QDRANT_URL", None)
        api_key = getattr(settings, "QDRANT_API_KEY", None)

        if url:
            return AsyncQdrantClient(url=url, api_key=api_key)
        if host and port:
            return AsyncQdrantClient(host=host, port=int(port), api_key=api_key)
    except Exception as exc:
        logger.warning("Could not initialise Qdrant for v2: %s", exc)
    return None


def _reset_v2_loop() -> None:
    """Test helper — clears the singleton so the next call rebuilds it."""
    global _v2_loop
    _v2_loop = None
