"""
V2 Pipeline Factory

Constructs the V2CognitiveLoop from app settings.
Called once at startup and cached for the process lifetime.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from app.core.config import settings
from app.v2.graph.client import V2GraphClient
from app.v2.llm.inference_engine import InferenceEngine
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

    # Dedicated vllm-embed pod for embeddings; falls back to main inference pod.
    embed_base_url = (
        getattr(settings, "VLLM_EMBEDDING_BASE_URL", None)
        or getattr(settings, "VLLM_BASE_URL_CPU", None)
        or getattr(settings, "VLLM_BASE_URL", "http://localhost:8000")
    )
    _default_base_url = getattr(settings, "VLLM_BASE_URL", "http://localhost:8000")
    _default_model    = getattr(settings, "VLLM_MODEL", "qwen2.5:7b")
    _timeout          = getattr(settings, "VLLM_TIMEOUT_SECONDS", 45.0)
    _embed_api        = getattr(settings, "EMBED_API_FORMAT", "openai")
    extract_base_url  = getattr(settings, "EXTRACT_BASE_URL", None)
    extract_model     = getattr(settings, "EXTRACT_MODEL", None)

    # Primary engine (used for embeddings, entity extraction, and default inference)
    engine = InferenceEngine(
        base_url=_default_base_url,
        model=_default_model,
        embed_model=getattr(settings, "LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        embed_base_url=embed_base_url,
        extract_base_url=extract_base_url,
        extract_model=extract_model,
        timeout=_timeout,
        embed_api=_embed_api,
    )

    # Per-tier engines — point at the correct vLLM pod URL for each model tier.
    # When VLLM_BASE_URL_FAST / VLLM_BASE_URL_REASONING are not set they
    # fall back to the primary engine (single-pod mode).
    slm_model = settings.get_llm_model("fast")
    llm_model = settings.get_llm_model("reasoning")

    _fast_url      = getattr(settings, "VLLM_BASE_URL_FAST", None) or _default_base_url
    _reasoning_url = getattr(settings, "VLLM_BASE_URL_REASONING", None) or _default_base_url

    slm_engine: InferenceEngine | None = None
    llm_engine: InferenceEngine | None = None

    if _fast_url != _default_base_url or slm_model != _default_model:
        slm_engine = InferenceEngine(
            base_url=_fast_url,
            model=slm_model,
            embed_model=getattr(settings, "LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
            embed_base_url=embed_base_url,
            timeout=_timeout,
            embed_api=_embed_api,
        )

    if _reasoning_url != _default_base_url or llm_model != _default_model:
        llm_engine = InferenceEngine(
            base_url=_reasoning_url,
            model=llm_model,
            embed_model=getattr(settings, "LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
            embed_base_url=embed_base_url,
            timeout=_timeout,
            embed_api=_embed_api,
        )

    logger.info(
        "Model tiers — SLM: %s @ %s | LLM: %s @ %s",
        slm_model, _fast_url, llm_model, _reasoning_url,
    )

    # Qdrant client — optional (graceful degrade when not configured)
    qdrant_client = _try_get_qdrant()

    router = DNCMemoryRouter(
        graph_client=graph_client,
        qdrant_service=qdrant_client,
        embedding_fn=engine.embed,
        entity_extractor=engine.extract_entities,
        top_k_qdrant=int(os.environ.get("QDRANT_TOP_K", "30")),
        top_m_graph=int(os.environ.get("GRAPH_TOP_M", "30")),
        graph_hops=int(os.environ.get("GRAPH_HOPS", "3")),
    )

    loop = V2CognitiveLoop(
        dnc_router=router,
        reasoning_engine=engine,
        slm_engine=slm_engine,
        llm_engine=llm_engine,
    )
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
        raw_key = getattr(settings, "QDRANT_API_KEY", None)
        # Reject keys that are whitespace-only (e.g. a stale \r\n secret value)
        api_key = raw_key.strip() if isinstance(raw_key, str) else raw_key
        if not api_key:
            api_key = None

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
