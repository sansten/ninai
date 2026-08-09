"""Embedding service.

Central place to generate embeddings (OpenAI) with a safe fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.core.config import settings


class EmbeddingService:
    _backend_sem: asyncio.Semaphore | None = None
    _cache_lock: asyncio.Lock | None = None
    _cache: "OrderedDict[str, tuple[datetime, list[float]]]" = OrderedDict()
    _cache_max_size: int = int(os.getenv("EMBEDDING_CACHE_MAX_SIZE", "2000") or 2000)
    _cache_ttl_seconds: int = int(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", "3600") or 3600)

    @staticmethod
    def utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @classmethod
    def _zeros(cls) -> list[float]:
        return [0.0] * int(settings.EMBEDDING_DIMENSIONS)

    @classmethod
    def _backend_semaphore(cls) -> asyncio.Semaphore:
        if cls._backend_sem is None:
            cls._backend_sem = asyncio.Semaphore(int(getattr(settings, "VLLM_MAX_CONCURRENCY", 2) or 2))
        return cls._backend_sem

    @classmethod
    def _embedding_cache_lock(cls) -> asyncio.Lock:
        if cls._cache_lock is None:
            cls._cache_lock = asyncio.Lock()
        return cls._cache_lock

    @classmethod
    def _cache_key(cls, provider: str, model: str, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"{provider}:{model}:{digest}"

    @classmethod
    async def _cache_get(cls, key: str) -> list[float] | None:
        if cls._cache_ttl_seconds <= 0 or cls._cache_max_size <= 0:
            return None

        async with cls._embedding_cache_lock():
            entry = cls._cache.get(key)
            if not entry:
                return None

            created_at, vector = entry
            age = (cls.utcnow() - created_at).total_seconds()
            if age > cls._cache_ttl_seconds:
                cls._cache.pop(key, None)
                return None

            # LRU refresh on hit
            cls._cache.move_to_end(key)
            return list(vector)

    @classmethod
    async def _cache_set(cls, key: str, vector: list[float]) -> None:
        if cls._cache_ttl_seconds <= 0 or cls._cache_max_size <= 0:
            return

        async with cls._embedding_cache_lock():
            cls._cache[key] = (cls.utcnow(), list(vector))
            cls._cache.move_to_end(key)
            while len(cls._cache) > cls._cache_max_size:
                cls._cache.popitem(last=False)

    @classmethod
    async def _embed_local_backend(cls, text: str) -> list[float]:
        # Dedicated embedding endpoint (vllm-embed) takes priority so inference
        # pods are not interrupted during mixed ingest/query workloads.
        base_url = str(
            getattr(settings, "VLLM_EMBEDDING_BASE_URL", None)
            or getattr(settings, "VLLM_BASE_URL_CPU", None)
            or getattr(settings, "VLLM_BASE_URL", None)
            or "http://localhost:8000"
        ).rstrip("/")
        model = str(getattr(settings, "LOCAL_EMBEDDING_MODEL", None) or "BAAI/bge-base-en-v1.5")
        timeout = float(getattr(settings, "VLLM_TIMEOUT_SECONDS", 30.0) or 30.0)
        api_fmt = str(getattr(settings, "EMBED_API_FORMAT", "openai") or "openai").lower()

        async with cls._backend_semaphore():
            async with httpx.AsyncClient(timeout=timeout) as client:
                if api_fmt == "openai":
                    payload: dict[str, Any] = {"model": model, "input": text}
                    resp = await client.post(f"{base_url}/v1/embeddings", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    emb = data["data"][0]["embedding"]
                else:
                    payload = {"model": model, "prompt": text}
                    resp = await client.post(f"{base_url}/api/embeddings", json=payload)
                    resp.raise_for_status()
                    emb = resp.json().get("embedding", [])

        if not isinstance(emb, list) or not emb:
            raise ValueError("Embedding response missing vector")

        return [float(x) for x in emb]

    @classmethod
    async def _embed_openai(cls, text: str) -> list[float]:
        if not settings.OPENAI_API_KEY:
            return cls._zeros()

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        resp = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text,
        )
        return resp.data[0].embedding

    @classmethod
    async def embed(cls, text: str) -> list[float]:
        # Safety: never embed empty text.
        cleaned = (text or "").strip()
        if not cleaned:
            return cls._zeros()

        provider = str(getattr(settings, "EMBEDDING_PROVIDER", "auto") or "auto").lower()

        # Local-first default: if no OpenAI key, try the local embedding backend.
        if provider == "auto":
            provider = "openai" if bool(settings.OPENAI_API_KEY) else "local"

        if provider in {"local", "vllm"}:
            model = str(getattr(settings, "LOCAL_EMBEDDING_MODEL", None) or "nomic-embed-text")
            key = cls._cache_key("local", model, cleaned)
            cached = await cls._cache_get(key)
            if cached is not None:
                return cached
            try:
                emb = await cls._embed_local_backend(cleaned)
                await cls._cache_set(key, emb)
                return emb
            except Exception:
                # If local embedding backend isn't available, fall back to OpenAI if configured.
                if settings.OPENAI_API_KEY:
                    try:
                        model = str(getattr(settings, "EMBEDDING_MODEL", None) or "text-embedding-3-small")
                        key = cls._cache_key("openai", model, cleaned)
                        cached = await cls._cache_get(key)
                        if cached is not None:
                            return cached
                        emb = await cls._embed_openai(cleaned)
                        await cls._cache_set(key, emb)
                        return emb
                    except Exception:
                        return cls._zeros()
                return cls._zeros()

        if provider == "openai":
            model = str(getattr(settings, "EMBEDDING_MODEL", None) or "text-embedding-3-small")
            key = cls._cache_key("openai", model, cleaned)
            cached = await cls._cache_get(key)
            if cached is not None:
                return cached
            try:
                emb = await cls._embed_openai(cleaned)
                await cls._cache_set(key, emb)
                return emb
            except Exception:
                # If OpenAI isn't configured/available, fall back to local backend; otherwise zeros.
                try:
                    model = str(getattr(settings, "LOCAL_EMBEDDING_MODEL", None) or "nomic-embed-text")
                    key = cls._cache_key("local", model, cleaned)
                    cached = await cls._cache_get(key)
                    if cached is not None:
                        return cached
                    emb = await cls._embed_local_backend(cleaned)
                    await cls._cache_set(key, emb)
                    return emb
                except Exception:
                    return cls._zeros()

        # Unknown provider -> safe fallback.
        return cls._zeros()
