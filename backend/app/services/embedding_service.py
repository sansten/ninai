"""Embedding service.

Central place to generate embeddings (OpenAI) with a safe fallback.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.core.config import settings


class EmbeddingService:
    _ollama_sem: asyncio.Semaphore | None = None

    @staticmethod
    def utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @classmethod
    def _zeros(cls) -> list[float]:
        return [0.0] * int(settings.EMBEDDING_DIMENSIONS)

    @classmethod
    def _ollama_semaphore(cls) -> asyncio.Semaphore:
        if cls._ollama_sem is None:
            cls._ollama_sem = asyncio.Semaphore(int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2) or 2))
        return cls._ollama_sem

    @classmethod
    async def _embed_ollama(cls, text: str) -> list[float]:
        base_url = str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434").rstrip("/")
        model = str(getattr(settings, "OLLAMA_EMBEDDING_MODEL", None) or "nomic-embed-text")
        timeout = float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0) or 5.0)

        payload: dict[str, Any] = {"model": model, "prompt": text}

        async with cls._ollama_semaphore():
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{base_url}/api/embeddings", json=payload)
                resp.raise_for_status()
                data = resp.json()

        emb = data.get("embedding") if isinstance(data, dict) else None
        if not isinstance(emb, list) or not emb:
            raise ValueError("Ollama embedding response missing 'embedding' list")

        # Normalize to floats.
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

        # Local-first default: if no OpenAI key, try Ollama.
        if provider == "auto":
            provider = "openai" if bool(settings.OPENAI_API_KEY) else "ollama"

        if provider == "ollama":
            try:
                return await cls._embed_ollama(cleaned)
            except Exception:
                # If Ollama isn't available, fall back to OpenAI if configured; otherwise zeros.
                if settings.OPENAI_API_KEY:
                    try:
                        return await cls._embed_openai(cleaned)
                    except Exception:
                        return cls._zeros()
                return cls._zeros()

        if provider == "openai":
            try:
                return await cls._embed_openai(cleaned)
            except Exception:
                # If OpenAI isn't configured/available, fall back to Ollama; otherwise zeros.
                try:
                    return await cls._embed_ollama(cleaned)
                except Exception:
                    return cls._zeros()

        # Unknown provider -> safe fallback.
        return cls._zeros()
