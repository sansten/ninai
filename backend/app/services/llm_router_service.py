"""Multi-model LLM routing service for per-org provider selection."""

from __future__ import annotations

import os
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.org_llm_config import OrgLlmConfig, VALID_PROVIDERS


@dataclass
class LlmCallResult:
    provider: str
    model: str
    content: str
    tokens_used: int


class LlmRouterService:
    def __init__(self, session: AsyncSession, org_id: str):
        self.session = session
        self.org_id = org_id

    async def get_config(self) -> OrgLlmConfig | None:
        res = await self.session.execute(
            select(OrgLlmConfig).where(
                OrgLlmConfig.organization_id == self.org_id,
                OrgLlmConfig.is_active.is_(True),
            )
        )
        return res.scalar_one_or_none()

    async def complete(
        self, *, prompt: str, system: str = "", max_tokens: int = 512
    ) -> LlmCallResult:
        """Route LLM call to org-configured provider, fallback to local backend."""
        cfg = await self.get_config()
        provider = (cfg.provider if cfg else "local").strip().lower()
        model = cfg.model if cfg else (settings.VLLM_MODEL or "qwen2.5:7b")

        if provider == "openai":
            return await self._call_openai(cfg, prompt, system, max_tokens)
        elif provider == "anthropic":
            return await self._call_anthropic(cfg, prompt, system, max_tokens)
        else:
            return await self._call_local_backend(model, prompt, system, max_tokens)

    async def _call_local_backend(
        self, model: str, prompt: str, system: str, max_tokens: int
    ) -> LlmCallResult:
        import httpx

        url = settings.VLLM_BASE_URL or "http://localhost:11434"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(f"{url}/api/generate", json=payload)
                r.raise_for_status()
                data = r.json()
                return LlmCallResult(
                    provider="local",
                    model=model,
                    content=data.get("response", ""),
                    tokens_used=data.get("eval_count", 0),
                )
        except Exception:
            return LlmCallResult(
                provider="local", model=model, content="", tokens_used=0
            )

    async def _call_openai(
        self, cfg: OrgLlmConfig, prompt: str, system: str, max_tokens: int
    ) -> LlmCallResult:
        api_key = (
            os.getenv(cfg.api_key_ref or "") if cfg.api_key_ref else ""
        )
        if not api_key:
            return LlmCallResult(
                provider="openai", model=cfg.model, content="[no api key]", tokens_used=0
            )
        import httpx

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                r.raise_for_status()
                data = r.json()
                return LlmCallResult(
                    provider="openai",
                    model=cfg.model,
                    content=data["choices"][0]["message"]["content"],
                    tokens_used=data["usage"]["total_tokens"],
                )
        except Exception:
            return LlmCallResult(
                provider="openai", model=cfg.model, content="", tokens_used=0
            )

    async def _call_anthropic(
        self, cfg: OrgLlmConfig, prompt: str, system: str, max_tokens: int
    ) -> LlmCallResult:
        api_key = (
            os.getenv(cfg.api_key_ref or "") if cfg.api_key_ref else ""
        )
        if not api_key:
            return LlmCallResult(
                provider="anthropic", model=cfg.model, content="[no api key]", tokens_used=0
            )
        import httpx

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": cfg.model,
            "system": system,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload,
                    headers=headers,
                )
                r.raise_for_status()
                data = r.json()
                return LlmCallResult(
                    provider="anthropic",
                    model=cfg.model,
                    content=data["content"][0]["text"],
                    tokens_used=data["usage"]["input_tokens"]
                    + data["usage"]["output_tokens"],
                )
        except Exception:
            return LlmCallResult(
                provider="anthropic", model=cfg.model, content="", tokens_used=0
            )
