"""Ollama client with circuit breaker integration.

Wraps OllamaClient to automatically use circuit breaker protection.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.llm.ollama import OllamaClient as BaseOllamaClient
from app.agents.llm.tool_events import ToolEventSink
from app.core.llm_integration import call_llm_with_breaker
from app.core.circuit_breaker import CircuitBreakerOpen
from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaClientWithCircuitBreaker(BaseOllamaClient):
    """OllamaClient wrapper that uses circuit breaker for protection."""

    async def complete_json(
        self,
        *,
        prompt: str,
        schema_hint: dict[str, Any],
        tool_event_sink: ToolEventSink | None = None,
    ) -> dict[str, Any]:
        """
        Complete JSON prompt with circuit breaker protection.

        If circuit breaker is open, returns empty dict (fail-closed).
        """
        try:
            # Call parent method through circuit breaker
            return await call_llm_with_breaker(
                provider="ollama",
                func=super().complete_json,
                prompt=prompt,
                schema_hint=schema_hint,
                tool_event_sink=tool_event_sink,
            )
        except CircuitBreakerOpen as e:
            logger.warning(f"Ollama circuit breaker open: {e}, failing closed")
            # Tool event for circuit open
            if tool_event_sink is not None:
                try:
                    await tool_event_sink(
                        {
                            "event_type": "tool_result",
                            "summary_text": "ollama.generate circuit_breaker_open",
                            "payload": {
                                "tool": "ollama.generate",
                                "ok": False,
                                "circuit_breaker_open": True,
                            },
                        }
                    )
                except Exception:
                    pass
            # Return empty dict (fail-closed behavior)
            return {}
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise

    async def complete_text(
        self,
        *,
        prompt: str,
        num_ctx: int = 32768,
        temperature: float = 0.1,
    ) -> str:
        """Complete plain-text prompt with circuit breaker protection."""
        try:
            result = await call_llm_with_breaker(
                provider="ollama",
                func=super().complete_text,
                prompt=prompt,
                num_ctx=num_ctx,
                temperature=temperature,
            )
            return str(result or "")
        except CircuitBreakerOpen as e:
            logger.warning(f"Ollama circuit breaker open: {e}, failing closed")
            self._last_error = f"circuit_breaker_open: {e}"
            return ""
        except Exception as e:
            self._last_error = repr(e)
            logger.error(f"Ollama text call failed: {e}")
            raise


def create_ollama_client(
    *,
    base_url: str = "http://localhost:11434",
    model: str = "llama3.1:8b",
    timeout_seconds: float = 30.0,
    max_concurrency: int | None = None,
    use_circuit_breaker: bool = True,
    purpose: str | None = None,
    auth_token: str | None = None,
) -> BaseOllamaClient:
    """
    Create an Ollama client with optional circuit breaker.

    Args:
        base_url: Ollama service URL
        model: Model name (e.g., "qwen2.5:7b")
        timeout_seconds: Request timeout
        max_concurrency: Max concurrent requests
        use_circuit_breaker: Whether to wrap with circuit breaker
        purpose: Optional logical purpose label (planning, reasoning, etc.)

    Returns:
        OllamaClient (with or without circuit breaker)
    """
    inferred_purpose = purpose
    if not inferred_purpose:
        purpose_candidates = {
            "planning": settings.get_ollama_model("planning"),
            "reasoning": settings.get_ollama_model("reasoning"),
            "distillation": settings.get_ollama_model("distillation"),
            "boundary": settings.get_ollama_model("boundary"),
            "uncertainty": settings.get_ollama_model("uncertainty"),
            "agents": settings.get_ollama_model("agents"),
            "default": settings.get_ollama_model(),
        }
        for candidate_purpose, candidate_model in purpose_candidates.items():
            if str(candidate_model).strip() == str(model).strip():
                inferred_purpose = candidate_purpose
                break

    logger.info(
        "llm.model_route provider=ollama purpose=%s model=%s base_url=%s circuit_breaker=%s",
        inferred_purpose or "default",
        model,
        base_url,
        use_circuit_breaker,
    )

    cpu_base_url = str(getattr(settings, "OLLAMA_BASE_URL_CPU", "") or "").strip()
    gpu_base_url = str(getattr(settings, "OLLAMA_BASE_URL_GPU", "") or "").strip()
    overflow_enabled = bool(getattr(settings, "OLLAMA_OVERFLOW_ENABLED", False)) and bool(gpu_base_url)
    overflow_primary_max_inflight = int(getattr(settings, "OLLAMA_OVERFLOW_PRIMARY_MAX_INFLIGHT", 0) or 0)

    # Honor explicit caller intent first. If a specific base_url is passed
    # (e.g., GPU service), do not silently override it with CPU defaults.
    explicit_base_url = str(base_url or "").strip()
    primary_base_url = explicit_base_url or cpu_base_url or "http://localhost:11434"

    secondary_candidates = [gpu_base_url, cpu_base_url]
    secondary_base_url = None
    for candidate in secondary_candidates:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        if candidate.rstrip("/") == primary_base_url.rstrip("/"):
            continue
        secondary_base_url = candidate
        break

    resolved_token = auth_token or getattr(settings, "OLLAMA_AUTH_TOKEN", None) or None
    fallback_models = [
        str(candidate).strip()
        for candidate in (
            settings.get_ollama_model("agents"),
            settings.get_ollama_model(),
            getattr(settings, "OLLAMA_MODEL_FAST", None),
        )
        if str(candidate or "").strip()
    ]

    client = BaseOllamaClient(
        base_url=primary_base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        max_concurrency=max_concurrency,
        auth_token=resolved_token,
        secondary_base_url=secondary_base_url,
        overflow_enabled=overflow_enabled,
        overflow_primary_max_inflight=overflow_primary_max_inflight,
        fallback_models=fallback_models,
    )

    if use_circuit_breaker:
        return OllamaClientWithCircuitBreaker(
            base_url=primary_base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            max_concurrency=max_concurrency,
            auth_token=resolved_token,
            secondary_base_url=secondary_base_url,
            overflow_enabled=overflow_enabled,
            overflow_primary_max_inflight=overflow_primary_max_inflight,
            fallback_models=fallback_models,
        )

    return client
