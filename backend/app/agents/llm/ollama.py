from __future__ import annotations

import asyncio
import json
from typing import Any
import time

import httpx

from app.agents.llm.base import LLMClient
from app.agents.llm.tool_events import ToolEventSink


class OllamaClient(LLMClient):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout_seconds: float = 30.0,
        max_concurrency: int | None = None,
        auth_token: str | None = None,
        secondary_base_url: str | None = None,
        overflow_enabled: bool = False,
        overflow_primary_max_inflight: int = 0,
    ):
        self._base_url = base_url.rstrip("/")
        self._secondary_base_url = (secondary_base_url or "").rstrip("/") or None
        self._model = model
        self._timeout = timeout_seconds
        self._max_concurrency = int(max_concurrency) if max_concurrency is not None else None
        self._auth_token = auth_token or None
        self._overflow_enabled = bool(overflow_enabled)
        self._overflow_primary_max_inflight = max(0, int(overflow_primary_max_inflight or 0))
        self._last_error = ""

    @property
    def last_error(self) -> str:
        return str(self._last_error or "")

    def _request_endpoints(self) -> list[str]:
        """Resolve ordered endpoints for this request.

        Default behavior: CPU/primary first, then GPU/secondary fallback.
        Overload behavior: if enabled and in-flight on primary exceeds threshold,
        prefer secondary first for this request.
        """
        if not self._overflow_enabled or not self._secondary_base_url:
            return [self._base_url]

        if (
            self._overflow_primary_max_inflight > 0
            and _get_inflight(self._base_url) >= self._overflow_primary_max_inflight
        ):
            return [self._secondary_base_url, self._base_url]
        return [self._base_url, self._secondary_base_url]

    async def _generate_once(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        sem: asyncio.Semaphore | None,
    ) -> dict:
        async def _request_with_fallback(client: httpx.AsyncClient) -> dict:
            try:
                r = await client.post(f"{endpoint}/api/generate", json=payload, headers=headers)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as exc:
                if exc.response is None or exc.response.status_code != 404:
                    raise
                # Compatibility fallback: some runtimes expose /api/chat but not /api/generate.
                chat_payload = {
                    "model": payload.get("model", self._model),
                    "messages": [{"role": "user", "content": str(payload.get("prompt", ""))}],
                    "stream": False,
                    "keep_alive": payload.get("keep_alive", -1),
                    "options": payload.get("options", {}),
                }
                if payload.get("format") == "json":
                    chat_payload["format"] = "json"
                try:
                    chat = await client.post(f"{endpoint}/api/chat", json=chat_payload, headers=headers)
                    chat.raise_for_status()
                    data = chat.json()
                    content = str((data.get("message") or {}).get("content") or "")
                    return {"response": content}
                except httpx.HTTPStatusError as chat_exc:
                    if chat_exc.response is None or chat_exc.response.status_code != 404:
                        raise
                    # Secondary compatibility fallback for OpenAI-compatible runtimes.
                    openai_payload = {
                        "model": chat_payload.get("model", self._model),
                        "messages": chat_payload.get("messages", []),
                        "temperature": float((payload.get("options") or {}).get("temperature", 0.2)),
                        "stream": False,
                    }
                    openai = await client.post(
                        f"{endpoint}/v1/chat/completions", json=openai_payload, headers=headers
                    )
                    openai.raise_for_status()
                    data = openai.json()
                    choices = data.get("choices") or []
                    message = choices[0].get("message") if choices else {}
                    content = str((message or {}).get("content") or "")
                    return {"response": content}

        _inflight_inc(endpoint)
        try:
            if sem is None:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    return await _request_with_fallback(client)
            async with sem:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    return await _request_with_fallback(client)
        finally:
            _inflight_dec(endpoint)

    async def complete_json(
        self,
        *,
        prompt: str,
        schema_hint: dict[str, Any],
        tool_event_sink: ToolEventSink | None = None,
    ) -> dict[str, Any]:
        # Ollama /api/generate supports JSON mode via "format": "json".
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": -1,
            "options": {
                "temperature": 0.2,
            },
        }
        sem = None
        if self._max_concurrency is not None:
            sem = await _get_semaphore(self._max_concurrency)

        t0 = time.perf_counter()

        if tool_event_sink is not None:
            try:
                await tool_event_sink(
                    {
                        "event_type": "tool_call",
                        "summary_text": f"ollama.generate model={self._model} prompt_chars={len(prompt or '')}",
                        "payload": {
                            "tool": "ollama.generate",
                            "base_url": self._base_url,
                            "model": self._model,
                            "prompt_chars": len(prompt or ""),
                            "schema_hint_keys": sorted(list((schema_hint or {}).keys())),
                        },
                    }
                )
            except Exception:
                pass

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        # For long-running generation windows (>=90s), multiple retries can
        # exceed caller HTTP timeouts and surface as transport failures.
        max_retries = 1 if self._timeout >= 90 else 2
        last_exc: Exception | None = None
        data: dict = {}
        endpoints = self._request_endpoints()
        for endpoint in endpoints:
            for attempt in range(max_retries):
                try:
                    data = await self._generate_once(endpoint=endpoint, payload=payload, headers=headers, sem=sem)
                    last_exc = None
                    break  # success for this endpoint
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 * (2 ** attempt))
            if last_exc is None:
                break

        if last_exc is not None:
            self._last_error = repr(last_exc)
            if tool_event_sink is not None:
                try:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    await tool_event_sink(
                        {
                            "event_type": "tool_result",
                            "summary_text": f"ollama.generate error duration_ms={dt_ms:.1f}",
                            "payload": {"tool": "ollama.generate", "ok": False, "duration_ms": dt_ms},
                        }
                    )
                except Exception:
                    pass
            # Fail closed: callers should fall back to heuristics.
            return {}
        self._last_error = ""

        # Ollama returns {response: "{...}"}
        raw = data.get("response")
        if not raw:
            if tool_event_sink is not None:
                try:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    await tool_event_sink(
                        {
                            "event_type": "tool_result",
                            "summary_text": f"ollama.generate empty duration_ms={dt_ms:.1f}",
                            "payload": {"tool": "ollama.generate", "ok": False, "duration_ms": dt_ms, "empty": True},
                        }
                    )
                except Exception:
                    pass
            return {}
        try:
            parsed = json.loads(raw)
            if tool_event_sink is not None:
                try:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    await tool_event_sink(
                        {
                            "event_type": "tool_result",
                            "summary_text": f"ollama.generate ok duration_ms={dt_ms:.1f}",
                            "payload": {
                                "tool": "ollama.generate",
                                "ok": True,
                                "duration_ms": dt_ms,
                                "result_keys": sorted(list(parsed.keys())) if isinstance(parsed, dict) else [],
                            },
                        }
                    )
                except Exception:
                    pass
            return parsed
        except json.JSONDecodeError:
            if tool_event_sink is not None:
                try:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    await tool_event_sink(
                        {
                            "event_type": "tool_result",
                            "summary_text": f"ollama.generate invalid_json duration_ms={dt_ms:.1f}",
                            "payload": {"tool": "ollama.generate", "ok": False, "duration_ms": dt_ms, "invalid_json": True},
                        }
                    )
                except Exception:
                    pass
            # Best-effort: if it returned already-parsed object or non-json, fail closed.
            return {}

    async def complete_text(
        self,
        *,
        prompt: str,
        num_ctx: int = 32768,
        temperature: float = 0.1,
    ) -> str:
        """Plain-text completion — returns the raw response string.

        Unlike complete_json, no format:json is set, so the model outputs
        natural language suitable for answer generation and judging tasks.
        Returns empty string on any failure (fail-closed).
        """
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        sem = None
        if self._max_concurrency is not None:
            sem = await _get_semaphore(self._max_concurrency)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        # Keep plain-text calls bounded; one long attempt is preferable to
        # several chained retries that outlive upstream request timeouts.
        max_retries = 1 if self._timeout >= 90 else 2
        last_exc: Exception | None = None
        data: dict = {}
        endpoints = self._request_endpoints()
        for endpoint in endpoints:
            for attempt in range(max_retries):
                try:
                    data = await self._generate_once(endpoint=endpoint, payload=payload, headers=headers, sem=sem)
                    last_exc = None
                    break
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 * (2 ** attempt))
            if last_exc is None:
                break

        if last_exc is not None:
            self._last_error = repr(last_exc)
            return ""
        response = str(data.get("response") or "").strip()
        if not response:
            self._last_error = "empty_response"
            return ""
        self._last_error = ""
        return response


_semaphore_lock = asyncio.Lock()
_semaphores: dict[int, asyncio.Semaphore] = {}
_inflight_counts: dict[str, int] = {}


def _get_inflight(endpoint: str) -> int:
    return int(_inflight_counts.get(endpoint, 0))


def _inflight_inc(endpoint: str) -> None:
    _inflight_counts[endpoint] = _get_inflight(endpoint) + 1


def _inflight_dec(endpoint: str) -> None:
    cur = _get_inflight(endpoint)
    if cur <= 1:
        _inflight_counts.pop(endpoint, None)
    else:
        _inflight_counts[endpoint] = cur - 1


async def _get_semaphore(max_concurrency: int) -> asyncio.Semaphore:
    if max_concurrency <= 0:
        # Treat <=0 as "no limit".
        return asyncio.Semaphore(10**9)

    async with _semaphore_lock:
        sem = _semaphores.get(max_concurrency)
        if sem is None:
            sem = asyncio.Semaphore(max_concurrency)
            _semaphores[max_concurrency] = sem
        return sem
