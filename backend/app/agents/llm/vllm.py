from __future__ import annotations

import asyncio
import json
from typing import Any
import time

import httpx

from app.agents.llm.base import LLMClient
from app.agents.llm.tool_events import ToolEventSink


class VLLMClient(LLMClient):
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
        fallback_models: list[str] | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._secondary_base_url = (secondary_base_url or "").rstrip("/") or None
        self._model = model
        self._fallback_models = [
            str(item).strip()
            for item in (fallback_models or [])
            if str(item).strip() and str(item).strip() != str(model).strip()
        ]
        self._timeout = timeout_seconds
        self._max_concurrency = int(max_concurrency) if max_concurrency is not None else None
        self._auth_token = auth_token or None
        self._overflow_enabled = bool(overflow_enabled)
        self._overflow_primary_max_inflight = max(0, int(overflow_primary_max_inflight or 0))
        self._last_error = ""
        self._last_model_used = str(model).strip()
        self._last_endpoint_used = self._base_url

    @property
    def last_error(self) -> str:
        return str(self._last_error or "")

    @property
    def last_model_used(self) -> str:
        return str(self._last_model_used or self._model or "")

    @property
    def last_endpoint_used(self) -> str:
        return str(self._last_endpoint_used or self._base_url or "")

    def _request_endpoints(self) -> list[str]:
        """Resolve ordered endpoints for this request.

        Default behavior: primary first, then secondary as a failover target
        when one is configured. Overload behavior: if enabled and in-flight on
        primary exceeds threshold, prefer secondary first for this request.
        """
        if not self._secondary_base_url:
            return [self._base_url]

        if (
            self._overflow_enabled
            and
            self._overflow_primary_max_inflight > 0
            and _get_inflight(self._base_url) >= self._overflow_primary_max_inflight
        ):
            return [self._secondary_base_url, self._base_url]
        return [self._base_url, self._secondary_base_url]

    async def _attempt_generation(
        self,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
        sem: asyncio.Semaphore | None,
        max_retries: int,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for endpoint in self._request_endpoints():
            candidate_models = await self._candidate_models_for_endpoint(endpoint, headers)
            for resolved_model in candidate_models:
                payload["model"] = resolved_model
                for attempt in range(max_retries):
                    try:
                        data = await self._generate_once(endpoint=endpoint, payload=payload, headers=headers, sem=sem)
                        self._last_model_used = resolved_model
                        self._last_endpoint_used = endpoint
                        return data
                    except (httpx.HTTPError, OSError, ValueError) as exc:
                        last_exc = exc
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.5 * (2 ** attempt))
        if last_exc is not None:
            raise last_exc
        return {}

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

    async def _available_models(self, endpoint: str, headers: dict[str, str]) -> list[str]:
        cached = _model_inventory_cache.get(endpoint)
        if cached is not None:
            return list(cached)

        async with _model_inventory_lock:
            cached = _model_inventory_cache.get(endpoint)
            if cached is not None:
                return list(cached)

            try:
                async with httpx.AsyncClient(timeout=min(self._timeout, 10.0)) as client:
                    resp = await client.get(f"{endpoint}/api/tags", headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                models = [
                    str(item.get("name") or "").strip()
                    for item in (data.get("models") or [])
                    if str(item.get("name") or "").strip()
                ]
            except Exception:
                models = []

            _model_inventory_cache[endpoint] = list(models)
            return list(models)

    async def _resolve_model_for_endpoint(self, endpoint: str, headers: dict[str, str]) -> str:
        available_models = await self._available_models(endpoint, headers)
        if not available_models:
            return self._model
        if self._model in available_models:
            return self._model
        for candidate in self._fallback_models:
            if candidate in available_models:
                return candidate
        return available_models[0]

    async def _candidate_models_for_endpoint(self, endpoint: str, headers: dict[str, str]) -> list[str]:
        available_models = await self._available_models(endpoint, headers)
        ordered: list[str] = []
        seen: set[str] = set()

        def _add(model_name: str) -> None:
            normalized = str(model_name or "").strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            ordered.append(normalized)

        if available_models:
            if self._model in available_models:
                _add(self._model)
            for candidate in self._fallback_models:
                if candidate in available_models:
                    _add(candidate)
            for candidate in available_models:
                _add(candidate)
            return ordered

        _add(self._model)
        for candidate in self._fallback_models:
            _add(candidate)
        return ordered

    async def complete_json(
        self,
        *,
        prompt: str,
        schema_hint: dict[str, Any],
        tool_event_sink: ToolEventSink | None = None,
    ) -> dict[str, Any]:
        # vLLM /api/generate supports JSON mode via "format": "json".
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
                        "summary_text": f"llm.generate model={self._model} prompt_chars={len(prompt or '')}",
                        "payload": {
                            "tool": "llm.generate",
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
        try:
            data = await self._attempt_generation(
                payload=payload,
                headers=headers,
                sem=sem,
                max_retries=max_retries,
            )
        except (httpx.HTTPError, OSError, ValueError) as last_exc:
            self._last_error = repr(last_exc)
            if tool_event_sink is not None:
                try:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    await tool_event_sink(
                        {
                            "event_type": "tool_result",
                            "summary_text": f"llm.generate error duration_ms={dt_ms:.1f}",
                            "payload": {"tool": "llm.generate", "ok": False, "duration_ms": dt_ms},
                        }
                    )
                except Exception:
                    pass
            # Fail closed: callers should fall back to heuristics.
            return {}
        self._last_error = ""

        # vLLM returns {response: "{...}"}
        raw = data.get("response")
        if not raw:
            if tool_event_sink is not None:
                try:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    await tool_event_sink(
                        {
                            "event_type": "tool_result",
                            "summary_text": f"llm.generate empty duration_ms={dt_ms:.1f}",
                            "payload": {"tool": "llm.generate", "ok": False, "duration_ms": dt_ms, "empty": True},
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
                            "summary_text": f"llm.generate ok duration_ms={dt_ms:.1f}",
                            "payload": {
                                "tool": "llm.generate",
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
                            "summary_text": f"llm.generate invalid_json duration_ms={dt_ms:.1f}",
                            "payload": {"tool": "llm.generate", "ok": False, "duration_ms": dt_ms, "invalid_json": True},
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
        try:
            data = await self._attempt_generation(
                payload=payload,
                headers=headers,
                sem=sem,
                max_retries=max_retries,
            )
        except (httpx.HTTPError, OSError, ValueError) as last_exc:
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
_model_inventory_lock = asyncio.Lock()
_model_inventory_cache: dict[str, list[str]] = {}


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
