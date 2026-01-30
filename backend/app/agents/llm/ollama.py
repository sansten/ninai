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
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._max_concurrency = int(max_concurrency) if max_concurrency is not None else None

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

        try:
            if sem is None:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    r = await client.post(f"{self._base_url}/api/generate", json=payload)
                    r.raise_for_status()
                    data = r.json()
            else:
                async with sem:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        r = await client.post(f"{self._base_url}/api/generate", json=payload)
                        r.raise_for_status()
                        data = r.json()
        except (httpx.HTTPError, OSError, ValueError):
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


_semaphore_lock = asyncio.Lock()
_semaphores: dict[int, asyncio.Semaphore] = {}


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
