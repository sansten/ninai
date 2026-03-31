"""External Connector Service — Phase 47.

Dispatches HTTP actions to external systems (webhooks, PagerDuty, Jira, Slack,
generic REST endpoints).  Implements exponential-backoff retry with a hard
attempt cap.

Design rules:
- Never logs or persists raw payloads; callers store safe summaries via
  ActionExecutionRecord.
- All I/O is async; the caller awaits dispatch().
- httpx is used directly; no third-party SDK dependencies in this layer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 1.0         # seconds; doubles each retry
_THROTTLE_BACKOFF_BASE = 5.0  # longer base for 429 throttle responses
_DEFAULT_TIMEOUT = 10.0     # seconds

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_TRANSIENT_STATUS_CODES = frozenset({500, 502, 503, 504})
_THROTTLED_STATUS_CODE = 429


class RetryClass(str, Enum):
    """Error classification for retry and backoff decisions.

    TRANSIENT  — network errors or 5xx server errors; retry with standard backoff.
    THROTTLED  — 429 Too Many Requests; retry with longer backoff.
    PERMANENT  — 4xx client errors (excl. 429); never retry.
    """
    TRANSIENT = "transient"
    THROTTLED = "throttled"
    PERMANENT = "permanent"

# Action-type to default content-type mapping
_CONTENT_TYPE_MAP: dict[str, str] = {
    "webhook":     "application/json",
    "pagerduty":   "application/json",
    "jira":        "application/json",
    "slack":       "application/json",
    "generic_rest": "application/json",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    status: str                    # "success" | "failed"
    http_status_code: int | None
    attempt_count: int
    error: str | None = None
    response_summary: dict = field(default_factory=dict)
    retry_class: str | None = None  # RetryClass value of the terminal failure, or None on success


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _build_headers(action_type: str, extra_headers: dict | None) -> dict[str, str]:
    content_type = _CONTENT_TYPE_MAP.get(str(action_type).lower(), "application/json")
    headers = {"Content-Type": content_type, "User-Agent": "Ninai-ActionEngine/1.0"}
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})
    return headers


def _backoff_seconds(attempt: int, base: float = _BACKOFF_BASE) -> float:
    """Exponential backoff for transient errors: 1s, 2s, 4s … capped at 30s."""
    return min(base * (2 ** attempt), 30.0)


def _throttle_backoff_seconds(attempt: int, base: float = _THROTTLE_BACKOFF_BASE) -> float:
    """Longer exponential backoff for throttled (429) responses: 5s, 10s, 20s … capped at 60s."""
    return min(base * (2 ** attempt), 60.0)


def _classify_result(status_code: int | None, exc: Exception | None) -> RetryClass:
    """Return the RetryClass for a failed-or-retryable HTTP result.

    Rules:
    - Any exception (network error, timeout) → TRANSIENT
    - 429 → THROTTLED
    - 5xx (500, 502, 503, 504) → TRANSIENT
    - Anything else (4xx, unexpected) → PERMANENT
    """
    if exc is not None:
        return RetryClass.TRANSIENT
    if status_code == _THROTTLED_STATUS_CODE:
        return RetryClass.THROTTLED
    if status_code in _TRANSIENT_STATUS_CODES:
        return RetryClass.TRANSIENT
    return RetryClass.PERMANENT


def _safe_response_summary(status_code: int, text: str) -> dict[str, Any]:
    return {
        "http_status": status_code,
        "body_len": len(text),
        "body_prefix": text[:200] if text else "",
    }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ExternalConnectorService:
    """Async HTTP dispatcher with retry/backoff.

    Designed to be instantiated once and reused (no per-request state).
    The underlying httpx.AsyncClient is created per-dispatch to keep
    lifecycle management simple and avoid connection pool leaks in tests.
    """

    def __init__(
        self,
        *,
        max_attempts: int = _MAX_ATTEMPTS,
        backoff_base: float = _BACKOFF_BASE,
        default_timeout: float = _DEFAULT_TIMEOUT,
        _http_client_factory: Any = None,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_base = float(backoff_base)
        self.default_timeout = float(default_timeout)
        # injectable for tests; if None, real httpx.AsyncClient is used
        self._http_client_factory = _http_client_factory

    async def dispatch(
        self,
        *,
        action_type: str,
        target_url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> DispatchResult:
        """POST payload to target_url with retry logic.

        Parameters
        ----------
        action_type:     one of the registered action types (used for header defaults)
        target_url:      full URL to POST to
        payload:         JSON-serialisable dict to send as request body
        headers:         additional HTTP headers (merged with defaults)
        timeout_seconds: per-request timeout override

        Returns
        -------
        DispatchResult with status, http_status_code, attempt_count, error
        """
        if not _HTTPX_AVAILABLE and self._http_client_factory is None:
            return DispatchResult(
                status="failed",
                http_status_code=None,
                attempt_count=0,
                error="httpx not available and no http_client_factory injected",
            )

        merged_headers = _build_headers(action_type, headers)
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout

        last_error: str | None = None
        last_code: int | None = None
        last_retry_class: RetryClass = RetryClass.TRANSIENT

        for attempt in range(self.max_attempts):
            exc_caught: Exception | None = None
            try:
                result = await self._send_once(
                    target_url=target_url,
                    payload=payload,
                    headers=merged_headers,
                    timeout=timeout,
                )
                last_code = result.get("status_code")
                summary = result.get("summary", {})

                if last_code is not None and last_code < 400:
                    return DispatchResult(
                        status="success",
                        http_status_code=last_code,
                        attempt_count=attempt + 1,
                        response_summary=summary,
                    )

                last_retry_class = _classify_result(last_code, None)

                # PERMANENT errors: do not retry — fail immediately.
                if last_retry_class == RetryClass.PERMANENT:
                    return DispatchResult(
                        status="failed",
                        http_status_code=last_code,
                        attempt_count=attempt + 1,
                        error=f"Permanent HTTP {last_code} — no retry",
                        response_summary=summary,
                        retry_class=last_retry_class.value,
                    )

                last_error = f"HTTP {last_code} ({last_retry_class.value})"

            except Exception as exc:
                exc_caught = exc
                last_retry_class = _classify_result(None, exc)
                last_error = f"{type(exc).__name__}: {exc}"
                logger.debug("dispatch attempt %d/%d failed: %s", attempt + 1, self.max_attempts, last_error)

            if attempt + 1 < self.max_attempts:
                # Use longer backoff for throttled responses.
                if last_retry_class == RetryClass.THROTTLED:
                    await asyncio.sleep(_throttle_backoff_seconds(attempt, self.backoff_base * 5))
                else:
                    await asyncio.sleep(_backoff_seconds(attempt, self.backoff_base))

        return DispatchResult(
            status="failed",
            http_status_code=last_code,
            attempt_count=self.max_attempts,
            error=last_error or "max retries exhausted",
            retry_class=last_retry_class.value,
        )

    async def _send_once(
        self,
        *,
        target_url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        """Send one HTTP POST.  Raises on network error."""
        if self._http_client_factory is not None:
            client = self._http_client_factory()
            resp = await client.post(target_url, json=payload, headers=headers, timeout=timeout)
            code = getattr(resp, "status_code", None)
            text = getattr(resp, "text", "") or ""
            return {"status_code": code, "summary": _safe_response_summary(code or 0, text)}

        # Real httpx path
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(target_url, json=payload, headers=headers)
            return {
                "status_code": resp.status_code,
                "summary": _safe_response_summary(resp.status_code, resp.text),
            }
