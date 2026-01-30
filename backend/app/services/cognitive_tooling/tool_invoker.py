"""Tool invocation orchestration for Cognitive Loop.

Flow:
- Look up tool spec
- Authorize via PolicyGuard (PermissionChecker-backed)
- Persist a tool_call_logs row with safe summaries
- Invoke the tool via ToolRegistry
- Persist success/failure outcome

Security defaults:
- Inputs/outputs are NOT persisted by default.
- Only length/count metadata is persisted unless ToolSensitivity permits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.services.cognitive_tooling.policy_guard import PolicyGuard, ToolContext
from app.services.cognitive_tooling.tool_call_log_service import ToolCallLogService
from app.services.cognitive_tooling.tool_registry import ToolRegistry, ToolSpec


def _truncate(s: str, max_len: int = 300) -> str:
    s = s or ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _safe_json_len(payload: Any) -> int:
    try:
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        return -1


def _summarize_payload(
    payload: dict[str, Any] | None,
    *,
    allow_persist: bool,
    redacted_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    payload = payload or {}

    if allow_persist:
        redacted = dict(payload)
        for key in redacted_fields:
            if key in redacted:
                redacted[key] = "[REDACTED]"
        return {
            "mode": "persisted",
            "payload": redacted,
        }

    # Default: summary only, no raw values.
    return {
        "mode": "summary",
        "keys_count": len(payload.keys()),
        "json_len": _safe_json_len(payload),
        "has_nested": any(isinstance(v, (dict, list)) for v in payload.values()),
    }


@dataclass(frozen=True)
class ToolInvocationResult:
    status: str  # success|denied|failed
    success: bool
    tool_name: str
    tool_call_id: str | None = None
    output: dict[str, Any] | None = None
    denial_reason: str | None = None
    error: str | None = None
    warnings: list[str] | None = None  # SelfModel reliability warnings, etc.


class ToolInvoker:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        guard: PolicyGuard,
        log_service: ToolCallLogService,
    ) -> None:
        self.registry = registry
        self.guard = guard
        self.log_service = log_service

    async def invoke(
        self,
        *,
        session_id: str,
        iteration_id: str,
        tool_name: str,
        tool_input: dict[str, Any] | None,
        ctx: ToolContext,
        swallow_exceptions: bool = False,
    ) -> ToolInvocationResult:
        started_at = datetime.now(timezone.utc)

        spec: ToolSpec = self.registry.get_spec(tool_name)
        decision = await self.guard.authorize(tool=spec, ctx=ctx)

        safe_input = _summarize_payload(
            tool_input,
            allow_persist=spec.sensitivity.allow_persist_input,
            redacted_fields=spec.sensitivity.redacted_input_fields,
        )

        # Extract reliability warnings from decision details
        warnings: list[str] = []
        if decision.details and isinstance(decision.details, dict):
            reliability_warning = decision.details.get("reliability_warning")
            if reliability_warning:
                warnings.append(str(reliability_warning))

        if not decision.allowed:
            finished_at = datetime.now(timezone.utc)
            row = await self.log_service.create(
                session_id=session_id,
                iteration_id=iteration_id,
                tool_name=tool_name,
                tool_input=safe_input,
                tool_output_summary={"mode": "denied", "reason": _truncate(decision.reason)},
                status="denied",
                denial_reason=_truncate(decision.reason),
                started_at=started_at,
                finished_at=finished_at,
            )
            return ToolInvocationResult(
                status="denied",
                success=False,
                tool_name=tool_name,
                tool_call_id=str(getattr(row, "id", "")) or None,
                output=None,
                denial_reason=_truncate(decision.reason),
                warnings=warnings if warnings else None,
            )

        try:
            output = await self.registry.invoke(tool_name, tool_input or {})
            safe_output = _summarize_payload(
                output,
                allow_persist=spec.sensitivity.allow_persist_output,
                redacted_fields=spec.sensitivity.redacted_output_fields,
            )
            finished_at = datetime.now(timezone.utc)
            row = await self.log_service.create(
                session_id=session_id,
                iteration_id=iteration_id,
                tool_name=tool_name,
                tool_input=safe_input,
                tool_output_summary=safe_output,
                status="success",
                denial_reason=None,
                started_at=started_at,
                warnings=warnings if warnings else None,
                finished_at=finished_at,
            )
            return ToolInvocationResult(
                status="success",
                success=True,
                tool_name=tool_name,
                tool_call_id=str(getattr(row, "id", "")) or None,
                output=output,
                warnings=warnings if warnings else None,
            )
        except Exception as e:
            finished_at = datetime.now(timezone.utc)
            err = _truncate(f"{type(e).__name__}: {e}")
            row = await self.log_service.create(
                session_id=session_id,
                iteration_id=iteration_id,
                tool_name=tool_name,
                tool_input=safe_input,
                tool_output_summary={"mode": "failed", "error": err},
                status="failed",
                denial_reason=None,
                started_at=started_at,
                finished_at=finished_at,
            )
            if swallow_exceptions:
                return ToolInvocationResult(
                    status="failed",
                    success=False,
                    tool_name=tool_name,
                    tool_call_id=str(getattr(row, "id", "")) or None,
                    output=None,
                    error=err,
                    warnings=warnings if warnings else None,
                )
            raise
